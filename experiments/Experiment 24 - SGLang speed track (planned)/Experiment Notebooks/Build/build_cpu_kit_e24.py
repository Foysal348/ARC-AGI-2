"""E24 CPU kit builder (WP5, rev 2, 2026-10-30): Build/cpu_kernel_e24/{q27-e24-cpu-kit.ipynb, kernel-metadata.json}.

The kit runs on a Kaggle CPU kernel (pinned image, no GPU quota, internet off) with the REAL /kaggle/input mounts of every
DESIGN §2 input, and checks what a FAKE local run cannot (BUILD_SPEC §6 + the rev-2 WP5 overrides (a)-(h);
kaggle_inputs_kits.md §6.2 checklist; review kaggle_feasibility F9):
  1  every DESIGN §2 input mounted at its exact path (a wrong model variation drops a mount silently: E23 kit v1);
  2  the offline SGLang 0.5.21 and vLLM 0.32 installs through the notebook's OWN CELL 1 + CELL 2 (sgl_install /
     vllm_install run as the notebook runs them; E14_LOADPATH=0: a 31 GB CPU box cannot stage 23 GB) (e);
  3  the three notebooks in FAKE mode on the real mounts (E24_FORCE_CONDITIONAL=1, small E24_TIME_SCALE, fresh TMP/OUT:
     the E23 kit v1 bug): input discovery, identity, FP8 vs INT4 id equality with the REAL tokenizers, and all of
     run_local_e24.py's positive checks (arm AND window statuses: E23's all_ok ignored them) (g);
  4  every runtime SGLang argv (R["argv"] of the FAKE runs + the install-time catalog + plan_argvs_e24.json) through the
     REAL 0.5.21 parser in the venv (exact option strings + prepare_server_args; method syntax_only on CPU), and the
     s1_a / s1_b argv through the REAL vLLM 0.32 parser (E21 flagcheck, stub platform) (b) (c);
  5  patches on the INSTALLED package: check -> apply each E24 name and the combos -> compile -> revert -> pristine sha256
     (43559 alone refused);
  6  AST of every hook target (e24_hook TARGETS / TARGET_METHODS) in the installed package, signatures = the 0.5.21
     reference source (f);
  7  every planned SGLANG_* env var present in the installed srt/environ.py (else read elsewhere in the package) (f);
  8  the NGRAM load_jit C++ module compiles and loads in the venv (f);
  9  hook import smoke in the venv python (sitecustomize active, env record written; `import sglang` may fail on CPU:
     recorded, not failed);
  10 fault modes on Linux (run_local_e24 scenarios: diverge, leak, mamba_assert, unsupported flag, logprob_oom,
     long-prefill census, crash_start, hard-stop watchdog) (d);
  11 every E24 unit-test file on Linux (+ the mock-input self-test, + a Thread-attribute shadowing scan) (a);
  12 the competition-rerun fallback submission format.
Packaging (h): every kit file carries its sha256 (META["files"]); the three notebooks travel as the SMOKE code cells
(their module blobs stored once, as the notebook has them) + the two baked lines (asserted here); the lab / patcher /
hook / unit / E14 modules are taken from the notebook's own embedded copies (BLOBS, sha-checked), so the kit tests exactly
the bytes that ship. --mode auto (default): every file gzip+base64 inside the notebook when that stays < 400 kB; otherwise
(the WP2 fakes alone are ~225 kB raw) the review-F9 fallback: ONE deterministic zip `cpu_kernel_e24/kit_data/
q27x_e24_kit_<sha12>.bin` + dataset-metadata.json (kit dataset foysalemonshanto/q27-e24-cpu-kit-data, placeholder
account), pinned by sha256 in a small notebook; upload that dataset version before pushing the kernel. Notebook < 400 kB
is asserted in both modes. Report: /kaggle/working/e24_cpu_kit_report.json (all_ok over every check) +
e24_cpu_kit_logs_tar.bin.

usage: /opt/homebrew/bin/python3.12 build_cpu_kit_e24.py [--nb-dir DIR] [--out DIR] [--sglang-src DIR]
Run it after every notebook rebuild. Kernel id foysalemonshanto/q27-e24-cpu-kit is a placeholder: the user picks the
account (the private models must be shared with it). Never pushed by the builder.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # Experiment Notebooks/Build
E24 = HERE.parents[1]
EXPS = E24.parent
KIT_REV = "e24 cpu kit rev 1 (WP5, DESIGN rev 2, 2026-10-30)"
VARIANTS = {"SMOKE": ("SMOKE", "ALL"), "FULL_A": ("FULL", "A"), "FULL_B": ("FULL", "B")}
NB_FILE = "Q27-XHIGH-E24_SGLANG_SPEED_{}.ipynb"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
KIT_ID = "foysalemonshanto/q27-e24-cpu-kit"            # placeholder account (E23 convention: <owner>/q27-e2x-cpu-kit)
KIT_DATASET_ID = "foysalemonshanto/q27-e24-cpu-kit-data"  # dataset mode only (kaggle_inputs_kits.md §4.2 naming)
DATASETS = ["foysalemonshanto/sglang-0-5-21-cu13-wheelhouse", "foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"]
MODELS = ["foysalemonshanto/qwen3-8-27b-int4-redhat/PyTorch/w4a16-g128/1",
          "foysalemonshanto/qwen3-8-27b-dflash2/PyTorch/bf16/1",
          "foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"]
COMPETITIONS = ["arc-prize-2026-arc-agi-2"]
INPUTS = {"SGL_WHEELHOUSE": "datasets/foysalemonshanto/sglang-0-5-21-cu13-wheelhouse",          # DESIGN §2 mount paths
          "VLLM_WHEELHOUSE": "datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse",
          "INT4": "models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1",
          "DFLASH2": "models/foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1",
          "FP8": "models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1",
          "ARC": "competitions/arc-prize-2026-arc-agi-2"}
KIT_MAX_BYTES = 400_000
SGL_SRC_DEFAULT = ("/private/tmp/claude-501/-Users-md-hamidhosen-Documents-Kaggle-ARC-AGI-2/e21dd5ac-4964-44ae-8fe0-b7d1156db571"
                   "/scratchpad/e24/downloads/sglang_source_0521/src/sglang")
# the override (f) targets: each must stay in the hook's tables (a hook edit cannot drop one silently)
REQUIRED_TARGETS = [("MambaComponent", "_alloc_mamba_slot"), ("SchedulerBatchResultProcessor", "_mamba_check_track_boundary"),
                    ("DFlashWorkerV2", "forward_batch_generation"), ("CandidateSelector", "sample_path"),
                    ("_grouped_conv", None), ("Scheduler", "__init__")]
BAKE_RE = {k: re.compile(r'(os\.environ\.get\(\s*"Q27X_E24_%s"\s*,\s*")([A-Z_]+)("\s*\))' % k) for k in ("MODE", "PART")}


def bake(code, mode, part):
    for k, v in (("MODE", mode), ("PART", part)):
        code, n = BAKE_RE[k].subn(lambda g: g.group(1) + v + g.group(3), code, count=1)
        assert n == 1, f"baked Q27X_E24_{k} default not found exactly once"
    return code


def code_cells(p):
    return ["".join(c["source"]) for c in json.loads(Path(p).read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def joined(cells):
    return "".join(c + "\n\n" for c in cells)


MODULE_EXTRACT = r'''
def extract_modules(code):
    """The notebook's own embedded modules: BLOBS {key: (file, sha256, b64 gzip)} (+ LAB_SRC text when the lab is
    embedded as plain text), each sha256-checked against BUILD_SHAS (a mismatch raises). Returns ({file: bytes},
    BUILD_SHAS, [files the notebook was built without])."""
    vals = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("LAB_SRC", "BLOBS", "BUILD_SHAS"):
            try:
                vals[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    shas, files, missing = vals.get("BUILD_SHAS") or {}, {}, []
    for key, (name, sha, b64) in (vals.get("BLOBS") or {}).items():
        if not b64 or sha == "MISSING":
            missing.append(name)
            continue
        data = gzip.decompress(base64.b64decode(b64))
        assert hashlib.sha256(data).hexdigest() == sha == shas.get(key, sha), f"notebook blob {name}: sha256 mismatch"
        files[name] = data
    if isinstance(vals.get("LAB_SRC"), str):
        files.setdefault("e24_lab.py", vals["LAB_SRC"].encode("utf-8"))
    assert "e24_lab.py" in files and hashlib.sha256(files["e24_lab.py"]).hexdigest() == shas.get("lab"), "lab sha256 mismatch"
    missing += [n for n in ("e24_sglang_patches.py", "e24_hook.py") if n not in files and n not in missing]
    return files, shas, missing
'''
exec(MODULE_EXTRACT)


def hook_tables(text):
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("TARGETS", "TARGET_METHODS"):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def ref_signatures(src, tables):
    """{"Target.method": [arg names]} of every hook target in the reference 0.5.21 source (None where absent)."""
    sigs = {}
    src = Path(src)
    for tgt, (mod, attr) in tables["TARGETS"].items():
        if not mod.startswith("sglang."):
            continue
        f = src / (mod.split(".", 1)[1].replace(".", "/") + ".py")
        if not f.is_file():
            sigs[tgt] = None
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        top = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
        node = top.get(attr)
        if node is None:
            sigs[tgt] = None
            continue
        if isinstance(node, ast.ClassDef):
            meth = {n.name: n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for m in tables["TARGET_METHODS"].get(tgt) or ():
                sigs[f"{tgt}.{m}"] = [a.arg for a in meth[m].args.args] if m in meth else None
        else:
            sigs[tgt] = [a.arg for a in node.args.args]
    return sigs


def emb(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return [hashlib.sha256(data).hexdigest(), base64.b64encode(gzip.compress(data, 9, mtime=0)).decode("ascii")]


KIT_MAIN = r'''# E24 CPU kit main (WP5 rev 1, DESIGN rev 2): a Kaggle CPU kernel on the pinned image with the REAL /kaggle/input mounts.
# Stages: 0 decode | 1 mounts | 2 offline SGLang 0.5.21 + vLLM 0.32 installs through the notebook's own CELL 1 + CELL 2 |
# 5 patches on the installed package | 6 hook-target AST | 8 NGRAM load_jit | 9 hook import smoke | 3 the three notebooks
# in FAKE mode on the real mounts | 4 every runtime argv through the real parsers | 7 SGLANG_* env vars in environ.py |
# 10 fault modes | 11 unit tests on Linux | 12 competition-rerun format. Report: /kaggle/working/e24_cpu_kit_report.json.
import ast, base64, gzip, hashlib, json, os, re, shutil, subprocess, sys, tarfile, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
T0 = time.time()
INP = Path(os.environ.get("E24KIT_INPUT", "/kaggle/input"))
WORKING = Path(os.environ.get("E24KIT_WORKING", "/kaggle/working"))
K = Path(os.environ.get("E24KIT_ROOT", "/tmp/e24kit"))
LOCAL = os.environ.get("E24KIT_LOCAL") == "1"        # local dry run of the kit itself (no installs, no rerun); never on Kaggle
SCALE = float(os.environ.get("E24KIT_SCALE") or 0.02)
NEGS = [x for x in (os.environ.get("E24KIT_NEGATIVES") or "all").split(",") if x]                  # "none" = no fault modes
VARS = [x for x in (os.environ.get("E24KIT_VARIANTS") or "SMOKE,FULL_A,FULL_B").split(",") if x and x != "none"]
# EMB (embedded files, or None), PAYLOAD (the kit dataset's files, or None) and META (build facts) are globals passed in
# by the bootstrap cell (embedded mode: EMB; dataset mode: PAYLOAD from the sha256-pinned kit .bin)
shutil.rmtree(K, ignore_errors=True)
K.mkdir(parents=True)
(K / "logs").mkdir()
WORKING.mkdir(parents=True, exist_ok=True)
REP = {"kit": META["kit"], "built": META["built"], "mode": "dataset" if PAYLOAD is not None else "embedded", "local": LOCAL, "scale": SCALE, "checks": {}, "stages": {}, "info": {}, "runs": {}}
S = {}
E24K = K / "repo" / "experiments" / "Experiment 24 - SGLang speed track (planned)"
KITD = K / "kit"


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def save():
    (WORKING / "e24_cpu_kit_report.json").write_text(json.dumps(REP, indent=1, default=str))


def check(name, ok, info=None):
    REP["checks"][name] = {"ok": bool(ok), "info": info}
    log("PASS" if ok else "FAIL", name, "" if ok else json.dumps(info, default=str)[:700])
    return bool(ok)


def stage(name, fn):
    t = time.time()
    log(">>>", name)
    try:
        fn()
        REP["stages"][name] = {"sec": round(time.time() - t, 1)}
    except BaseException as e:
        REP["stages"][name] = {"sec": round(time.time() - t, 1), "error": f"{type(e).__name__}: {e}"[:800],
                               "tb": traceback.format_exc()[-3000:]}
        check(f"{name}: stage completed", False, REP["stages"][name]["error"])
    save()


def sh(cmd, timeout=600, env=None, cwd=None, name=None):
    t = time.time()
    try:
        r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        rc, out = r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, out = -9, (e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + "\nKIT TIMEOUT"
    if name:
        (K / "logs" / f"{name}.log").write_text(out)
    return rc, out, round(time.time() - t, 1)


def jline(out, tag):
    for l in reversed((out or "").splitlines()):
        if l.startswith(tag + " "):
            try:
                return json.loads(l[len(tag) + 1:])
            except ValueError:
                return None
    return None


def unpack(rel):
    """A kit file: from the kit dataset .bin (PAYLOAD; the bootstrap checked the .bin sha256) or the notebook's own
    gzip+base64 (EMB); either way its sha256 must equal the build's (META["files"])."""
    data = PAYLOAD[rel] if PAYLOAD is not None else gzip.decompress(base64.b64decode(EMB[rel][1]))
    assert hashlib.sha256(data).hexdigest() == META["files"][rel], f"kit file {rel}: sha256 mismatch"
    return data


def clean_env(**kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E24_", "FAKE_", "Q27X_", "E14_", "KAGGLE_IS_COMPETITION"))}
    env.update({k: str(v) for k, v in kw.items()})
    return env


def df(p):
    try:
        u = shutil.disk_usage(p)
        return {"free_gb": round(u.free / 2 ** 30, 1), "total_gb": round(u.total / 2 ** 30, 1)}
    except Exception as e:
        return repr(e)


BAKE = {k: re.compile(r'(os\.environ\.get\(\s*"Q27X_E24_%s"\s*,\s*")([A-Z_]+)("\s*\))' % k) for k in ("MODE", "PART")}


def bake(code, m, p):
    for k, v in (("MODE", m), ("PART", p)):
        code, n = BAKE[k].subn(lambda g: g.group(1) + v + g.group(3), code, count=1)
        assert n == 1, k
    return code
__MODULE_EXTRACT__

# ---------------------------------------------------------------- 0. decode the embedded tree + the notebook's own modules
def s0():
    for rel in META["files"]:
        dst = (KITD / rel[4:]) if rel.startswith("kit/") else E24K / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(unpack(rel))
    check("0a every kit file unpacked, sha256 verified", True, {"files": len(META["files"]), "mode": REP["mode"]})
    check("0d the kit was built with every work-package file (not a structure-only build)", not META["missing_at_build"],
          META["missing_at_build"])
    cells = json.loads(unpack("kit/nb_cells_SMOKE.json"))
    for tok, b64 in META["nb_blobs"].items():             # the notebook's own gzip+b64 module blobs, stored once
        cells = [c.replace(tok, b64) for c in cells]
    S["cells"] = cells
    S["code"] = {"SMOKE": "".join(c + "\n\n" for c in cells)}
    for v, (m, p) in (("FULL_A", ("FULL", "A")), ("FULL_B", ("FULL", "B"))):
        S["code"][v] = bake(S["code"]["SMOKE"], m, p)
    ok = {v: hashlib.sha256(c.encode("utf-8")).hexdigest() == META["nb_code_sha256"][v] for v, c in S["code"].items()}
    check("0b the three notebooks rebuilt from SMOKE + the baked MODE / PART lines equal the built notebooks", all(ok.values()), ok)
    mods, shas, nb_missing = extract_modules(S["code"]["SMOKE"])
    for name, data in mods.items():
        d = E24K / "Base Engine" / "src_e14_modules" if name.startswith("e14_") else E24K / "src"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(data)
    check("0c modules taken from the notebook's own embedded copies (BLOBS, sha256 = BUILD_SHAS = the kit build), none missing",
          shas == META["build_shas"] and not nb_missing, {"modules": sorted(mods), "missing_in_notebook": nb_missing,
                                                          "build_shas_equal": shas == META["build_shas"]})
    for p in (E24K / "Base Engine" / "src_e14_modules", E24K / "src", E24K / "src" / "harness"):
        sys.path.insert(0, str(p))
    global L, P, RL
    import e24_lab as L
    import e24_sglang_patches as P
    import run_local_e24 as RL


# ---------------------------------------------------------------- 1. the DESIGN §2 mounts
def s1():
    mounts = {}
    pm = Path("/proc/mounts").read_text().splitlines() if Path("/proc/mounts").exists() else []
    for k, rel in META["inputs"].items():
        p = INP / rel
        files = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else []
        fs = max(((l.split()[1], l.split()[2]) for l in pm if len(l.split()) > 2 and str(p).startswith(l.split()[1])),
                 key=lambda x: len(x[0]), default=(None, None))[1]
        mounts[k] = {"path": str(p), "exists": p.is_dir(), "files": len(files), "gb": round(sum(f.stat().st_size for f in files) / 1e9, 2),
                     "fstype": fs}
    REP["info"]["mounts"] = mounts
    REP["info"]["host"] = {"python": sys.version.split()[0], "nproc": os.cpu_count(), "tmp": df("/tmp"), "working": df(str(WORKING)),
                           "shm": df("/dev/shm")}
    check("1 every DESIGN §2 input mounted at its exact path (a wrong handle variation drops a mount silently)",
          all(m["exists"] and m["files"] > 0 for m in mounts.values()), mounts)


# ---------------------------------------------------------------- 2. installs through the notebook's own CELL 1 + CELL 2
DRIVER = r"""
import json, os, sys, time, traceback
cells = json.load(open(sys.argv[1]))
pick = [c for c in cells if "CELL 1 " in c[:400] or "CELL 2 " in c[:400]]
pick = pick if len(pick) == 2 else cells[:2]
G = {"__name__": "__main__"}
t0, ok, err = time.time(), True, None
try:
    exec(compile(pick[0], "<notebook CELL 1>", "exec"), G)
    if G.get("L") is not None:            # a 31 GB-RAM CPU box: no FP8 page-cache prefetch (installs are untouched)
        G["L"].prefetch = lambda *a, **k: {"skipped": "CPU kit: no FP8 page-cache prefetch"}
    exec(compile(pick[1], "<notebook CELL 2>", "exec"), G)
except BaseException:
    ok, err = False, traceback.format_exc()[-3000:]
R = G.get("R") or {}
cat = {a: {"engine": s.get("engine"), "flags": s.get("flags"), "section": s.get("section")} for a, s in (G.get("CAT") or {}).items()}
out = {"ok": ok, "err": err, "sec": round(time.time() - t0, 1), "engines": G.get("ENGINES"), "setup": R.get("setup"),
       "inputs": R.get("inputs"), "identity": R.get("identity"), "gates": R.get("gates"), "errors": R.get("errors"),
       "psite": G.get("PSITE"), "patch_broken": G.get("PATCH_BROKEN"), "catalog": cat, "variant": G.get("VARIANT")}
print("E24KIT_INSTALL " + json.dumps(out, default=str), flush=True)
os._exit(0)
"""


def s2():
    if LOCAL:
        site = os.environ.get("E24KIT_SGLANG_SITE")
        S["sgl"] = {"vpy": sys.executable, "env": clean_env(), "version": "LOCAL", "site": str(Path(site).parent) if site else None}
        S["sgl_pkg"], S["vllm"] = site, None
        REP["info"]["install"] = "LOCAL dry run: no installs"
        return
    (K / "nb_cells.json").write_text(json.dumps(S["cells"]))
    (K / "install_driver.py").write_text(DRIVER)
    base = K / "inst"
    for d in ("out", "tmp", "home"):
        (base / d).mkdir(parents=True, exist_ok=True)
    env = clean_env(E24_INPUT=INP, E24_OUT=base / "out", E24_TMP=base / "tmp", HOME=base / "home", E14_LOADPATH="0",
                    PYTHONUNBUFFERED="1")
    REP["info"]["disk_before_install"] = df("/tmp")
    rc, out, sec = sh([sys.executable, K / "install_driver.py", K / "nb_cells.json"], timeout=5400, env=env, name="install_driver")
    REP["info"]["disk_after_install"] = df("/tmp")
    j = jline(out, "E24KIT_INSTALL") or {}
    E = j.get("engines") or {}
    S["install"], S["sgl"], S["vllm"] = j, E.get("sglang") or {}, E.get("vllm") or {}
    S["sgl_pkg"] = j.get("psite") if not j.get("patch_broken") else None
    S["catalog"] = j.get("catalog") or {}
    REP["info"]["install"] = {"rc": rc, "sec": sec, "ok": j.get("ok"), "err": j.get("err"),
                              "sglang": {k: v for k, v in S["sgl"].items() if k != "env"},
                              "vllm": {k: v for k, v in S["vllm"].items() if k != "env"}, "setup": j.get("setup")}
    check("2a the notebook's own CELL 1 + CELL 2 ran on the real mounts (sgl_install / vllm_install as the notebook calls them)",
          rc == 0 and j.get("ok"), {"rc": rc, "err": j.get("err"), "tail": out[-1500:]})
    check("2b offline SGLang 0.5.21 venv (wheelhouse, '+'-stripped names restored by sha256)", S["sgl"].get("version") == "0.5.21",
          {k: S["sgl"].get(k) for k in ("failed", "version", "vpy", "cuda_home", "shims", "sec")})
    vf = str(S["vllm"].get("failed") or "")
    check("2c offline vLLM 0.32 venv (install_wheelhouse.py), or a recorded kit-disk reason",
          str(S["vllm"].get("version", "")).startswith("0.32") or "No space left" in vf,
          {k: S["vllm"].get(k) for k in ("failed", "version", "vpy")})
    check("2d CELL 1 input gate GO on the real mounts", (j.get("gates") or {}).get("inputs") == "GO",
          {"gates": j.get("gates"), "inputs": j.get("inputs")})
    idn = j.get("identity") or {}
    check("2e checkpoint identity recorded for fp8 / int4 / dflash2 (+ tokenizer sha256)",
          all(idn.get(k) for k in ("fp8", "int4", "dflash2")), {k: idn.get(k) for k in ("fp8", "int4", "dflash2", "tokenizer_sha256")})
    check("2f patch site = the installed sglang package, pristine at start", bool(S["sgl_pkg"]),
          {"psite": j.get("psite"), "patch_broken": j.get("patch_broken")})
    bc = ((j.get("setup") or {}).get("base_check")) or {}
    check("2g BASE argv = DESIGN §2 inside the notebook (real staged / mount paths)", bc.get("ok") is True, bc)


# ---------------------------------------------------------------- 3. the three notebooks in FAKE mode on the real mounts
def s3():
    if S.get("sgl_pkg"):
        os.environ["E24_SGLANG_SRC"] = str(S["sgl_pkg"])      # E24_FAKE_SGLANG_SITE = a copy of the INSTALLED package
    S["runs"] = {}
    for v in VARS:
        r = RL.run_notebook(f"pos_{v}", S["code"][v], K / "runs", INP, v, scale=SCALE, site=bool(S.get("sgl_pkg")),
                            timeout_min=180, fake_dir=E24K / "src" / "harness")
        S["runs"][v] = r
        for label, ok, info in RL.positive_checks(r, require_site=True):
            check(f"3 {v}: {label}", ok, info)
        R = r["R"]
        tok = (R.get("prompts") or {}).get("tokenizer") or {}
        check(f"3 {v}: input discovery GO on the real mounts", (R.get("gates") or {}).get("inputs") == "GO", R.get("inputs"))
        check(f"3 {v}: REAL tokenizers (system `tokenizers`): FP8 and INT4 ids identical on every prompt",
              (R.get("gates") or {}).get("tokenizer") == "GO" and tok.get("fp8_int4_ids_identical") is True
              and (LOCAL or tok.get("kind") == "tokenizers"), tok)
        check(f"3 {v}: checkpoint identity recorded", all((R.get("identity") or {}).get(k) for k in ("fp8", "int4", "dflash2")),
              {k: (R.get("identity") or {}).get(k) for k in ("fp8", "int4", "dflash2")})
        REP["runs"][f"pos_{v}"] = {"rc": r["rc"], "sec": r["sec"], "statuses": {a: x.get("status") for a, x in RL.arms_of(R).items()},
                                   "windows": dict(RL.windows(R)), "errors": R.get("errors"), "site_end": r.get("site_end")}
        save()


# ---------------------------------------------------------------- 4. every runtime argv through the REAL parsers
def _engine_of(argv, rec_engine=None):
    """The engine of a server argv, or None for anything that is not a server launch (e.g. the s6_unit test commands)."""
    a = [str(x) for x in argv]
    head = " ".join(a[:4])
    if "fake_vllm_server" in head or "serve" in a[:3] or (a and a[0].endswith("/vllm")):
        return "vllm"
    if "fake_sglang_server" in head or "sglang.launch_server" in head:
        return "sglang"
    if a and a[0].startswith("--"):                         # bare flags (catalog / plan): SGLang argvs start with --model-path
        return rec_engine or "sglang"
    if rec_engine == "vllm" and a and not a[0].startswith("-"):
        return "vllm"                                       # catalog / plan vLLM flags: [model, --flags...]
    return None


def _vllm_flags(argv):
    a = [str(x) for x in argv]
    if "serve" in a:
        return a[a.index("serve") + 1:]
    if len(a) > 1 and a[1].endswith("fake_vllm_server.py"):
        return a[2:]
    return a


def s4():
    sg, vl = {}, {}

    def add(src, arm, argv, eng=None):
        e = _engine_of(argv, eng)
        if e is None:
            REP["info"].setdefault("argv_not_a_server", []).append(f"{src}:{arm}")
            return
        if e == "sglang":
            f = tuple(L._flags_only("sglang", argv))
            sg.setdefault(f, []).append(f"{src}:{arm}")
        else:
            f = tuple(_vllm_flags(argv))
            vl.setdefault(f, []).append(f"{src}:{arm}")
    for v, r in (S.get("runs") or {}).items():
        A = RL.arms_of(r["R"])
        for arm, argv in (r["R"].get("argv") or {}).items():
            add(f"fake_{v}", arm, argv, (A.get(arm) or {}).get("engine"))
    for arm, spec in (S.get("catalog") or {}).items():         # every arm the notebook can launch, real staged / mount paths
        if spec.get("flags"):
            add("catalog", arm, list(spec["flags"]), spec.get("engine"))
    try:                                                       # the builder's static plan (both branches; placeholders -> mounts)
        pa = json.loads((E24K / "Experiment Notebooks" / "Build" / "plan_argvs_e24.json").read_text())
        sub = {"<FP8>": str(INP / META["inputs"]["FP8"]), "<INT4>": str(INP / META["inputs"]["INT4"]),
               "<DFLASH>": str(INP / META["inputs"]["DFLASH2"])}
        for v, arms in (pa.get("variants") or {}).items():
            for arm, spec in arms.items():
                if isinstance(spec, dict) and spec.get("flags"):
                    add(f"plan_{v}", arm, [sub.get(str(x), str(x)) for x in spec["flags"]], spec.get("engine"))
    except Exception as e:
        REP["info"]["plan_argvs"] = repr(e)
    REP["info"]["argv_counts"] = {"sglang_unique": len(sg), "vllm_unique": len(vl)}
    vpy, env = (S.get("sgl") or {}).get("vpy"), (S.get("sgl") or {}).get("env")
    res = {}
    if vpy:
        with ThreadPoolExecutor(3) as ex:
            futs = {f: ex.submit(L.flag_check, "sglang", vpy, list(f), env=env, timeout=900) for f in sg}
            res = {f: fu.result() for f, fu in futs.items()}
    bad = {", ".join(sg[f][:3]): {k: r.get(k) for k in ("ran", "unknown", "ambiguous", "prefix", "parse_error", "error", "method")}
           for f, r in res.items() if not (r.get("ran") and r.get("ok"))}
    methods = sorted({str(r.get("method")) for r in res.values()})
    check("4a every runtime SGLang argv: exact 0.5.21 option strings + prepare_server_args (REAL parser in the venv)",
          bool(res) and not bad, {"checked": len(res), "methods": methods, "failed": dict(list(bad.items())[:12])})
    check("4b every check ran in the venv and its CPU method is syntax_only (resolve_once needs the GPU box)",
          bool(res) and all(r.get("ran") for r in res.values()) and methods == ["syntax_only"],
          {"methods": methods, "not_ran": sum(1 for r in res.values() if not r.get("ran"))})
    want = {"s1_a", "s1_b"}
    seen = {x.split(":", 1)[1] for refs in vl.values() for x in refs}
    vv = S.get("vllm") or {}
    if not vv.get("vpy"):
        check("4c s1_a / s1_b argv through the REAL vLLM 0.32 parser", "No space left" in str(vv.get("failed") or ""),
              {"reason": vv.get("failed") or "no vLLM venv"})
        return
    fj = {f"{'|'.join(refs[:2])}#{i}": {"vllm": "0.32.0", "flags": list(f)} for i, (f, refs) in enumerate(vl.items())}
    (K / "vllm_flags.json").write_text(json.dumps(fj))
    rc, out, sec = sh([vv["vpy"], KITD / "e21_flagcheck.py", K / "vllm_flags.json", "0.32.0"], timeout=1800, env=vv.get("env"),
                      name="vllm_flagcheck")
    summ = jline(out, "FLAGCHECK") or {}
    check("4c s1_a / s1_b argv through the REAL vLLM 0.32 parser (E21 flagcheck: argparse + AsyncEngineArgs, stub platform)",
          bool(summ.get("all_ok")) and want <= seen, {"summary": summ, "arms": sorted(seen), "tail": out[-1200:] if not summ else ""})
    lab = [L.flag_check("vllm", vv["vpy"], ["vllm", "serve"] + list(f), env=vv.get("env"), timeout=600) for f in list(vl)[:1]]
    REP["info"]["vllm_lab_flag_check"] = [{k: r.get(k) for k in ("ran", "ok", "unknown", "error", "method")} for r in lab]


# ---------------------------------------------------------------- 5. patches on the INSTALLED package
def _compile(site, chk):
    bad = {}
    for rel in (chk.get("files") or {}):
        p = Path(site) / rel
        try:
            compile(p.read_text(encoding="utf-8"), str(p), "exec")
        except Exception as e:
            bad[rel] = f"{type(e).__name__}: {e}"[:200]
    return {"ok": not bad, "files": len(chk.get("files") or {}), "bad": bad}


def s5():
    site = S.get("sgl_pkg")
    if not site:
        check("5 patches on the installed package", False, "no patch site (install failed)")
        return
    c0 = P.check(site)
    check("5a installed sglang pristine at start (7 names not_applied, file sha256 = 0.5.21)", P.state_ok(c0, expect_applied=()),
          {"names": c0.get("names"), "files_verified": c0.get("files_verified")})

    def cycle(names):
        a = P.apply(site, list(names))
        c = P.check(site)
        applied = P.state_ok(c, expect_applied=tuple(names))
        comp = _compile(site, c)
        rv = P.revert(site, list(names))
        c2 = P.check(site)
        pristine = P.state_ok(c2, expect_applied=())
        info = {"applied": applied, "names": c.get("names"), "compile": comp, "pristine_after_revert": pristine,
                "apply_ok": a.get("ok"), "revert_ok": rv.get("ok")}
        return applied and comp["ok"] and pristine, info
    for n in [x for x in P.E24_NAMES if x != "43559"]:
        ok, info = cycle([n])
        check(f"5b {n}: apply -> state -> compile -> revert -> pristine sha256 on the installed package", ok, info)
    r = P.apply(site, ["43559"])
    c = P.check(site)
    check("5c 43559 alone is refused (REQUIRES 43567) and the package stays pristine",
          P.state_ok(c, expect_applied=()) and (r.get("ok") is False or "refus" in json.dumps(r, default=str).lower()),
          {"apply": {k: r.get(k) for k in ("ok", "error", "results")}, "names": c.get("names")})
    for label, combo in (("43567+43559", ("43567", "43559")), ("all five E24", tuple(P.E24_NAMES)), ("all seven", tuple(P.NAMES))):
        ok, info = cycle(combo)
        check(f"5d {label}: apply -> compile -> revert -> pristine sha256", ok, info)
    cF = P.check(site)
    check("5e installed sglang pristine at the end (the FAKE runs and unit tests use it)", P.state_ok(cF, expect_applied=()),
          cF.get("names"))


# ---------------------------------------------------------------- 6. hook targets (AST) in the installed package
def _ast_targets(pkg_root, tables, mod_prefix):
    rows, bad = {}, {}
    for tgt, (mod, attr) in tables["TARGETS"].items():
        if not mod.startswith(mod_prefix + "."):
            continue
        f = Path(pkg_root) / (mod.split(".", 1)[1].replace(".", "/") + ".py")
        if not f.is_file():
            bad[tgt] = f"module file missing: {f}"
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"))
        top = {n.name: n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
        node = top.get(attr)
        if node is None:
            bad[tgt] = f"{attr} not defined at top level of {f.name}"
            continue
        if isinstance(node, ast.ClassDef):
            meth = {n.name: n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            for m in tables["TARGET_METHODS"].get(tgt) or ():
                key = f"{tgt}.{m}"
                if m not in meth:
                    bad[key] = f"method {m} missing in class {attr}"
                    continue
                rows[key] = [a.arg for a in meth[m].args.args]
                ref = META["ref_sigs"].get(key)
                if ref is not None and rows[key] != ref:
                    bad[key] = {"installed": rows[key], "reference": ref}
        else:
            rows[tgt] = [a.arg for a in node.args.args]
            ref = META["ref_sigs"].get(tgt)
            if ref is not None and rows[tgt] != ref:
                bad[tgt] = {"installed": rows[tgt], "reference": ref}
    return rows, bad


def s6():
    tables = {}
    for node in ast.parse((E24K / "src" / "e24_hook.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("TARGETS", "TARGET_METHODS"):
            tables[node.targets[0].id] = ast.literal_eval(node.value)
    req = [(t, m) for t, m in META["required_targets"] if t not in tables.get("TARGETS", {}) or
           (m and m not in (tables.get("TARGET_METHODS", {}).get(t) or ()))]
    check("6a the hook still wraps every override-(f) target (_alloc_mamba_slot, _mamba_check_track_boundary, "
          "DFlashWorkerV2.forward_batch_generation, CandidateSelector.sample_path, _grouped_conv, Scheduler.__init__)", not req, req)
    if not S.get("sgl_pkg"):
        check("6b hook targets in the installed package (AST)", False, "no installed sglang")
        return
    rows, bad = _ast_targets(S["sgl_pkg"], tables, "sglang")
    check("6b every SGLang hook target + wrapped method exists in the INSTALLED package, arguments = the 0.5.21 reference",
          bool(rows) and not bad, {"bad": bad, "checked": sorted(rows)})
    vv = S.get("vllm") or {}
    if vv.get("vpy"):
        rc, out, _ = sh([vv["vpy"], "-c", "import importlib.util as u; print(u.find_spec('vllm').submodule_search_locations[0])"],
                        timeout=120, env=vv.get("env"))
        vroot = out.strip().splitlines()[-1] if rc == 0 and out.strip() else None
        if vroot:
            rows2, bad2 = _ast_targets(vroot, tables, "vllm")
            check("6c vLLM hook target (GPUModelRunner.load_model) exists in the installed vLLM 0.32", bool(rows2) and not bad2,
                  {"bad": bad2, "checked": sorted(rows2)})


# ---------------------------------------------------------------- 7. SGLANG_* env vars the plan sets
def s7():
    if not S.get("sgl_pkg"):
        check("7 planned SGLANG_* env vars exist in the installed environ.py", False, "no installed sglang")
        return
    names = set(re.findall(r"\bSGLANG_[A-Z0-9_]+", S["code"]["SMOKE"] + (E24K / "src" / "e24_lab.py").read_text()))
    for r in (S.get("runs") or {}).values():
        for x in r.get("launches") or []:
            names |= {k for k in (x.get("env") or {}) if k.startswith("SGLANG_")}
    pkg = Path(S["sgl_pkg"])
    envpy = (pkg / "srt" / "environ.py").read_text(encoding="utf-8")
    where = {}
    for n in sorted(names):
        prefix = n.endswith("_")                                   # e.g. SGLANG_TEST_ (a knob family the notebook guards)
        pat = (r"\b" + re.escape(n) + r"[A-Z0-9_]+") if prefix else (r"\b" + re.escape(n) + r"\b")
        if re.search(pat, envpy):
            where[n] = "srt/environ.py"
            continue
        cmd = ["grep", "-rlE", "--include=*.py", re.escape(n) + "[A-Z0-9_]+", pkg] if prefix else \
            ["grep", "-rlwF", "--include=*.py", n, pkg]
        rc, out, _ = sh(cmd, timeout=300)
        hits = [l for l in out.splitlines() if l.endswith(".py")]
        where[n] = (str(Path(hits[0]).relative_to(pkg)) if hits else "MISSING")
    miss = [n for n, w in where.items() if w == "MISSING"]
    check("7 every SGLANG_* env var the notebook sets or names is read by the installed 0.5.21 (environ.py, else elsewhere)",
          bool(names) and not miss, {"missing": miss, "where": where})


# ---------------------------------------------------------------- 8. NGRAM load_jit (s6_gu / s6_gp pay this C++ build)
NGRAM = r"""
import json, time, traceback
o, t = {"ok": False}, time.time()
try:
    from sglang.kernels.jit.utils import override_jit_cuda_arch
    from sglang.kernels.ops.speculative.ngram_corpus import get_ngram_corpus_cls
    with override_jit_cuda_arch(8, 9):
        cls = get_ngram_corpus_cls()
    o.update(ok=True, cls=getattr(cls, "__name__", str(cls)))
    try:
        c = cls(1 << 16, 8, 1, 1, 4, "BFS")
        c.insert([[1, 2, 3, 4, 5, 1, 2, 3, 4]])
        o["object"] = "constructed + insert ok"
    except BaseException as e:
        o["object"] = (type(e).__name__ + ": " + str(e))[:300]
except BaseException as e:
    o["error"], o["tb"] = (type(e).__name__ + ": " + str(e))[:600], traceback.format_exc()[-2500:]
o["sec"] = round(time.time() - t, 1)
print("E24KIT_NGRAM " + json.dumps(o), flush=True)
"""


def s8():
    sg = S.get("sgl") or {}
    if not sg.get("vpy"):
        check("8 NGRAM load_jit compiles", False, "no SGLang venv")
        return
    env = dict(sg.get("env") or clean_env())
    env.update(SGLANG_JIT_CACHE_DIR=str(K / "jit"), SGLANG_CACHE_DIR=str(K / "jit_sglang"), HOME=str(K / "home_jit"))
    rc, out, sec = sh([sg["vpy"], "-c", NGRAM], timeout=1800, env=env, name="ngram_jit")
    j = jline(out, "E24KIT_NGRAM") or {}
    so = [str(p) for p in (K / "jit").rglob("*.so")]
    check("8 the NGRAM C++ corpus (load_jit, 5 C++ files) compiles and loads in the SGLang venv (image g++, venv ninja)",
          bool(j.get("ok")) and bool(so), {"result": j, "so": so[:3], "compiler_errors": re.findall(r".*error:.*", out)[:5],
                                           "tail": out[-1500:] if not j.get("ok") else ""})


# ---------------------------------------------------------------- 9. hook import smoke in the venv python
def s9():
    sg = S.get("sgl") or {}
    if not sg.get("vpy"):
        check("9 hook import smoke", False, "no SGLang venv")
        return
    hs, ho = K / "hook_site", K / "hook_out"
    hs.mkdir(exist_ok=True)
    ho.mkdir(exist_ok=True)
    (hs / "sitecustomize.py").write_bytes((E24K / "src" / "e24_hook.py").read_bytes())
    env = dict(sg.get("env") or clean_env())
    env.update(PYTHONPATH=str(hs), E24_HOOK_DIR=str(ho), E24_HOOK_COUNTERS="1", E24_HOOK_FAST_SEC="5", HOME=str(K / "home_hook"))
    code = ("import json, time, sys\nr = {}\ntry:\n    import sglang\n    r['sglang'] = sglang.__version__\n"
            "    import sglang.srt.managers.scheduler\n    r['scheduler'] = 'imported'\n    time.sleep(3)\n"
            "except BaseException as e:\n    r['import_error'] = (type(e).__name__ + ': ' + str(e))[:300]\n"
            "print('E24KIT_IMPORT ' + json.dumps(r), flush=True)\n")
    rc, out, sec = sh([sg["vpy"], "-c", code], timeout=900, env=env, name="hook_import")
    recs = sorted(ho.glob("env_*.json"))
    rec = json.loads(recs[0].read_text()) if recs else {}
    imp = jline(out, "E24KIT_IMPORT") or {}
    REP["info"]["hook_import"] = {"import": imp, "counters_files": [p.name for p in ho.glob("counters_*.json")],
                                  "chain": (rec.get("hook") or {}).get("chain"), "rc": rc}
    check("9 the hook (sitecustomize) is active in the venv python: env record with pid / argv / cache vars / role "
          "(import sglang on CPU recorded, not required)", bool(recs) and bool(rec.get("pid")) and "env" in rec,
          {"records": [p.name for p in recs][:4], "keys": sorted(rec)[:20], "import": imp, "tail": out[-800:] if not recs else ""})


# ---------------------------------------------------------------- 9b. s6_unit overlays on the INSTALLED package (CPU: no CUDA)
def s9b():
    sg = S.get("sgl") or {}
    unit = E24K / "src" / "e24_unit_41736_41737.py"
    if not (sg.get("vpy") and S.get("sgl_pkg") and unit.is_file()):
        check("9b s6_unit overlay preparation on the installed package", False,
              {"vpy": sg.get("vpy"), "site": S.get("sgl_pkg"), "unit_script": unit.is_file()})
        return
    root = K / "unit_root"
    env = dict(sg.get("env") or clean_env())
    env.update(E24_UNIT_ROOT=str(root), HOME=str(K / "home_unit_gpu"))
    for t in ("41736", "41737"):
        rc, out, sec = sh([sg["vpy"], unit, "--test", t, "--out", K / f"unit_{t}.json", "--root", root, "--site", S["sgl_pkg"],
                           "--patcher", E24K / "src" / "e24_sglang_patches.py", "--timeout", "600"], timeout=1800, env=env,
                          name=f"unit_gpu_{t}")
        try:
            j = json.loads((K / f"unit_{t}.json").read_text())
        except Exception as e:
            j = {"error": repr(e)}
        ovl = sorted(p.parent.name for p in root.glob("overlay_*/sglang"))
        prep_ok = j.get("status") not in ("INVALID_OVERLAY_STATE", None) and not str(j.get("reason") or "").startswith("prepare")
        check(f"9b s6_unit {t}: overlays of the installed package prepared and patched (copies only), the test fails "
              f"gracefully without CUDA", rc == 0 and prep_ok and {"overlay_none", f"overlay_{t}"} <= set(ovl)
              and (j.get("ran") or "CUDA" in str(j.get("reason")) or "torch" in str(j.get("reason"))),
              {"rc": rc, "status": j.get("status"), "ran": j.get("ran"), "reason": str(j.get("reason"))[:300], "overlays": ovl,
               "sec": sec, "tail": out[-800:] if rc else ""})
    shutil.rmtree(root, ignore_errors=True)                          # ~2 x 3 package copies


# ---------------------------------------------------------------- 10. fault modes on Linux
def s10():
    sc = RL.scenarios(SCALE)
    names = [n for n in sc if not sc[n].get("local_only")] if NEGS == ["all"] else [n for n in NEGS if n in sc]
    for n in names:
        ctx = dict(sc[n].get("ctx") or {})
        if sc[n].get("armed"):
            ctx["armed_ok"], ctx["armed"] = RL.armed_logprob_control(K / "runs", fake_dir=E24K / "src" / "harness", input_root=INP)
        r = RL.run_notebook(f"neg_{n}", S["code"][sc[n]["variant"]], K / "runs", INP, sc[n]["variant"], scale=SCALE,
                            rules=sc[n]["rules"], extra_env=sc[n].get("env"), site=bool(S.get("sgl_pkg")), timeout_min=120,
                            fake_dir=E24K / "src" / "harness", unset=sc[n].get("unset", ()))
        try:
            res = sc[n]["check"](r, ctx)
        except Exception as e:
            res = [("scenario check raised", False, repr(e))]
        for label, ok, info in res:
            check(f"10 {n}: {label}", ok, info)
        REP["runs"][f"neg_{n}"] = {"rc": r["rc"], "sec": r["sec"], "statuses": {a: x.get("status") for a, x in RL.arms_of(r["R"]).items()}}
        save()


# ---------------------------------------------------------------- 11. unit tests on Linux
SHADOW = ("_stop", "_started", "_target", "_args", "_kwargs", "_name", "_ident", "_daemonic", "_tstate_lock", "_is_stopped")
# test classes that read repo data the kit cannot carry under 400 kB (the 7 EMS server logs = 1.24 MB): excluded BY NAME
# and reported; they are pure text parsing (platform independent) and run in the local Mac run.
KIT_EXCLUDE = {"test_e24_lab": ("TestLogs",)}
UNIT_DRIVER = r"""
import json, sys, unittest
mod, excl = sys.argv[1], tuple(x for x in sys.argv[2].split(",") if x)
sys.path.insert(0, ".")
suite = unittest.defaultTestLoader.loadTestsFromName(mod)
def flat(s):
    for t in s:
        if isinstance(t, unittest.TestSuite):
            yield from flat(t)
        else:
            yield t
tests = list(flat(suite))
keep = [t for t in tests if not any("." + c + "." in "." + t.id() + "." for c in excl)]
r = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(unittest.TestSuite(keep))
print("E24KIT_UNIT " + json.dumps({"ran": r.testsRun, "failures": [t.id() for t, _ in r.failures], "errors": [t.id() for t, _ in r.errors],
      "skipped": len(r.skipped), "skips": [[t.id(), str(why)[:160]] for t, why in r.skipped],
      "excluded": sorted(t.id() for t in tests if t not in keep), "ok": r.wasSuccessful()}), flush=True)
"""


def s11():
    tests = sorted((E24K / "src").glob("test_*.py")) + sorted((E24K / "src" / "harness").glob("test_*.py"))
    env = clean_env(HOME=K / "home_unit", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    if S.get("sgl_pkg"):
        env["E24_SGLANG_SRC"] = str(S["sgl_pkg"])               # test_e24_patches: a PRISTINE sglang package (the installed one)
    (K / "unit_driver.py").write_text(UNIT_DRIVER)
    for t in tests:
        excl = [] if (E24K / "Base Engine" / "server_logs").is_dir() else list(KIT_EXCLUDE.get(t.stem, ()))
        rc, out, sec = sh([sys.executable, K / "unit_driver.py", t.stem, ",".join(excl)], timeout=3000, env=env, cwd=t.parent,
                          name=f"unit_{t.stem}")
        j = jline(out, "E24KIT_UNIT") or {}
        src_skips = [x for x in (j.get("skips") or []) if "E24_SGLANG_SRC" in x[1] or "SGLang 0.5.21 source" in x[1]]
        check(f"11 {t.relative_to(E24K)} on Linux" + (f" (excluded: {', '.join(excl)})" if excl else ""),
              rc == 0 and j.get("ok") is True and j.get("ran", 0) > len(j.get("skips") or []) and not src_skips,
              {"rc": rc, "ran": j.get("ran"), "failures": j.get("failures"), "errors": j.get("errors"), "skipped": j.get("skipped"),
               "skips": (j.get("skips") or [])[:6], "skipped_for_missing_source": len(src_skips),
               "E24_SGLANG_SRC": env.get("E24_SGLANG_SRC"), "excluded": len(j.get("excluded") or []), "sec": sec,
               "tail": out[-1200:] if not j else ""})
    check("11 at least the four E24 test files exist", len(tests) >= 4, [str(t.name) for t in tests])
    bad = []
    for f in sorted((E24K / "src").rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for c in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            if any("Thread" in ast.unparse(b) for b in c.bases):
                for n in ast.walk(c):
                    if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name) \
                            and n.value.id == "self" and n.attr in SHADOW:
                        bad.append(f"{f.name}:{c.name}.{n.attr}")
                    if isinstance(n, ast.FunctionDef) and n.name in SHADOW:
                        bad.append(f"{f.name}:{c.name}.{n.name}()")
    check("11 no threading.Thread subclass shadows a Thread internal (_stop / _started / ...: the E14A 3.12 crash)", not bad, bad)


# ---------------------------------------------------------------- 12. competition rerun format
def s12():
    if LOCAL:
        REP["info"]["rerun"] = "LOCAL: skipped (writes /kaggle/working/submission.json)"
        return
    sub = Path("/kaggle/working/submission.json")
    if sub.exists():
        sub.unlink()
    (K / "rerun").mkdir(exist_ok=True)
    (K / "rerun" / "nb.py").write_text(S["code"]["SMOKE"])
    env = clean_env(KAGGLE_IS_COMPETITION_RERUN="1", E24_INPUT=INP, E24_OUT=K / "rerun" / "out", E24_TMP=K / "rerun" / "tmp")
    rc, out, sec = sh([sys.executable, K / "rerun" / "nb.py"], timeout=600, env=env, name="rerun")
    ok_sub = False
    if sub.is_file():
        s = json.loads(sub.read_text())
        tasks = json.loads(next(iter(sorted(INP.rglob("arc-agi_test_challenges.json")))).read_text())
        ok_sub = set(s) == set(tasks) and all(len(s[k]) == len(tasks[k]["test"]) and all(set(a) == {"attempt_1", "attempt_2"}
                                                                                         for a in s[k]) for k in s)
        sub.unlink()                                             # a CPU kit leaves no submission behind
    servers = sorted(p.name for p in (K / "rerun" / "out").rglob("server_*.log"))
    check("12 competition rerun: valid fallback submission.json, no exception, no server launched",
          rc == 0 and "Traceback" not in out and ok_sub and not servers, {"rc": rc, "submission_ok": ok_sub, "servers": servers,
                                                                          "tail": out[-600:]})


for _name, _fn in (("0 decode", s0), ("1 mounts", s1), ("2 installs", s2), ("5 patches", s5), ("6 hook AST", s6),
                   ("8 NGRAM jit", s8), ("9 hook smoke", s9), ("9b unit overlays", s9b), ("3 FAKE notebooks", s3),
                   ("4 flag checks", s4),
                   ("7 environ", s7), ("10 fault modes", s10), ("11 unit tests", s11), ("12 rerun", s12)):
    if _name != "0 decode" and "0 decode" in REP["stages"] and REP["stages"]["0 decode"].get("error"):
        break
    stage(_name, _fn)
REP["n_checks"] = len(REP["checks"])
REP["failed"] = [k for k, c in REP["checks"].items() if not c["ok"]]
REP["all_ok"] = bool(REP["checks"]) and not REP["failed"]
REP["sec"] = round(time.time() - T0)
save()
with tarfile.open(WORKING / "e24_cpu_kit_logs_tar.bin", "w:gz") as tf:      # .bin: Kaggle does not unpack it
    for p in sorted(K.rglob("*")):
        if p.is_file() and p.suffix in (".log", ".json", ".md", ".jsonl") and p.stat().st_size < 50e6 \
                and "venv" not in str(p) and "/site/" not in str(p) and "/repo/" not in str(p):
            tf.add(p, str(p.relative_to(K)), recursive=False)
print("failed checks:", json.dumps(REP["failed"], indent=1)[:6000])
print("ALL OK" if REP["all_ok"] else "NOT OK", "|", REP["n_checks"], "checks |", REP["sec"], "s", flush=True)
'''


HEADER = """# E24 CPU kit (WP5 rev 1, DESIGN rev 2) - Kaggle CPU kernel, pinned image, internet off, the REAL /kaggle/input mounts.
# Stages: 0 unpack | 1 mounts | 2 offline SGLang 0.5.21 + vLLM 0.32 installs through the notebook's own CELL 1 + CELL 2 |
# 5 patches apply/compile/revert on the INSTALLED package | 6 hook-target AST | 8 NGRAM load_jit | 9 hook import smoke |
# 3 the three notebooks in FAKE mode on the real mounts (E24_FORCE_CONDITIONAL=1) | 4 every runtime argv through the REAL
# SGLang 0.5.21 / vLLM 0.32 parsers | 7 SGLANG_* env vars in environ.py | 10 fault modes on Linux | 11 unit tests on Linux |
# 12 competition-rerun format. Report: /kaggle/working/e24_cpu_kit_report.json (+ e24_cpu_kit_logs_tar.bin).
# Kit logic: kit/e24_cpu_kit_main.py (source: experiments/Experiment 24 - SGLang speed track (planned)/Experiment Notebooks/
# Build/build_cpu_kit_e24.py, KIT_MAIN). Every kit file carries its sha256 (META["files"]).
"""
BOOT_EMBED = HEADER + r"""# Embedded mode: every kit file is gzip+base64 in EMB.
import base64, gzip, hashlib
EMB = __EMB__
META = __META__
_src = gzip.decompress(base64.b64decode(EMB["kit/e24_cpu_kit_main.py"][1]))
assert hashlib.sha256(_src).hexdigest() == META["files"]["kit/e24_cpu_kit_main.py"], "kit main: sha256 mismatch (corrupt notebook)"
exec(compile(_src.decode("utf-8"), "e24_cpu_kit_main.py", "exec"), {"__name__": "__main__", "EMB": EMB, "META": META, "PAYLOAD": None})
"""
BOOT_DATASET = HEADER + r"""# Dataset mode (the kit files do not fit a < 400 kB notebook): they travel as ONE deterministic zip, the kit dataset file
# __BIN__ (the E14-E22 kit pattern; '.bin' so Kaggle does not unpack it), pinned here by its sha256.
import hashlib, io, json, os, zipfile
from pathlib import Path
KIT_BIN, KIT_SHA, KIT_DATASET = "__BIN__", "__SHA__", "__DS__"
_root = Path(os.environ.get("E24KIT_PAYLOAD_DIR") or os.environ.get("E24KIT_INPUT") or "/kaggle/input")
_hits = sorted({p for d in range(0, 5) for p in _root.glob("/".join(["*"] * d + [KIT_BIN]))})
assert _hits, f"attach the kit dataset {KIT_DATASET} (file {KIT_BIN}, the version built with this notebook)"
_blob = _hits[0].read_bytes()
assert hashlib.sha256(_blob).hexdigest() == KIT_SHA, f"{_hits[0]}: sha256 mismatch: attach the kit dataset version of this build"
_z = zipfile.ZipFile(io.BytesIO(_blob))
PAYLOAD = {n: _z.read(n) for n in _z.namelist()}
META = json.loads(PAYLOAD.pop("kit/META.json"))
_src = PAYLOAD["kit/e24_cpu_kit_main.py"]
assert hashlib.sha256(_src).hexdigest() == META["files"]["kit/e24_cpu_kit_main.py"], "kit main: sha256 mismatch"
exec(compile(_src.decode("utf-8"), "e24_cpu_kit_main.py", "exec"), {"__name__": "__main__", "EMB": None, "META": META, "PAYLOAD": PAYLOAD})
"""


def notebook_json(code):
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1)


def kit_zip(files):
    """One deterministic zip (sorted names, fixed timestamps, deflate 9): the same inputs give the same bytes and sha256."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(files):
            zi = zipfile.ZipInfo(name, date_time=(2026, 10, 30, 0, 0, 0))
            zi.compress_type, zi.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
            data = files[name]
            z.writestr(zi, data if isinstance(data, bytes) else data.encode("utf-8"))
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description="E24 CPU kit builder")
    ap.add_argument("--nb-dir", default=str(HERE.parent), help="dir with the three built notebooks (default: the repo)")
    ap.add_argument("--out", default=str(HERE / "cpu_kernel_e24"))
    ap.add_argument("--sglang-src", default=os.environ.get("E24_SGLANG_SRC") or SGL_SRC_DEFAULT,
                    help="the 0.5.21 reference source (hook-target signatures); optional")
    ap.add_argument("--allow-missing", action="store_true",
                    help="structure-only build when a parallel work-package file is absent (the kit then fails its checks)")
    ap.add_argument("--no-plan-argvs", action="store_true", help="do not carry plan_argvs_e24.json (size)")
    ap.add_argument("--mode", choices=("auto", "embed", "dataset"), default="auto",
                    help="auto: embed when the notebook stays < 400 kB, else the kit dataset .bin (review F9 fallback)")
    a = ap.parse_args(argv)
    nbdir = Path(a.nb_dir).resolve()
    src_root = nbdir.parent if (nbdir.parent / "src" / "harness").is_dir() else E24    # the notebooks' own E24 tree (a mirror
    print("sources from", src_root)                                                   # snapshot stays self-consistent)
    cells = {v: code_cells(nbdir / NB_FILE.format(v)) for v in VARIANTS}
    code = {v: joined(c) for v, c in cells.items()}
    for v, (m, p) in VARIANTS.items():               # the kit carries SMOKE only; the others are SMOKE with two baked lines
        assert bake(code["SMOKE"], m, p) == code[v], f"{v} differs from SMOKE beyond the baked MODE / PART lines"
    mods, build_shas, nb_missing = extract_modules(code["SMOKE"])
    if nb_missing:
        assert a.allow_missing, f"the notebooks were built without {nb_missing}: rebuild them when every module exists"
        print("WARNING: the notebooks were built without", nb_missing)
    stale = []
    for name, data in mods.items():                   # the notebook must embed the CURRENT repo modules
        src = (src_root / "Base Engine" / "src_e14_modules" / name) if name.startswith("e14_") else (src_root / "src" / name)
        if src.is_file() and hashlib.sha256(src.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest() \
                != hashlib.sha256(data).hexdigest():
            stale.append(name)
    assert not stale, f"the notebooks embed stale modules {stale}: rebuild the notebooks first"
    blobs, stripped = {}, list(cells["SMOKE"])
    for node in ast.walk(ast.parse(code["SMOKE"])):           # BLOBS = {key: (file, sha, b64)}: keep each b64 once, as is
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "BLOBS":
            for key, (_n, _s, b64) in ast.literal_eval(node.value).items():
                if b64:
                    tok = f"@@E24KIT_BLOB_{key}@@"
                    assert sum(c.count(b64) for c in stripped) == 1 and not any(tok in c for c in stripped), key
                    stripped = [c.replace(b64, tok) for c in stripped]
                    blobs[tok] = b64
    rebuilt = list(stripped)
    for tok, b64 in blobs.items():
        rebuilt = [c.replace(tok, b64) for c in rebuilt]
    assert rebuilt == cells["SMOKE"], "blob stripping is not reversible"
    files = {"kit/nb_cells_SMOKE.json": json.dumps(stripped)}
    harness = src_root / "src" / "harness"
    for f in sorted(harness.glob("*.py")):
        if f.name != "mock_inputs.py":                       # local-only (the kit runs on the real mounts)
            files[f"src/harness/{f.name}"] = f.read_bytes()
    for f in sorted((src_root / "src").glob("test_*.py")):
        files[f"src/{f.name}"] = f.read_bytes()
    for f in sorted((src_root / "src" / "patches").glob("*.diff")):
        files[f"src/patches/{f.name}"] = f.read_bytes()
    pa = nbdir / "Build" / "plan_argvs_e24.json"
    if pa.is_file() and not a.no_plan_argvs:
        files["Experiment Notebooks/Build/plan_argvs_e24.json"] = pa.read_bytes()
    bb = nbdir / "Build" / "build_e24_notebook.py"           # src/test_e24_decisions.py reads the cells from it (ast only)
    if bb.is_file():
        files["Experiment Notebooks/Build/build_e24_notebook.py"] = bb.read_bytes()
    fc = next(iter(sorted(EXPS.glob("Experiment 21 - */src/e21_flagcheck.py"))), None)
    assert fc is not None, "Experiment 21 src/e21_flagcheck.py (the proven vLLM 0.32 CPU parser check) not found"
    files["kit/e21_flagcheck.py"] = fc.read_bytes()
    need = ["src/harness/fake_sglang_server.py", "src/harness/fake_vllm_server.py", "src/harness/run_local_e24.py",
            "src/harness/test_fakes.py", "src/test_e24_lab.py", "src/test_e24_patches.py", "src/test_e24_hook.py"]
    missing = [n for n in need if n not in files] + [f"notebook:{n}" for n in nb_missing]
    if missing:
        assert a.allow_missing, f"missing files {missing} (parallel work packages not finished?); --allow-missing to build anyway"
        print("WARNING: structure-only kit, missing", missing)
    tables = hook_tables(mods["e24_hook.py"].decode("utf-8"))
    sigs = ref_signatures(a.sglang_src, tables) if Path(a.sglang_src, "srt", "server_args.py").is_file() else {}
    meta = {"kit": KIT_REV, "built": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "nb_code_sha256": {v: hashlib.sha256(c.encode("utf-8")).hexdigest() for v, c in code.items()},
            "build_shas": build_shas, "inputs": INPUTS, "required_targets": REQUIRED_TARGETS, "ref_sigs": sigs,
            "ref_sigs_source": str(a.sglang_src) if sigs else None, "missing_at_build": missing, "nb_blobs": blobs}
    kit_main = KIT_MAIN.replace("__MODULE_EXTRACT__", MODULE_EXTRACT, 1)
    compile(kit_main, "e24_cpu_kit_main.py", "exec")
    files["kit/e24_cpu_kit_main.py"] = kit_main
    files = {k: (v if isinstance(v, bytes) else v.encode("utf-8")) for k, v in files.items()}
    meta["files"] = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kd = out / "kit_data"
    EMB = {k: emb(v) for k, v in files.items()}
    code_embed = BOOT_EMBED.replace("__EMB__", repr(EMB), 1).replace("__META__", repr(meta), 1)
    text = notebook_json(code_embed)
    size_embed = len(text.encode("utf-8"))
    mode = "embed" if a.mode == "embed" or (a.mode == "auto" and size_embed < KIT_MAX_BYTES) else "dataset"
    datasets = list(DATASETS)
    for old_bin in sorted(kd.glob("q27x_e24_kit_*.bin")) if kd.is_dir() else []:
        old_bin.unlink()
    if mode == "dataset":
        blob = kit_zip(dict(files, **{"kit/META.json": json.dumps(meta, sort_keys=True).encode("utf-8")}))
        bsha = hashlib.sha256(blob).hexdigest()
        bname = f"q27x_e24_kit_{bsha[:12]}.bin"
        kd.mkdir(parents=True, exist_ok=True)
        (kd / bname).write_bytes(blob)
        (kd / "dataset-metadata.json").write_text(json.dumps({"title": "Q27 E24 CPU kit data", "id": KIT_DATASET_ID,
                                                              "licenses": [{"name": "apache-2.0"}]}, indent=1) + "\n",
                                                  encoding="utf-8", newline="\n")
        code_ = BOOT_DATASET.replace("__BIN__", bname).replace("__SHA__", bsha).replace("__DS__", KIT_DATASET_ID)
        compile(code_, "<e24 cpu kit>", "exec")
        text = notebook_json(code_)
        datasets.append(KIT_DATASET_ID)
    else:
        compile(code_embed, "<e24 cpu kit>", "exec")
        for f in (kd / "dataset-metadata.json",):
            if f.exists():
                f.unlink()
    size = len(text.encode("utf-8"))
    assert size < KIT_MAX_BYTES, f"kit notebook {size} B >= {KIT_MAX_BYTES} B: Kaggle commits above ~407-444 kB never start"
    (out / "q27-e24-cpu-kit.ipynb").write_text(text, encoding="utf-8", newline="\n")
    md = {"id": KIT_ID, "title": "Q27 E24 CPU Kit", "code_file": "q27-e24-cpu-kit.ipynb", "language": "python",
          "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
          "keywords": [], "dataset_sources": datasets, "kernel_sources": [], "competition_sources": COMPETITIONS,
          "model_sources": MODELS, "docker_image": IMAGE, "machine_shape": "None"}
    for g in sorted((nbdir / "Build" / "kernels").glob("*/kernel-metadata.json")):   # every GPU variant input is attached here
        gm = json.loads(g.read_text(encoding="utf-8"))
        for k in ("dataset_sources", "model_sources", "competition_sources"):
            extra = sorted(set(gm.get(k) or []) - set(md[k]))
            assert not extra, f"{g.parent.name} attaches {extra} that the kit does not"
        assert gm.get("docker_image") == IMAGE, f"{g.parent.name}: docker image differs from the kit's"
    (out / "kernel-metadata.json").write_text(json.dumps(md, indent=2) + "\n", encoding="utf-8", newline="\n")
    sha8 = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    print(f"kit {out / 'q27-e24-cpu-kit.ipynb'} | mode {mode} | sha256 {sha8} | {size / 1000:.1f} kB (< {KIT_MAX_BYTES / 1000:.0f} kB; "
          f"embedded would be {size_embed / 1000:.1f} kB) | {len(files)} kit files | hook-target reference signatures: {len(sigs)}")
    if mode == "dataset":
        print(f"  kit dataset {KIT_DATASET_ID}: {kd / bname} ({len(blob) / 1000:.1f} kB, sha256 {bsha[:12]}) -> upload it as a new "
              f"dataset version BEFORE pushing the kernel (the notebook pins this exact sha256)")
    for k, (sh_, b64) in sorted(EMB.items()):
        print(f"  {k:60} {sh_[:8]} {len(b64) / 1000:7.1f} kB gz+b64")
    return 0


if __name__ == "__main__":
    sys.exit(main())
