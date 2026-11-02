# E30 gentail CPU kit (pinned image, Py3.12, no GPU, no internet): GPU_TEST_SPEC_gentail.md section 0.
#   R1 inputs (exact Kaggle mount paths, as the NBDF K100 notebook), R2 the Merlin offline install (the NBDF K100 notebook's
#   merlin_install, verbatim apart from the kit's TMP / logs), R3 libcuda stub + FakeL4 platform (the NBDF kit's), R4 the
#   CPU tests (test_gentail_seam.py with the INSTALLED pristine tree as GENTAIL_PRISTINE, test_gentail_logic.py, bundled
#   CSVs), R5 gentail_seam apply + verify on the installed tree, R6 the real-class drive (on / off) in the Merlin venv,
#   R7 revert + verify pristine. One line: @@KIT gentail ALL OK@@ or @@KIT gentail NOT OK: <checks>@@; report JSON in
#   /kaggle/working/gentail_kit_report.json.
import os, sys, json, time, subprocess, shutil, traceback, hashlib, re, base64, lzma, tarfile, glob
from pathlib import Path
T0 = time.time()
KIT = "gentail"
INP = Path(os.environ.get("KIT_INPUT", "/kaggle/input"))
K = Path(os.environ.get("KIT_TMP", "/tmp/gentailkit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
TMP = K / "tmp"; TMP.mkdir()
LOGS = W / "logs"; LOGS.mkdir(exist_ok=True)
REP = {"what": "E30 gentail CPU kit", "python": sys.version, "checks": {}, "kit_files": "__FILESHA__"}
def save():
    (W / "gentail_kit_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def unpack(b):
    return json.loads(lzma.decompress(base64.b85decode(b)).decode("utf-8"))
def run(name, cmd, timeout=3600, env=None):                       # the NBDF notebook's run() (CELL 2), logs under /kaggle/working
    t = time.time()
    try:
        with open(LOGS / f"{name}.log", "w") as f:
            r = subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                               env={**os.environ, **(env or {})})
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = -9
    txt = (LOGS / f"{name}.log").read_text(errors="replace")
    log(name, "rc", rc, f"{time.time() - t:.0f}s")
    return rc, txt
def verdict(ok, recorded=None):
    REP["verdict"] = {"checks": ok, "ALL_OK": all(ok.values()), "recorded": recorded or {}, "minutes": round((time.time() - T0) / 60, 1)}
    save()
    bad = [k for k, v in ok.items() if not v]
    print("VERDICT", json.dumps(REP["verdict"], default=str)[:6000], flush=True)
    print(f"@@KIT {KIT} ALL OK@@" if not bad else f"@@KIT {KIT} NOT OK: {', '.join(bad)}@@", flush=True)

F = unpack("__BLOB__")
B = K / "kit"
for rel, txt in F.items():
    (B / rel).parent.mkdir(parents=True, exist_ok=True)
    (B / rel).write_text(txt, encoding="utf-8", newline="\n")
REP["kit_file_shas"] = {rel: hashlib.sha256(txt.encode("utf-8")).hexdigest() for rel, txt in F.items()}
sys.path.insert(0, str(B))
import gentail_seam as GS
save()

# ---- R1 inputs: the NBDF K100 notebook's exact mount paths (CELL 1)
R1 = {}
MD = INP / "models" / "foysalemonshanto"
MERLIN_PAY = MD / "e23-merlin-env/other/v1/1/merlin_offline_tar.bin"
MERLIN_PAY = MERLIN_PAY if MERLIN_PAY.is_file() else None
K100_DIR = MD / "qwen38-flash-next-klee100-3bpw-mtp/pytorch/autoround-3bpw-mtp-nople/1"
if not (K100_DIR / "KAGGLE_MIRROR_MANIFEST.json").is_file():
    _g = sorted((MD / "qwen38-flash-next-klee100-3bpw-mtp").glob("*/*/*/KAGGLE_MIRROR_MANIFEST.json"))
    K100_DIR = _g[0].parent if len(_g) == 1 else None
R1.update(merlin_pay=str(MERLIN_PAY), klee100_dir=str(K100_DIR), klee100_config=bool(K100_DIR and (K100_DIR / "config.json").is_file()),
          mounts=sorted(str(p) for p in MD.glob("*/*/*/*"))[:20] if MD.is_dir() else [])
REP["checks"]["R1"] = R1; log("R1", json.dumps(R1)[:900]); save()

# ---- R2 the Merlin install: the NBDF K100 notebook's merlin_install() (CELL 2), verbatim but for the kit's TMP and timeout
R2 = {}
eng = None
try:
    if os.environ.get("KIT_LOCAL_TREE"):          # local control-flow check only (never set on Kaggle): no install, no venv
        eng = {"kv": {"MERLIN_TREE": os.environ["KIT_LOCAL_TREE"], "MERLIN_VENV": str(Path(sys.executable).parent.parent)}, "cuda_home": ""}
        raise RuntimeError("KIT_LOCAL_TREE set: local control-flow run, no Merlin install")
    if MERLIN_PAY is None:
        raise RuntimeError("Merlin checkpoint / env not attached or incomplete")
    pay_root = TMP / "merlin_pay"
    shutil.rmtree(pay_root, ignore_errors=True)
    pay_root.mkdir()
    t = time.time()
    with tarfile.open(MERLIN_PAY) as tf:
        tf.extractall(pay_root, filter="data")
    R2["merlin_extract_sec"] = round(time.time() - t, 1)
    pay = pay_root / "merlin_offline"
    shutil.rmtree(TMP / "merlin_off", ignore_errors=True)
    rc, txt = run("install_merlin", f"bash {pay}/install_merlin_offline.sh {pay} {TMP}/merlin_off", timeout=3600)
    kv = dict(re.findall(r"^(MERLIN_[A-Z]+)=(.*)$", txt, re.M))
    shutil.rmtree(pay_root, ignore_errors=True)
    if rc or "INSTALL OK" not in txt:
        raise RuntimeError("Merlin offline install failed (logs/install_merlin.log)")
    site = subprocess.run([kv["MERLIN_VENV"] + "/bin/python", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                          capture_output=True, text=True).stdout.strip()
    eng = {"kv": kv, "cuda_home": site + "/nvidia/cu13"}
    R2.update(ok=True, kv=kv, cuda_home=eng["cuda_home"])
except Exception as e:
    R2.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
REP["checks"]["R2"] = R2; log("R2 install", json.dumps({k: v for k, v in R2.items() if k != "tb"})[:900]); save()
TREE = Path(eng["kv"]["MERLIN_TREE"]) if eng else None
VPY = (sys.executable if os.environ.get("KIT_LOCAL_TREE") else eng["kv"]["MERLIN_VENV"] + "/bin/python") if eng else None

# ---- R3 the NBDF kit's libcuda stub + venv runner with the FakeL4 platform preamble
STUB = {}
try:
    import ctypes
    try:
        ctypes.CDLL("libcuda.so.1")
        STUB["libcuda_so_1"] = "loadable"
    except OSError as e:
        STUB["libcuda_so_1"] = f"not loadable: {e}"[:200]
    cands = []
    for root in [eng["cuda_home"] if eng else "", "/usr/local/cuda", "/usr/local/cuda/targets/x86_64-linux"]:
        if root:
            cands += glob.glob(f"{root}/lib*/stubs/libcuda.so") + glob.glob(f"{root}/**/stubs/libcuda.so", recursive=True)
    if cands:
        d_ = K / "cudastub"
        d_.mkdir(exist_ok=True)
        if not (d_ / "libcuda.so.1").exists():
            os.symlink(cands[0], d_ / "libcuda.so.1")
        STUB.update(stub=cands[0], dir=str(d_))
except Exception as e:
    STUB["error"] = repr(e)[:300]
REP["checks"]["libcuda_stub"] = STUB
PLATFORM = F["platform_preamble.py"]


def venv_py(code, args=(), name="probe", timeout=1800, env_extra=None, platform=False):
    tree, py = eng["kv"]["MERLIN_TREE"], eng["kv"]["MERLIN_VENV"] + "/bin/python"
    if platform:
        code = PLATFORM + code
        env_extra = dict(env_extra or {}, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")   # = the NBDF kit R9 probe (no VLLM_TARGET_DEVICE)
        if STUB.get("dir"):
            env_extra["LD_LIBRARY_PATH"] = STUB["dir"] + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_PRELOAD")}
    env.update(PYTHONPATH=tree, CUDA_HOME=eng["cuda_home"], VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PATH=f"{eng['kv']['MERLIN_VENV']}/bin:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}")
    if eng["kv"].get("MERLIN_SHIM"):
        env["LD_PRELOAD"] = eng["kv"]["MERLIN_SHIM"]
    env.update(env_extra or {})
    (K / f"{name}.py").write_text(code)
    try:
        r = subprocess.run([py, str(K / f"{name}.py"), *args], capture_output=True, text=True, timeout=timeout, env=env, cwd=str(K))
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (LOGS / f"kit_{name}.log").write_text(out)
    js = next((json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("@JSON ")), None)
    return {"rc": rc, "json": js, "tail": out[-1500:]}


# ---- R4 the CPU tests (BEFORE the seam touches the installed tree: it is the tests' pristine reference)
R4 = {}
try:
    R4["pre_state"] = GS.verify_gentail_patch(TREE)["state"] if TREE else None
    tenv = {k: v for k, v in os.environ.items() if not k.startswith("GENTAIL_")}
    tenv.update(GENTAIL_PRISTINE=str(TREE), GENTAIL_TREE_COPY=str(K / "tree_copy"), GENTAIL_DUCK_TIMELINE=str(B / "duck_timeline"))
    for name, script in (("seam", "test_gentail_seam.py"), ("logic", "test_gentail_logic.py")):
        r = subprocess.run([sys.executable, "-I", str(B / script)], capture_output=True, text=True, timeout=1800, env=tenv, cwd=str(B))
        o = r.stdout + "\n--- stderr ---\n" + r.stderr
        (LOGS / f"test_{name}.log").write_text(o)
        lines = o.splitlines()
        R4[name] = {"rc": r.returncode, "pass": sum(l.startswith("PASS ") for l in lines), "fail": [l[:300] for l in lines if l.startswith("FAIL")][:30],
                    "summary": next((l for l in lines[::-1] if re.match(r"\d+ / \d+ passed", l)), None), "tail": o[-1200:]}
    R4["seam_ok"] = R4["seam"]["rc"] == 0 and R4["seam"]["summary"] == "31 / 31 passed"
    R4["logic_ok"] = R4["logic"]["rc"] == 0 and R4["logic"]["pass"] == 11 and not R4["logic"]["fail"]
    R4["tree_untouched_after_tests"] = GS.verify_gentail_patch(TREE)["state"] == "not_applied"
except Exception as e:
    R4.update(error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
REP["checks"]["R4"] = R4; log("R4 tests", json.dumps({k: (v if not isinstance(v, dict) else {x: v.get(x) for x in ("rc", "pass", "summary", "fail")}) for k, v in R4.items() if k != "tb"})[:1500]); save()

# ---- R5 apply + verify on the INSTALLED tree (py_compile with the Merlin venv python)
R5 = {}
try:
    rec = GS.apply_gentail_patch(TREE, py=VPY)
    R5.update(ok=rec.get("ok") is True and rec["after"]["state"] == "applied", applied=rec.get("applied"), after=rec["after"],
              py_compile=rec.get("py_compile"), tree_state=GS.tree_state(TREE))
except Exception as e:
    R5.update(ok=False, error=f"{type(e).__name__}: {e}"[:1200], tb=traceback.format_exc()[-1500:])
REP["checks"]["R5"] = R5; log("R5 apply", json.dumps({k: v for k, v in R5.items() if k != "tb"})[:900]); save()

# ---- R6 the real-class drive, on and off (only on a patched tree)
R6 = {}
if R5.get("ok"):
    for mode in ("on", "off"):
        r = venv_py(F["kit_realdrive.py"], [mode, str(B / "k100_fixture.json"), "__PIN__", str(K100_DIR) if R1["klee100_config"] else "None"],
                    name=f"realdrive_{mode}", platform=True)
        R6[mode] = {"rc": r["rc"], "json": r["json"], "tail": None if r["json"] else r["tail"]}
        log(f"R6 {mode}", json.dumps({"rc": r["rc"], "ok": (r["json"] or {}).get("ok"), "checks": (r["json"] or {}).get("checks"),
                                      "error": (r["json"] or {}).get("error"), "turns": (r["json"] or {}).get("turns"),
                                      "stats": (r["json"] or {}).get("stats")}, default=str)[:3000])
        REP["checks"]["R6"] = R6; save()
REP["checks"]["R6"] = R6; save()

# ---- R7 revert + verify pristine
R7 = {}
try:
    if TREE is not None and GS.verify_gentail_patch(TREE)["state"] != "not_applied":
        rec = GS.revert_gentail_patch(TREE, py=VPY)
        R7["reverted"] = rec.get("reverted")
    v = GS.verify_gentail_patch(TREE) if TREE else {"state": None}
    R7.update(state=v["state"], ok=v["state"] == "not_applied", shas={k: x["sha"] for k, x in (v.get("files") or {}).items()})
except Exception as e:
    R7.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
REP["checks"]["R7"] = R7; log("R7 revert", json.dumps({k: v for k, v in R7.items() if k != "tb"})[:900]); save()

on, off = (R6.get("on") or {}).get("json") or {}, (R6.get("off") or {}).get("json") or {}
verdict({"merlin_pay_found": MERLIN_PAY is not None, "merlin_install_ok": R2.get("ok") is True,
         "installed_tree_pristine_before": R4.get("pre_state") == "not_applied",
         "test_gentail_seam_31_31": R4.get("seam_ok") is True, "test_gentail_logic_11_11": R4.get("logic_ok") is True,
         "tests_left_tree_untouched": R4.get("tree_untouched_after_tests") is True,
         "seam_apply_verify_ok": R5.get("ok") is True,
         "real_drive_on_ok": on.get("ok") is True, "real_drive_off_ok": off.get("ok") is True,
         "revert_pristine_ok": R7.get("ok") is True},
        recorded={"on_checks": on.get("checks"), "off_checks": off.get("checks"), "on_turns": on.get("turns"), "off_turns": off.get("turns"),
                  "on_stats": on.get("stats"), "published": on.get("published"), "kv_config": on.get("kv_config"),
                  "managers": on.get("managers"), "coordinator": on.get("coordinator"), "ctl": on.get("ctl"),
                  "on_error": on.get("error"), "off_error": off.get("error"), "klee100_config": R1.get("klee100_config")})
