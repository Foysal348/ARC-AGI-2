"""E29_DFLASH_D3S rev 2d adversarial regression tests (Experiment 29, 2026-11-02): every audit regression of the four rev 2c
audits (A1 RISK_AUDIT, A2 = A3 GPT_AUDIT, A4 INDEPENDENT_AUDIT; FIX_LIST_E29_REV2C_AUDITS_2026-11-02.md F1..F14) as an
executable plain-assert case against the REAL modules (no reimplementation of the logic under test):
  e29_lib.py, e29_direct.py, e29d_sched_patch.py (+ e29_sched_patch.py / e28_sched_patch.py / the Duck payload arc2_duck.py),
  the generated e29_nbx.py (exec'd like the notebook's e29_extras), and the notebook's own functions extracted by AST from
  e29_dflash_d3s_code_export.py (= audit_repro_e29.py's technique), dflash_seam.py, e28k100_lib.py, nb5_lib.py.
Stubs are only the environment (a scripted nb5_boot_try, a fake server object, a fake seam sequence, a DIRECT stub run, a fake clock).

usage:  python e29_rev2d_adversarial_tests.py [--dir DIR] [--export FILE] [--only F1,F10,...] [--strict] [--no-fallback]
                                              [--json-out FILE] [-v]
  --dir          the folder with the modules (default: this file's folder = Build, or the extracted audit zip)
  --export       the code export (default DIR/e29_dflash_d3s_code_export.py)
  --no-fallback  never look outside DIR (the audit-zip self-containment check); default: the Experiment 15 / 25 / 27 / 28
                 Build folders are searched for e28_sched_patch.py / nb5_lib.py / e28k100_lib.py / dflash_seam.py / the Duck
                 payload q27x_e15b_duck_9e4b89285136.bin / the notebook when DIR lacks them
  --strict       SKIP_NOT_IMPLEMENTED / SKIP_NO_INPUT also count as failures (use this on the rebuilt rev 2d)
Result per case: PASS | FAIL | SKIP_NOT_IMPLEMENTED (a rev 2d function / record is not in this build yet) | SKIP_NO_INPUT.
Exit 0 = no FAIL (and, with --strict, no SKIP); 1 otherwise.

EXPECTED AFTER THE rev 2d REBUILD: every case PASS with --strict (no FAIL, no SKIP). The column "rev 2c" below is what the
rev 2c code (notebook ed2e23a7) gives: FAIL = the audit bug reproduced; SKIP_NOT_IMPLEMENTED = the rev 2d function is absent.

RESULTS RECORDED 2026-11-02 (Python 3.14.6, Windows; the runs are re-done by the coordinator after the rebuild):
  rev 2c = the files of Audit Zips/E29_DFLASH_D3S_REV2C_audit_FLAT_2026-11-02.zip (notebook ed2e23a7) + e28_sched_patch.py,
  nb5_lib.py, e28k100_lib.py and the Duck payload (folder _scratch/e29_rev2d_adv/rev2c_dir):
     58 cases: 13 PASS, 25 FAIL, 20 SKIP_NOT_IMPLEMENTED, 0 SKIP_NO_INPUT
     (58/58 equal to the rev 2c column; exit 1). FAIL = F10.1 F10.2 F10.6 (in-run XID -> same DFlash config), F4.1 F4.2 F4.3
     (no E28 boot gates armed on the MTP relaunch), F3.1-F3.7 (QSA read-back record-only: no label, no RPC fallback, no
     256/256 same-rung relaunch), F2.1 (no low-free ladder step), F1.7 F1.8b F1.1b F1.2-F1.5 (late phase C: DIRECT
     unconditionally, no E29D_CAN_START), A4-4 A4-5 A4-6 A4-7 (the four audit_repro bugs). SKIP_NOT_IMPLEMENTED = F4.4-F4.7,
     F11.1-F11.5, F2.3, F12.1-F12.7, F5.1, F5.3, F1.9. PASS = F10.3-5 F10.7 F10.8 F2.2 F5.2 F1.8a F1.1 F1.6 A4-1..3.
  Build folder at 2026-11-02 05:18 (local machine clock) (already rebuilt by E29-B / SCHED: e29_lib 99f97226, e29d_sched_patch 4cc807f2,
  e29_nbx 1388aa90, code export 01787a21, notebook 80d4aeff), python e29_rev2d_adversarial_tests.py --strict:
     58 cases: 58 PASS, 0 FAIL, 0 SKIP (exit 0). The coordinator re-runs it on the final rev 2d build (expected: the same).
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import types
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(r"D:/ARC AGI 2")
PAYLOAD_NAME = "q27x_e15b_duck_9e4b89285136.bin"
PAYLOAD_SHA = "9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566"
NB_NAME = "Q27-XHIGH-E29_DFLASH_D3S.ipynb"
_E28B = ROOT / "experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build"
FALLBACK = {
    "e28_sched_patch.py": [_E28B], "e28k100_lib.py": [_E28B],
    "nb5_lib.py": [ROOT / "experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build"],
    "dflash_seam.py": [ROOT / "experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build"],
    PAYLOAD_NAME: [ROOT / "experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload"],
    NB_NAME: [SCRIPT_DIR.parent / "Experiment Notebooks"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(SCRIPT_DIR))
ap.add_argument("--export", default=None)
ap.add_argument("--only", default="")
ap.add_argument("--strict", action="store_true")
ap.add_argument("--no-fallback", action="store_true")
ap.add_argument("--json-out", default=None)
ap.add_argument("-v", action="store_true")
ARGS = ap.parse_args()
DIR = Path(ARGS.dir).resolve()
if DIR != SCRIPT_DIR and str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(DIR))


def find(name):
    p = DIR / name
    if p.is_file():
        return p
    if not ARGS.no_fallback:
        for d in FALLBACK.get(name, []):
            if (Path(d) / name).is_file():
                return Path(d) / name
    return None


def load(modname, required=True):
    p = find(modname + ".py")
    if p is None:
        if required:
            raise SystemExit(f"FATAL: {modname}.py not found in {DIR}" + ("" if ARGS.no_fallback else " (nor in the fallback folders)"))
        return None
    spec = importlib.util.spec_from_file_location(modname, str(p))
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    m.__adv_path__ = str(p)
    return m


def sha256(b):
    return hashlib.sha256(b).hexdigest()


E28S = load("e28_sched_patch")        # imported by name by e29_sched_patch / e29d_sched_patch
E28L = load("e28k100_lib")
NB5L = load("nb5_lib")
DSM = load("dflash_seam")
L = load("e29_lib")
D = load("e29_direct")
E29S = load("e29_sched_patch")
DS29 = load("e29d_sched_patch")
EXPORT = Path(ARGS.export) if ARGS.export else DIR / "e29_dflash_d3s_code_export.py"
EXPORT_SRC = EXPORT.read_text(encoding="utf-8")
FN, LITERALS = {}, {}
for _n in ast.parse(EXPORT_SRC).body:
    if isinstance(_n, ast.FunctionDef):
        FN[_n.name] = _n                  # the last definition wins (= exec)
    elif isinstance(_n, ast.Assign) and len(_n.targets) == 1 and isinstance(_n.targets[0], ast.Name):
        try:                              # module-level literal state the functions use (e.g. E29_SEAM_BYTES = {})
            LITERALS[_n.targets[0].id] = ast.literal_eval(_n.value)
        except Exception:
            pass
FACTS = json.loads((DIR / "e29_dflash_d3s_build_facts.json").read_text(encoding="utf-8"))
STEPS = [float(x) for x in FACTS["pin_steps_gib"]]
TOP = int(FACTS["pins"][0])
NBX = find("e29_nbx.py")


# ------------------------------------------------------------------ the case runner
class NotImpl(Exception):
    pass


class NoInput(Exception):
    pass


CASES, RESULTS = [], []


def case(cid, title, rev2c):
    """rev2c = what the rev 2c code gives (FAIL = the audit bug reproduced). Expected after the rev 2d rebuild: PASS."""
    def deco(fn):
        CASES.append((cid, title, rev2c, fn))
        return fn
    return deco


def need_fn(*names):
    miss = [n for n in names if n not in FN]
    if miss:
        raise NotImpl("the code export has no " + ", ".join(miss))


def need_lib(*names):
    miss = [n for n in names if not hasattr(L, n)]
    if miss:
        raise NotImpl("e29_lib has no " + ", ".join(miss))


def fns_exec(ns, names, optional=()):
    need_fn(*names)
    body = [FN[n] for n in names] + [FN[n] for n in optional if n in FN]
    used = {x.id for f in body for x in ast.walk(f) if isinstance(x, ast.Name)}
    for n in sorted(used):
        if n not in ns and n in LITERALS:
            ns[n] = copy.deepcopy(LITERALS[n])
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "<E29 export: " + ",".join(names) + ">",
                 "exec"), ns)
    return ns


def src_of(name):
    return ast.get_source_segment(EXPORT_SRC, FN[name]) or ""


def noop(*a, **k):
    return None


# ------------------------------------------------------------------ shared fixtures (environment only)
TAIL = L.df_spec_argv('{"method":"dflash","num_speculative_tokens":3}', '{"cudagraph_capture_sizes":[4,8,12]}')
CFG_DF = {"drafter": "dflash_d3s", "embed_host": True, "attempts_supply": True}    # = the E29_DFLASH_D3S variant
QSA_SWITCH = {"max_logits_mb": 128, "granule_mb": 64}                                # = CELL 1's E29_QSA
LEVER_ENV = {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}
MTP_LEVERS = dict(LEVER_ENV, DRAFT_HEAD="int8")
K0, K1 = L.QSA_KEYS
OOM = "ERROR [gpu_model_runner.py] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
XID = "kernel: NVRM: Xid (PCI:0000:00:05): 79, pid=4242, name=VLLM::Worker, GPU has fallen off the bus."
HEALTHY = "INFO [launch.py] EngineCore stopped: SIGTERM from the watchdog (health probe timeout)"


def df_rt():
    return {"k": 3, "tail": list(TAIL), "mask_row_sha": "ab" * 32, "sw_block": 1024}


def base_ns():
    return {"__name__": "e29_adv_ns", "json": json, "time": time, "os": os, "re": re, "hashlib": hashlib, "Path": Path,
            "traceback": traceback, "E29L": L, "DS": DSM, "E28L": E28L, "NB5L": NB5L, "E29_CFG": L.check_cfg(CFG_DF),
            "E29_QSA": L.qsa_cfg(QSA_SWITCH), "E28_I8_ENV": {"DRAFT_HEAD": "int8"},
            "E28_I8": {"want": True, "on": False, "suspended": "N/A_DFLASH", "dropped": None}, "K100_PIN_STEPS_GIB": STEPS,
            "NB5_ACTIVE_BOOT": {}, "SSEQ": None, "FAKE": False, "NOTEBOOK_T0": time.perf_counter(), "log": noop, "E11_SERVER": None,
            "E29_RT": {"cfg": dict(CFG_DF), "on": ["df"], "dropped": [], "errors": [], "df": df_rt(), "df_try": True, "df_live": True,
                       "df_cur": None, "relaunch_plans": [], "seam": None, "tries": [], "canary": []}}


class Old:
    """The dead server the E15B watchdog hands to e21_relaunch (only .log())."""
    def __init__(self, text):
        self.text = text

    def log(self):
        return self.text


RELAUNCH_FNS = ["e29_relaunch_plan", "e29_mtp_boot", "e29_drop", "e29_env"]
RELAUNCH_OPT = ["e29_seam_restore", "e29_seam_snapshot"]


def relaunch_ns(rung):
    ns = fns_exec(base_ns(), RELAUNCH_FNS, RELAUNCH_OPT)
    boot_at(ns, rung)
    return ns


def boot_at(ns, rung):
    """NB5_ACTIVE_BOOT = the booted D rung `rung` (what nb5_boot leaves for the watchdog: the lever env + the try's e29_env)."""
    t = next(x for x in L.df_tries(True) if x["df"]["rung"] == rung)
    ns["E29_RT"]["df_cur"] = dict(t["df"])
    ns["NB5_ACTIVE_BOOT"].clear()
    ns["NB5_ACTIVE_BOOT"].update(name=f"merlin_fp8_mtp_pin{int(round(t['gib'] * 100))}_df3r{rung}", kv_bytes=t["kv_bytes"],
                                 pin_gib=t["gib"], levers=dict(LEVER_ENV, **ns["e29_env"](t["df"])),
                                 extra=["--max-num-seqs", "24"] + list(TAIL), kv="fp8")
    return t


def mtp_top_problems(ns):
    a, p = ns["NB5_ACTIVE_BOOT"], []
    if a.get("kv_bytes") != 7933526016 or a.get("kv_bytes") != L.D_PIN0 or a.get("kv_bytes") != TOP:
        p.append(f"kv_bytes {a.get('kv_bytes')} != the top pin 7,933,526,016")
    if a.get("pin_gib") is None or abs(float(a["pin_gib"]) - STEPS[0]) > 1e-9:
        p.append(f"pin_gib {a.get('pin_gib')} != {STEPS[0]}")
    if a.get("extra") != ["--max-num-seqs", "24"]:
        p.append(f"argv {a.get('extra')} != ['--max-num-seqs', '24'] (DFlash tail must be gone, 24 seqs kept)")
    if a.get("levers") != MTP_LEVERS:
        p.append(f"levers {a.get('levers')} != {MTP_LEVERS} (no DFLASH_* / QSA env, INT8 head)")
    if "df" in ns["E29_RT"]["on"]:
        p.append("DFlash still on")
    if ns["E28_I8"].get("on") is not True:
        p.append(f"E28 I8 not restored: {ns['E28_I8']}")
    return p


# ==================================================================== F10: in-run relaunch plan (e29_lib.df_relaunch_plan)
TRIES = L.df_tries(True)
RUNGS = [t["df"]["rung"] for t in TRIES]


@case("F10.1", "df_relaunch_plan: XID only (NVRM: Xid 79) -> 'mtp' (device fault first: DFlash dropped, never the same config)", "FAIL")
def _f10_1():
    p = L.df_relaunch_plan({"rung": 0}, XID, TRIES)
    assert p == "mtp", f"plan {p!r} (rev 2c: None = the watchdog relaunches the SAME DFlash config)"


@case("F10.2", "df_relaunch_plan: XID + OOM in one log -> 'mtp' (the device fault wins over the OOM rung step)", "FAIL")
def _f10_2():
    for rung in RUNGS:
        p = L.df_relaunch_plan({"rung": rung}, XID + "\n" + OOM, TRIES)
        assert p == "mtp", f"rung {rung}: plan {p!r}"
    p = L.df_relaunch_plan({"rung": 0}, "CUDA error: an illegal memory access was encountered\n" + OOM, TRIES)
    assert p == "mtp", f"illegal memory access + OOM: plan {p!r}"


@case("F10.3", "df_relaunch_plan: a plain OOM on every rung but the last -> the next D rung (EMBED_HOST kept)", "PASS")
def _f10_3():
    for j, rung in enumerate(RUNGS[:-1]):
        p = L.df_relaunch_plan({"rung": rung}, OOM, TRIES)
        assert isinstance(p, dict) and p["df"]["rung"] == RUNGS[j + 1] and p["kv_bytes"] == TRIES[j + 1]["kv_bytes"], (rung, p)
        assert p["df"]["embed_host"] is True, p


@case("F10.4", "df_relaunch_plan: an OOM on the last D rung -> 'mtp'", "PASS")
def _f10_4():
    assert L.df_relaunch_plan({"rung": RUNGS[-1]}, OOM, TRIES) == "mtp"


@case("F10.5", "df_relaunch_plan: a healthy non-OOM death (no XID) -> None (E15B: the same config again)", "PASS")
def _f10_5():
    for t in (HEALTHY, "", "RuntimeError: Engine core initialization failed"):
        assert L.df_relaunch_plan({"rung": 0}, t, TRIES) is None, repr(t)


@case("F10.6", "notebook e29_relaunch_plan: in-run XID -> the E28 MTP top rung (pin 7,933,526,016, 24 seqs, I8, no DFlash / QSA "
      "env), recorded GPU_DEVICE_FAULT", "FAIL")
def _f10_6():
    for log_ in (XID, XID + "\n" + OOM):
        ns = relaunch_ns(RUNGS[0])
        ns["e29_relaunch_plan"](Old(log_))
        pr = mtp_top_problems(ns)
        assert not pr, f"log {log_[:40]!r}...: {pr} (rev 2c: the DFlash plan is left unchanged)"
        assert ns["E29_RT"]["relaunch_plans"][-1]["plan"] == "mtp", ns["E29_RT"]["relaunch_plans"]
        rec = json.dumps({k: ns["E29_RT"].get(k) for k in ("gpu_device_fault", "dropped")}, default=str)
        assert "GPU_DEVICE_FAULT" in rec or ns["E29_RT"].get("gpu_device_fault"), "no GPU_DEVICE_FAULT record: " + rec[:300]
        assert not ns["E29_RT"]["errors"], ns["E29_RT"]["errors"]


@case("F10.7", "notebook e29_relaunch_plan: a plain in-run OOM on each rung -> the next D rung (pin, EMBED_HOST, seam + QSA env, "
      "argv unchanged); the last rung -> the MTP top rung", "PASS")
def _f10_7():
    for j, rung in enumerate(RUNGS):
        ns = relaunch_ns(rung)
        ns["e29_relaunch_plan"](Old(OOM))
        a = ns["NB5_ACTIVE_BOOT"]
        if j + 1 < len(RUNGS):
            nxt = TRIES[j + 1]
            assert a["kv_bytes"] == nxt["kv_bytes"] and ns["E29_RT"]["df_cur"]["rung"] == nxt["df"]["rung"], (rung, a["kv_bytes"])
            assert a["levers"].get("DFLASH_DRAFT_EMBED_HOST") == "1" and a["levers"].get("DFLASH_SEAM") == "1", a["levers"]
            assert a["levers"].get(K0) == "128" and a["levers"].get(K1) == "64", a["levers"]
            assert a["extra"] == ["--max-num-seqs", "24"] + list(TAIL) and "df" in ns["E29_RT"]["on"], a["extra"]
        else:
            assert not mtp_top_problems(ns), mtp_top_problems(ns)


@case("F10.8", "notebook e29_relaunch_plan: a healthy death -> the booted DFlash config unchanged (plan None recorded)", "PASS")
def _f10_8():
    ns = relaunch_ns(2)
    before = json.loads(json.dumps(ns["NB5_ACTIVE_BOOT"]))
    ns["e29_relaunch_plan"](Old(HEALTHY))
    assert ns["NB5_ACTIVE_BOOT"] == before and "df" in ns["E29_RT"]["on"], ns["NB5_ACTIVE_BOOT"]
    assert ns["E29_RT"]["relaunch_plans"][-1]["plan"] is None, ns["E29_RT"]["relaunch_plans"]


# ==================================================================== F4: the in-run DFlash -> MTP relaunch + the E28 boot gates
@case("F4.1", "e29_mtp_boot: the E28 top rung (7,933,526,016 B, 24 seqs, INT8 head, no DFlash argv / seam / QSA env) AND the "
      "relaunch is marked for the E28 boot gates (E29_RT['mtp_relaunch'])", "FAIL")
def _f4_1():
    ns = relaunch_ns(RUNGS[0])
    ns["e29_mtp_boot"]("TEST_INRUN_FALLBACK")
    assert not mtp_top_problems(ns), mtp_top_problems(ns)
    assert ns["E29_RT"].get("mtp_relaunch") is True, "the relaunch is not marked for the boot gates (C-11 / method map / I8 read-back)"


@case("F4.2", "in-run zero-accept canary (e29_inrun_check): server killed once, the MTP top rung + I8, gates armed", "FAIL")
def _f4_2():
    ns = relaunch_ns(RUNGS[0])
    fns_exec(ns, ["e29_inrun_check"])
    kills = []
    ns["E11_SERVER"] = types.SimpleNamespace(kill=lambda: kills.append(1))
    rows = [{"t": 0.0, "sd_drafts": 0, "sd_draft_tok": 0, "sd_acc": 0}, {"t": 15.0, "sd_drafts": 120, "sd_draft_tok": 360, "sd_acc": 0}]
    ns["e29_inrun_check"](rows)
    ns["e29_inrun_check"](rows)                                      # fires once
    assert len(kills) == 1 and ns["E29_RT"].get("inrun_fired"), (kills, ns["E29_RT"].get("inrun"))
    assert not mtp_top_problems(ns), mtp_top_problems(ns)
    assert ns["E29_RT"].get("mtp_relaunch") is True, "gates not armed for the relaunch"


@case("F4.3", "injected in-run OOM on every rung in turn (0 -> 2 -> 3 -> 4 -> MTP): the final relaunch = the MTP top rung, gates "
      "armed; then a further death does not touch the MTP plan", "FAIL")
def _f4_3():
    ns = relaunch_ns(RUNGS[0])
    seen = [ns["E29_RT"]["df_cur"]["rung"]]
    for _ in range(len(RUNGS)):
        ns["e29_relaunch_plan"](Old(OOM))
        if "df" not in ns["E29_RT"]["on"]:
            break
        seen.append(ns["E29_RT"]["df_cur"]["rung"])
    assert seen == RUNGS, (seen, RUNGS)
    assert not mtp_top_problems(ns), mtp_top_problems(ns)
    assert ns["E29_RT"].get("mtp_relaunch") is True, "gates not armed"
    before = json.loads(json.dumps(ns["NB5_ACTIVE_BOOT"]))
    ns["e29_relaunch_plan"](Old(OOM))                                # DFlash is off: e29_relaunch_plan is a no-op
    assert ns["NB5_ACTIVE_BOOT"] == before


I8_ARGV = ["--max-num-seqs", "24", "--speculative-config",
           json.dumps({"method": "mtp", "num_speculative_tokens": 3, "draft_lm_head_quant": "int8"})]
I8_LOG = "\n".join(f"(Worker_PP{p}_TP{t} pid={100 + 2 * p + t}) INFO Draft lm_head: INT8 weight-only copy of the 124160x2048 "
                   f"shard (242.5 MiB) built" for p in (0, 1) for t in (0, 1))


class Srv:
    def __init__(self, log_text, argv=None):
        self.text, self.argv, self.kills, self.port = log_text, list(argv or I8_ARGV), 0, 8000

    def log(self):
        return self.text

    def kill(self):
        self.kills += 1


def gates_ns(c11=True, mm_fatal=None):
    ns = base_ns()
    fns_exec(ns, ["e29_relaunch_gates"])
    ns["NB5L"] = types.SimpleNamespace(config_readback=lambda log_, argv, kv, mtp: {"ok": bool(c11), "verdict": "PASS" if c11 else "X"})
    ns["k100_boot_gates"] = lambda srv, x: {"ok": mm_fatal is None, "fatal": mm_fatal}
    ns["NB5_BOOT_KV"] = "fp8"
    ns["E28_I8"] = {"want": True, "on": True, "dropped": None}
    ns["E29_RT"].update(on=[], df_live=False, df_try=False, mtp_relaunch=True)
    ns["NB5_ACTIVE_BOOT"].update(name="inrun_mtp", kv_bytes=TOP, pin_gib=STEPS[0], levers=dict(MTP_LEVERS), extra=["--max-num-seqs", "24"])
    return ns


@case("F4.4", "e29_relaunch_gates on the NEW MTP server: C-11 + method map + I8 read-back pass -> ok, no kill, gates disarmed, "
      "recorded", "SKIP_NOT_IMPLEMENTED")
def _f4_4():
    assert E28L.i8_readback(I8_LOG, I8_ARGV, "int8")["label"] == "I8_ENGAGED", "fixture: the I8 log is not I8_ENGAGED"
    ns, s = gates_ns(), Srv(I8_LOG)
    ok, why = ns["e29_relaunch_gates"](s, "relaunch ok")
    assert ok is True and s.kills == 0, (ok, why, s.kills)
    assert ns["E29_RT"]["mtp_relaunch"] is False and ns["E28_I8"]["on"] is True
    g = ns["E29_RT"]["mtp_relaunch_gates"][-1]
    assert g.get("c11") is True and g.get("i8") == "I8_ENGAGED" and g.get("kv_bytes") == TOP, g


@case("F4.5", "e29_relaunch_gates: the INT8 head NOT engaged on the relaunch -> E28's own I8 rule (server killed, the same pin with "
      "the BF16 head, gates stay armed for that retry)", "SKIP_NOT_IMPLEMENTED")
def _f4_5():
    no_head = "INFO non-INT8 boot: no draft head line"
    assert E28L.i8_readback(no_head, I8_ARGV, "int8")["label"] != "I8_ENGAGED"
    ns, s = gates_ns(), Srv(no_head)
    ok, why = ns["e29_relaunch_gates"](s, "relaunch ok")
    assert ok is False and s.kills == 1, (ok, why, s.kills)
    assert ns["E28_I8"]["on"] is False and (ns["E28_I8"].get("dropped") or {}).get("kind"), ns["E28_I8"]
    lv = ns["NB5_ACTIVE_BOOT"]["levers"]
    assert "DRAFT_HEAD" not in lv and lv == LEVER_ENV and ns["NB5_ACTIVE_BOOT"]["kv_bytes"] == TOP, lv
    assert ns["E29_RT"]["mtp_relaunch"] is True


@case("F4.6", "e29_relaunch_gates: C-11 failed or a method-map fatal -> this relaunch failed (killed, gates armed); I8 untouched",
      "SKIP_NOT_IMPLEMENTED")
def _f4_6():
    for kw in ({"c11": False}, {"mm_fatal": ["method map: mtp expected, got dflash"]}):
        ns, s = gates_ns(**kw), Srv(I8_LOG)
        ok, why = ns["e29_relaunch_gates"](s, "relaunch ok")
        assert ok is False and s.kills == 1 and ns["E29_RT"]["mtp_relaunch"] is True, (kw, ok, why, s.kills)
        assert ns["E28_I8"]["on"] is True and ns["NB5_ACTIVE_BOOT"]["levers"] == MTP_LEVERS, (kw, ns["E28_I8"])


def _calls_to(name):
    out = []
    for fname, node in FN.items():
        if fname == name:
            continue
        for x in ast.walk(node):
            if isinstance(x, ast.Call) and getattr(x.func, "id", getattr(x.func, "attr", None)) == name:
                out.append(fname)
                break
    return out


@case("F4.7", "static: the watchdog relaunch path calls e29_relaunch_gates when E29_RT['mtp_relaunch'] is set", "SKIP_NOT_IMPLEMENTED")
def _f4_7():
    need_fn("e29_relaunch_gates")
    who = _calls_to("e29_relaunch_gates")
    assert who, "e29_relaunch_gates is never called"
    assert any("mtp_relaunch" in src_of(w) for w in who), f"callers {who} do not test E29_RT['mtp_relaunch']"


# ==================================================================== F11: the seam revert (pristine-SHA verify + re-restore + inert read-back)
class FakeSeam:
    """The SeamSeq stand-in: finish() moves the fake Merlin tree as a real revert would (all / part / a file made unwritable),
    then returns or raises."""
    def __init__(self, tree, rels, mode):
        self.tree, self.rels, self.mode, self.ring, self.state = Path(tree), list(rels), mode, True, "applied"

    def finish(self):
        half = self.rels[: len(self.rels) // 2]
        todo = self.rels if self.mode == "clean" else half
        for r in todo:
            (self.tree / r).write_bytes(_orig_bytes(r))
        if self.mode == "unrestorable":
            bad = self.tree / self.rels[-1]
            bad.unlink()
            bad.mkdir()                                       # a directory where the file was: no write can restore it
            (bad / "x").write_text("x")
        if self.mode == "clean":
            self.state = "reverted"
            return {"reverted": todo}
        raise RuntimeError("injected: SeamSeq.finish() failed after a partial revert")


def _orig_bytes(rel):
    return f"# pristine Merlin file {rel}\nX = 1\n".encode()


def seam_case(mode):
    need_fn("e29_seam_snapshot", "e29_seam_restore")
    need_lib("seam_dirty")
    tree = Path(tempfile.mkdtemp(prefix="e29adv_seam_"))
    rels = list(DSM._rels(True, True))
    for r in rels:
        (tree / r).parent.mkdir(parents=True, exist_ok=True)
        (tree / r).write_bytes(_orig_bytes(r))
    ns = fns_exec(base_ns(), ["e29_drop", "e29_seam_snapshot", "e29_seam_restore"])
    ns["e29_seam_snapshot"](tree)
    for r in rels:                                            # = SeamSeq.apply(): the seam edits on the tree
        (tree / r).write_bytes(_orig_bytes(r) + b"# E27 DFLASH_SEAM patched\n")
    ns["SSEQ"] = FakeSeam(tree, rels, mode)
    ns["E28_I8"] = {"want": True, "on": False, "suspended": "N/A_DFLASH", "dropped": None}
    ns["e29_drop"]("df", "TEST_" + mode.upper(), "D_r0")
    return ns, tree, rels


def _pristine(tree, rels):
    return [r for r in rels if not (tree / r).is_file() or (tree / r).read_bytes() != _orig_bytes(r)]


@case("F11.1", "SSEQ.finish() raises after a partial revert -> the rest is re-restored from the pre-apply snapshot, every seam "
      "file back at its pristine sha (SEAM_PRISTINE); the run continues on MTP (I8 on, DFlash dropped)", "SKIP_NOT_IMPLEMENTED")
def _f11_1():
    ns, tree, rels = seam_case("raise_partial")
    rr = ns["E29_RT"].get("seam_restore") or {}
    assert not _pristine(tree, rels), f"not pristine: {_pristine(tree, rels)}"
    assert rr.get("label") == "SEAM_PRISTINE" and rr.get("dirty_after_finish"), rr
    assert ns["E29_RT"].get("seam_unverified") is False, ns["E29_RT"].get("seam_unverified")
    assert any("seam finish" in e for e in ns["E29_RT"]["errors"]), ns["E29_RT"]["errors"]
    assert "df" not in ns["E29_RT"]["on"] and ns["E28_I8"]["on"] is True


@case("F11.2", "partial restore impossible (a seam file replaced by a directory) -> SEAM_REVERT_UNVERIFIED, seam_unverified True "
      "(the next MTP boot needs the q_seam_inert read-back); never raises, the run continues", "SKIP_NOT_IMPLEMENTED")
def _f11_2():
    ns, tree, rels = seam_case("unrestorable")
    rr = ns["E29_RT"].get("seam_restore") or {}
    assert ns["E29_RT"].get("seam_unverified") is True and rr.get("label") == "SEAM_REVERT_UNVERIFIED", rr
    assert rels[-1] in (rr.get("dirty_after_restore") or []), rr
    assert "df" not in ns["E29_RT"]["on"] and ns["E28_I8"]["on"] is True


@case("F11.3", "a clean SSEQ.finish() -> nothing dirty after finish, SEAM_PRISTINE, no error", "SKIP_NOT_IMPLEMENTED")
def _f11_3():
    ns, tree, rels = seam_case("clean")
    rr = ns["E29_RT"].get("seam_restore") or {}
    assert rr.get("dirty_after_finish") == [] and rr.get("label") == "SEAM_PRISTINE", rr
    assert not _pristine(tree, rels) and not ns["E29_RT"]["errors"], ns["E29_RT"]["errors"]
    assert ns["E29_RT"].get("seam_unverified") is False


@case("F11.4", "seam_inert (= nbdf q_seam_inert): an MTP boot log -> SEAM_REVERT_UNVERIFIED_INERT; a DFlash2 model or seam events "
      "-> SEAM_NOT_INERT", "SKIP_NOT_IMPLEMENTED")
def _f11_4():
    need_lib("seam_inert")
    ok = L.seam_inert("INFO Resolved architecture: Qwen4ExpForCausalLM\nINFO Resolved architecture: Qwen4ExpMTP\n")
    assert ok["ok"] is True and ok["label"] == "SEAM_REVERT_UNVERIFIED_INERT", ok
    for bad in ("INFO Resolved architecture: DFlash2DraftModel", 'INFO E27_DFLASH {"event": "taps", "rank": 0}'):
        r = L.seam_inert("INFO Resolved architecture: Qwen4ExpForCausalLM\n" + bad)
        assert r["ok"] is False and r["label"] == "SEAM_NOT_INERT", (bad, r)


@case("F11.5", "static: snapshot before SeamSeq.apply, restore after every finish (drop + setup failure), the inert read-back in "
      "nb5_boot_try under seam_unverified (E29_SEAM_NOT_INERT = a real failure of that try in nb5_boot)", "SKIP_NOT_IMPLEMENTED")
def _f11_5():
    need_fn("e29_seam_snapshot", "e29_seam_restore")
    s = src_of("e29_df_setup")
    assert "e29_seam_snapshot(" in s and s.index("e29_seam_snapshot(") < s.index("SSEQ.apply()"), "no snapshot before SSEQ.apply()"
    assert s.count("e29_seam_restore()") >= 1, "the setup failure path does not restore"
    assert "e29_seam_restore()" in src_of("e29_drop"), "e29_drop does not restore"
    t = src_of("nb5_boot_try")
    assert "seam_inert(" in t and "seam_unverified" in t and "E29_SEAM_NOT_INERT" in t, "no inert gate in nb5_boot_try"
    assert "E29_SEAM_NOT_INERT" in src_of("nb5_boot"), "nb5_boot does not fail the try on E29_SEAM_NOT_INERT"


# ==================================================================== F3 / F14: the QSA 128/64 read-back gate
def wenv(a, b):
    return {K0: a, K1: b}


@case("F3.1", "qsa_readback: 4 ranks at 128/64 -> env_ok, QSA_VERIFIED", "FAIL")
def _f3_1():
    r = L.qsa_readback([{"env": wenv("128", "64")}] * 4, L.qsa_env(L.qsa_cfg(QSA_SWITCH)))
    assert r["env_ok"] is True and r.get("label") == "QSA_VERIFIED", r


@case("F3.2", "qsa_readback: one rank at Merlin's 256 -> not ok, QSA_ENV_MISMATCH", "FAIL")
def _f3_2():
    r = L.qsa_readback([{"env": wenv("128", "64")}] * 3 + [{"env": wenv("256", "256")}], L.qsa_env(L.qsa_cfg(QSA_SWITCH)))
    assert r["env_ok"] is False and r.get("label") == "QSA_ENV_MISMATCH", r


@case("F3.3", "qsa_readback: a rank missing (3 of 4) -> not ok, UNVERIFIED_QSA (F14: no QSA speed claim)", "FAIL")
def _f3_3():
    r = L.qsa_readback([{"env": wenv("128", "64")}] * 3, L.qsa_env(L.qsa_cfg(QSA_SWITCH)))
    assert r["env_ok"] is False and r.get("label") == "UNVERIFIED_QSA", r


class QSrv:
    def __init__(self, text=""):
        self.text, self.proc, self.port, self.name, self.env = text, types.SimpleNamespace(pid=999999999), 8000, "D_r0", {}

    def log(self):
        return self.text

    def alive(self):
        return True


def rpc_with(ranks):
    calls = []

    def rpc(port, method, *a, **k):
        calls.append((method, a))
        return {"results": [{"rank": i, "env": e} for i, e in enumerate(ranks)]}
    return rpc, calls


@case("F3.4", "notebook e29_qsa_rb: /proc gives < 4 workers -> the worker-rank RPC read-back (4 ranks ok -> env_ok; a 256 rank or "
      "a missing rank -> not ok)", "FAIL")
def _f3_4():
    want = L.qsa_env(L.qsa_cfg(QSA_SWITCH))
    for ranks, exp in (([wenv("128", "64")] * 4, True), ([wenv("128", "64")] * 3 + [wenv("256", "256")], False),
                       ([wenv("128", "64")] * 3, False)):
        ns = fns_exec(base_ns(), ["e29_qsa_rb"])
        ns["rpc"], calls = rpc_with(ranks)
        try:
            r = ns["e29_qsa_rb"](QSrv(), want)
        except TypeError:
            r = ns["e29_qsa_rb"](QSrv())                      # rev 2c signature (s) only
        assert bool(r.get("env_ok")) is exp, (len(ranks), r, calls)


class DSFake:
    ALL_ENVS = DSM.ALL_ENVS

    @staticmethod
    def readback_label(text, **kw):
        return "ENGAGED", {"aux_taps": 4}

    @staticmethod
    def verify_dflash_patch(tree, ring=True):
        return {"files": {L.GRD_REL: {"state": "applied", "sha": L.GRD_POST}}}


def readback_ns(ranks, qsa_off=False):
    ns = fns_exec(base_ns(), ["e29_df_readback", "e29_qsa_rb"], ["e29_ledger"])
    ns["DS"], ns["rpc"] = DSFake, rpc_with(ranks)[0]
    ns["SSEQ"] = types.SimpleNamespace(ring=True, check=lambda name: (True, "ok"))
    ns["E29_RT"]["df"]["tree"] = "/merlin"
    ns["e29_ledger"] = lambda port: {"ledger": "stub"}
    if qsa_off:
        ns["E29_RT"]["qsa_off"] = True
    return ns


@case("F3.5", "notebook e29_df_readback: an ENGAGED DFlash boot whose workers do not all show 128/64 -> QSA_ENV_MISMATCH (a real "
      "failure before the first prefill); all 4 ok -> ENGAGED; qsa_off (256/256 relaunch) -> ENGAGED, qsa_state recorded", "FAIL")
def _f3_5():
    r = readback_ns([wenv("128", "64")] * 4)["e29_df_readback"](QSrv("Resolved architecture: DFlash2DraftModel"))
    assert r["label"] == "ENGAGED", r
    for ranks in ([wenv("128", "64")] * 3 + [wenv("256", "256")], [wenv("128", "64")] * 3):
        ns = readback_ns(ranks)
        r = ns["e29_df_readback"](QSrv("Resolved architecture: DFlash2DraftModel"))
        assert r["label"] == "QSA_ENV_MISMATCH", (len(ranks), r.get("label"), r.get("qsa"))
    ns = readback_ns([wenv("256", "256")] * 4, qsa_off=True)
    r = ns["e29_df_readback"](QSrv("Resolved architecture: DFlash2DraftModel"))
    assert r["label"] == "ENGAGED" and ns["E29_RT"].get("qsa_state") == "QSA_OFF_MERLIN_256_256", (r["label"], ns["E29_RT"].get("qsa_state"))


def ladder(script, on=("df",)):
    """The notebook's own nb5_boot (+ e29_env / e29_extra / e29_ntries / e29_drop) with a scripted nb5_boot_try (= the kit's
    e29_kit_lib_tests ladder): each scripted entry is a status or (status, log text)."""
    g = base_ns()
    g.update({"NB5_KV_DTYPE": "fp8", "NB5_KV_BYTES": {"fp8": L.FP8_PIN}, "NB5_BOOTS": [], "NB5_EXTRA_ARGS": [],
              "NB5_MTP_OFF_ARGS": ["--speculative-config", "None"], "FAKE": None,
              "P": {"pin_ladder": True, "patch_60490": True, "k100": True, "i8": True, "top_rung_seqs": 24},
              "NB5_PIN": {"start_gib": None, "levers": None, "patch_60490": {"ok": True}}, "NB5_LEVER_ENV": dict(LEVER_ENV),
              "NB5_SEQS_RULE": "k100", "K100_SEQS_TABLE": {round(float(k), 2): int(v) for k, v in FACTS["seqs_table"].items()},
              "left": lambda: 1e9, "nb5_write_report": noop, "ABORT": {}, "NB5_READY_TIMEOUT": 1800, "NB5_BOOT_LEVERS": {},
              "NB5_BOOT_EXTRA": [], "NB5_BOOT_KV": None,
              "E28_I8": {"want": True, "on": False, "static": None, "dropped": None, "readback": None, "suspended": "N/A_DFLASH"}})
    g["E29_RT"].update(on=list(on), df_try=False, df_live=False)
    seen, it = [], iter(script)

    def nb5_boot_try(trec, name, kv_bytes, phases, mtp_on):
        st = next(it)
        st, st_log = st if isinstance(st, tuple) else (st, "")
        av = list(g["NB5_BOOT_EXTRA"])
        seen.append({"name": name, "kv_bytes": kv_bytes, "levers": dict(g["NB5_BOOT_LEVERS"]), "extra": av,
                     "df_try": g["E29_RT"]["df_try"], "e29_df": trec.get("e29_df")})
        if st == "DONE":
            trec.update(status="DONE", phases_started=True, log_text_for_ladder="")
        else:
            trec.update(status=st, log_text_for_ladder=st_log)
        return trec
    g["nb5_boot_try"] = nb5_boot_try
    fns_exec(g, ["nb5_boot", "e29_env", "e29_extra", "e29_ntries", "e29_drop"], ["e29_seam_restore"])
    b = g["nb5_boot"]("merlin_fp8_mtp", [["Q", "all", None, "on", "A", "fp8"]], True, "fp8")
    return b, seen, g


@case("F3.6", "nb5_boot: E29_QSA_ENV_MISMATCH on D rung 0 -> the SAME rung relaunched WITHOUT the QSA env (Merlin 256/256) + "
      "EMBED_HOST + seam env; DFlash kept, qsa_off recorded", "FAIL")
def _f3_6():
    b, seen, g = ladder(["E29_QSA_ENV_MISMATCH", "DONE"])
    assert len(seen) == 2, [(s["name"], s["kv_bytes"], s["df_try"]) for s in seen]
    a, c = seen
    assert a["df_try"] and a["levers"].get(K0) == "128" and a["levers"].get(K1) == "64", a["levers"]
    assert c["df_try"] and c["kv_bytes"] == a["kv_bytes"] == TRIES[0]["kv_bytes"] == L.D_START_PIN, (c["df_try"], c["kv_bytes"])   # rev 2f
    assert K0 not in c["levers"] and K1 not in c["levers"], c["levers"]
    assert c["levers"].get("DFLASH_SEAM") == "1" and c["levers"].get("DFLASH_DRAFT_EMBED_HOST") == "1" and c["extra"][-4:] == TAIL
    assert b["status"] == "DONE" and "df" in g["E29_RT"]["on"] and g["E29_RT"].get("qsa_off") is True, (b["status"], g["E29_RT"]["on"])
    assert not [x for x in g["E29_RT"]["dropped"] if x["lever"] == "df"], g["E29_RT"]["dropped"]


@case("F3.7", "nb5_boot: a second QSA mismatch after the 256/256 relaunch does not loop: DFlash ends -> the MTP top rung (I8)",
      "FAIL")
def _f3_7():
    b, seen, g = ladder(["E29_QSA_ENV_MISMATCH", "E29_QSA_ENV_MISMATCH", "DONE"])
    assert [s["df_try"] for s in seen] == [True, True, False], [(s["name"], s["df_try"]) for s in seen]
    assert seen[2]["kv_bytes"] == TOP and seen[2]["levers"].get("DRAFT_HEAD") == "int8" and seen[2]["extra"] == ["--max-num-seqs", "24"]
    assert b["status"] == "DONE" and [x["lever"] for x in g["E29_RT"]["dropped"]] == ["df"]


# ==================================================================== F2: the forced first-prefill probe decisions
@case("F2.1", "nb5_boot: E29_DFLASH_LOW_FREE (a worker below the free floor after the forced first prefill) -> ONE D ladder step "
      "(7,665,090,560 B, EMBED_HOST), DFlash kept", "FAIL")
def _f2_1():
    b, seen, g = ladder(["E29_DFLASH_LOW_FREE", "DONE"])
    assert len(seen) == 2 and seen[1]["df_try"] and seen[1]["kv_bytes"] == TRIES[1]["kv_bytes"] == 7396655104, \
        [(s["kv_bytes"], s["df_try"]) for s in seen]
    assert seen[1]["levers"].get("DFLASH_DRAFT_EMBED_HOST") == "1" and "df" in g["E29_RT"]["on"]


@case("F2.2", "nb5_boot: E29_DFLASH_PROBE_FAILED with an OOM line -> one D ladder step; without an OOM (dead) -> DFlash ends -> the "
      "MTP top rung", "PASS")
def _f2_2():
    b, seen, g = ladder([("E29_DFLASH_PROBE_FAILED", OOM), "DONE"])
    assert seen[1]["df_try"] and seen[1]["kv_bytes"] == TRIES[1]["kv_bytes"], [(s["kv_bytes"], s["df_try"]) for s in seen]
    b, seen, g = ladder([("E29_DFLASH_PROBE_FAILED", "server died: no output"), "DONE"])
    assert not seen[1]["df_try"] and seen[1]["kv_bytes"] == TOP and seen[1]["levers"].get("DRAFT_HEAD") == "int8", seen[1]


@case("F2.3", "e29_lib probe facts: PROBE_TOKENS fits max_model_len 131,072 with >= 8 output tokens, PROBE_FREE_MIB 512 (NB-S1), "
      "worker_free_mib parses nb1_cache_info results; static: nb5_boot_try runs e29_prefill_probe on DFlash tries", "SKIP_NOT_IMPLEMENTED")
def _f2_3():
    need_lib("PROBE_TOKENS", "PROBE_FREE_MIB", "worker_free_mib")
    need_fn("e29_prefill_probe")
    assert 116000 <= L.PROBE_TOKENS <= 131072 - 8 and L.PROBE_FREE_MIB == 512, (L.PROBE_TOKENS, L.PROBE_FREE_MIB)
    res = {"results": [{"rank": i, "mem": {"free_mib": v}} for i, v in enumerate((900, 600, 577, 1047))] + [{"rank": 9, "error": "x"}]}
    assert L.worker_free_mib(res) == [900, 600, 577, 1047] and L.worker_free_mib(None) == []
    assert "e29_prefill_probe(" in src_of("nb5_boot_try"), "nb5_boot_try does not run the probe"


# ==================================================================== F12: submission_check
def _tasks12():
    return {"aa000001": {"train": [], "test": [{"input": [[1, 2], [3, 4]]}, {"input": [[0]]}]},
            "bb000002": {"train": [], "test": [{"input": [[5, 5, 5]]}]}}


def _sub_ok():
    return {"aa000001": [{"attempt_1": [[1, 2], [3, 4]], "attempt_2": [[4, 3], [2, 1]]}, {"attempt_1": [[0]], "attempt_2": [[9]]}],
            "bb000002": [{"attempt_1": [[5, 5, 5]], "attempt_2": [[0, 0, 0]]}]}


def _check(sub, raw=None):
    need_lib("submission_check")
    p = Path(tempfile.mkdtemp(prefix="e29adv_sub_")) / "submission.json"
    p.write_text(raw if raw is not None else json.dumps(sub), encoding="utf-8")
    return L.submission_check(p, _tasks12())


@case("F12.1", "submission_check: a valid submission -> ok (every task, one row per test input, 0-9 grids)", "SKIP_NOT_IMPLEMENTED")
def _f12_1():
    r = _check(_sub_ok())
    assert r["ok"] is True and r["n_missing"] == 0 and r["rows"] == 2, r


@case("F12.2", "submission_check: a missing task id -> not ok (n_missing 1)", "SKIP_NOT_IMPLEMENTED")
def _f12_2():
    s = _sub_ok()
    s.pop("bb000002")
    r = _check(s)
    assert r["ok"] is False and r["n_missing"] == 1 and r["missing"] == ["bb000002"], r


@case("F12.3", "submission_check: a wrong row count for a task (1 row for 2 test inputs) -> not ok (bad_rows)", "SKIP_NOT_IMPLEMENTED")
def _f12_3():
    s = _sub_ok()
    s["aa000001"] = s["aa000001"][:1]
    r = _check(s)
    assert r["ok"] is False and r["bad_rows"] == ["aa000001"], r


@case("F12.4", "submission_check: bad cells (10, -1, True, '3', 1.0, None), a ragged grid, an empty grid, 31 rows, a missing "
      "attempt_2 -> not ok (bad_grids)", "SKIP_NOT_IMPLEMENTED")
def _f12_4():
    for bad in ([[10]], [[-1]], [[True]], [["3"]], [[1.0]], [[None]], [[1, 2], [3]], [], [[]], [[0]] * 31):
        s = _sub_ok()
        s["bb000002"][0]["attempt_2"] = bad
        r = _check(s)
        assert r["ok"] is False and r["bad_grids"] == ["bb000002"], (bad, r)
    s = _sub_ok()
    del s["bb000002"][0]["attempt_2"]
    assert _check(s)["ok"] is False


@case("F12.5", "the notebook's CELL 0 fallback submission (identity: attempt_1 = attempt_2 = the test input), written by the real "
      "CELL 0 code -> submission_check ok", "SKIP_NOT_IMPLEMENTED")
def _f12_5():
    need_lib("submission_check")
    nb = find(NB_NAME)
    if nb is not None:
        c0 = "".join(next(c for c in json.loads(nb.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code")["source"])
    else:
        c0 = EXPORT_SRC[:EXPORT_SRC.index("# ==================================================================== CELL 1")]
    c0 = c0[:c0.index("\nE29_DIRECT2 = ")] if "\nE29_DIRECT2 = " in c0 else c0
    assert "identity_submission" in c0 and "write_json_atomic(SUBMISSION_PATH" in c0, "CELL 0 not found"
    d = Path(tempfile.mkdtemp(prefix="e29adv_cell0_"))
    (d / "arc").mkdir()
    (d / "arc" / "arc-agi_test_challenges.json").write_text(json.dumps(_tasks12()), encoding="utf-8")
    old = {k: os.environ.get(k) for k in ("Q27X_ARC_ROOT", "Q27X_WORKING", "KAGGLE_IS_COMPETITION_RERUN")}
    os.environ.update(Q27X_ARC_ROOT=str(d / "arc"), Q27X_WORKING=str(d / "w"))
    os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    try:
        exec(compile(c0, "<CELL 0>", "exec"), {"__name__": "cell0"})
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    r = L.submission_check(d / "w" / "submission.json", _tasks12())
    assert r["ok"] is True and r["rows"] == 2, r


@case("F12.6", "submission_check after an interruption: a truncated JSON / a missing file -> ok False with the error, never raises",
      "SKIP_NOT_IMPLEMENTED")
def _f12_6():
    r = _check(None, raw=json.dumps(_sub_ok())[:57])
    assert r["ok"] is False and r["exists"] is False and "JSONDecodeError" in r.get("error", ""), r
    r = L.submission_check(Path(tempfile.mkdtemp()) / "nope.json", _tasks12())
    assert r["ok"] is False and r["exists"] is False and r.get("error"), r


@case("F12.7", "static: the notebook records E29L.submission_check (CELL 7/9 / after the writes) in the report", "SKIP_NOT_IMPLEMENTED")
def _f12_7():
    need_lib("submission_check")
    assert "E29L.submission_check(" in EXPORT_SRC, "the code export never calls E29L.submission_check"


# ==================================================================== F5: effective algorithm record (DIRECT off never stops the run)
@case("F5.1", "effective(): DIRECT OFF_SETUP / not set up -> experiment_validity DIRECT_OFF with the reason; DIRECT ON or the hidden "
      "rerun -> OK; drafter / EMBED_HOST from the live levers", "SKIP_NOT_IMPLEMENTED")
def _f5_1():
    need_lib("effective")
    rt = {"on": ["df"], "df_live": True, "cfg": {"embed_host": True}, "direct": {"state": "OFF_SETUP", "error": "ModuleNotFoundError: e29_direct"}}
    e = L.effective(rt)
    assert e["experiment_validity"] == "DIRECT_OFF" and "e29_direct" in str(e["direct_why"]) and e["drafter"].startswith("DFlash"), e
    assert e["embed_host"] is True and L.effective(rt, rerun=True)["experiment_validity"] == "OK"
    e = L.effective({"on": [], "extras_error": "FileNotFoundError: e29_nbx.py"})
    assert e["experiment_validity"] == "DIRECT_OFF" and "e29_nbx" in str(e["direct_why"]) and e["drafter"] == "MTP K3", e
    assert L.effective({"on": ["df"], "direct": {"state": "ON"}})["experiment_validity"] == "OK"


@case("F5.2", "notebook e29_extras: the payload e29_nbx.py missing -> no raise, extras_error recorded, E29D_RUN None, E29_DIRECT env "
      "removed (the run continues = phase C on the E28 Duck path)", "PASS")
def _f5_2():
    ns = fns_exec(base_ns(), ["e29_extras"])
    ns["E12A"] = types.SimpleNamespace(E29D_RUN=lambda *a: None, E29D_CAN_START=None)
    ns["E29D_FILES"] = {"nbx": ("e29_nbx.py", "0" * 64)}

    def q27x_payload(name, sha_):
        raise FileNotFoundError(f"{name} {sha_[:12]} not in the payload dataset (old version)")
    ns["q27x_payload"] = q27x_payload
    old = os.environ.get("E29_DIRECT")
    os.environ["E29_DIRECT"] = "1"
    try:
        ns["e29_extras"]()
        assert "E29_DIRECT" not in os.environ
    finally:
        os.environ.pop("E29_DIRECT", None) if old is None else os.environ.__setitem__("E29_DIRECT", old)
    assert ns["E12A"].E29D_RUN is None and "FileNotFoundError" in ns["E29_RT"].get("extras_error", ""), ns["E29_RT"].get("extras_error")


@case("F5.3", "static: the notebook computes + prints E29L.effective (the effective_algorithm record)", "SKIP_NOT_IMPLEMENTED")
def _f5_3():
    need_lib("effective")
    assert "E29L.effective(" in EXPORT_SRC or "E29L.effective(" in (NBX.read_text(encoding="utf-8") if NBX else ""), \
        "E29L.effective is never called"


# ==================================================================== F1 / F1+: late phase C (the notebook hook + the real scheduler)
class FakeTime:
    def __init__(self, now):
        self.now = float(now)

    def monotonic(self):
        return self.now

    def time(self):
        return time.time()

    def perf_counter(self):
        return time.perf_counter()

    def sleep(self, s):
        return None


def nbx_ns(direct2, sched_ok=True, stale=False):
    if NBX is None:
        raise NoInput("e29_nbx.py not found")
    E12A = types.SimpleNamespace(E29D_RUN=(lambda *a: "STALE") if stale else None, E29D_CAN_START=(lambda d: True) if stale else None,
                                 SERVER_DOWN=threading.Event(), solve_all_r6=lambda *a, **k: {})
    files = {k: (n, sha256(find(n).read_bytes())) for k, n in (("direct", "e29_direct.py"), ("sched", "e29d_sched_patch.py"),
                                                                 ("nbx", "e29_nbx.py"))}
    ns = {"__name__": "e29_nbx_adv", "E12A": E12A, "E29_DIRECT2": dict(direct2), "E29_RT": {},
          "E29D_SCHED_REC": ({"want": True, "ok": True, "applied": "applied", "path": "E29"} if sched_ok
                             else {"want": True, "ok": False, "error": "injected: the DIRECT diff is not applied", "path": "E29"}),
          "TMP": Path(tempfile.mkdtemp(prefix="e29adv_nbx_")), "E29D_FILES": files,
          "q27x_payload": lambda n, s: find(n).read_bytes(), "log": noop, "os": os, "json": json, "Path": Path,
          "threading": threading, "time": time, "sys": sys, "traceback": traceback, "hashlib": hashlib, "E29L": L,
          "NOTEBOOK_T0": time.perf_counter(), "E11_SERVER": None, "SGL_URL": None, "E11_HARNESS": False, "RERUN": False,
          "TASK_IDS": [], "NB5_LOCAL_TOK": {}, "SOLUTIONS": None}
    old = os.environ.get("E29_DIRECT")
    old_ph = os.environ.get("E29_DIRECT_PHASE")
    os.environ["E29_DIRECT"] = "1"
    try:
        exec(compile(NBX.read_text(encoding="utf-8"), "e29_nbx.py", "exec"), ns)
        ns["e29_direct_setup"]()
        ns["_phase_env_after_setup"] = os.environ.get("E29_DIRECT_PHASE")   # rev 2f: what the setup exported
    finally:
        os.environ.pop("E29_DIRECT", None) if old is None else os.environ.__setitem__("E29_DIRECT", old)
        os.environ.pop("E29_DIRECT_PHASE", None) if old_ph is None else os.environ.__setitem__("E29_DIRECT_PHASE", old_ph)
    return ns


DIRECT2 = {"on": True, "think": 104000, "answer": 8192, "temperature": 1.0, "drain_min": 26}   # = CELL 1's switch


@case("F1.7", "notebook hook (e29_direct_setup ON): E29D_RUN + E29D_CAN_START set; fake clock, drain_min 26 (+ 60 s margin): "
      "31 / 27.02 min left -> DIRECT; 27.00 (the exact boundary) / 26.5 / 25 / 20 / 1 min -> Duck C", "FAIL")
def _f1_7():
    ns = nbx_ns(DIRECT2)
    E12A = ns["E12A"]
    assert ns["E29_RT"]["direct"]["state"] == "ON" and E12A.E29D_RUN is ns["e29_direct_run"], ns["E29_RT"]["direct"].get("state")
    h = E12A.E29D_CAN_START
    assert callable(h), "E12A.E29D_CAN_START is not set by e29_direct_setup (rev 2c: no hook -> late C is a silent no-op)"
    now = time.monotonic()
    ns["time"] = FakeTime(now)
    for mins, want in ((31, True), (27 + 1.2 / 60, True), (27, False), (26.5, False), (25, False), (20, False), (1, False)):
        got = bool(h(now + mins * 60.0))
        assert got is want, f"{mins} min left: hook {got}, want {want}"


@case("F1.8a", "notebook hook: the switch OFF -> no E29D_RUN, no callable E29D_CAN_START (phase C = the E28 Duck attempt 2)", "PASS")
def _f1_8a():
    ns = nbx_ns(dict(DIRECT2, on=False))
    assert ns["E29_RT"]["direct"]["state"].startswith("OFF_SWITCH") and ns["E12A"].E29D_RUN is None, ns["E29_RT"]["direct"]["state"]
    assert not callable(ns["E12A"].E29D_CAN_START)


@case("F1.8b", "notebook hook: setup failure (the DIRECT diff not applied) with stale hooks -> both E29D_RUN and E29D_CAN_START "
      "cleared, OFF_SETUP recorded", "FAIL")
def _f1_8b():
    ns = nbx_ns(DIRECT2, sched_ok=False, stale=True)
    assert ns["E29_RT"]["direct"]["state"] == "OFF_SETUP" and ns["E12A"].E29D_RUN is None, ns["E29_RT"]["direct"]
    assert not callable(ns["E12A"].E29D_CAN_START), "a stale E29D_CAN_START survives the failed setup"


SCHED = {}


def sched():
    if "M" in SCHED:
        return SCHED
    if "err" in SCHED:
        raise NoInput(SCHED["err"])
    pay = Path(os.environ["E28_TEST_PAYLOAD"]) if os.environ.get("E28_TEST_PAYLOAD") else find(PAYLOAD_NAME)
    if pay is None or not pay.is_file() or not find("e29_sched_tests.py"):
        SCHED["err"] = f"the Duck payload {PAYLOAD_NAME} or e29_sched_tests.py not found"
        raise NoInput(SCHED["err"])
    os.environ["E28_TEST_PAYLOAD"] = str(pay)
    os.environ.setdefault("E29_TEST_E28_DIR", str(Path(E28S.__adv_path__).parent))
    ST = load("e29_sched_tests")                    # the E29 scheduler harness (real arc2_duck + BranchSession, fake Duck agent)
    ST.setUpModule()
    tmp = ST.FIX["tmp"]
    with zipfile.ZipFile(pay) as z:
        z.extractall(tmp / "adv_e29d")
    assert E29S.apply_e29_sched(tmp / "adv_e29d")["ok"]
    r = DS29.apply_direct(tmp / "adv_e29d")
    assert r["ok"] and r["applied"] == "applied", r
    SCHED.update(ST=ST, M=ST._load(tmp / "adv_e29d" / "arc2_duck.py", "arc2_duck_adv_e29d"), runs={})
    return SCHED


def scen():
    ST = SCHED["ST"]
    tasks = ST._tasks(6)
    ST.SCRIPTS.clear()
    ST.SCRIPTS.update({("t000", "A"): {"kind": "pass", "at": 3000}, ("t001", "A"): {"kind": "pass", "at": 1000},
                       ("t002", "A"): {"kind": "partial", "at": 2000}, ("t002", "B"): {"kind": "partial", "at": 2000},
                       ("t003", "A"): {"kind": "pass", "at": 2000}, ("t004", "A"): {"kind": "fail"}, ("t004", "B"): {"kind": "fail"},
                       ("t005", "A"): {"kind": "pass", "at": 5000}})
    return tasks, list(tasks)


def stub_direct(calls):
    """The DIRECT request stand-in (= e29d_direct_tests._stub_hook): one valid grid per test input."""
    def run(tid, task, plan, deadline, base_url, model_id, out_dir):
        calls.append((tid, plan.get("c_priority"), deadline))
        n = len(task["test"])
        return {"task_id": tid, "branch": 1, "status": "done", "accepted": False, "generated_tokens": 10, "e29_direct": True,
                "submissions": [{"test_outputs": [[[7]]] * n, "accepted": False, "train_exact": [], "t": time.time()}],
                "e31": {"mode": "natural", "end_reason": "natural"}}
    return run


def run_c(mode, can_start, env_direct=True, deadline_sec=60.0):
    S = sched()
    if mode in S["runs"]:
        return S["runs"][mode]
    ST, M = S["ST"], S["M"]
    tasks, order = scen()
    calls, rel, hook_args = [], [], []
    M.E29D_RUN = stub_direct(calls)
    M.E29D_CAN_START = can_start(hook_args) if callable(can_start) and getattr(can_start, "factory", False) else can_start
    M.E28_RELEASE = lambda key: rel.append(tuple(key))
    t_before = time.monotonic()
    try:
        env = dict(E29_SUPPLY_MAX_ATTEMPTS="3", **({"E29_DIRECT": "1"} if env_direct else {}))
        env["E29_DIRECT_PHASE"] = "C"   # rev 2f: these cases test the rev 2d routing of EVERY C item (scope C); the C1
        # default (DIRECT only on train-fail C items) is tested in e29d_direct_tests.TestC1Scope and case C1.1 below
        with ST.Env(supply=True, **env):
            kit = ST.RunLog(M)
            try:
                recs, _, sl = ST._solve(M, tasks, order, 1, deadline_sec=deadline_sec)
            finally:
                kit.close()
                runs = list(ST.RUNLOG)
            subs = M.build_submission(tasks, recs)
    finally:
        M.E29D_RUN, M.E29D_CAN_START, M.E28_RELEASE = None, None, None
    out = dict(tasks=tasks, recs=recs, sl=sl, subs=subs, calls=calls, rel=rel, runs=runs, hook_args=hook_args,
               t_before=t_before, deadline_sec=deadline_sec, M=M)
    S["runs"][mode] = out
    return out


def factory(fn):
    fn.factory = True
    return fn


@factory
def hook_true(args):
    def h(deadline):
        args.append(deadline)
        return True
    return h


def _common_c(r):
    e, n = r["sl"]["e28"], len(r["tasks"])
    assert not e["setup_failed"] and r["sl"]["worker_errors"] == 0, (e["setup_failed"][:2], r["sl"]["worker_errors"])
    assert e["started"]["C"] == n and e["done"]["C"] == n, f"phase C started {e['started']['C']} / done {e['done']['C']} != {n}"
    for t in r["tasks"]:
        k = r["rel"].count((t, 1))
        assert k == 1, f"{t}: E28_RELEASE((t, 1)) called {k} times (want exactly once)"
        rows = [a for a in e["attempts"] if a["phase"] == "C" and a["task"] == t]
        assert len(rows) == 1, f"{t}: {len(rows)} phase-C rows"
        for q in r["subs"][t]:
            assert r["M"].valid_grid(q["attempt_1"]) and r["M"].valid_grid(q["attempt_2"]), (t, q)
    return e, n


def expect_direct(r):
    e, n = _common_c(r)
    assert sorted(c[0] for c in r["calls"]) == sorted(r["tasks"]), f"DIRECT calls {[c[0] for c in r['calls']]}"
    duck_c = [x for x in r["runs"] if x[0] == "start" and x[3] == "C"]
    assert not duck_c, f"{len(duck_c)} Duck C sessions ran next to DIRECT"
    d = e.get("direct") or {}
    assert (d.get("started"), d.get("records"), d.get("errors")) == (n, n, 0), d
    assert all(a.get("e29_direct") for a in e["attempts"] if a["phase"] == "C")
    for t in r["tasks"]:
        assert any(x.get("e29_direct") and x["branch"] == 1 for x in r["recs"][t]), t


def expect_duck_c(r, why):
    e, n = _common_c(r)
    assert not r["calls"], f"DIRECT ran {len(r['calls'])} x (want 0: {why}; rev 2c dispatches every phase-C item to DIRECT)"
    duck_c = [x for x in r["runs"] if x[0] == "start" and x[3] == "C"]
    assert len(duck_c) == n and sorted(x[1] for x in duck_c) == sorted(r["tasks"]), f"{len(duck_c)} Duck C sessions, want {n}"
    for t in r["tasks"]:
        assert any(x["branch"] == 1 and not x.get("e29_direct") for x in r["recs"][t]), f"{t}: no Duck attempt-2 record"
    assert not any(a.get("e29_direct") for a in e["attempts"])
    d = e.get("direct") or {}
    assert d.get("duck_c") == n, f"e28['direct']['duck_c'] = {d.get('duck_c')} (want {n}: the late / no-hook route recorded)"


@case("F1.1", "scheduler: E29D_CAN_START True -> every phase-C item = ONE DIRECT request (no Duck C), C started / done once, "
      "E28_RELEASE((t, 1)) once, valid attempt_2", "PASS")
def _f1_1():
    expect_direct(run_c("true", hook_true))


@case("F1.1b", "scheduler: the hook is consulted once per phase-C item with the scheduler's own deadline", "FAIL")
def _f1_1b():
    r = run_c("true", hook_true)
    n = len(r["tasks"])
    assert len(r["hook_args"]) == n, f"hook called {len(r['hook_args'])} x, want {n}"
    want = r["t_before"] + r["deadline_sec"]
    assert all(abs(float(a) - want) < 10.0 for a in r["hook_args"]), (r["hook_args"][:3], want)


@case("F1.2", "scheduler: E29D_CAN_START False (late: < drain + 60 s left) -> the SAME item runs the original Duck C session: one "
      "start, one completion, one release, a valid attempt_2, recorded duck_c", "FAIL")
def _f1_2():
    expect_duck_c(run_c("false", lambda d: False), "hook False")


@case("F1.3", "scheduler: no hook (E29D_CAN_START None) -> Duck C", "FAIL")
def _f1_3():
    expect_duck_c(run_c("none_attr", None), "no hook")


@case("F1.4", "scheduler: the hook returns None -> Duck C", "FAIL")
def _f1_4():
    expect_duck_c(run_c("ret_none", lambda d: None), "hook returned None")


@case("F1.5", "scheduler: the hook raises -> Duck C, never a worker error", "FAIL")
def _f1_5():
    def boom(d):
        raise RuntimeError("injected hook failure")
    expect_duck_c(run_c("raises", boom), "hook raised")


@case("F1.6", "scheduler: E29_DIRECT env unset (switch off) with a True hook -> the E29 post Duck path (DIRECT never called)", "PASS")
def _f1_6():
    r = run_c("env_off", hook_true, env_direct=False)
    e, n = _common_c(r)
    assert not r["calls"] and not r["hook_args"], (len(r["calls"]), len(r["hook_args"]))
    assert len([x for x in r["runs"] if x[0] == "start" and x[3] == "C"]) == n


@case("F1.9", "integration: the REAL notebook hook (e29_nbx, drain_min 1 + 60 s margin = 120 s) on the REAL scheduler: a 400 s "
      "solve window -> DIRECT; a 100 s window -> Duck C for the same items", "SKIP_NOT_IMPLEMENTED")
def _f1_9():
    ns = nbx_ns(dict(DIRECT2, drain_min=1))
    h = ns["E12A"].E29D_CAN_START
    if not callable(h):
        raise NotImpl("e29_nbx.e29_direct_setup sets no E29D_CAN_START")
    expect_direct(run_c("nbx_early", h, deadline_sec=400.0))
    expect_duck_c(run_c("nbx_late", h, deadline_sec=100.0), "real hook, 100 s left")


@case("C1.1", "rev 2f: the REAL notebook setup exports E29_DIRECT_PHASE (absent / C1 -> C1, C -> C, OFF -> unset); the C1 scope on the REAL scheduler: DIRECT only on the train-fail C items, Duck C (c2_duck) on the rest", "FAIL")
def _c1_1():
    for d2, want in ((dict(DIRECT2), "C1"), (dict(DIRECT2, phase="C1"), "C1"), (dict(DIRECT2, phase="C"), "C"),
                     (dict(DIRECT2, on=False), None)):
        ns = nbx_ns(L.direct2_cfg(d2))
        assert ns["_phase_env_after_setup"] == want, (d2, ns["_phase_env_after_setup"], want)
        assert ns["E29_DIRECT_RT"].get("phase") == want, (d2, ns["E29_DIRECT_RT"].get("phase"))
    S = sched()
    ST, M = S["ST"], S["M"]
    tasks, order = scen()
    calls, rel = [], []
    M.E29D_RUN, M.E29D_CAN_START = stub_direct(calls), (lambda d: True)
    M.E28_RELEASE = lambda key: rel.append(tuple(key))
    try:
        with ST.Env(supply=True, E29_SUPPLY_MAX_ATTEMPTS="3", E29_DIRECT="1", E29_DIRECT_PHASE="C1"):
            kit = ST.RunLog(M)
            try:
                recs, _, sl = ST._solve(M, tasks, order, 1, deadline_sec=60.0)
            finally:
                kit.close()
                runs = list(ST.RUNLOG)
    finally:
        M.E29D_RUN, M.E29D_CAN_START, M.E28_RELEASE = None, None, None
    e = sl["e28"]
    c_rows = [x for x in e["attempts"] if x["phase"] == "C"]
    c1 = sorted(x["task"] for x in c_rows if x.get("c_priority") == 1)
    c2 = sorted(x["task"] for x in c_rows if x.get("c_priority") != 1)
    assert c1 and c2, (c1, c2)
    assert sorted(c[0] for c in calls) == c1, (calls, c1)
    assert sorted(x[1] for x in runs if x[0] == "start" and x[3] == "C") == c2, c2
    assert e["direct"]["duck_c_why"] == {"c2_duck": len(c2)}, e["direct"]
    assert sorted(k for k in rel if k[1] == 1) == sorted((t_, 1) for t_ in c1 + c2), rel   # one C release per item
    assert (e["started"]["C"], e["done"]["C"]) == (len(c_rows), len(c_rows)), (e["started"], e["done"])


# ==================================================================== A4 (INDEPENDENT_AUDIT audit_repro_e29.py): the 7 scenarios
@case("A4-1", "audit_repro 1: e29_mtp_boot -> the top pin + I8 + clean env / argv", "PASS")
def _a4_1():
    ns = relaunch_ns(RUNGS[-1])
    ns["e29_mtp_boot"]("TEST_DIRECT_MTP_FALLBACK")
    assert not mtp_top_problems(ns), mtp_top_problems(ns)
    assert not (set(ns["NB5_ACTIVE_BOOT"]["levers"]) & set(tuple(DSM.ALL_ENVS) + tuple(L.QSA_KEYS)))


@case("A4-2", "audit_repro 2: e29_relaunch_plan, OOM on rung 0 -> the next D rung, EMBED_HOST kept", "PASS")
def _a4_2():
    ns = relaunch_ns(RUNGS[0])
    ns["e29_relaunch_plan"](Old("torch.OutOfMemoryError: CUDA out of memory"))
    a = ns["NB5_ACTIVE_BOOT"]
    assert a["kv_bytes"] == TRIES[1]["kv_bytes"] and a["levers"]["DFLASH_DRAFT_EMBED_HOST"] == "1", (a["kv_bytes"], a["levers"])   # rev 2f


@case("A4-3", "audit_repro 3: e29_relaunch_plan, OOM on the last rung -> MTP top + I8", "PASS")
def _a4_3():
    ns = relaunch_ns(RUNGS[-1])
    ns["e29_relaunch_plan"](Old("torch.OutOfMemoryError: CUDA out of memory"))
    assert not mtp_top_problems(ns), mtp_top_problems(ns)


@case("A4-4", "audit_repro 4 (inverted, H2): XID + OOM in-run -> MTP top (rev 2c left the DFlash plan unchanged)", "FAIL")
def _a4_4():
    ns = relaunch_ns(RUNGS[0])
    ns["e29_relaunch_plan"](Old("NVRM: Xid (PCI:0000:01:00): 79\ntorch.OutOfMemoryError: CUDA out of memory"))
    assert not mtp_top_problems(ns), mtp_top_problems(ns)


@case("A4-5", "audit_repro 5 (inverted, H1): the solver alone admits nothing at T-20 min (drain 26: precondition, unchanged) AND the "
      "DIRECT scheduler diff gates the phase-C DIRECT branch on E29D_CAN_START(deadline)", "FAIL")
def _a4_5():
    def enc(s):
        if s == D.STOP_THINKING_TEXT:
            return list(D.FN_STOP_IDS)
        if s == "</think>":
            return [D.FN_THINK_CLOSE_ID]
        return [1000 + ord(c) for c in s]
    tok = {"render": lambda msgs: msgs[0]["content"], "encode": enc, "decode": lambda ids, k: "", "eos_ids": list(D.FN_EOS_IDS)}
    tasks = {"late001": {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[2]]}]}}
    with tempfile.TemporaryDirectory() as td:
        out = D.solve_all_direct(tasks, ["late001"], out_dir=td, deadline=time.monotonic() + 20 * 60, base_url="http://127.0.0.1:9",
                                 model_id="mock", workers=1, cfg={"think": 104000, "answer": 8192, "temperature": 1.0, "drain_min": 26},
                                 tok=tok, harness=True, health=lambda: True, log=noop)
        rep = json.loads((Path(td) / "e31_direct_report.json").read_text(encoding="utf-8"))
    assert out["late001"] == [] and rep["sched"]["admitted"] == 0 and rep["sched"]["not_admitted"] == 1, rep["sched"]
    p = DS29.PATCH
    i = p.index('if p["phase"] == "C" and _e29d_on()')
    win = p[i:i + 400]
    assert "E29D_CAN_START" in p and ("_e29d_can_start(deadline)" in win or "_e29d_can_start(deadline, p)" in win   # rev 2f: + the plan
                                      or "E29D_CAN_START(deadline)" in win), \
        "the phase-C DIRECT branch is not gated on E29D_CAN_START(deadline) (rev 2c: `phase == 'C' and _e29d_on()` only)"


@case("A4-6", "audit_repro 6 (inverted, H2): e29_lib.df_relaunch_plan('NVRM: Xid 79 + OOM') == 'mtp' (rev 2c: None)", "FAIL")
def _a4_6():
    p = L.df_relaunch_plan({"rung": 0}, "NVRM: Xid 79\nCUDA out of memory", L.df_tries(True))
    assert p == "mtp", f"plan {p!r}"


@case("A4-7", "audit_repro 7 (inverted, H3 as decided in F11): SSEQ.finish() raising is not swallowed silently: the seam files are "
      "verified against the pre-apply snapshot (seam_restore recorded) before the MTP boot", "FAIL")
def _a4_7():
    class BrokenSeam:
        state, ring = "applied", True

        def finish(self):
            raise RuntimeError("injected revert failed; original Merlin tree not restored")
    ns = fns_exec(base_ns(), ["e29_drop"], ["e29_seam_restore", "e29_seam_snapshot"])
    tree = Path(tempfile.mkdtemp(prefix="e29adv_a47_"))
    rels = list(DSM._rels(True, True))
    for r in rels:
        (tree / r).parent.mkdir(parents=True, exist_ok=True)
        (tree / r).write_bytes(_orig_bytes(r))
    if "e29_seam_snapshot" in ns:
        ns["e29_seam_snapshot"](tree)
    ns["SSEQ"] = BrokenSeam()
    ns["e29_drop"]("df", "TEST_SEAM_REVERT_THROW", "D")
    assert any("seam finish" in e for e in ns["E29_RT"]["errors"]), ns["E29_RT"]["errors"]
    rr = ns["E29_RT"].get("seam_restore")
    assert rr and rr.get("label") in ("SEAM_PRISTINE", "SEAM_REVERT_UNVERIFIED"), \
        "finish() raised and nothing verified the seam files (rev 2c: the error is only appended, MTP boots on an unverified tree)"
    assert "df" not in ns["E29_RT"]["on"] and ns["E28_I8"]["on"] is True     # F11 decision: not fail-closed, the run continues


# ==================================================================== run
def main():
    only = [x.strip() for x in ARGS.only.split(",") if x.strip()]
    print(f"E29 rev 2d adversarial tests | dir {DIR} | export {EXPORT.name} sha {sha256(EXPORT.read_bytes())[:12]} | e29_lib "
          f"{getattr(L, 'LIB_VERSION', '?')} sha {sha256(Path(L.__adv_path__).read_bytes())[:12]} | e29d_sched_patch POST "
          f"{DS29.POST[:12]} | nbx {sha256(NBX.read_bytes())[:12] if NBX else None} | python {sys.version.split()[0]}", flush=True)
    for cid, title, rev2c, fn in CASES:
        if only and not any(cid == o or cid.startswith(o + ".") or cid.startswith(o + "-") or cid == o.rstrip(".") for o in only):
            continue
        t0 = time.time()
        try:
            fn()
            res, why = "PASS", ""
        except NotImpl as e:
            res, why = "SKIP_NOT_IMPLEMENTED", str(e)
        except NoInput as e:
            res, why = "SKIP_NO_INPUT", str(e)
        except AssertionError as e:
            res, why = "FAIL", (str(e) or "assert") + ("  | " + traceback.format_exc()[-600:].replace("\n", " ") if ARGS.v else "")
        except Exception as e:
            res, why = "FAIL", f"ERROR {type(e).__name__}: {e}"[:400] + " | " + traceback.format_exc()[-700:].replace("\n", " ")
        RESULTS.append({"id": cid, "result": res, "expected_rev2c": rev2c, "expected_rev2d": "PASS", "why": why[:1500],
                        "sec": round(time.time() - t0, 2), "title": title})
        print(f"{res:<21} {cid:<6} {title}" + (f"\n{'':<28}<- {why[:900]}" if why else ""), flush=True)
    if "ST" in SCHED:
        try:
            SCHED["ST"].tearDownModule()
        except Exception:
            pass
    n = {k: sum(1 for r in RESULTS if r["result"] == k) for k in ("PASS", "FAIL", "SKIP_NOT_IMPLEMENTED", "SKIP_NO_INPUT")}
    as2c = sum(1 for r in RESULTS if r["result"] == r["expected_rev2c"])
    print(f"\n{len(RESULTS)} cases: {n['PASS']} PASS, {n['FAIL']} FAIL, {n['SKIP_NOT_IMPLEMENTED']} SKIP_NOT_IMPLEMENTED, "
          f"{n['SKIP_NO_INPUT']} SKIP_NO_INPUT | expected after rev 2d: all PASS | {as2c}/{len(RESULTS)} equal to the rev 2c column",
          flush=True)
    print("compact: " + " ".join(f"{r['id']}={ {'PASS': 'P', 'FAIL': 'F', 'SKIP_NOT_IMPLEMENTED': 'NI', 'SKIP_NO_INPUT': 'NO'}[r['result']] }"
                                 for r in RESULTS), flush=True)
    if ARGS.json_out:
        Path(ARGS.json_out).write_text(json.dumps({"dir": str(DIR), "results": RESULTS, "counts": n}, indent=1), encoding="utf-8")
    bad = n["FAIL"] + ((n["SKIP_NOT_IMPLEMENTED"] + n["SKIP_NO_INPUT"]) if ARGS.strict else 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
