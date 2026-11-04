"""Regression tests of nbdf_gate_verify.py (audit 2026-11-01: per_k_success None passed). Each case runs the real CLI on temp
files; only the first may pass: valid K100 gate; per_k_success missing; D4s entry missing; kv_configs_ok False;
k100_r9_run_ok False; notebook sha mismatch; plus a W4 gate without per_k_success (passes) and a missing notebook file.
Run: python nbdf_gate_verify_test.py"""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = {"pass": 0, "fail": 0}


def ok(name, cond, info=""):
    RES["pass" if cond else "fail"] += 1
    if not cond:
        print("FAIL", name, str(info)[:400], flush=True)


def run(td, facts, nb=b'{"cells": []}', nb_path=None, arms=("D3s", "D4s")):
    fp, bp, np_ = td / "facts.json", td / "body.py", td / "nb.ipynb"
    fp.write_text(json.dumps(facts))
    bp.write_text('KV_KIT_GATE = json.loads(r"""' + json.dumps(facts["kv_kit_gate"]) + '""")\n')
    np_.write_bytes(nb)
    r = subprocess.run([sys.executable, str(HERE / "nbdf_gate_verify.py"), str(fp), str(bp), str(nb_path or np_), *arms],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


NB = b'{"cells": [], "nbformat": 4}'
SUCC = {a: {"success": True, "kv_configs_ok": True, "k100_r9_run_ok": True, "run": f"K{a[1]}_swblock1024"} for a in ("D3s", "D4s")}
GOOD = {"variant": "k100", "sha256": hashlib.sha256(NB).hexdigest(), "gpu_ready": True,
        "kv_kit_gate": {"passed": True, "per_k": {"D3s": True, "D4s": True}, "per_k_success": SUCC}}
with tempfile.TemporaryDirectory() as t:
    td = Path(t)
    rc, out = run(td, GOOD, NB)
    ok("valid K100 gate with D3s / D4s success -> PASS", rc == 0 and json.loads(out)["ok"] is True, out)
    cases = {}
    f = copy.deepcopy(GOOD); del f["kv_kit_gate"]["per_k_success"]; cases["per_k_success missing"] = f
    f = copy.deepcopy(GOOD); del f["kv_kit_gate"]["per_k_success"]["D4s"]; cases["D4s entry missing"] = f
    f = copy.deepcopy(GOOD); f["kv_kit_gate"]["per_k_success"]["D3s"]["kv_configs_ok"] = False; cases["kv_configs_ok False"] = f
    f = copy.deepcopy(GOOD); f["kv_kit_gate"]["per_k_success"]["D4s"]["k100_r9_run_ok"] = False; cases["k100_r9_run_ok False"] = f
    f = copy.deepcopy(GOOD); f["kv_kit_gate"]["per_k_success"] = None; cases["per_k_success None"] = f
    f = copy.deepcopy(GOOD); f["sha256"] = "0" * 64; cases["notebook sha mismatch"] = f
    for nm, f in cases.items():
        rc, out = run(td, f, NB)
        ok(f"{nm} -> FAIL (exit 1)", rc == 1 and json.loads(out)["ok"] is False, out)
    rc, out = run(td, GOOD, NB, nb_path=td / "missing.ipynb")
    ok("notebook file missing -> FAIL", rc == 1, out)
    w4 = copy.deepcopy(GOOD); w4.pop("variant"); del w4["kv_kit_gate"]["per_k_success"]
    w4["kv_kit_gate"]["per_k"].update(D4=True)
    rc, out = run(td, w4, NB, arms=("D3s", "D4s", "D4"))
    ok("W4 gate (no per_k_success, not k100) with per_k True -> PASS", rc == 0, out)
    rc, out = run(td, dict(w4, kv_kit_gate=dict(w4["kv_kit_gate"], per_k={"D3s": True, "D4s": True, "D4": None})), NB, arms=("D3s", "D4s", "D4"))
    ok("W4 per_k None for an arm -> FAIL", rc == 1, out)
print(f"{RES['pass']}/{RES['pass'] + RES['fail']} PASS", flush=True)
sys.exit(1 if RES["fail"] else 0)
