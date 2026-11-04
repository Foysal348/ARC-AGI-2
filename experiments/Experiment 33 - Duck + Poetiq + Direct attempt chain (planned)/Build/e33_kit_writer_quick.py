"""E33 kit (local quick unit, seconds): the real kit's WRITER section (build_e33_cpu_kit.E33_REAL, the block from `E33W = {}` to its
REP line) executed against the REAL patched arc2_duck (Duck payload + E29 + E33 diffs, e29_sched_tests' harness setup), the
notebook's own e12_write_submission / _e12_write_json_atomic (cut verbatim by the builder) and CELL 0's identity_submission,
on the competition-shaped eval challenges. Proves the section's expected slots match the E33 answer rule before the Kaggle kit.
usage: python e33_kit_writer_quick.py   (run from a folder holding e29_sched_tests.py, e29_sched_patch.py, e33_sched_patch.py;
       env E29_TEST_E28_DIR, E33KWQ_BUILDER = build_e33_cpu_kit.py, E33KWQ_EVAL = arc-agi_evaluation_challenges.json)
"""
import json
import os
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e29_sched_tests as ST     # noqa: E402
import e33_sched_patch as S33    # noqa: E402

bns = {"__file__": os.environ["E33KWQ_BUILDER"], "__name__": "e33kwq"}
exec(compile(Path(os.environ["E33KWQ_BUILDER"]).read_text(encoding="utf-8"), "build_e33_cpu_kit.py", "exec"), bns)
src = bns["E33_REAL"]
blk = src[src.index("E33W = {}\n"):src.index('REP["checks"]["e33_writer"] = E33W; save()')]
nbp = bns["discover"]()["notebook"]
rd = bns["readable"](bns["nb_code"](nbp))
writer_src = bns["cut_nested"](next(c for c in rd if "def e21_cell6():" in c), "e21_cell6", ["_e12_write_json_atomic", "e12_write_submission"])
c0 = rd[0]
g0 = {}
import ast  # noqa: E402
t = ast.parse(c0)
exec(compile(ast.Module([n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "identity_submission"], []), "<cell0>", "exec"), g0)

ST.setUpModule()
tmp = ST.FIX["tmp"]
with zipfile.ZipFile(ST.PAYLOAD) as z:
    z.extractall(tmp / "e33w")
assert ST.E29S.apply_e29_sched(tmp / "e33w")["ok"] and S33.apply_e33(tmp / "e33w")["ok"]
M = ST._load(tmp / "e33w" / "arc2_duck.py", "arc2_duck_e33w")
os.environ.update(ST.E29S.server_env(supply=False))
os.environ.update(S33.server_env(None))
tc = json.loads(Path(os.environ["E33KWQ_EVAL"]).read_text(encoding="utf-8"))
import threading, tempfile, traceback, hashlib, time   # noqa: E402,E401
K = tmp / "kwq"
K.mkdir()
REP = {"checks": {}}
G = {"test_challenges": tc, "identity_submission": g0["identity_submission"], "E12A": M}
E33V = {}
ns = dict(G=G, K=K, KF={"writer_src": writer_src}, os=os, json=json, Path=Path, traceback=traceback, _th33=threading, _tf33=tempfile,
          E33V=E33V, REP=REP, strict_schema=lambda s, t: [k for k in t if k not in s])
exec(compile(blk, "<E33_REAL writer block>", "exec"), ns)
w = ns["E33W"]
print(json.dumps({k: w.get(k) for k in ("commit", "rerun", "env", "error")}, default=str)[:3000])
print(json.dumps(E33V))
ok = E33V.get("e33_writer_commit_mode_identity_fallback") is True and E33V.get("e33_writer_rerun_mode_e33_answer_rule_slots_strict_schema") is True
ST.tearDownModule()
print("E33 KIT WRITER QUICK:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
