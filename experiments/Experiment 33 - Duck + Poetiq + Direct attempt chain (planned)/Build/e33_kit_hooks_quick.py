"""E33 kit (local quick unit, seconds; LOCAL DEBUG ONLY, never in a kit): the real kit's HOOKS section (build_e33_cpu_kit.E33_REAL,
from the fake-server block to its REP line) executed in the chain e2e's notebook-like namespace (the notebook's own e33_nbx.py,
e33_setup -> the real e33_poetiq / e29_direct) with the stub tokenizer (Flash-Next spec ids) instead of the klee100 one and an
in-process sandbox (Windows has no seccomp). The Kaggle real kit runs the same block with the real tokenizer and sandbox.
usage: python e33_kit_hooks_quick.py   (the chain e2e's folder; env E29_TEST_E28_DIR, E33KWQ_BUILDER)
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["E33KT_LOCAL_SANDBOX"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e33_kit_chain_e2e as T    # noqa: E402

bns = {"__file__": os.environ["E33KWQ_BUILDER"], "__name__": "e33khq"}
exec(compile(Path(os.environ["E33KWQ_BUILDER"]).read_text(encoding="utf-8"), "build_e33_cpu_kit.py", "exec"), bns)
src = bns["E33_REAL"]
blk = src[src.index('FK = {"log": [], "lock": _th33.Lock()}'):src.index('REP["checks"]["e33_hooks"] = E33H; save()')]
T.setUpModule()
work = T.FIX["tmp"] / "khq"
work.mkdir()
g = T.nb_namespace("http://127.0.0.1:1", {}, work)            # the notebook's CELL 1 switch WITHOUT a kit override (60k / 104k)
g["E11_HARNESS"] = False
M = T.FIX["e33"]
os.environ.update(T.S33.server_env(g["E33_CHAIN"]))
g["e33_setup"]()
g["e29d_tok"] = lambda: (g["E33_RT"].__setitem__("tokenizer", {"stop_ids_equal_spec": True, "think_close_ids": [T.CLOSE]}), T.TOK)[1]
K = T.FIX["tmp"] / "khq_k"
K.mkdir()
E33V, REP = {}, {"checks": {}}
import threading, types, tempfile   # noqa: E402,E401
ns = dict(G=g, K=K, KF={"tok_update_src": "pass\n"}, os=os, json=json, Path=Path, time=time, traceback=traceback, _th33=threading,
          _ty33=types, _tf33=tempfile, E33V=E33V, REP=REP)
exec(compile(blk, "<E33_REAL hooks block>", "exec"), ns)
h = ns["E33H"]
print(json.dumps({k: h.get(k) for k in ("c", "d", "e", "e_sub", "can_start", "can_start_late", "fake", "error", "tb")}, default=str)[:4000])
print(json.dumps(E33V, indent=0))
ok = all(E33V.values()) and len(E33V) == 7
T.tearDownModule()
print("E33 KIT HOOKS QUICK:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
