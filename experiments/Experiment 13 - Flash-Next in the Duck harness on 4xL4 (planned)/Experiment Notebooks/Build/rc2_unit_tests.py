"""RC2 fault-injection regression tests (the three pre-GPU audits of 2026-10-31). Runs the exported notebook code with
RC2_UNIT=1 (definitions + A0 + prompts only, no arms) against a fake input tree, then injects each fault.

Usage: python rc2_unit_tests.py <fake input dir> <work dir>
"""
import json
import os
import runpy
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INP, WORK = Path(sys.argv[1]), Path(sys.argv[2])
shutil.rmtree(WORK, ignore_errors=True)
(WORK / "out").mkdir(parents=True)
(WORK / "tmp").mkdir()
(WORK / "shm").mkdir()
os.environ.update(RC2_UNIT="1", E23_FAKE_SERVER=str(HERE / "fake_marker_not_used.py"), E23_OUT=str(WORK / "out"),
                  E23_TMP=str(WORK / "tmp"), E23_INPUT=str(INP), RC2_SUBMISSION=str(WORK / "out/submission.json"),
                  RC2_SHM=str(WORK / "shm"), RC2_BUDGET_SEC="3600")
os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
ns = runpy.run_path(str(HERE / "rc2_code_export.py"), run_name="rc2")["gated"].__globals__   # live module globals
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))


def ev(**kw):
    """A clean A1-like record; kw overrides."""
    o = {"status": "DONE", "probes": {"ok": True}, "sampled": {"ok": True, "garbage": None},
         "oracle": {"errors": [None] * 4, "garbage": [None] * 4, "row_count": 4, "expected_rows": 4, "complete": True}}
    o.update(kw)
    return o


gated = ns["gated"]
ns["FAKE"] = None                                   # gates as on the real GPU (probes enforced)
# 1. oracle / sampled / probes evidence
check("missing oracle -> not CLEAN", gated(ev(oracle=None))["correctness"] == "INCONCLUSIVE_NO_ORACLE")
o = ev(); del o["oracle"]
check("absent oracle key -> INCONCLUSIVE_NO_ORACLE", gated(o)["correctness"] == "INCONCLUSIVE_NO_ORACLE")
check("sampled.ok False -> NO_GO_SAMPLED_PATH", gated(ev(sampled={"ok": False, "tokens": 0, "error": "HTTP 500"}))["correctness"] == "NO_GO_SAMPLED_PATH")
check("sampled missing -> INCONCLUSIVE_NO_SAMPLED_TEST", gated(ev(sampled=None))["correctness"] == "INCONCLUSIVE_NO_SAMPLED_TEST")
check("sampled malformed (string) -> not CLEAN", gated(ev(sampled="ok"))["correctness"] != "CLEAN")
check("sampled garbage -> NO_GO_SAMPLED_PATH", gated(ev(sampled={"ok": True, "garbage": "BANG_GARBAGE"}))["correctness"] == "NO_GO_SAMPLED_PATH")
check("probe failed -> NO_GO_SHORT_PROBE", gated(ev(probes={"ok": False}))["correctness"] == "NO_GO_SHORT_PROBE")
check("probes missing -> INCONCLUSIVE_PROBES_MISSING", gated(ev(probes=None))["correctness"] == "INCONCLUSIVE_PROBES_MISSING")
part = ev(oracle={"errors": [None] * 3, "garbage": [None] * 3, "row_count": 3, "expected_rows": 4, "complete": False})
check("3 of 4 oracle rows -> INCONCLUSIVE_ORACLE_INCOMPLETE", gated(part)["correctness"] == "INCONCLUSIVE_ORACLE_INCOMPLETE")
short = ev(oracle={"errors": [None] * 4, "garbage": [None] * 4, "row_count": 4, "expected_rows": 4, "complete": False})
check("short-token row (complete False) -> INCONCLUSIVE", gated(short)["correctness"] == "INCONCLUSIVE_ORACLE_INCOMPLETE")
err = ev(oracle={"errors": [None, "HTTPError 500", None, None], "garbage": [None] * 4, "row_count": 4, "complete": False})
check("oracle HTTP error -> NO_GO_ORACLE_REQUEST_ERROR", gated(err)["correctness"] == "NO_GO_ORACLE_REQUEST_ERROR")
check("clean 4-row oracle -> CLEAN", gated(ev())["correctness"] == "CLEAN")
st = ev(oracle={"errors": [None] * 3, "garbage": [None] * 3, "row_count": 3, "expected_rows": 3, "complete": True})
check("stock 3-row oracle with expected_rows=3 -> CLEAN", gated(st, expected_rows=3)["correctness"] == "CLEAN")
check("FP8 need=bf16 with vs_bf16 missing -> not CLEAN", gated(ev(), need=("bf16",))["correctness"] != "CLEAN")
# 2. census_long garbage is gated
g = gated(ev(census_long={"116k@4": {"status": "FAIL_GARBAGE_UNDER_LOAD"}}))
check("census_long garbage -> NO_GO_GARBAGE_UNDER_LOAD", g["correctness"] == "NO_GO_GARBAGE_UNDER_LOAD", g["correctness"])

# 3. oracle() completeness from real rows
L = ns["L"]
real_greedy = L.greedy
NT = ns["NTOK"]
L.greedy = lambda port, prompts, n_tok=0, timeout=0: [{"text": "word " * 40, "tokens": list(range(NT)), "error": None,
                                                       "margins": [1.0] * NT} for _ in prompts[:-1]]
r = ns["oracle"](1, key="unit_partial")
check("oracle() with a missing row -> complete False, no reference", r["complete"] is False and "unit_partial" not in ns["ROWS"])
L.greedy = lambda port, prompts, n_tok=0, timeout=0: [{"text": "word " * 40, "tokens": list(range(NT - 5)), "error": None,
                                                       "margins": [1.0] * NT} for _ in prompts]
r = ns["oracle"](1, key="unit_short")
check("oracle() with short rows -> complete False", r["complete"] is False)
L.greedy = lambda port, prompts, n_tok=0, timeout=0: [{"text": "alpha beta gamma delta " * 10, "tokens": list(range(NT)),
                                                       "error": None, "margins": [1.0] * NT} for _ in prompts]
ns["L"].template_probe = lambda *a, **k: {"status": "MATCH"}
r = ns["oracle"](1, key="unit_ok")
check("oracle() complete rows -> complete True + reference stored", r["complete"] is True and "unit_ok" in ns["ROWS"])
L.greedy = real_greedy

# 4. FP8 gate: anything but PASS never runs the load tests
calls = {"levels": 0, "long_rows": 0}
ns["levels"] = lambda *a, **k: calls.__setitem__("levels", calls["levels"] + 1) or {}
ns["long_rows"] = lambda *a, **k: calls.__setitem__("long_rows", calls["long_rows"] + 1) or {}


class FakeS:
    port = 1
    def alive(self): return True
    def log(self): return ""


ns["kv_census"] = lambda s: {}
ns["L"].chat_probes = lambda p: {"ok": True}
ns["sampled_check"] = lambda s: {"ok": True, "garbage": None}
ns["tf_last"] = lambda s, **k: {"alive_after": True}
ns["PREFILL"]["measured"] = True
captured = {}


def fake_boot(name, kv, port, timeout, body, need=(), expected_rows=4, require_sampled=True):
    captured["o"] = body([FakeS()])
    return captured["o"]


ns["merlin_boot"] = fake_boot
for vs, want in (("INCONCLUSIVE_NO_ORACLE", "INCONCLUSIVE"), ("FAIL_HIGH_MARGIN", "FAIL")):
    calls.update(levels=0, long_rows=0)
    ns["oracle"] = lambda p, key=None, refs=(), prompts=None, vs=vs: {"errors": [None] * 4, "garbage": [None] * 4,
                                                                         "row_count": 4, "complete": True, "vs_bf16": {"status": vs}}
    o = ns["arm_fp8"]()
    check(f"FP8 vs_bf16 {vs} -> gate {want}, no levels / long_rows", o["fp8_gate"] == want and calls == {"levels": 0, "long_rows": 0},
          f"{o['fp8_gate']} {calls} {o.get('status')}")
calls.update(levels=0, long_rows=0)
ns["oracle"] = lambda p, key=None, refs=(), prompts=None: {"errors": [None] * 4, "garbage": [None] * 4, "row_count": 4,
                                                             "complete": False, "vs_bf16": {"status": "EXACT"}}
o = ns["arm_fp8"]()
check("FP8 EXACT but incomplete oracle -> INCONCLUSIVE, no load", o["fp8_gate"] == "INCONCLUSIVE" and calls["levels"] == 0)
ns["sampled_check"] = lambda s: {"ok": False, "error": "HTTP 500"}
ns["oracle"] = lambda p, key=None, refs=(), prompts=None: {"errors": [None] * 4, "garbage": [None] * 4, "row_count": 4,
                                                             "complete": True, "vs_bf16": {"status": "EXACT"}}
o = ns["arm_fp8"]()
check("FP8 sampled failure -> FAIL_SAMPLED, no load", o["fp8_gate"] == "FAIL_SAMPLED" and calls["levels"] == 0)
ns["sampled_check"] = lambda s: {"ok": True, "garbage": None}
o = ns["arm_fp8"]()
check("FP8 PASS -> load tests run", o["fp8_gate"] == "PASS" and calls["levels"] >= 1 and calls["long_rows"] == 1, str(calls))

# 5. host_phase failure stops the boot before any server
started = []
ns["MServer"] = lambda *a, **k: started.append(a) or None


class Plan:
    status = "untested"
    env = {}
    def argv(self): return ["/v/bin/vllm", "serve", "/ckpt"]


class ML:
    @staticmethod
    def build(*a, **k): return Plan()
    @staticmethod
    def host_phase(plan):
        raise RuntimeError("LaunchError: scales sidecar missing")
    @staticmethod
    def distributed_ports(p): return 29500, 30000


ns["merlin_install"] = lambda: {"kv": {"MERLIN_VENV": "/v", "MERLIN_TREE": "/t", "MERLIN_SHIM": "/s"}, "ML": ML, "cuda_home": "/c"}
ns["FAKE"] = None
try:
    ns["merlin_fn_server"]("unit", "fp8", 8001, 1)
    raised = False
except RuntimeError as e:
    raised = "MERLIN_HOST_PHASE_FAILED" in str(e)
check("host_phase LaunchError -> raises, no server started, recorded",
      raised and not started and "unit" in ns["R"]["setup"].get("host_phase_error", {}))

# 6. stock fallback: unavailable inputs and the boot gate
ns["INTEL"], ns["WHEELHOUSE"] = None, None
check("stock fallback without Intel / wheelhouse -> FALLBACK_UNAVAILABLE", ns["arm_stock"]()["status"] == "FALLBACK_UNAVAILABLE")
ns["INTEL"], ns["WHEELHOUSE"] = Path("/intel"), Path("/wh")
stock_started = []
ns["stock_fn_server"] = lambda *a, **k: stock_started.append(a)
for gv in (False, None):
    ns["boot_gate"] = lambda gv=gv: {"ok": gv, "gpus": []}
    check(f"stock boot gate ok={gv} -> GPU_NOT_EMPTY, no server", ns["arm_stock"]()["status"] == "GPU_NOT_EMPTY" and not stock_started)

# 7. TF death classification (the real merlin_boot, with_servers stubbed)
ns2 = runpy.run_path(str(HERE / "rc2_code_export.py"), run_name="rc2b")["gated"].__globals__
ns2["FAKE"] = None
ns2["boot_gate"] = lambda: {"ok": True, "gpus": []}
ns2["merlin_fn_server"] = lambda *a, **k: None
for tail, xid, want in (("torch.OutOfMemoryError: CUDA out of memory", False, "TF_OOM"),
                        ("CUDA error: an illegal memory access was encountered", True, "TF_SERVER_DIED"),
                        ("Segmentation fault", False, "TF_SERVER_DIED")):
    ns2["with_servers"] = lambda servers, timeout, body, tail=tail, xid=xid: dict(
        ev(status="SERVER_DIED"), tf_died=True, servers={"m": {"log_tail": tail, "xid": xid}})
    o = ns2["merlin_boot"]("m", "bf16", 8000, 60, None)
    check(f"TF death '{tail[:30]}' -> {want}", o["status"] == want, o["status"])

# 8. /dev/shm: only entries created during the boot are removed
shm = WORK / "shm"
(shm / "torch_test_unrelated").write_text("x")
(shm / "other_file").write_text("x")
before = ns["shm_names"]()
(shm / "torch_new_from_boot").write_text("x")
(shm / "psm_newboot").write_text("x")
(shm / "unknown_new").write_text("x")
r = ns["shm_cleanup"](before)
names = {p.name for p in shm.iterdir()}
check("shm: pre-existing torch_ entry kept, boot's torch_/psm_ removed, unknown kept",
      "torch_test_unrelated" in names and "other_file" in names and "unknown_new" in names
      and "torch_new_from_boot" not in names and "psm_newboot" not in names, str(sorted(names)))

# 9. checkpoint merge: a wrong size and a wrong sha are both refused (real GPU mode, FAKE off)
ns3 = runpy.run_path(str(HERE / "rc2_code_export.py"), run_name="rc2c")["gated"].__globals__
ns3["FAKE"] = None
_sym = Path.symlink_to
Path.symlink_to = lambda self, src, *a, **k: shutil.copyfile(src, self)   # Windows: no symlink privilege
try:
    import hashlib
    for n in list(ns3["PLE_EXPECTED"]):                     # the fake tree's PLE shards (tiny files)
        f = ns3["PLE_DIR"] / n
        ns3["PLE_EXPECTED"][n] = [f.stat().st_size, hashlib.sha256(f.read_bytes()).hexdigest()]
    p1 = ns3["PARTS"][1]
    for n in list(p1[1]["files"]):                           # part 1 small files: real sizes / sha of the local copies
        f = p1[0] / n
        p1[1]["files"][n] = [f.stat().st_size, hashlib.sha256(f.read_bytes()).hexdigest()]
    k1 = next(iter(ns3["PARTS"][2][1]["files"]))
    size, sha = ns3["PARTS"][2][1]["files"][k1]
    ns3["PARTS"][2][1]["files"][k1] = [size + 1, sha]
    m = ns3["merge_ckpt"]()
    check("merge: wrong size -> merge not ok", m["ok"] is False and any(k1 in b for b in m["bad"]), str(m["bad"][:2]))
    ns3["PARTS"][2][1]["files"][k1] = [size, "0" * 64]
    m = ns3["merge_ckpt"]()
    ok, why = ns3["sha_gate"](120)
    check("merge: wrong sha256 -> sha gate refuses the boot", m["ok"] is True and ok is False and "mismatch" in why, why)
    ns3["PARTS"][2][1]["files"][k1] = [size, sha]
    m = ns3["merge_ckpt"]()
    ok, why = ns3["sha_gate"](120)
    check("merge: correct sizes + sha256 -> sha gate ok", m["ok"] is True and ok is True, why)
finally:
    Path.symlink_to = _sym

n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
(WORK / "unit_results.json").write_text(json.dumps([{"test": n, "ok": ok, "detail": d} for n, ok, d in RESULTS], indent=1))
sys.exit(1 if n_fail else 0)
