"""E25 NB6Q SUBMIT unit tests (quick, local; the CPU units kit (S1, S3) and real kit (S2, S4) run the same file).

NB6Q = the proven NB6 SUBMIT notebook (87afcc84) + ONE change, the Q-defer patch (nb4_ppack.py rev 2 + nb6q_lib.py, packed).
  S1 static     only the markdown and the engine cell differ from NB6; the engine cell only gains lines (Q2, Q3, the NB6Q
                block at its end); the packed literal = nb4_ppack.py (the NB4S module, sha 8dadf076) + nb6q_lib.py on disk;
                placements (AST): Q2 after the warm-up counters, Q3 after the boot's server stopped; CELL 8's first
                merlin_install() (step A0) runs before the sha gate, #60490 and any boot; NB6's merlin_fn_server calls
                merlin_install() first; e21_relaunch (NB6, unchanged) calls merlin_fn_server; the only env key set is NB4_PPDEFER
  S2 functional the notebook's own NB6Q block text, exec'd with stubs, on copies of the REAL Merlin ad42e37 source files:
                applied path (pre / post shas, env on every launch, once only), env only when applied (inherited NB4_* removed),
                failure paths (foreign tree -> refused, files untouched; failure after the write -> byte-for-byte restore;
                restore failing -> FAILED_REVERT_INCOMPLETE; embed sha mismatch; load failure; install failure), every one
                continuing without the patch and without NB4_PPDEFER and never raising; read-back recorded (ENGAGED /
                ENGAGED_FALLBACK / NOT_INTEGRATED / NOT_APPLIED / READBACK_ERROR), never raising; the non-FAKE path (venv
                python; POSIX only, the CPU real kit)
  S3 NB6's own unit tests (nb6_unit_tests.py) on the NB6Q notebook: every check PASS except exactly the NB6-only push-id check
                (replaced by S3b: the NB6Q push folder = this notebook, every other metadata field = NB6's)
  S4 nb4_ppack_tests.py on the real Merlin source copy (reused: 247 checks)
NB6QT_SECTIONS (default S1,S2,S3,S4) selects sections: the units kit runs S1,S3; the real kit runs S2,S4 with NB6QT_MERLIN_SRC =
the real Merlin tree installed from the payload (pristine, after the kit's revert).
usage: python nb6q_unit_tests.py   (exit 0 = all PASS; last line @@NB6Q_UNITS_OK@@)"""
import ast
import contextlib
import difflib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = os.environ.get
NB = Path(ENV("NB6QT_NB") or HERE.parent / "Q27-XHIGH-E25_NB6Q_SUBMIT.ipynb")
NB6 = Path(ENV("NB6QT_NB6") or HERE.parent / "Q27-XHIGH-E25_NB6_SUBMIT.ipynb")
MSRC = Path(ENV("NB6QT_MERLIN_SRC") or "D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
PPACK = Path(ENV("NB6QT_PPACK") or HERE / "nb4_ppack.py")
LIB = Path(ENV("NB6QT_LIB") or HERE / "nb6q_lib.py")
PPTESTS = Path(ENV("NB6QT_PPACK_TESTS") or HERE / "nb4_ppack_tests.py")
NB6UNITS = Path(ENV("NB6QT_NB6_UNITS") or HERE / "nb6_unit_tests.py")
PUSH = Path(ENV("NB6QT_PUSH") or "D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb6q-submit__foysalemonshanto")
PUSH6 = Path(ENV("NB6QT_PUSH6") or "D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb6-submit__foysalemonshanto")
NB6_SHA = "87afcc848cd60eee290509b6a1171f04a754015130d69363d9046d265b2c1e2b"
PPACK_SHA = "8dadf0763d43d22966603745198ad327ed6ed36001cc60a79925ea18818f2dea"      # = the module NB4S fa9f35ca embeds
FILES = {"vllm/v1/worker/gpu/pp_utils.py": ("6be931deaf8bee6dc9571c3c84d7e92c9f26c618c09dcb58a8b384bff487a662",
                                            "8b7fa4a6e4598ff4d4a56eb6e595efb1e494276b12b187b099973c262e90c69a"),
         "vllm/v1/worker/gpu/model_runner.py": ("05e889afba5dd60b6995b78aaa1f69051770ab02b638680097eccbf4e592d64c",
                                                "2c25fda60dbd5f2506ff989c29ccfe8fc0ff082fbad457fc3ae9c89ed9a185d6"),
         "vllm/v1/worker/gpu_worker.py": ("a97d08ef042a53fcd139bae041fff407fce75796650d936c200c699f0d63a489",
                                          "5642b89d918b30e5b70b1067d53776e379613035b7b6e91e93cf0ee4279cfef5")}
NB6_ID_CHECK = "kernel-metadata: id / code_file / the notebook in the folder = this NB6 build"
RESULTS, SKIPPED = [], []
SEC = set((ENV("NB6QT_SECTIONS") or "S1,S2,S3,S4").split(","))   # units kit: S1,S3 | real kit: S2,S4 (the real Merlin tree)


def check(name, cond, detail=""):
    if name.split(" ", 1)[0].rstrip("b") not in SEC:     # a section this run does not select: neither evaluated nor counted
        return None
    try:
        ok = bool(cond() if callable(cond) else cond)
    except Exception as e:
        ok, detail = False, f"raised {type(e).__name__}: {e} | {detail}"
    RESULTS.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  [{str(detail)[:900]}]"), flush=True)
    return ok


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def sha_f(p):
    return sha_b(Path(p).read_bytes())


def cells_of(p):
    return [(c["cell_type"], "".join(c["source"])) for c in json.loads(Path(p).read_text(encoding="utf-8"))["cells"]]


raw = NB.read_bytes()
C, C6 = cells_of(NB), cells_of(NB6)
code = [t for k, t in C if k == "code"]
ENG, ENG6 = C[5][1], C6[5][1]
print("NB6Q", NB.name, "sha256", sha_b(raw), "|", len(raw), "bytes", flush=True)

# ============================================================================ S1 static
check("S1 NB6Q notebook < 400 kB (Kaggle commit limit)", len(raw) < 400_000, len(raw))
check("S1 no CR byte in the notebook", b"\r" not in raw.replace(b"\\r", b""))
check("S1 the NB6 reference is the GPU-committed 87afcc84", sha_f(NB6) == NB6_SHA)
check("S1 same cell layout as NB6 (9 cells, same kinds)", [k for k, _ in C] == [k for k, _ in C6], [k for k, _ in C])
_dc = [i for i, (a, b) in enumerate(zip(C6, C)) if a != b]
check("S1 only the markdown (0) and the engine cell (5) differ from NB6: CELL 0, CELL 1 (profiles / levers / budgets), Duck, "
      "task order, CELL 6/7, CELL 8, CELL 9 are NB6's bytes", _dc == [0, 5], _dc)


def changed(a, b):
    rm, add = [], []
    for ln in difflib.ndiff(a.splitlines(), b.splitlines()):
        if ln.startswith("- "):
            rm.append(ln[2:])
        elif ln.startswith("+ "):
            add.append(ln[2:])
    return rm, add


_rm, _add = changed(ENG6, ENG)
HDR = "# ==================================================================== NB6Q: the Q-defer patch (deferred PP receive)"
check("S1 engine cell: no NB6 line removed or changed (insertions only)", _rm == [], _rm[:5])
_blk_i = ENG.index(HDR) if HDR in ENG else -1
BLOCK = ENG[_blk_i:] if _blk_i >= 0 else ""
_blk_lines = set(BLOCK.splitlines())
_outside = [ln for ln in _add if ln not in _blk_lines and ln.strip()]
check("S1 engine cell: the added lines = the NB6Q block at the end of the cell + the two marked read-back lines (Q2, Q3)",
      _blk_i > 0 and ENG[:_blk_i].rstrip("\n") == ENG6.rstrip("\n").replace(
          '        trec["counters_after_warmup"] = e21_counters()\n',
          '        trec["counters_after_warmup"] = e21_counters()\n'
          '        nb6q_readback(trec, s, "after_warmup")   # NB6Q Q2: Q-defer read-back after the warm-up (record only)\n').replace(
          '    trec["log_text_for_ladder"] = srv.log() or ""\n',
          '    trec["log_text_for_ladder"] = srv.log() or ""\n'
          '    nb6q_readback(trec, srv, "final")           # NB6Q Q3: Q-defer read-back, whole log after the server stopped (record only)\n')
      and len(_outside) == 2 and all("NB6Q Q2" in ln or "NB6Q Q3" in ln for ln in _outside), _outside)
_mrm, _madd = changed(C6[0][1], C[0][1])
check("S1 markdown: only the title line replaced, the NB6Q bullet added",
      _mrm == ["# E25 NB6 SUBMIT · FN-Duck production submission · NB6COMMIT"] and _madd[0].startswith("# E25 NB6Q SUBMIT")
      and any("Q-defer" in ln for ln in _madd), (_mrm, _madd[:2]))
_defs = [n.name for n in ast.parse(ENG).body if isinstance(n, ast.FunctionDef)]
check("S1 the NB6Q block comes after every NB6 definition of the engine cell (wrappers bind after the originals)",
      _blk_i > max(ENG.index(f"\ndef {n}(") for n in ("merlin_install", "merlin_fn_server", "nb5_boot_try", "nb5_boot",
                                                      "e21_relaunch", "nb5_write_report")))
_pk = re.search(r'b85decode\(\n\s+"([^"]+)"\)\)', BLOCK)
PACK = None
try:
    import base64
    import lzma
    PACK = json.loads(lzma.decompress(base64.b85decode(_pk.group(1))).decode("utf-8"))
except Exception as e:
    PACK = {"_error": repr(e)}
_lib_sha = re.search(r'NB6Q_LIB_SHA = "([0-9a-f]{64})"', BLOCK)
_pp_sha = re.search(r'NB6Q_PPACK_SHA = "([0-9a-f]{64})"', BLOCK)
PP_DISK = PPACK.read_bytes().decode("utf-8").replace("\r\n", "\n")
LIB_DISK = LIB.read_bytes().decode("utf-8").replace("\r\n", "\n")
check("S1 packed literal = {nb4_ppack.py, nb6q_lib.py}, verbatim = the files on disk (LF), shas in the cell match",
      sorted(PACK) == ["nb4_ppack.py", "nb6q_lib.py"] and PACK["nb4_ppack.py"] == PP_DISK and PACK["nb6q_lib.py"] == LIB_DISK
      and _lib_sha and _pp_sha and sha_b(LIB_DISK.encode()) == _lib_sha.group(1) and sha_b(PP_DISK.encode()) == _pp_sha.group(1),
      sorted(PACK))
check("S1 nb4_ppack.py = the rev 2 module the GPU-run NB4S notebook embeds (sha 8dadf076)", _pp_sha and _pp_sha.group(1) == PPACK_SHA)
_ppns = {"__name__": "nb4_ppack_t"}
exec(compile(PP_DISK, "nb4_ppack.py", "exec"), _ppns)
check("S1 nb4_ppack PATCHES = the three files with the expected pre / post shas; server_env('Q-defer') = {NB4_PPDEFER: '1'}",
      {r: (p[0], p[1]) for r, p in _ppns["PATCHES"].items()} == FILES and _ppns["server_env"]("Q-defer") == {"NB4_PPDEFER": "1"})
_libns_tree = ast.parse(LIB_DISK)
_lf = next(ast.literal_eval(n.value) for n in _libns_tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "NB6Q_FILES")
check("S1 nb6q_lib NB6Q_FILES = nb4_ppack PATCHES (pre / post)", {r: (v["pre"], v["post"]) for r, v in _lf.items()} == FILES)
_env_sets = re.findall(r'"(NB4_[A-Z]+)"', LIB_DISK)
check("S1 no other lever: nb6q_lib names only NB4_PPDEFER / NB4_PPACK, and NB4_PPACK only in the removal tuple; the env set = "
      "server_env('Q-defer')", set(_env_sets) == {"NB4_PPDEFER", "NB4_PPACK"} and LIB_DISK.count('"NB4_PPACK"') == 1
      and 'for k in ("NB4_PPDEFER", "NB4_PPACK") if k in s.env' in LIB_DISK and 'env=PQ.server_env("Q-defer")' in LIB_DISK
      and "os.environ[" not in LIB_DISK and "os.environ.update" not in LIB_DISK)


def _fn(text, name):
    return next(n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name)


def _after(body, pred):
    """the statement right after the first statement matching pred (searched in nested bodies too)"""
    for i, st in enumerate(body):
        if pred(st):
            return body[i + 1] if i + 1 < len(body) else None
        for f in ("body", "orelse", "finalbody"):
            sub = getattr(st, f, None)
            if isinstance(sub, list) and sub and not isinstance(st, ast.FunctionDef):
                r = _after(sub, pred)
                if r is not None:
                    return r
    return None


_bt = _fn(ENG, "nb5_boot_try")
_body = next(n for n in _bt.body if isinstance(n, ast.FunctionDef) and n.name == "body")
_seg = lambda n: ast.unparse(n) if n is not None else None
_q2 = _after(_body.body, lambda st: "counters_after_warmup" in _seg(st) and isinstance(st, ast.Assign))
_q2n = _after(_body.body, lambda st: _seg(st) == _seg(_q2))
check("S1 Q2 placement: nb6q_readback(trec, s, 'after_warmup') right after the warm-up counters, before phases_started",
      _seg(_q2) == "nb6q_readback(trec, s, 'after_warmup')" and "phases_started" in _seg(_q2n), (_seg(_q2), _seg(_q2n)))
_q3 = _after(_bt.body, lambda st: "log_text_for_ladder" in _seg(st) and isinstance(st, ast.Assign))
check("S1 Q3 placement: nb6q_readback(trec, srv, 'final') right after the boot's with_servers() returned (server stopped)",
      _seg(_q3) == "nb6q_readback(trec, srv, 'final')" and ENG.index("o = with_servers(NB5_SERVERS") < ENG.index('nb6q_readback(trec, srv, "final")'),
      _seg(_q3))
C8 = code[6]
check("S1 CELL 8 (NB6 bytes): the first merlin_install() (step A0) runs before the sha gate, #60490 and the first boot",
      "CELL 8" in C8.splitlines()[0] and C8.index("merlin_install())") < C8.index("sha_gate(") < C8.index("nb5_apply_60490()")
      < C8.index("nb5_boot(f\"merlin_") and "merlin_install(" not in C8[:C8.index("step(\"A0\", \"merlin_install\"")])
_mfs = _fn(ENG6, "merlin_fn_server")
check("S1 NB6's merlin_fn_server calls merlin_install() first (every launch plan comes after the apply)",
      _seg(_mfs.body[1]) == "eng = merlin_install()", _seg(_mfs.body[1]))
check("S1 NB6's e21_relaunch is unchanged and relaunches through merlin_fn_server (the relaunched server gets the same env)",
      _seg(_fn(ENG, "e21_relaunch")) == _seg(_fn(ENG6, "e21_relaunch")) and "merlin_fn_server(old.name" in _seg(_fn(ENG, "e21_relaunch")))
check("S1 the load-failure branch defines nb6q_readback (Q2 / Q3 never raise NameError)",
      "except Exception as _nb6q_e:" in BLOCK and "    def nb6q_readback(trec, s, stage):" in BLOCK)

WORK = Path(tempfile.mkdtemp(prefix="nb6qt_"))
# ============================================================================ S2 functional (the notebook's own block)
if "S2" in SEC:
    _src_ok = all((MSRC / r).is_file() and sha_f(MSRC / r) == FILES[r][0] for r in FILES)
    check("S2 the real Merlin ad42e37 source copy is present with the pre shas", _src_ok, str(MSRC))


    def tree_copy(tag, foreign=False):
        t = WORK / f"tree_{tag}"
        for r in FILES:
            (t / r).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(MSRC / r, t / r)
        if foreign:
            with open(t / "vllm/v1/worker/gpu/pp_utils.py", "a", encoding="utf-8", newline="\n") as fh:
                fh.write("# a foreign edit\n")
        return t


    class Srv(types.SimpleNamespace):
        pass


    def make_ns(tag, fake=True, install=None, block=None, base_env=None):
        tmp = WORK / f"tmp_{tag}"
        tmp.mkdir(parents=True, exist_ok=True)
        logs, calls = [], {"install": 0, "server": 0}
        ns = {"__name__": "nb6q_t", "os": os, "sys": sys, "json": json, "time": time, "hashlib": hashlib, "traceback": traceback,
              "Path": Path, "TMP": tmp, "FAKE": "fake_server.py" if fake else None, "R": {"setup": {}}, "save": lambda: None,
              "log": lambda *a: logs.append(" ".join(str(x) for x in a)), "NB5_REPORT": {"phases": {}}}

        def merlin_install():
            calls["install"] += 1
            return install() if install else {"fake": True}

        def merlin_fn_server(name, kv, port, kv_bytes):
            ns["merlin_install"]()                       # = NB6's first statement (global lookup -> the NB6Q wrapper)
            calls["server"] += 1
            return Srv(name=name, env=dict(base_env or {}, PATH="x"), log_path=tmp / f"server_{name}.log", argv=["serve"])
        ns.update(merlin_install=merlin_install, merlin_fn_server=merlin_fn_server)
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()) as out:
            exec(compile(block or BLOCK, "<nb6q block>", "exec"), ns)
        ns["_logs"], ns["_calls"], ns["_stderr"], ns["_stdout"] = logs, calls, err, out
        return ns


    def run_q(ns, fn, *a):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            r = fn(*a)
        ns["_stderr_run"] = ns.get("_stderr_run", "") + err.getvalue()
        return r


    def ppq(rec):
        return "(Worker pid=1) NB4_PPQ " + json.dumps(rec, sort_keys=True)


    def ppq_log(kind="engaged"):
        L = ["INFO some vllm line"]
        for pid, last in ((101, True), (102, True), (201, False), (202, False)):
            if kind == "three" and pid == 202:
                continue
            L.append(ppq({"event": "init", "pid": pid, "ppack": False, "ppdefer": not last, "is_last_rank": last, "k": 3}))
        for pid in (201, 202):
            ex, cons = (990, 0) if kind != "fallback" else (500, 500)
            st = {"defer_park": 1000, "flush_exec": ex, "flush_empty": 10, "flush_recv": 0, "flush_consume": cons, "flush_rpc": 0,
                  "flush_shutdown": 0}
            L.append(ppq({"event": "stats", "pid": pid, "key": "defer_park", "stats": st}))
            L.append(ppq({"event": "stats", "pid": pid, "key": "flush_exec", "stats": st}))
        return "\n".join(L) + "\n"


    os.environ.pop("NB6Q_FAKE_TREE", None)
    try:
        # ---- T1 the applied path (FAKE + NB6Q_FAKE_TREE = a copy of the real Merlin source)
        T1 = tree_copy("t1")
        os.environ["NB6Q_FAKE_TREE"] = str(T1)
        ns = make_ns("t1", base_env={"NB4_PPACK": "1"})
        check("S2 block exec'd: NB6Q state NOT_ATTEMPTED, merlin_install / merlin_fn_server wrapped, nothing applied at load",
              ns["NB6Q"]["state"] == "NOT_ATTEMPTED" and ns["merlin_install"].__doc__.startswith("NB6Q")
              and ns["merlin_fn_server"].__doc__.startswith("NB6Q") and all(sha_f(T1 / r) == FILES[r][0] for r in FILES))
        PQ = ns["nb6q_module"]()
        _orig_apply, _n_apply = PQ.apply_ppack_patch, [0]

        def _count_apply(*a, **k):
            _n_apply[0] += 1
            return _orig_apply(*a, **k)
        PQ.apply_ppack_patch = _count_apply
        check("S2 nb6q_module: the embedded nb4_ppack written to TMP with its sha and loaded from that path",
              sha_f(Path(ns["TMP"]) / "nb4_ppack.py") == PPACK_SHA and Path(PQ.__file__).resolve() == (Path(ns["TMP"]) / "nb4_ppack.py").resolve())
        eng = run_q(ns, ns["merlin_install"])
        Q = ns["NB6Q"]
        check("S2 applied path: right after merlin_install -> APPLIED, pre shas = Merlin ad42e37, post shas = the patched files on disk",
              Q["state"] == "APPLIED" and Q["applied"] is True and Q["pre"] == {r: v[0] for r, v in FILES.items()}
              and Q["post"] == {r: v[1] for r, v in FILES.items()} and all(sha_f(T1 / r) == FILES[r][1] for r in FILES)
              and Q["apply"]["ok"] is True and Q["verify_state"] == "applied", json.dumps({k: Q.get(k) for k in ("state", "error", "post")})[:600])
        check("S2 applied path: env = {NB4_PPDEFER: '1'}; recorded in R setup and the report (same dict); logged",
              Q["env"] == {"NB4_PPDEFER": "1"} and ns["R"]["setup"]["nb6q_ppdefer"] is Q and ns["NB5_REPORT"]["nb6q_ppdefer"] is Q
              and any("APPLIED" in ln for ln in ns["_logs"]))
        run_q(ns, ns["merlin_install"])
        s1 = run_q(ns, ns["merlin_fn_server"], "merlin_fp8_mtp_pin100", "fp8", 8000, 3553580524)
        check("S2 the patch is applied ONCE (second install / a launch do not re-apply); NB6's install ran each time",
              _n_apply[0] == 1 and ns["_calls"]["install"] == 3, (_n_apply, ns["_calls"]))
        check("S2 launch env when applied: NB4_PPDEFER=1 set, an inherited NB4_PPACK removed (recorded), nothing else touched",
              s1.env.get("NB4_PPDEFER") == "1" and "NB4_PPACK" not in s1.env and s1.env.get("PATH") == "x"
              and Q["launch_env"]["merlin_fp8_mtp_pin100"] == {"NB4_PPDEFER": "1", "removed_inherited": {"NB4_PPACK": "1"}}, s1.env)
        check("S2 the patched tree verifies as 'applied' with nb4_ppack.verify_ppack_patch", PQ.verify_ppack_patch(T1)["state"] == "applied")
        # read-back
        tr = {}
        Path(s1.log_path).write_text(ppq_log("engaged"), encoding="utf-8")
        rb = ns["nb6q_readback"](tr, s1, "after_warmup")
        check("S2 read-back: 4 init lines + clean flush ratios -> ENGAGED, recorded in the try, NB6Q readbacks / last_label, logged",
              rb["label"] == "ENGAGED" and tr["nb6q_readback"]["after_warmup"]["label"] == "ENGAGED" and Q["last_label"] == "ENGAGED"
              and Q["readbacks"][-1]["stage"] == "after_warmup" and rb["ppq_lines"] == 8 and rb["env"] == "1"
              and any("read-back after_warmup" in ln for ln in ns["_logs"]), json.dumps(rb, default=str)[:600])
        Path(s1.log_path).write_text(ppq_log("fallback"), encoding="utf-8")
        check("S2 read-back: deferral collapsed to post-at-consume -> ENGAGED_FALLBACK (recorded only)",
              ns["nb6q_readback"](tr, s1, "final")["label"] == "ENGAGED_FALLBACK" and tr["nb6q_readback"]["final"]["label"] == "ENGAGED_FALLBACK")
        Path(s1.log_path).write_text(ppq_log("three"), encoding="utf-8")
        check("S2 read-back: 3 of 4 init lines -> NOT_INTEGRATED (recorded only, no raise)", ns["nb6q_readback"]({}, s1, "x")["label"] == "NOT_INTEGRATED")
        Path(s1.log_path).write_text("no NB4 line here\n", encoding="utf-8")
        check("S2 read-back: no NB4_PPQ line -> NOT_INTEGRATED", ns["nb6q_readback"]({}, s1, "x")["label"] == "NOT_INTEGRATED")
        Path(s1.log_path).unlink()
        check("S2 read-back: no log file -> READBACK_ERROR, never raises", ns["nb6q_readback"]({}, s1, "final")["label"] == "READBACK_ERROR")
        check("S2 read-back: a broken trec (not a dict) never raises", ns["nb6q_readback"](None, s1, "final")["label"] == "READBACK_ERROR")
        rv = PQ.revert_ppack_patch(T1, sys.executable)
        check("S2 round trip on the real source: nb4_ppack.revert_ppack_patch -> every pre sha again", rv["ok"] is True
              and all(sha_f(T1 / r) == FILES[r][0] for r in FILES))

        # ---- T2 env only when applied: FAKE without a tree
        os.environ.pop("NB6Q_FAKE_TREE", None)
        ns = make_ns("t2", base_env={"NB4_PPDEFER": "1", "NB4_PPACK": "1"})
        s2 = run_q(ns, ns["merlin_fn_server"], "m2", "fp8", 8000, 1)
        check("S2 no tree (FAKE without NB6Q_FAKE_TREE) -> SKIPPED_FAKE_NO_TREE; the launch carries NO NB4_PPDEFER (inherited ones removed)",
              ns["NB6Q"]["state"] == "SKIPPED_FAKE_NO_TREE" and ns["NB6Q"]["applied"] is False and "NB4_PPDEFER" not in s2.env
              and "NB4_PPACK" not in s2.env and ns["NB6Q"]["env"] == {}, s2.env)
        check("S2 not applied -> read-back NOT_APPLIED (recorded)", ns["nb6q_readback"]({}, s2, "final")["label"] == "NOT_APPLIED")

        # ---- T3 failure path: a foreign tree (pre-sha mismatch) -> refused before any write
        T3 = tree_copy("t3", foreign=True)
        before = {r: (T3 / r).read_bytes() for r in FILES}
        os.environ["NB6Q_FAKE_TREE"] = str(T3)
        ns = make_ns("t3")
        run_q(ns, ns["merlin_install"])
        s3 = run_q(ns, ns["merlin_fn_server"], "m3", "fp8", 8000, 1)
        Q = ns["NB6Q"]
        check("S2 foreign tree: PATCH_PPACK_PRE_SHA_MISMATCH -> FAILED_REVERTED (nothing written: every file 'unchanged', bytes equal)",
              Q["state"] == "FAILED_REVERTED" and "PRE_SHA_MISMATCH" in Q["error"] and Q["applied"] is False
              and all(v["action"] == "unchanged" for v in Q["revert"]["files"].values())
              and all((T3 / r).read_bytes() == before[r] for r in FILES), json.dumps({k: Q.get(k) for k in ("state", "error")})[:500])
        check("S2 foreign tree: loud log (log + stderr), no exception, the launch has NO NB4_PPDEFER, read-back NOT_APPLIED",
              any("PATCH FAILED" in ln for ln in ns["_logs"]) and "PATCH FAILED" in ns.get("_stderr_run", "") and "NB4_PPDEFER" not in s3.env
              and ns["nb6q_readback"]({}, s3, "final")["label"] == "NOT_APPLIED")

        # ---- T4 failure AFTER the write (py_compile fails) -> byte-for-byte restore
        T4 = tree_copy("t4")
        orig = {r: (T4 / r).read_bytes() for r in FILES}
        os.environ["NB6Q_FAKE_TREE"] = str(T4)
        ns = make_ns("t4")
        PQ4 = ns["nb6q_module"]()
        PQ4._py_compile = lambda py, files: {"rc": 1, "err": "forced by the test"}
        for r in FILES:                                  # a stale pyc of a patched file must not survive the restore
            pc = (T4 / r).parent / "__pycache__"
            pc.mkdir(exist_ok=True)
            (pc / (Path(r).stem + ".cpython-312.pyc")).write_bytes(b"stale")
        run_q(ns, ns["merlin_install"])
        s4 = run_q(ns, ns["merlin_fn_server"], "m4", "fp8", 8000, 1)
        Q = ns["NB6Q"]
        check("S2 failure after the write (VERIFY_FAILED): every file 'restored' to its exact original bytes, pyc dropped, FAILED_REVERTED",
              Q["state"] == "FAILED_REVERTED" and "VERIFY_FAILED" in Q["error"]
              and all(v["action"] == "restored" and v["sha_equal_snapshot"] for v in Q["revert"]["files"].values())
              and all((T4 / r).read_bytes() == orig[r] for r in FILES)
              and not [p for r in FILES for p in ((T4 / r).parent / "__pycache__").glob(Path(r).stem + ".*.pyc")],
              json.dumps({k: Q.get(k) for k in ("state", "error", "revert")})[:700])
        check("S2 failure after the write: no NB4_PPDEFER on the launch, env {}, no exception", "NB4_PPDEFER" not in s4.env and Q["env"] == {})

        # ---- T5 the restore itself fails -> FAILED_REVERT_INCOMPLETE, still no env, still no raise
        T5 = tree_copy("t5")
        os.environ["NB6Q_FAKE_TREE"] = str(T5)
        ns = make_ns("t5")
        PQ5 = ns["nb6q_module"]()
        PQ5._py_compile = lambda py, files: {"rc": 1, "err": "forced by the test"}

        def _bad_replace(a, b):
            raise OSError("forced: replace refused")
        ns["os"] = types.SimpleNamespace(**{k: getattr(os, k) for k in ("environ", "path", "getpid")}, replace=_bad_replace)
        run_q(ns, ns["merlin_install"])
        s5 = run_q(ns, ns["merlin_fn_server"], "m5", "fp8", 8000, 1)
        check("S2 restore failing -> FAILED_REVERT_INCOMPLETE, applied False, no NB4_PPDEFER, no exception (env unset = the patched "
              "files run the original code paths; nothing restored = all patched = consistent)",
              ns["NB6Q"]["state"] == "FAILED_REVERT_INCOMPLETE" and ns["NB6Q"]["applied"] is False and ns["NB6Q"]["consistent"] is True
              and "NB4_PPDEFER" not in s5.env, ns["NB6Q"].get("revert"))
        ns["os"] = os
        for r in FILES:
            shutil.copyfile(MSRC / r, T5 / r)

        # ---- T5b the k-th restore write fails (k = 1, 2, 3): the reverse-order restore stops there -> a CONSISTENT tree
        # (the patched files are a prefix of pp_utils -> model_runner -> gpu_worker), no NB4_PPDEFER, no exception
        CHAIN = ["vllm/v1/worker/gpu/pp_utils.py", "vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu_worker.py"]
        WANT = {1: ["patched", "patched", "patched"], 2: ["patched", "patched", "original"], 3: ["patched", "original", "original"]}

        def fail_on(k):
            calls = [0]

            def rep(a, b):
                calls[0] += 1
                if calls[0] == k:
                    raise OSError(f"forced: restore write {k} refused")
                return os.replace(a, b)
            return types.SimpleNamespace(**{n_: getattr(os, n_) for n_ in ("environ", "path", "getpid")}, replace=rep)

        def disk_states(t):
            return ["original" if sha_f(t / r) == FILES[r][0] else "patched" if sha_f(t / r) == FILES[r][1] else "other" for r in CHAIN]
        for k in (1, 2, 3):
            Tk = tree_copy(f"t5b{k}")
            os.environ["NB6Q_FAKE_TREE"] = str(Tk)
            ns = make_ns(f"t5b{k}")
            PQk = ns["nb6q_module"]()
            PQk._py_compile = lambda py, files: {"rc": 1, "err": "forced by the test"}
            for r in FILES:
                pc = (Tk / r).parent / "__pycache__"
                pc.mkdir(exist_ok=True)
                (pc / (Path(r).stem + ".cpython-312.pyc")).write_bytes(b"stale")
            ns["os"] = fail_on(k)
            run_q(ns, ns["merlin_install"])
            ns["os"] = os
            sk = run_q(ns, ns["merlin_fn_server"], f"m5b{k}", "fp8", 8000, 1)
            Qk = ns["NB6Q"]
            check(f"S2 restore write {k} of 3 fails (order gpu_worker -> model_runner -> pp_utils, stop at the failure): "
                  f"FAILED_REVERT_INCOMPLETE, disk = {WANT[k]} = consistent (recorded), pyc of all three dropped, no NB4_PPDEFER, no raise",
                  Qk["state"] == "FAILED_REVERT_INCOMPLETE" and Qk["consistent"] is True and Qk["revert"]["consistent"] is True
                  and Qk["revert"]["order"] == CHAIN[::-1] and disk_states(Tk) == WANT[k]
                  and Qk["revert"]["files"][CHAIN[::-1][k - 1]]["action"] == "error"
                  and len(Qk["revert"]["files"]) == k
                  and not [p for r in FILES for p in ((Tk / r).parent / "__pycache__").glob(Path(r).stem + ".*.pyc")]
                  and "NB4_PPDEFER" not in sk.env and Qk["applied"] is False,
                  json.dumps({"state": Qk.get("state"), "disk": disk_states(Tk), "revert": Qk.get("revert")}, default=str)[:900])

        # ---- negative control: the OLD (apply) order with a failure on the 2nd write -> pp_utils original + model_runner
        # patched = INCONSISTENT (patched model_runner would call the missing pp_handler.flush_deferred_receive)
        Tn = tree_copy("t5neg")
        snap_n = {r: (Tn / r).read_bytes() for r in FILES}
        ns = make_ns("t5neg")
        PQn = ns["nb6q_module"]()
        PQn.apply_ppack_patch(Tn, sys.executable)
        ns["os"] = fail_on(2)
        rn = ns["nb6q_restore"](Tn, snap_n, order=tuple(CHAIN))
        ns["os"] = os
        check("S2 negative control: the old pp_utils -> model_runner -> gpu_worker restore order, 2nd write fails -> "
              "pp_utils original + model_runner patched = INCONSISTENT (detected: consistent False)",
              disk_states(Tn) == ["original", "patched", "patched"] and rn["consistent"] is False and rn["ok"] is False,
              json.dumps({"disk": disk_states(Tn), "rn": rn}, default=str)[:600])
        check("S2 consistency rule: all original / all patched / prefix-patched are consistent; a patched dependent over an "
              "original dependency, or a foreign file, is not",
              ns["nb6q_consistency"](tree_copy("t5c0"))["consistent"] is True and ns["nb6q_consistency"](Tn)["consistent"] is False
              and ns["nb6q_consistency"](tree_copy("t5c1", foreign=True))["consistent"] is False)
        os.environ.pop("NB6Q_FAKE_TREE", None)

        # ---- T6 embedded nb4_ppack sha mismatch -> FAILED_NOT_TOUCHED
        T6 = tree_copy("t6")
        os.environ["NB6Q_FAKE_TREE"] = str(T6)
        ns = make_ns("t6")
        ns["NB6Q_PPACK_SHA"] = "0" * 64
        run_q(ns, ns["merlin_install"])
        s6 = run_q(ns, ns["merlin_fn_server"], "m6", "fp8", 8000, 1)
        check("S2 embedded nb4_ppack sha mismatch -> FAILED_NOT_TOUCHED, files untouched, no NB4_PPDEFER, no exception",
              ns["NB6Q"]["state"] == "FAILED_NOT_TOUCHED" and "EMBED_SHA_MISMATCH" in ns["NB6Q"]["error"] and "NB4_PPDEFER" not in s6.env
              and all(sha_f(T6 / r) == FILES[r][0] for r in FILES))

        # ---- T7 the block's load failure (nb6q_lib sha mismatch) -> NB6 behaviour, a stub read-back, no wrapping
        bad = re.sub(r'NB6Q_LIB_SHA = "[0-9a-f]{64}"', 'NB6Q_LIB_SHA = "' + "0" * 64 + '"', BLOCK)
        ns = make_ns("t7", block=bad)
        tr7 = {}
        ns["nb6q_readback"](tr7, None, "after_warmup")
        s7 = ns["merlin_fn_server"]("m7", "fp8", 8000, 1)
        check("S2 load failure: NB6Q FAILED_LOAD, merlin_install / merlin_fn_server stay NB6's, read-back stub records NOT_APPLIED, "
              "loud print, no exception", ns["NB6Q"]["state"] == "FAILED_LOAD" and ns["merlin_install"].__doc__ is None
              and tr7["nb6q_readback"]["after_warmup"]["label"] == "NOT_APPLIED" and "LOAD FAILED" in ns["_stdout"].getvalue()
              and "NB4_PPDEFER" not in s7.env)

        # ---- T8 NB6's install fails -> raises exactly as NB6, no apply attempted, no env
        def _fail():
            raise RuntimeError("Merlin offline install failed (logs/install_merlin.log)")
        ns = make_ns("t8", install=_fail)
        try:
            ns["merlin_install"]()
            raised = None
        except RuntimeError as e:
            raised = str(e)
        check("S2 NB6 install failure: the same exception as NB6 (unchanged behaviour), patch not attempted, no env",
              raised and "install failed" in raised and ns["NB6Q"]["attempted"] is False and ns["NB6Q"]["env"] == {})

        # ---- T9 the real (non-FAKE) path: MERLIN_TREE + the venv python (POSIX: a venv/bin/python symlink to this python)
        os.environ.pop("NB6Q_FAKE_TREE", None)
        if os.name == "posix":
            T9 = tree_copy("t9")
            venv = WORK / "venv"
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            os.symlink(sys.executable, venv / "bin" / "python")
            os.environ["NB6Q_FAKE_TREE"] = str(WORK / "must_not_be_read")
            ns = make_ns("t9", fake=False, install=lambda: {"kv": {"MERLIN_TREE": str(T9), "MERLIN_VENV": str(venv)}})
            run_q(ns, ns["merlin_install"])
            s9 = run_q(ns, ns["merlin_fn_server"], "m9", "fp8", 8000, 1)
            check("S2 real path (FAKE None): MERLIN_TREE patched with the venv python's py_compile, NB6Q_FAKE_TREE ignored, env set",
                  ns["NB6Q"]["state"] == "APPLIED" and ns["NB6Q"]["tree"] == str(T9) and ns["NB6Q"]["apply"]["py_compile"]["rc"] == 0
                  and all(sha_f(T9 / r) == FILES[r][1] for r in FILES) and s9.env.get("NB4_PPDEFER") == "1", ns["NB6Q"].get("error"))
            os.environ.pop("NB6Q_FAKE_TREE", None)
        else:
            SKIPPED.append("S2 real path (venv/bin/python symlink): POSIX only (the CPU real kit runs it)")
            print("SKIP " + SKIPPED[-1], flush=True)
    except Exception:
        check("S2 no unexpected exception", False, traceback.format_exc()[-2000:])
    finally:
        os.environ.pop("NB6Q_FAKE_TREE", None)

# ============================================================================ S3 NB6's unit tests on NB6Q + the push folder
if "S3" in SEC:
    env = dict(os.environ, NB6T_NB=str(NB), NB6T_PUSH=str(PUSH), PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, str(NB6UNITS)], capture_output=True, text=True, timeout=3000, env=env, cwd=str(NB6UNITS.parent),
                       encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    fails = [ln[5:].split("  [")[0] for ln in out.splitlines() if ln.startswith("FAIL ")]
    m = re.search(r"(\d+) / (\d+) PASS", out)
    check(f"S3 nb6_unit_tests.py on NB6Q: every check PASS except exactly the NB6-only push-id check ({m.group(0) if m else '?'})",
          m and int(m.group(2)) >= 86 and int(m.group(1)) == int(m.group(2)) - 1 and fails == [NB6_ID_CHECK], (fails, out[-1500:]))
if "S3" in SEC:
    _mq = json.loads((PUSH / "kernel-metadata.json").read_text(encoding="utf-8"))
    _m6 = json.loads((PUSH6 / "kernel-metadata.json").read_text(encoding="utf-8"))
    check("S3b NB6Q push folder: id / title / code_file NB6Q, the notebook in it = this build, every other field = NB6's (sources, "
          "image, GPU, internet off, private)", _mq["id"] == "foysalemonshanto/q27-xhigh-e25-nb6q-submit" and _mq["title"] == "Q27 Xhigh E25 NB6Q Submit"
          and _mq["code_file"] == "q27-xhigh-e25-nb6q-submit.ipynb" and (PUSH / _mq["code_file"]).read_bytes() == raw
          and {k: v for k, v in _mq.items() if k not in ("id", "title", "code_file")} == {k: v for k, v in _m6.items() if k not in ("id", "title", "code_file")})

# ============================================================================ S4 nb4_ppack_tests.py on the real source (reused)
if "S4" in SEC:
    r = subprocess.run([sys.executable, str(PPTESTS), "--merlin-src", str(MSRC)], capture_output=True, text=True, timeout=1800,
                       cwd=str(PPTESTS.parent), env=dict(os.environ, PYTHONIOENCODING="utf-8"), encoding="utf-8", errors="replace")
    out = r.stdout + r.stderr
    js = re.search(r'\{\s*"checks": (\d+),\s*"pass": (\d+),\s*"fail": (\d+)', out)
    check("S4 nb4_ppack_tests.py on the real Merlin source copy: ALL PASS (>= 247 checks, 0 fail)",
          r.returncode == 0 and "ALL PASS" in out and js and int(js.group(1)) >= 247 and js.group(3) == "0", out[-1500:])

shutil.rmtree(WORK, ignore_errors=True)
bad = [n for n, ok in RESULTS if not ok]
print("\nsections " + ",".join(sorted(SEC)), flush=True)
print(f"{len(RESULTS) - len(bad)} / {len(RESULTS)} PASS ({len(SKIPPED)} skipped)" + (f" | FAIL: {bad}" if bad else ""), flush=True)
print("@@NB6Q_UNITS_OK@@" if not bad else "@@NB6Q_UNITS_FAIL@@", flush=True)
sys.exit(1 if bad else 0)
