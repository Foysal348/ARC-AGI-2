"""E25 NB6Q SUBMIT (lead 2026-11-01): the proven NB6 SUBMIT notebook (sha 87afcc84, GPU-committed) + exactly ONE change, the
Q-defer patch (deferred PP receive, nb4_ppack.py rev 2), the winner of the NB4S GPU test (notebook fa9f35ca, run v1
2026-11-01: +21.1 % at 8k@8, +19.8 % at 40k@8, oracle EXACT vs B, stage-0 stall 19.81 -> 0.035 ms, read-back ENGAGED with
exec share 0.965 and fallback share 0).

A thin wrapper, like build_e26_k100_quality.py over NB6: build_e25_nb6.py is imported (it re-derives the Kaggle-run NB5 QUALITY
sha 9c71ba89 first); its cells are rebuilt and asserted byte for byte against the NB6 SUBMIT notebook on disk (87afcc84);
build_e25_nb6.py and the NB6 notebook are never written. Only the marked "NB6Q" edits below are applied:

  Q1 (engine cell, appended at its end = after every NB6 definition, before CELL 8 runs anything)
     a short loader + ONE packed literal (json -> lzma -> base85; the notebook stays < 400 kB, the NB5 nb5_lib pattern) holding
     nb4_ppack.py rev 2 verbatim (sha256 = the module the GPU-run NB4S notebook embeds, asserted here) and nb6q_lib.py
     verbatim (the NB6Q functions, sha256 checked, exec'd into the notebook namespace; a load failure = NB6 unchanged):
       nb6q_apply      once, right after the first successful merlin_install() (CELL 8 step A0: before the sha gate, #60490
                       and any boot; commit AND rerun): snapshot of pp_utils.py / model_runner.py / gpu_worker.py, pre shas,
                       apply_ppack_patch (strict pre-sha, post-sha, marker, py_compile with the Merlin venv), post shas
                       verified again; ANY failure -> the three files restored byte for byte from the snapshot (pyc dropped),
                       a loud log line, the run goes on WITHOUT the patch and WITHOUT NB4_PPDEFER
       merlin_install  wrapper: NB6's install, then nb6q_apply (once)
       merlin_fn_server wrapper: NB6's launch plan, then NB4_PPDEFER / NB4_PPACK removed from the server env and
                       NB4_PPDEFER=1 set only when the patch is verified applied (the 4 workers inherit the server env);
                       e21_relaunch (NB6's watchdog relaunch, unchanged) calls merlin_fn_server, so a relaunch gets the same env
       nb6q_readback   nb4_ppack.readback_label on the launch log's NB4_PPQ lines (ENGAGED / ENGAGED_FALLBACK / NOT_ENGAGED /
                       NOT_INTEGRATED; NOT_APPLIED without the patch): recorded in the boot try, NB6Q["readbacks"], the report
                       (nb5_report.json "nb6q_ppdefer") and the log; never raises, never stops a run (commit or rerun)
       FAKE-only hook  NB6Q_FAKE_TREE (the CPU kits' fake server has no Merlin tree): read only when E11_HARNESS fake mode is on
  Q2 (engine, nb5_boot_try) one inserted line after the warm-up: nb6q_readback(trec, s, "after_warmup")
  Q3 (engine, nb5_boot_try) one inserted line after the boot's server stopped: nb6q_readback(trec, srv, "final")
  Q4 (markdown) the NB6Q header line + one bullet

Nothing else changes: CELL 0 (E18 R2 = E15B), CELL 1 (profiles RERUN_PROD 717 min / NB6COMMIT 50 min, levers), the Duck
path, the task order, the runner, the writer, the server ladder / readiness / fast kill, no sha gate in the rerun: NB6 bytes.

usage: python build_e25_nb6q.py  ->  ../Q27-XHIGH-E25_NB6Q_SUBMIT.ipynb + nb6q_code_export.py + nb6q_build_facts.json +
       the GPU push folder D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb6q-submit__foysalemonshanto/ (nothing is pushed)
"""
import ast
import base64
import hashlib
import json
import lzma
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_e25_nb6 as N6   # noqa: E402  (module import only: build_e25_nb6.build() is never called here)

B = N6.B
NB6_SHA = "87afcc848cd60eee290509b6a1171f04a754015130d69363d9046d265b2c1e2b"       # Q27-XHIGH-E25_NB6_SUBMIT.ipynb (GPU commit)
NB6_NB = B.NBDIR / "Q27-XHIGH-E25_NB6_SUBMIT.ipynb"
NB4S_NB = B.NBDIR / "Q27-XHIGH-E25_NB4S_STALL.ipynb"
NB4S_SHA = "fa9f35ca43ec0c8108327a840a9f6c28d2891e1313c131b3ec56b379048f964a"     # NB4S STALL rev 3 (ran on GPU 2026-11-01)
PPACK = HERE / "nb4_ppack.py"
OUT = B.NBDIR / "Q27-XHIGH-E25_NB6Q_SUBMIT.ipynb"
EXPORT = HERE / "nb6q_code_export.py"
FACTS = HERE / "nb6q_build_facts.json"
PUSH_ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
PUSH6 = PUSH_ROOT / "q27-xhigh-e25-nb6-submit__foysalemonshanto"
SLUG = "q27-xhigh-e25-nb6q-submit"
PUSH = PUSH_ROOT / f"{SLUG}__foysalemonshanto"
PUSHED: set = set()   # sha256 of every NB6Q notebook that ran on Kaggle (never overwrite one: rebuild -> *_REBUILD)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def once(text, old, new, label):
    assert text.count(old) == 1, (label, text.count(old))
    return text.replace(old, new)


# ------------------------------------------------------------------ provenance (fail the build on any drift)
NB6_CELLS = N6.cells(N6.DEFAULT_PROFILE)
assert sha(N6.notebook_bytes(NB6_CELLS)[0]) == NB6_SHA, "build_e25_nb6.py no longer reproduces NB6 SUBMIT 87afcc84"
assert sha(NB6_NB.read_bytes()) == NB6_SHA, "NB6 SUBMIT notebook changed on disk"
assert sha(NB4S_NB.read_bytes()) == NB4S_SHA, "NB4S STALL notebook changed on disk (not the GPU-run fa9f35ca)"
PP_SRC = PPACK.read_bytes().decode("utf-8").replace("\r\n", "\n")          # LF (Windows checkouts may carry CRLF)
PP_SHA = sha(PP_SRC.encode("utf-8"))
_nb4s_code = "\n".join("".join(c["source"]) for c in json.loads(NB4S_NB.read_bytes().decode("utf-8"))["cells"] if c["cell_type"] == "code")
_m = re.findall(r'PPACK_SHA = "([0-9a-f]{64})"', _nb4s_code)
assert _m == [PP_SHA], ("nb4_ppack.py != the module embedded in the GPU-run NB4S notebook", _m, PP_SHA)
compile(PP_SRC, "nb4_ppack.py", "exec")
_ppns = {"__name__": "nb4_ppack_build"}
exec(compile(PP_SRC, "nb4_ppack.py", "exec"), _ppns)
PP_FILES = {rel: {"pre": p[0], "post": p[1]} for rel, p in _ppns["PATCHES"].items()}
assert sorted(PP_FILES) == sorted(["vllm/v1/worker/gpu/pp_utils.py", "vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu_worker.py"])
assert _ppns["server_env"]("Q-defer") == {"NB4_PPDEFER": "1"}
assert {"apply_ppack_patch", "verify_ppack_patch", "revert_ppack_patch", "readback_label", "parse_log"} <= {
    n.name for n in ast.parse(PP_SRC).body if isinstance(n, ast.FunctionDef)}
LIB = HERE / "nb6q_lib.py"
LIB_SRC = LIB.read_bytes().decode("utf-8").replace("\r\n", "\n")
LIB_SHA = sha(LIB_SRC.encode("utf-8"))
compile(LIB_SRC, "nb6q_lib.py", "exec")
_lib_tree = ast.parse(LIB_SRC)
_lib_files = next(ast.literal_eval(n.value) for n in _lib_tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "NB6Q_FILES")
assert _lib_files == PP_FILES, "nb6q_lib.NB6Q_FILES != nb4_ppack.PATCHES (pre / post shas)"
assert {"nb6q_module", "nb6q_tree", "nb6q_consistency", "nb6q_clear_pyc", "nb6q_restore", "nb6q_apply", "merlin_install",
        "merlin_fn_server", "nb6q_readback"} == {n.name for n in _lib_tree.body if isinstance(n, ast.FunctionDef)}
_chain = next(ast.literal_eval(n.value) for n in _lib_tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "NB6Q_CHAIN")
assert list(_chain) == list(PP_FILES) == list(_ppns["PATCHES"]), "NB6Q_CHAIN != nb4_ppack apply order (pp_utils, model_runner, gpu_worker)"
# one packed literal (the commit notebook stays < 400 kB): {"nb4_ppack.py": verbatim rev 2, "nb6q_lib.py": verbatim}
PACK = json.dumps({"nb4_ppack.py": PP_SRC, "nb6q_lib.py": LIB_SRC}, sort_keys=True)
PP_B85 = base64.b85encode(lzma.compress(PACK.encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
assert json.loads(lzma.decompress(base64.b85decode(PP_B85)).decode("utf-8")) == {"nb4_ppack.py": PP_SRC, "nb6q_lib.py": LIB_SRC}
assert '"' not in PP_B85 and "\\" not in PP_B85

# ------------------------------------------------------------------ Q1: the NB6Q loader (appended to the engine cell)
Q1 = r"""


# ==================================================================== NB6Q: the Q-defer patch (deferred PP receive)
# NB6Q Q1 (lead 2026-11-01): the ONE change over NB6 SUBMIT (sha 87afcc84) = nb4_ppack.py rev 2 Q-defer, the winner of the
# NB4S GPU test (fa9f35ca: +21.1 % at 8k@8, +19.8 % at 40k@8, oracle EXACT, stage-0 stall 19.81 -> 0.035 ms, read-back
# ENGAGED). nb4_ppack.py and nb6q_lib.py (the NB6Q functions: apply once right after merlin_install with pre / post shas,
# byte-for-byte restore + loud log + no NB4_PPDEFER on any failure, NB4_PPDEFER=1 in the server env only when verified,
# record-only read-back) are packed verbatim below (sha256 checked) and exec'd into this namespace. A load failure leaves
# NB6 unchanged (no patch, no env) and is logged; it never stops the run.
NB6Q_PPACK_SHA = "__PPSHA__"   # = the module NB4S fa9f35ca ran
NB6Q_LIB_SHA = "__LIBSHA__"
try:
    NB6Q_SRC = json.loads(__import__("lzma").decompress(__import__("base64").b85decode(
        "__PPB85__")).decode("utf-8"))
    if hashlib.sha256(NB6Q_SRC["nb6q_lib.py"].encode("utf-8")).hexdigest() != NB6Q_LIB_SHA:
        raise RuntimeError("NB6Q_EMBED_SHA_MISMATCH: nb6q_lib.py")
    exec(compile(NB6Q_SRC["nb6q_lib.py"], "nb6q_lib.py", "exec"), globals())
except Exception as _nb6q_e:
    NB6Q = {"state": "FAILED_LOAD", "applied": False, "attempted": True, "error": repr(_nb6q_e)[:600], "readbacks": []}
    print("!!!!!!!!!! NB6Q LOAD FAILED: " + NB6Q["error"] + " -> NB6 behaviour (no patch, no NB4_PPDEFER) !!!!!!!!!!", flush=True)

    def nb6q_readback(trec, s, stage):
        trec.setdefault("nb6q_readback", {})[stage] = {"label": "NOT_APPLIED", "state": NB6Q["state"], "stage": stage}
"""
assert sorted(set(re.findall(r"__[A-Z0-9]+__", Q1))) == ["__LIBSHA__", "__PPB85__", "__PPSHA__"], "Q1 tokens"
Q1 = Q1.replace("__PPSHA__", PP_SHA).replace("__LIBSHA__", LIB_SHA).replace("__PPB85__", PP_B85)

# ------------------------------------------------------------------ Q2 / Q3: two inserted lines in nb5_boot_try
_Q2_OLD = '        trec["counters_after_warmup"] = e21_counters()\n'
_Q2_NEW = _Q2_OLD + '        nb6q_readback(trec, s, "after_warmup")   # NB6Q Q2: Q-defer read-back after the warm-up (record only)\n'
_Q3_OLD = '    trec["log_text_for_ladder"] = srv.log() or ""\n'
_Q3_NEW = _Q3_OLD + '    nb6q_readback(trec, srv, "final")           # NB6Q Q3: Q-defer read-back, whole log after the server stopped (record only)\n'
# ------------------------------------------------------------------ Q4: markdown
_Q4_OLD = "# E25 NB6 SUBMIT · FN-Duck production submission · NB6COMMIT\n"
_Q4_NEW = ("# E25 NB6Q SUBMIT · FN-Duck production submission + Q-defer · NB6COMMIT\n\n"
           "- **NB6Q = NB6 SUBMIT (sha 87afcc84) + ONE change, the Q-defer patch** (nb4_ppack rev 2, deferred PP receive). NB4S GPU\n"
           "  test: +21.1 % at 8k@8, +19.8 % at 40k@8, oracle EXACT, stage-0 stall 19.8 -> 0.035 ms, read-back ENGAGED. Applied right\n"
           "  after the Merlin install (pre / post shas of pp_utils.py, model_runner.py, gpu_worker.py); `NB4_PPDEFER=1` in the server\n"
           "  env only when verified; a failed apply is restored byte for byte and the run continues without it. Read-back recorded only.\n")


def cells():
    cl = [list(c) for c in NB6_CELLS]
    assert cl[0][0] == "markdown" and "CELL 5 — NB5 engine" in cl[5][1] and "CELL 8 — NB5: boots and phases" in cl[7][1]
    cl[0][1] = once(cl[0][1], _Q4_OLD, _Q4_NEW, "NB6Q Q4 markdown")
    t = cl[5][1]
    t = once(t, _Q2_OLD, _Q2_NEW, "NB6Q Q2")
    t = once(t, _Q3_OLD, _Q3_NEW, "NB6Q Q3")
    cl[5][1] = t.rstrip("\n") + Q1
    return [tuple(c) for c in cl]


def kernel_metadata():
    m = json.loads((PUSH6 / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert m["id"] == "foysalemonshanto/q27-xhigh-e25-nb6-submit" and m["enable_gpu"] is True and m["machine_shape"] == "NvidiaL4"
    m.update(id=f"foysalemonshanto/{SLUG}", title="Q27 Xhigh E25 NB6Q Submit", code_file=f"{SLUG}.ipynb")
    return m


def diff_text(cl):
    import difflib
    a = "".join(f"# ---- cell {i} ({k})\n" + B.norm(t).rstrip("\n") + "\n" for i, (k, t) in enumerate(NB6_CELLS))
    b = "".join(f"# ---- cell {i} ({k})\n" + B.norm(t).rstrip("\n") + "\n" for i, (k, t) in enumerate(cl))
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), "NB6_SUBMIT_87afcc84 (cells)",
                                        "NB6Q_SUBMIT (cells)", n=3))


def build():
    cl = cells()
    B.undefined_names_check(cl + [("code", LIB_SRC)])     # nb6q_lib.py is exec'd into the notebook namespace: checked in context
    data, n = N6.notebook_bytes(cl)
    assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
    assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))
    diff_cells = [i for i, (x, y) in enumerate(zip(NB6_CELLS, cl)) if x != y]
    assert diff_cells == [0, 5], diff_cells
    out = OUT
    if out.exists() and sha(out.read_bytes()) in PUSHED:
        out = out.with_name(out.stem + "_REBUILD.ipynb")
        assert not (out.exists() and sha(out.read_bytes()) in PUSHED), out
    out.write_bytes(data)
    EXPORT.write_bytes("\n\n".join(B.norm(t) for k, t in cl if k == "code").encode("utf-8"))
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / f"{SLUG}.ipynb").write_bytes(data)
    meta = kernel_metadata()
    (PUSH / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    dt = diff_text(cl)
    facts = {"notebook": str(out), "sha256": sha(data), "bytes": len(data), "cells": n, "nb6_sha256": NB6_SHA,
             "nb6_bytes": NB6_NB.stat().st_size, "nb4s_sha256": NB4S_SHA,
             "ppack": {"file": "nb4_ppack.py", "sha256": PP_SHA, "bytes_lf": len(PP_SRC.encode("utf-8")), "embedded": "packed with nb6q_lib.py: json -> lzma -> base85",
                       "b85_chars": len(PP_B85), "files": PP_FILES},
             "nb6q_lib": {"file": "nb6q_lib.py", "sha256": LIB_SHA, "bytes_lf": len(LIB_SRC.encode("utf-8"))},
             "changed_cells_vs_nb6": diff_cells, "edits": ["Q1 engine-cell NB6Q block", "Q2 nb5_boot_try after_warmup read-back",
                                                           "Q3 nb5_boot_try final read-back", "Q4 markdown"],
             "q1_bytes": len(Q1.encode("utf-8")), "diff_lines": {"added": sum(1 for l in dt.splitlines() if l.startswith("+") and not l.startswith("+++")),
                                                                 "removed": sum(1 for l in dt.splitlines() if l.startswith("-") and not l.startswith("---"))},
             "export": str(EXPORT), "push": str(PUSH), "kernel_metadata": meta}
    FACTS.write_bytes((json.dumps(facts, indent=1) + "\n").encode("utf-8"))
    assert sha(NB6_NB.read_bytes()) == NB6_SHA, "NB6 notebook changed during the build"
    print("wrote", out.name, "sha256", sha(data), "|", len(data), "bytes |", n, "cells | nb4_ppack", PP_SHA[:12], "| diff +%d -%d"
          % (facts["diff_lines"]["added"], facts["diff_lines"]["removed"]))
    return out


if __name__ == "__main__":
    build()
