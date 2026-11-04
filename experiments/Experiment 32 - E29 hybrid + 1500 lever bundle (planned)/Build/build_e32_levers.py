"""E32 LEVERS (2026-11-02): the E29_DFLASH_D3S production notebook + the 1500 lever bundle, ONE 6-h EVAL120 notebook (dual
mode: commit = EVAL120 scored, the competition rerun = every hidden task, as E29).

A thin wrapper over Experiment 29's Build/build_e29_final.py (imported; its build() / write_dataset() / build_direct() are
never called, nothing in Experiment 29 or its push / payload folders is written):
  0. the E29 builder's cells for E29_DFLASH_D3S, assembled exactly as its build() does, must equal the E29 production notebook
     ON DISK byte for byte (so a later E29 rebuild = one rebuild here); the DIRECT / scheduler / nbx files come from the E29
     build outputs (Build/e29_direct.py, e29d_sched_patch.py, e29_nbx.py; sha of the sched file = the E29 builder's
     SCHED_PIN; each = its copy in the E29 payload folder = what the notebook pins)
  1. the marked once-only E32 edits below on those cells (each `once` asserts exactly one match):
     X1  (markdown) the E32 header
     X2  (CELL 1)   the lever switch lines E32_SEQS / E32_SPEC / E32_LP / E32_SPC / E32_IDX / E32_G1 / E32_G2 + E32_SPEED
     X3  (CELL 1)   E29_SMOKE budget 70 -> 100 min (boot + the speed pass + one lever retry + the same 40-min window)
     X4  (engine, the packed E29 block) the E32 block: E32_FILES (payload name + sha), E32_RT, e32_call / e32_setup /
                    e32_report; e29_env + G1 / G2, e29_extra + the LP / SPC / IDX argv (before the DFlash tail), e29_drop('df')
                    and e29_mtp_boot hooks (the bundle leaves with DFlash)
     X5  (engine)   nb5_capacity: the D try's E32 seqs count as the launched top rung (workers = seqs, as E28's 24)
     X6  (engine)   nb5_boot_try: the bundle read-back after the DFlash read-back; the speed pass after the F2 probe
     X7  (engine)   nb5_boot: E32 seqs on the D tries, the try name tag, the drop rule (one E32 lever, the same rung) after
                    E29's QSA rule and before E29's OOM / lever rule
     X8  (engine)   e21_relaunch: an in-run death with E32 levers on drops one lever first (else E29's in-run plan)
     X9  (CELL 8)   e32_setup() after e29_setup()
     X10 (CELL 9)   E28_OUT["e32"] = e32_report(the E29 run summary)
  2. e32_lib.py = a verbatim, sha-pinned cut of E25 nbd1_lib.py (NB-D1's argv / env / captures / read-back helpers) + Build/
     e32_core.py; e32_nbx.py = Build/e32_nbx.py; both go to the NEW payload dataset folder (name + sha256 in the notebook).

  python build_e32_levers.py  ->  ../Experiment Notebooks/Q27-XHIGH-E32_LEVERS.ipynb (+ _SMOKE: ONE line differs, the profile
  line E29SMOKE), Build/e32_levers_code_export.py, Build/e32_build_facts.json, Build/e29_vs_e32_export.diff, the push folders
  D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e32-levers[-smoke|-submit]__foysalemonshanto/ and the payload dataset folder
  D:/ARC AGI 2/_scratch/push_e25/ds_q27-e32-levers-payload/. Nothing is pushed.
"""
import difflib
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
E29_BUILD = Path(r"D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build")
E29_NB_DIR = E29_BUILD.parent / "Experiment Notebooks"
E25_BUILD = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
PUSH_ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
OWNER = "foysalemonshanto"
BASE_VARIANT = "E29_DFLASH_D3S"
NB_STEM = "Q27-XHIGH-E32_LEVERS"
SLUG = "q27-xhigh-e32-levers"
DS_SLUG = "q27-e32-levers-payload"
DS_ID = f"{OWNER}/{DS_SLUG}"
DS_DIR = PUSH_ROOT / f"ds_{DS_SLUG}"
PUSHED: set = set()          # sha256 of every E32 notebook that ran on Kaggle (never overwrite one: rebuild -> *_REBUILD)

# the E29 production files this build must never change (task spec, 2026-11-02); verified before and after the build
E29_PROTECTED = {
    "notebook": (E29_NB_DIR / "Q27-XHIGH-E29_DFLASH_D3S.ipynb", "9a0978474fb2686209ac922c212b2c457cf7d63d7321ba4bbce3118f03bc0885"),
    "smoke": (E29_NB_DIR / "Q27-XHIGH-E29_DFLASH_D3S_SMOKE.ipynb", "19928629ac35276fa4652757bfbb4a0eddc66265e88c017a3b3cac5e43701b29"),
}
E29_PROTECTED_DIRS = [PUSH_ROOT / "q27-xhigh-e29-dflash-d3s__foysalemonshanto", PUSH_ROOT / "q27-xhigh-e29-dflash-d3s-submit__foysalemonshanto",
                      PUSH_ROOT / "ds_q27-e29-final-payload"]


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def lf(p):
    b = Path(p).read_bytes()
    assert b"\r" not in b, f"CR bytes in {p}"
    return b.decode("utf-8")


def tree_shas(d):
    return {str(f.relative_to(d)).replace("\\", "/"): sha(f.read_bytes()) for f in sorted(Path(d).rglob("*")) if f.is_file()}


def protected_state():
    st = {k: sha(p.read_bytes()) for k, (p, _) in E29_PROTECTED.items()}
    st.update({d.name: tree_shas(d) for d in E29_PROTECTED_DIRS})
    return st


PROTECTED_BEFORE = protected_state()
for _k, (_p, _s) in E29_PROTECTED.items():
    assert PROTECTED_BEFORE[_k] == _s, ("the E29 production file changed before this build", _k, PROTECTED_BEFORE[_k])

_argv = sys.argv
sys.argv = [_argv[0]]                         # the E29 / E28 builders read sys.argv: import them without our args
sys.path.insert(0, str(E29_BUILD))
import build_e29_final as E29B   # noqa: E402  (module import only: its build() / write_dataset() / build_direct() never run)
sys.argv = _argv
E28B, E29L = E29B.E28B, E29B.E29L
once = E29B.once


# ------------------------------------------------------------------ 0. the E29 base, read-only
def e29_direct_from_build():
    """The DIRECT / scheduler / nbx files = the E29 build outputs (never regenerated or written here): the sched file sha =
    the E29 builder's SCHED_PIN; each file = its copy in the E29 payload folder (what the notebook's E29D_FILES pins)."""
    if E29B.DIRECT_TEXTS:
        return E29B.DIRECT_TEXTS
    t = {n: lf(E29_BUILD / n) for n in ("e29_direct.py", "e29d_sched_patch.py", "e29_nbx.py")}
    assert sha(t["e29d_sched_patch.py"]) == E29B.SCHED_PIN, ("Build/e29d_sched_patch.py != the E29 builder's SCHED_PIN", sha(t["e29d_sched_patch.py"]))
    sums = lf(E29B.DS_DIR / "SHA256SUMS")
    for n, x in t.items():
        assert sha((E29B.DS_DIR / n).read_bytes()) == sha(x), ("the E29 payload folder copy differs from the E29 build output", n)
        assert f"{sha(x)}  {n}\n" in sums, ("the E29 payload SHA256SUMS does not list", n)
    E29B.DIRECT_TEXTS.update(t)
    E29B.DIRECT_FILES.update(direct=("e29_direct.py", sha(t["e29_direct.py"])), sched=("e29d_sched_patch.py", sha(t["e29d_sched_patch.py"])),
                             nbx=("e29_nbx.py", sha(t["e29_nbx.py"])))
    return t


E29B.build_direct = e29_direct_from_build        # nothing in this process may regenerate / write the E29 files
BASE_CFG = E29L.check_cfg(E29B.VARIANTS[BASE_VARIANT])
e29_direct_from_build()


def assemble(cl):
    """= E29's build(): pyflakes over the readable cells, e21_cell7 packed (E28), the E29 engine block + the E21 donor block
    packed, the compact JSON, e21_cell6 packed when >= 395,000 B. -> (bytes, notebook cells, pack records)."""
    E29B.B.undefined_names_check(cl)
    ncl, pack7 = E28B.notebook_cells(cl)
    assert E29B.unpack_engine(ncl[5][1]) == ncl[5][1]
    c5, packe = E29B.pack_engine(ncl[5][1])
    ncl[5] = ("code", c5)
    data, n, fmt = E28B.notebook_bytes(ncl)
    pack6 = None
    if len(data) >= 395_000:
        c6, pack6 = E29B.pack_cell6(ncl[6][1])
        ncl[6] = ("code", c6)
        data, n, fmt = E28B.notebook_bytes(ncl)
    assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
    return data, ncl, {"e21_cell7": pack7, "engine": packe, "e21_cell6": pack6, "cells": n, "json_format": fmt}


E29_CELLS = E29B.cells_for(BASE_VARIANT, BASE_CFG)
E29_DATA, E29_NCL, _ = assemble(list(E29_CELLS))
E29_ON_DISK = (E29_NB_DIR / "Q27-XHIGH-E29_DFLASH_D3S.ipynb").read_bytes()
assert E29_DATA == E29_ON_DISK, ("the E29 builder no longer reproduces the E29 production notebook on disk", sha(E29_DATA), sha(E29_ON_DISK))
E29_SHA = sha(E29_DATA)

# ------------------------------------------------------------------ 2. e32_lib.py / e32_nbx.py (the payload)
NBD1_SRC = lf(E25_BUILD / "nbd1_lib.py")
NBD1_NAMES = ("SEQS", "GDN_ENV", "IDX_ARGS", "LP_ARGS", "SPC_ARGS", "K4", "DYN_SCHEDULE", "DYN_FAIL_RX", "dflash_captures",
              "dense_schedule", "dyn_captures", "dyn_coverage", "_nondefault", "logged_int", "logged_bool", "logged_indexer_fp8",
              "idx_readback")
NBD1_CUT, _ = E29B._cut(NBD1_SRC, NBD1_NAMES, "nbd1_lib")
NBD1_CUT_SHA = "92fd3e8bd938bcae3951858fd8f86ed5190de8259ba7c609eef50fc54cfb18f0"   # set at the first build (2026-11-02)
CORE = lf(HERE / "e32_core.py")
_i = CORE.index("\nLIB_VERSION = ")
LIB_TEXT = (CORE[:_i] + "\n\n# ---- verbatim from E25 nbd1_lib.py (sha " + sha(NBD1_SRC)[:12] + ", NB-D1's per-lever argv / env / captures / "
            "read-back helpers; cut by name with ast, sha-pinned in build_e32_levers.py)\n" + NBD1_CUT
            + "\n\n# ---- E32 core (Build/e32_core.py)" + CORE[_i:])
NBX_TEXT = lf(HERE / "e32_nbx.py")
for _n, _t in (("e32_lib.py", LIB_TEXT), ("e32_nbx.py", NBX_TEXT)):
    assert "\r" not in _t
    compile(_t, _n, "exec")
_libns = {"__name__": "e32_lib_build"}
exec(compile(LIB_TEXT, "e32_lib.py", "exec"), _libns)
E32_FILES = {"lib": ["e32_lib.py", sha(LIB_TEXT)], "nbx": ["e32_nbx.py", sha(NBX_TEXT)]}

# build-time facts the levers rely on (each = the E29 / NB-D1 / nbdf value)
_N = {}
exec(compile(E29B.E29_NBDF, "e29_nbdf.py", "exec"), _N)
assert _libns["IDX_ARGS"] == E29L.IDX_ARGS and _libns["GDN_ENV"] == {"G1": E29L.GDN_ENV["gdn_fastpath"], "G2": E29L.GDN_ENV["gdn_align"]}
for _s in (24, 28, 32):
    for _k in (3, 4):
        assert _N["capture_sizes"](_s, _k) == _libns["dflash_captures"](_s, _k), (_s, _k)
    _dc = _libns["dense_schedule"](_libns["dyn_schedule"](_s), _s, 4)
    assert _dc == _libns["dense_schedule"](_libns["DYN_SCHEDULE"], _s, 4) and _dc[1:9] == [4] * 8 and set(_dc[9:]) == {3}, _s
    assert _libns["dyn_coverage"](_libns["dyn_captures"](_s), _s)["covers_all"]
assert _libns["dyn_schedule"](24) == _libns["DYN_SCHEDULE"]          # 24 seqs: NB-D1's kit-validated schedule exactly
assert all(t["kv_bytes"] != E29L.D_PIN0 for t in E29L.df_tries(True)), "a D try on the E28 top pin would force seqs 24"
assert E29L.df_tries(True)[0]["kv_bytes"] == E29L.D_START_PIN == 7665090560

# ------------------------------------------------------------------ 1. the E32 edits
X1_MD = ("# E32 LEVERS · E29_DFLASH_D3S (sha " + E29_SHA[:8] + ") + the 1500 lever bundle · E28EVAL120\n\n"
         "- Levers = the `E32_*` lines on top of CELL 1 (DFlash tries only; a bad value = off). A real failure of a D try drops ONE "
         "lever (same D rung); none left = E29's rules. Build/build_e32_levers.py + README.\n")
X2_SWITCH = (
    "E32_SEQS = 28   # E32 lever S28/S32: --max-num-seqs + captures of the DFlash tries (24 = off = E29; 28 | 32)\n"
    "E32_SPEC = 'DYN'   # E32 lever D4s/DYN: 'K3' = off (E29 D3s) | 'D4s' (K4) | 'DYN' (K4 + per-batch K [[1,8,4],[9,seqs,3]])\n"
    "E32_LP = True   # E32 lever LP: --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 (+ adaptive)\n"
    "E32_SPC = True   # E32 lever SPC: --enable-mamba-shared-prefix-checkpoint\n"
    "E32_IDX = True   # E32 lever IDX: --attention-config {\"indexer_kv_dtype\":\"fp8\"}\n"
    "E32_G1 = True   # E32 lever G1: VLLM_GDN_DECODE_FASTPATH=1\n"
    "E32_G2 = True   # E32 lever G2: VLLM_GDN_SHARED_ALIGN_INDEX=1\n"
    "E32_SPEED = {'on': True, 'windows': [['8k', 16], ['20k', 24], ['90k', 8], ['116k', 6]], 'cap_min': 15}   # E32: speed pass "
    "(record only, commit only)\n")
X2_ANCHOR = "E29_QSA = {'max_logits_mb': 128, 'granule_mb': 64}   # E29 rev 2c: QSA prefill workspace, DFlash tries (Merlin 256/256)\n"
X3_OLD = "'budget_min': 70}   # E29 rev 2e: the E29SMOKE profile"
X3_NEW = "'budget_min': 100}   # E32: 70 -> 100; E29 rev 2e: the E29SMOKE profile"
SMOKE_BUDGET, SMOKE_WINDOW, BOOT_MIN, SPEED_MIN, RETRY_MIN = 100, E29B.SMOKE_WINDOW_MIN, 24, 15, 7
# the window rule (room = HARD_STOP (budget - 5) - 3 - 1 - the boot end): the E29 smoke booted (one rung step) by 26.2 min; + the
# speed pass (<= 15) + one E32 lever retry (~6.5 min, a recompile boot) -> 48 min -> the full 40-min window is kept
assert SMOKE_BUDGET - 5 - 3 - 1 - SMOKE_WINDOW >= BOOT_MIN + SPEED_MIN + RETRY_MIN - 2
assert 415 - 5 - 3 - 1 - 360 >= BOOT_MIN + SPEED_MIN + RETRY_MIN - 2     # production E28EVAL120: the 6-h window kept the same way

X4_BLOCK = r'''


# ==================================================================== E32 (Build/build_e32_levers.py): the 1500 lever bundle
# CELL 1's E32_<LEVER> lines = the switches. e32_lib.py (pure: switch parse, argv / env, the DFlash plan, the drop rule, the
# read-back, the fingerprint) + e32_nbx.py (the hook bodies _e32x_*) come from the E32 payload dataset by name + sha. Every
# hook is e32_call(<name>, <default>, ...): no payload / not ON / an exception -> the default = E29_DFLASH_D3S exactly. The
# levers act on the DFlash tries only; a real failure of a D try drops ONE lever and boots the same D rung again.
E32_FILES = __E32_FILES__
E32_RT = {"state": None, "errors": [], "files": E32_FILES}
E32L = None


def e32_call(fn, default, *a):
    f = globals().get("_e32x_" + fn)
    if f is None or E32_RT.get("state") != "ON":
        return default
    try:
        return f(*a)
    except Exception as e:
        E32_RT["errors"].append(f"{fn}: {type(e).__name__}: {e}"[:300])
        return default


def e32_setup():
    """After e29_setup(): e32_lib (import) + e32_nbx (exec into this namespace), then _e32x_setup(). A miss = every lever
    off (= E29_DFLASH_D3S), recorded; never raises."""
    global E32L
    try:
        (TMP / E32_FILES["lib"][0]).write_bytes(q27x_payload(*E32_FILES["lib"]))
        import e32_lib as _l
        E32L = _l
        exec(compile(q27x_payload(*E32_FILES["nbx"]).decode("utf-8"), "e32_nbx.py", "exec"), globals())
        E32_RT["state"] = "ON"
        globals()["_e32x_setup"]()
        if not E32_RT.get("active"):
            E32_RT["state"] = "OFF_NO_DFLASH"
    except Exception as e:
        E32_RT.update(state="OFF_SETUP", error=f"{type(e).__name__}: {e}"[:600])
        try:
            globals()["_e32x_rollback"]()
        except Exception:
            pass
        log("!!!!!!!!!! E32 levers off (= E29_DFLASH_D3S):", E32_RT["error"][:300])
    E29_RT["e32"] = E32_RT


def e32_report(sm=None):
    f = globals().get("_e32x_summary")
    try:
        return f(sm) if f is not None else dict(E32_RT)
    except Exception as e:
        return dict(E32_RT, summary_error=f"{type(e).__name__}: {e}"[:300])
'''.replace("__E32_FILES__", json.dumps(E32_FILES))

ENGINE_EDITS = [   # (label, old, new): exact-once text edits of the engine cell (E29's final text)
    ("X4 block", "\n\n\ndef e29_extras():\n", X4_BLOCK + "\n\n\ndef e29_extras():\n"),
    ("X4 env", '        env.update({} if E29_RT.get("qsa_off") else E29L.qsa_env(E29_QSA))   # E29 rev 2c / 2d: the QSA workspace (DFlash tries)\n',
     '        env.update({} if E29_RT.get("qsa_off") else E29L.qsa_env(E29_QSA))   # E29 rev 2c / 2d: the QSA workspace (DFlash tries)\n'
     '        env.update(e32_call("env", {}, df))   # E32: G1 / G2 (DFlash tries)\n'),
    ("X4 extra", '    return (list(E29L.IDX_ARGS) if "idx" in E29_RT["on"] else []) + (list(E29_RT["df"]["tail"]) if df else [])\n',
     '    return ((list(E29L.IDX_ARGS) if "idx" in E29_RT["on"] else []) + (e32_call("argv", [], df) if df else [])   # E32: LP / SPC / IDX\n'
     '            + (list(E29_RT["df"]["tail"]) if df else []))\n'),
    ("X4 drop", '    if lever == "df":\n        E29_RT["df_try"] = E29_RT["df_live"] = False\n',
     '    if lever == "df":\n        e32_call("df_dropped", None, why)   # E32: the bundle levers go with DFlash\n'
     '        E29_RT["df_try"] = E29_RT["df_live"] = False\n'),
    ("X4 mtp", '    lv = {k_: v_ for k_, v_ in dict(NB5_ACTIVE_BOOT.get("levers") or {}).items() if k_ not in DS.ALL_ENVS + E29L.QSA_KEYS}\n'
               '    NB5_ACTIVE_BOOT.update(extra=E29L.strip_df_argv(',
     '    e32_call("mtp", None)   # E32: the bundle argv / env / seqs leave the booted config (the E28 top rung = E29)\n'
     '    lv = {k_: v_ for k_, v_ in dict(NB5_ACTIVE_BOOT.get("levers") or {}).items() if k_ not in DS.ALL_ENVS + E29L.QSA_KEYS}\n'
     '    NB5_ACTIVE_BOOT.update(extra=E29L.strip_df_argv('),
    ("X5 capacity", 'int(_seq) == int(P["top_rung_seqs"]):   # E28 K100 (user 2026-11-01): the top',
     'int(_seq) in (int(P["top_rung_seqs"]), e32_call("seqs_top", -1)):   # E32; E28 K100 (user 2026-11-01): the top'),
    ("X6 readback", '                trec["status"] = "E29_QSA_ENV_MISMATCH" if trec["e29_df_readback"]["label"] == "QSA_ENV_MISMATCH" else "E29_DFLASH_NOT_ENGAGED"\n'
                    '                return {}\n',
     '                trec["status"] = "E29_QSA_ENV_MISMATCH" if trec["e29_df_readback"]["label"] == "QSA_ENV_MISMATCH" else "E29_DFLASH_NOT_ENGAGED"\n'
     '                return {}\n'
     '            if e32_call("readback", False, s, trec):   # E32\n'
     '                return {}\n'),
    ("X6 speed", '            _e29p = None if FAKE else e29_prefill_probe(s, trec)   # E29 rev 2d F2\n            if _e29p:\n'
                 '                trec["status"] = _e29p\n                return {}\n',
     '            _e29p = None if FAKE else e29_prefill_probe(s, trec)   # E29 rev 2d F2\n            if _e29p:\n'
     '                trec["status"] = _e29p\n                return {}\n'
     '            e32_call("speed", None, s, trec)   # E32 speed pass\n'),
    ("X7 seqs", '        seqs = int(E29_CFG["workers"]) if _e29d else',
     '        seqs = e32_call("seqs", int(E29_CFG["workers"])) if _e29d else'),
    ("X7 top", '        if P.get("top_rung_seqs") and abs(float(tr["gib"]) - K100_PIN_STEPS_GIB[0]) < 1e-9:   # E28 K100 (user 2026-11-01):\n',
     '        if P.get("top_rung_seqs") and not _e29d and abs(float(tr["gib"]) - K100_PIN_STEPS_GIB[0]) < 1e-9:   # E32; E28 K100:\n'),
    ("X7 name", '                + "".join("_" + x for x in E29_RT["on"] if x != "df"))   # E28 K100; E29\n',
     '                + "".join("_" + x for x in E29_RT["on"] if x != "df") + e32_call("tag", "", _e29d))   # E28 K100; E29; E32\n'),
    ("X7 pick", '                E29_RT["qsa_off"] = True\n                continue\n',
     '                E29_RT["qsa_off"] = True\n                continue\n'
     '            if e32_call("pick", False, trec, _e28_log, _e29s):   # E32: drop one lever, same rung\n'
     '                continue\n'),
    ("X8 relaunch", '        e29_relaunch_plan(old)   # E29 rev 2: an in-run DFlash OOM -> the next D rung / MTP K3\n',
     '        e32_call("relaunch", False, old) or e29_relaunch_plan(old)   # E32; E29 rev 2\n'),
]
X9 = ('    e29_setup()   # E29 FINAL: the switchable levers (vocab / DFlash drafter + seam / indexer / GDN); a miss = that lever off\n',
      '    e29_setup()   # E29 FINAL: the switchable levers (vocab / DFlash drafter + seam / indexer / GDN); a miss = that lever off\n'
      '    e32_setup()   # E32 lever bundle\n')
X10 = ('        print(json.dumps(_sm, default=str), flush=True)\n',
       '        print(json.dumps(_sm, default=str), flush=True)\n        E28_OUT["e32"] = e32_report(_sm)   # E32 (record only)\n')


def e32_cells(base):
    cl = list(base)
    assert cl[0][0] == "markdown" and cl[1][1] == E29_CELLS[1][1]
    cl[0] = ("markdown", X1_MD)
    c1 = once(cl[2][1], X2_ANCHOR, X2_ANCHOR + X2_SWITCH, "X2 switches")
    cl[2] = ("code", once(c1, X3_OLD, X3_NEW, "X3 smoke budget"))
    c5 = cl[5][1]
    for lab, a, b in ENGINE_EDITS:
        c5 = once(c5, a, b, lab)
    cl[5] = ("code", c5)
    cl[7] = ("code", once(cl[7][1], X9[0], X9[1], "X9 setup"))
    cl[8] = ("code", once(cl[8][1], X10[0], X10[1], "X10 report"))
    return cl


CELLS = e32_cells(E29_CELLS)
assert E29B.block_of(CELLS[5][1]).count("def e32_call(") == 1, "the E32 block must sit inside the packed E29 engine block"
E29B.B.undefined_names_check(list(CELLS) + [("code", NBX_TEXT)])      # the nbx names resolve in the notebook namespace too


# ------------------------------------------------------------------ outputs
def kernel_metadata(slug, title):
    km = E29B.kernel_metadata(BASE_VARIANT, BASE_CFG)
    assert E29B.DS_ID in km["dataset_sources"] and E29B.DRAFTER_MODEL in km["model_sources"]
    return dict(km, id=f"{OWNER}/{slug}", title=title, code_file=f"{slug}.ipynb", dataset_sources=list(km["dataset_sources"]) + [DS_ID])


def write_push(slug, title, data):
    push = PUSH_ROOT / f"{slug}__{OWNER}"
    push.mkdir(parents=True, exist_ok=True)
    (push / f"{slug}.ipynb").write_bytes(data)
    km = kernel_metadata(slug, title)
    (push / "kernel-metadata.json").write_bytes((json.dumps(km, indent=2) + "\n").encode("utf-8"))
    return {"push": str(push), "slug": f"{OWNER}/{slug}", "sha256": sha(data), "kernel_metadata": km}


def smoke_bytes(data):
    """= E29's write_smoke: ONE line differs (the CELL 1 profile line E28EVAL120 -> E29SMOKE), asserted cell by cell."""
    a = json.dumps(E29B.PROFILE_LINE.format(E29B.PROD_PROFILE))[1:-1].encode("utf-8")
    b = json.dumps(E29B.PROFILE_LINE.format(E29B.SMOKE_PROFILE))[1:-1].encode("utf-8")
    assert data.count(a) == 1, data.count(a)
    sm = data.replace(a, b)
    s0 = ["".join(c_["source"]) for c_ in json.loads(data)["cells"]]
    s1 = ["".join(c_["source"]) for c_ in json.loads(sm)["cells"]]
    diff = [(i, x, y) for i, (c0, c1) in enumerate(zip(s0, s1)) for x, y in zip(c0.splitlines(), c1.splitlines()) if x != y]
    assert len(s0) == len(s1) and len(diff) == 1 and all(len(c0.splitlines()) == len(c1.splitlines()) for c0, c1 in zip(s0, s1)), diff
    assert len(sm) < 400_000
    return sm, list(diff[0])


KIT_FIXTURES = ("kit_q27-xhigh-e32-levers.ipynb", "kit_q27-xhigh-e29-dflash-d3s.ipynb")   # the units kit's inputs (kit only)


def write_dataset(nb_bytes):
    """The E32 payload: e32_lib.py + e32_nbx.py (the notebook loads them by name + sha) + the units kit's two notebook fixtures
    (this build's E32 notebook, the E29 production notebook; never read by the notebook)."""
    DS_DIR.mkdir(parents=True, exist_ok=True)
    files = {"e32_lib.py": LIB_TEXT.encode("utf-8"), "e32_nbx.py": NBX_TEXT.encode("utf-8")}
    for n, b in files.items():
        (DS_DIR / n).write_bytes(b)
        assert sha((DS_DIR / n).read_bytes()) == sha(b) == dict(E32_FILES.values())[n], n
    files.update({KIT_FIXTURES[0]: nb_bytes, KIT_FIXTURES[1]: E29_ON_DISK})
    for n in KIT_FIXTURES:
        (DS_DIR / n).write_bytes(files[n])
        assert sha((DS_DIR / n).read_bytes()) == sha(files[n]), n
    extra = [f.name for f in DS_DIR.iterdir() if f.is_file() and f.name not in files and f.name not in ("SHA256SUMS", "dataset-metadata.json")]
    assert not extra, ("unexpected files in the E32 payload folder", extra)
    sums = "".join(f"{sha(b)}  {n}\n" for n, b in sorted(files.items()))
    (DS_DIR / "SHA256SUMS").write_bytes(sums.encode("utf-8"))
    meta = {"title": DS_SLUG, "id": DS_ID, "licenses": [{"name": "CC0-1.0"}], "isPrivate": True}
    (DS_DIR / "dataset-metadata.json").write_bytes((json.dumps(meta, indent=1) + "\n").encode("utf-8"))
    return {"dir": str(DS_DIR), "id": DS_ID, "files": {n: {"bytes": len(b), "sha256": sha(b)} for n, b in files.items()},
            "sha256sums": sha(sums)}


def readable(cl):
    return "\n\n".join(E29B.B.norm(t) for k, t in cl if k == "code")


def build():
    assert sha(NBD1_CUT) == NBD1_CUT_SHA, ("E25 nbd1_lib.py's cut pieces changed (review them, then re-pin)", sha(NBD1_CUT))
    data, ncl, packs = assemble(list(CELLS))
    assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))
    nbs = json.loads(data.decode("utf-8"))
    e29n = json.loads(E29_DATA.decode("utf-8"))
    src = lambda nb, i: "".join(nb["cells"][i]["source"])
    for i in (1, 3, 4, 6):            # CELL 0 (dual mode + fallback) / CELL 3 (the scheduler apply) / cell 4 / CELL 6/7 (the writer)
        assert src(nbs, i) == src(e29n, i), ("must stay byte-identical to E29", i)
    out = ROOT / "Experiment Notebooks" / f"{NB_STEM}.ipynb"
    if out.exists() and sha(out.read_bytes()) in PUSHED:
        out = out.with_name(out.stem + "_REBUILD.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    sm, diff = smoke_bytes(data)
    out_sm = ROOT / "Experiment Notebooks" / f"{NB_STEM}_SMOKE.ipynb"
    if out_sm.exists() and sha(out_sm.read_bytes()) in PUSHED:
        out_sm = out_sm.with_name(out_sm.stem + "_REBUILD.ipynb")
    out_sm.write_bytes(sm)
    rec_p = write_push(SLUG, "Q27 Xhigh E32 Levers", data)
    rec_s = write_push(SLUG + "-smoke", "Q27 Xhigh E32 Levers Smoke", sm)
    rec_u = write_push(SLUG + "-submit", "Q27 Xhigh E32 Levers Submit", sm)
    ds = write_dataset(data)
    (HERE / "e32_lib.py").write_bytes(LIB_TEXT.encode("utf-8"))
    exp = readable(CELLS)
    (HERE / "e32_levers_code_export.py").write_bytes(exp.encode("utf-8"))
    d = "".join(difflib.unified_diff(readable(E29_CELLS).splitlines(keepends=True), exp.splitlines(keepends=True),
                                     "E29_DFLASH_D3S (readable cells)", "E32_LEVERS (readable cells)", n=2))
    (HERE / "e29_vs_e32_export.diff").write_bytes(d.encode("utf-8"))
    facts = {"notebook": str(out), "sha256": sha(data), "bytes": len(data), "smoke": {"notebook": str(out_sm), "sha256": sha(sm), "bytes": len(sm),
             "diff": diff, "budget_min": SMOKE_BUDGET, "window_min": SMOKE_WINDOW, "tasks": E29B.SMOKE_TASKS},
             "push": {"prod": rec_p, "smoke": rec_s, "submit": rec_u}, "dataset": ds, "e32_files": E32_FILES,
             "base": {"variant": BASE_VARIANT, "e29_sha256": E29_SHA, "cfg": BASE_CFG, "direct_files": {k: list(v) for k, v in E29B.DIRECT_FILES.items()},
                      "e29_builder_sha256": sha((E29_BUILD / "build_e29_final.py").read_bytes()), "sched_pin": E29B.SCHED_PIN},
             "nbd1": {"lib_sha256": sha(NBD1_SRC), "cut": list(NBD1_NAMES), "cut_sha256": sha(NBD1_CUT)},
             "e32_core_sha256": sha(CORE), "lib_version": _libns["LIB_VERSION"], "switch_lines": X2_SWITCH.splitlines(),
             "defaults": _libns["DEFAULT"], "off": _libns["OFF"], "general_order": _libns["GENERAL_ORDER"],
             "memory_order": _libns["MEMORY_ORDER"], "edits": [x[0] for x in ENGINE_EDITS] + ["X1 md", "X2 switches", "X3 smoke budget",
                                                                                              "X9 setup", "X10 report"],
             "packs": packs, "export": str(HERE / "e32_levers_code_export.py"), "diff_file": str(HERE / "e29_vs_e32_export.diff"),
             "export_sha256": sha(exp)}
    (HERE / "e32_build_facts.json").write_bytes((json.dumps(facts, indent=1, default=str) + "\n").encode("utf-8"))
    after = protected_state()
    assert after == PROTECTED_BEFORE, "an E29 production file changed during the build"
    print("wrote", out.name, sha(data), len(data), "B |", out_sm.name, sha(sm), len(sm), "B | dataset", ds["dir"])
    print("E29 production files byte-identical:", json.dumps({k: v if isinstance(v, str) else len(v) for k, v in after.items()}))
    return facts


if __name__ == "__main__":
    build()
