"""E33 CHAIN (2026-11-02, DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md): Duck + Poetiq + Direct attempt chain on the E29 Flash-Next
engine. A THIN WRAPPER over Experiment 29's build_e29_final.py (imported, its build() / build_direct() / write_* never called
on E29's paths). It first rebuilds E29_DFLASH_D3S in memory and asserts the bytes == the GPU-smoked notebook 9a097847 (so
the verbatim E29 code is exactly what ran), then applies only the marked once-only E33 edits:

  W1 (CELL 1 top)  the switch line E29_DIRECT2 -> E33_CHAIN (plain literal); E29_QSA / the profile line / E29_SMOKE unchanged
  W2 (CELL 1)      E29's DIRECT block (E29D_FILES + direct2_cfg) -> E33D_FILES (name, sha256 of the E33 payload dataset) +
                   e33_lib.py from the payload + check_chain(E33_CHAIN) (a bad value turns only that item off; a missing lib
                   = the chain off)
  W3 (CELL 3)      E29's DIRECT diff step (e29d_sched_patch) -> the E33 chain diff step (e33_sched_patch.cell3, agent S);
                   chain off / a miss = no diff = the E29 post scheduler
  W4 (engine)      e29_extras(): exec e33_nbx.py (payload, name + sha) -> e33_setup() + e33_live_setup() (instead of E29's
                   e29_nbx.py -> e29_direct_setup / e29_live_setup); a miss = the chain off + no LIVE, recorded
  W5 (CELL 9)      the E33 chain report (E28_OUT["e33"], [E33 CHAIN REPORT]) after E29's report, before the report write
  W6 (markdown)    the E33 header
Everything else is E29_DFLASH_D3S verbatim: CELL 0 (E18 R2), the writer, the rerun path (RERUN_PROD), the clock, the boot
ladder (D_START_PIN 7,665,090,560), the F2 probe, QSA 128/64, EMBED_HOST, the watchdog / relaunch, the MTP fallback, the
submission check, LIVE / report of E29 (extended by e33_nbx).

The notebook-side functions (hooks wiring, LIVE, report, traces) = Build/e33_nbx_src.py -> e33_nbx.py (generated here:
E29's e29_live.py LiveLog + E29's e29d_tok verbatim + e33_nbx_src.py), loaded from the E33 payload dataset (the 400-kB limit).

  python build_e33.py            -> Experiment Notebooks/Q27-XHIGH-E33_CHAIN.ipynb + _SMOKE + the push folders
                                    D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e33-chain{,-smoke,-submit}__foysalemonshanto/ and the
                                    payload folder D:/ARC AGI 2/_scratch/push_e25/ds_q27-e33-chain-payload/ (nothing is pushed),
                                    Build/e33_nbx.py, Build/e33_chain_code_export.py, Build/e33_chain_build_facts.json
  python build_e33.py --draft    -> the same checks in memory with placeholder pins for agent P / S files not there yet;
                                    writes nothing outside D:/ARC AGI 2/_scratch/e33_nb/
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
NBDIR = ROOT / "Experiment Notebooks"
E29B = Path(r"D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build")
E29_ROOT = E29B.parent
PUSH_ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
SCRATCH = Path(r"D:/ARC AGI 2/_scratch/e33_nb")
OWNER = "foysalemonshanto"
DS_SLUG = "q27-e33-chain-payload"
DS_ID = f"{OWNER}/{DS_SLUG}"
DS_DIR = PUSH_ROOT / f"ds_{DS_SLUG}"
NB_NAME = "Q27-XHIGH-E33_CHAIN.ipynb"
SLUG = "q27-xhigh-e33-chain"
TITLE = "Q27 Xhigh E33 Chain"
VARIANT = "E29_DFLASH_D3S"                       # the E29 engine notebook E33 is built on
E29_NB_SHA = "9a0978474fb2686209ac922c212b2c457cf7d63d7321ba4bbce3118f03bc0885"      # GPU-smoked E29_DFLASH_D3S
E29_SMOKE_SHA = "19928629ac35276fa4652757bfbb4a0eddc66265e88c017a3b3cac5e43701b29"
DRAFT = "--draft" in sys.argv[1:]
PUSHED: set = set()          # sha256 of every E33 notebook that ran on Kaggle (never overwrite one: rebuild -> *_REBUILD)
AGENT_FILES = {"sched": "e33_sched_patch.py", "poetiq": "e33_poetiq.py"}   # agent S / agent P (Build/, never edited here)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def tree_shas(*roots):
    """{path: sha256} of every file under the roots (no __pycache__): the E29 byte-identity proof (before == after)."""
    out = {}
    for r in roots:
        r = Path(r)
        for f in (sorted(r.rglob("*")) if r.is_dir() else []):
            if f.is_file() and "__pycache__" not in f.parts:
                out[str(f)] = sha(f.read_bytes())
    return out


E29_TREES = [E29_ROOT, PUSH_ROOT / "ds_q27-e29-final-payload"] + sorted(p for g in ("q27-xhigh-e29-*", "q27-e29-*") for p in PUSH_ROOT.glob(g) if p.is_dir())
E29_BEFORE = tree_shas(*E29_TREES)

_argv = sys.argv
sys.argv = [_argv[0]]                       # build_e29_final / build_e28_k100 read sys.argv: import them without our args
sys.path.insert(0, str(E29B))
import build_e29_final as F   # noqa: E402  (module import only)
sys.argv = _argv
sys.path.insert(0, str(HERE))
import e33_lib as E33L        # noqa: E402

KQ, E28B = F.KQ, F.E28B
CFG = F.E29L.check_cfg(F.VARIANTS[VARIANT])


def lf_text(p):
    b = Path(p).read_bytes()
    assert b"\r" not in b, f"CR bytes in {p}"
    return b.decode("utf-8")


def once(text, old, new, label):
    assert text.count(old) == 1, f"E33 {label}: x{text.count(old)}"
    return text.replace(old, new)


def write_lf(path, data):
    data = data.encode("utf-8") if isinstance(data, str) else data
    assert b"\r" not in data, path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
    assert Path(path).read_bytes() == data
    return sha(data)


# ------------------------------------------------------------------ 0. E29 DIRECT texts in memory (build_direct redirected)
def e29_direct_texts():
    """F.build_direct() with its HERE redirected to a scratch folder (it reads e29_live.py there and writes its three files
    there, never into E29's Build): the texts + F.DIRECT_FILES of the GPU-smoked notebook; each == E29's file on disk."""
    red = SCRATCH / "_e29_build_direct_redirect"
    if red.exists():
        shutil.rmtree(red)
    red.mkdir(parents=True)
    shutil.copyfile(E29B / "e29_live.py", red / "e29_live.py")
    here0 = F.HERE
    F.HERE = red
    try:
        F.DIRECT_TEXTS.clear()
        F.DIRECT_FILES.clear()
        t = F.build_direct()
    finally:
        F.HERE = here0
    for n in ("e29_direct.py", "e29d_sched_patch.py", "e29_nbx.py"):
        assert (red / n).read_bytes() == (E29B / n).read_bytes() == t[n].encode("utf-8"), ("E29 build_direct redirect != E29 Build", n)
        assert sha(t[n]) == F.DIRECT_FILES[{"e29_direct.py": "direct", "e29d_sched_patch.py": "sched", "e29_nbx.py": "nbx"}[n]][1]
    return t


def notebook_data(cl):
    """= the data path of F.build() for a DFlash notebook (no writes): undefined names, cell 6/7 + engine (+ donor) packs,
    compact JSON, cell 6 packed when >= 395 kB."""
    F.B.undefined_names_check(cl)
    ncl, pack7 = E28B.notebook_cells(cl)
    assert F.unpack_engine(ncl[5][1]) == ncl[5][1]
    c5, packe = F.pack_engine(ncl[5][1])
    ncl[5] = ("code", c5)
    data, n, fmt = E28B.notebook_bytes(ncl)
    pack6 = None
    if len(data) >= 395_000:
        c6, pack6 = F.pack_cell6(ncl[6][1])
        ncl[6] = ("code", c6)
        data, n, fmt = E28B.notebook_bytes(ncl)
    assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
    return data, {"cells": n, "json_format": fmt, "e21_cell7_packed": pack7, "e29_engine_packed": packe, "e21_cell6_packed": pack6}


E29_TEXTS = e29_direct_texts()
E29_CELLS = F.cells_for(VARIANT, CFG)
E29_DATA, _ = notebook_data(E29_CELLS)
assert sha(E29_DATA) == E29_NB_SHA, ("the E29 builder no longer reproduces the GPU-smoked E29_DFLASH_D3S", sha(E29_DATA))
assert sha((E29_ROOT / "Experiment Notebooks" / f"Q27-XHIGH-{VARIANT}.ipynb").read_bytes()) == E29_NB_SHA
assert sha((E29_ROOT / "Experiment Notebooks" / f"Q27-XHIGH-{VARIANT}_SMOKE.ipynb").read_bytes()) == E29_SMOKE_SHA

# ------------------------------------------------------------------ 1. the E33 payload files
LIB_TEXT = lf_text(HERE / "e33_lib.py")
compile(LIB_TEXT, "e33_lib.py", "exec")
NBX_SRC = lf_text(HERE / "e33_nbx_src.py")
E29D_TOK_SRC, _ = F._cut(F.E29_DIRECT_FNS, ("e29d_tok",), "E29_DIRECT_FNS")
assert E29D_TOK_SRC.rstrip("\n") in E29_TEXTS["e29_nbx.py"], "e29d_tok is not E29's (verbatim cut)"
LIVE_BODY = lf_text(E29B / "e29_live.py").split('"""', 2)[2]
assert LIVE_BODY in E29_TEXTS["e29_nbx.py"], "e29_live.py body is not the one in E29's e29_nbx.py"
NBX = ('"""E33 CHAIN: the notebook-side functions of Q27-XHIGH-E33_CHAIN (generated by Build/build_e33.py, do not edit; the notebook\n'
       'executes this file into its own namespace via e29_extras(): it reads the notebook\'s globals). = E29 Build/e29_live.py\n'
       '(LiveLog, verbatim) + E29\'s e29d_tok (verbatim, the Flash-Next template / tokenizer of the DIRECT branch) +\n'
       'Build/e33_nbx_src.py (the chain hooks, LIVE, report, traces)."""\n' + LIVE_BODY
       + once(NBX_SRC, "__E29D_TOK__", E29D_TOK_SRC.rstrip("\n"), "nbx e29d_tok"))
compile(NBX, "e33_nbx.py", "exec")
assert "\r" not in NBX


def agent_file(key):
    """Agent S / P files (Build/, read only): bytes, or None when not there yet (--draft only)."""
    p = HERE / AGENT_FILES[key]
    if not p.is_file():
        assert DRAFT, f"{p.name} (agent {'S' if key == 'sched' else 'P'}) is not in Build/ yet: python build_e33.py --draft"
        return None
    b = p.read_bytes()
    assert b"\r" not in b, f"CR bytes in {p.name}"
    compile(b.decode("utf-8"), p.name, "exec")
    return b


PAYLOAD = {"lib": ("e33_lib.py", LIB_TEXT.encode("utf-8")), "nbx": ("e33_nbx.py", NBX.encode("utf-8")),
           "direct": ("e29_direct.py", E29_TEXTS["e29_direct.py"].encode("utf-8"))}
for _k in AGENT_FILES:
    PAYLOAD[_k] = (AGENT_FILES[_k], agent_file(_k))
E33D_FILES = {k: [n, sha(b) if b is not None else "0" * 64] for k, (n, b) in sorted(PAYLOAD.items())}
assert E33D_FILES["direct"][1] == F.DIRECT_FILES["direct"][1], "stage E = E29's e29_direct.py (the same bytes)"

# ------------------------------------------------------------------ 2. the E33 edits (texts)
E33_SWITCH = ("E33_CHAIN = {'on': True, 'poetiq': True, 'direct': True, 'attempt2': True, 'poetiq_think': 60000, "
              "'poetiq_answer': 8192, 'direct_think': 104000, 'direct_answer': 8192, 'temperature': 1.0, 'drain_min': 26}"
              "   # E33: per attempt A -> B -> C (Poetiq 1) -> D (Poetiq 2) -> E (Direct); 'on': False = no chain (the E29 post"
              " scheduler); 'poetiq' = C + D, 'direct' = E, 'attempt2' = the attempt-2 chain; thinking >= 60,000 (user rule);"
              " a bad value turns only that item off (edit freely, no rebuild)\n")
assert E33L.check_chain(json.loads(E33_SWITCH.split(" = ", 1)[1].split("   #")[0].replace("'", '"').replace("True", "true")
                                   .replace("False", "false")))["on"] is True
E29_SWITCH_LINE = F.E29_DIRECT_SWITCH.splitlines(True)[0]
assert E29_SWITCH_LINE.startswith("E29_DIRECT2 = {") and F.E29_DIRECT_SWITCH.count(E29_SWITCH_LINE) == 1
W1_SWITCH = once(F.E29_DIRECT_SWITCH, E29_SWITCH_LINE, E33_SWITCH, "W1 switch")
# W1b (user 2026-11-02): the E33 smoke = boot + the SMOKE-ONLY solver probe (Poetiq C -> D on X, Direct E on Y, full budgets,
# record only) + ~25 min of the 12-task chain run. E29's fixed-window rule: window = min(window_min, HARD_STOP - 3 - 1 - boot
# end); HARD_STOP = budget - 5; the probe runs at the start of the solve window (inside solve_all_r6's LIVE wrapper), so the
# window = probe cap + the scheduler minutes. A boot ending by 21 min keeps the full window: 21 + 95 + 25 + 3 + 1 + 5 = 150.
SMOKE_BUDGET_MIN, PROBE_MIN, PROBE_SCHED_MIN, SMOKE_SCHED_MIN, BOOT_MIN = 150, 95, 20, 25, 21
SMOKE_WINDOW_MIN = PROBE_MIN + SMOKE_SCHED_MIN                     # 120
PROBE_TASKS = {"poetiq": "a47bf94d", "direct": "b6f77b65"}           # both K100-failed smoke tasks (see the report)
assert BOOT_MIN + SMOKE_WINDOW_MIN + 3 + 1 + F.SMOKE_HARD_MARGIN_MIN <= SMOKE_BUDGET_MIN, "smoke arithmetic"
assert SMOKE_BUDGET_MIN - F.SMOKE_HARD_MARGIN_MIN - 3 - 1 - SMOKE_WINDOW_MIN >= 18, "E29's notebook-side smoke assert"
assert PROBE_SCHED_MIN <= SMOKE_WINDOW_MIN - PROBE_MIN and set(PROBE_TASKS.values()) <= set(F.SMOKE_TASKS)
E33_SMOKE_LINE = ("E29_SMOKE = " + repr({"tasks": sorted(F.SMOKE_TASKS), "window_min": SMOKE_WINDOW_MIN, "budget_min": SMOKE_BUDGET_MIN,
                                          "probe_min": PROBE_MIN, "probe_sched_min": PROBE_SCHED_MIN, "probe": PROBE_TASKS})
                  + "   # E33: the E29SMOKE profile (12 eval tasks; the solver probe first: Poetiq C+D on probe.poetiq, Direct E on"
                    " probe.direct, deadline = start + probe_min (<= solve end - probe_sched_min); then the chain run)\n")
assert W1_SWITCH.count(F.E29_SMOKE_LINE) == 1
W1_SWITCH = once(W1_SWITCH, F.E29_SMOKE_LINE, E33_SMOKE_LINE, "W1b smoke")

W2_CELL1 = r'''E33D_FILES = __E33FILES__   # E33: the chain files of the E33 payload dataset (name, sha256)
E33L = None
try:                               # E33: the switch checker (a bad item = that item off; no lib = the chain off)
    (TMP / E33D_FILES["lib"][0]).write_bytes(q27x_payload(*E33D_FILES["lib"]))
    import e33_lib as E33L
    E33_CHAIN = E33L.check_chain(E33_CHAIN, os.environ.get("E33_CHAIN_JSON") if os.environ.get("E11_HARNESS") == "1" else None)
except Exception as _e:
    E33_CHAIN = {"on": False, "poetiq": False, "direct": False, "attempt2": False, "given": E33_CHAIN,
                 "invalid": {"chain": f"E33_LIB {type(_e).__name__}: {_e}"[:300]}}
E29_QSA = E29L.qsa_cfg(E29_QSA)
print("E33 CHAIN", json.dumps(E33_CHAIN, default=str), "| QSA", json.dumps(E29_QSA), flush=True)
'''.replace("__E33FILES__", json.dumps(E33D_FILES, sort_keys=True))
assert "__FILES__" not in W2_CELL1

W3_CELL3 = r"""# ---- E33 (switch E33_CHAIN): the chain diff on the E29 post arc2_duck.py (e33_sched_patch.cell3; chain off / a miss = no
# diff = the E29 post scheduler)
E33_SCHED_REC = {"want": bool(E33_CHAIN.get("on")), "ok": False, "path": E28_SCHED_REC.get("path")}
try:
    if E33_SCHED_REC["want"]:
        (TMP / E33D_FILES["sched"][0]).write_bytes(q27x_payload(*E33D_FILES["sched"]))
        import e33_sched_patch as E33DS
        E33_SCHED_REC = E33DS.cell3(E12_DIR, E28_SCHED_REC, E33_SCHED_REC, E33_CHAIN)
except Exception as _e:
    E33_SCHED_REC["error"] = f"{type(_e).__name__}: {_e}"[:600]
CONFIG["e33"] = {"chain": E33_CHAIN, "sched": {k: v for k, v in E33_SCHED_REC.items() if k != "text"}}
print("E33 chain sched", json.dumps({k: E33_SCHED_REC.get(k) for k in ("want", "ok", "path", "error")}, default=str), flush=True)
"""

W4_EXTRAS = r"""


def e29_extras():
    # E33 (the 400-kB commit limit): the chain + LIVE notebook functions = e33_nbx.py of the E33 payload dataset (name + sha,
    # E33D_FILES), executed into this namespace, then e33_setup() / e33_live_setup(). A miss = the chain off (hooks None, the
    # chain env removed = the E29 post scheduler) and no LIVE lines, recorded; never raises.
    try:
        exec(compile(q27x_payload(*E33D_FILES["nbx"]).decode("utf-8"), "e33_nbx.py", "exec"), globals())
        globals()["e33_setup"]()
        globals()["e33_live_setup"]()
    except Exception as e:
        for _k in (globals().get("E33_SCHED_REC") or {}).get("env") or ():
            os.environ.pop(_k, None)
        try:
            E12A.E33_RUN_POETIQ = E12A.E33_RUN_DIRECT = E12A.E33_CAN_START = None
        except Exception:
            pass
        E29_RT["extras_error"] = f"{type(e).__name__}: {e}"[:600]
        log("!!!!!!!!!! E29 extras (DIRECT / LIVE) off:", E29_RT["extras_error"][:300])
"""
assert W4_EXTRAS.endswith(F.BLOCK_END)

W5_ANCHOR = '    write_json_atomic(BASE_OUT / "e28_k100_report.json", E28_OUT)\n'
W5_REPORT = ('    try:                                     # E33: the chain report (record only; e33_nbx.e33_report)\n'
             '        E28_OUT["e33"] = globals()["e33_report"]()\n'
             '        print("[E33 CHAIN REPORT]", "\\n" + globals()["e33_report_text"](E28_OUT["e33"]), flush=True)\n'
             '    except BaseException:\n'
             '        E28_OUT["e33_error"] = traceback.format_exc()[-1500:]\n' + W5_ANCHOR)

MD0 = """# E33 CHAIN · Duck + Poetiq + Direct attempt chain · E29_DFLASH_D3S engine (klee100 3bpw, DFlash2 D3s) · E28EVAL120

- **Chain per attempt (DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md):** A Duck (T1 104k) -> B Duck re-run (E28 B1 / B2) -> C Poetiq
  iteration 1 (60k think + 8,192 answer) -> D Poetiq iteration 2 (feedback) -> E Direct (104k + 8,192, temperature 1.0); the
  chain stops at the first train pass; attempt 2 = the same chain on separate branches. Switch: `E33_CHAIN` on top of CELL 1
  (`'on': False` = the E29 post scheduler; a bad value turns only that item off; thinking >= 60,000).
- **Engine = E29_DFLASH_D3S verbatim** (Merlin TP2 x PP2, DFlash2 D3s seam 341d859b, EMBED_HOST, QSA 128/64, D ladder from
  7,665,090,560 B, F2 probe, MTP K3 fallback, 24 seqs). CELL 0 / writer / rerun (RERUN_PROD, 12 h - 3 min) / clock = E29.
- **Smoke (E29SMOKE, 150 min):** the solver probe first (Poetiq C -> D on a47bf94d + Direct E on b6f77b65 at the same time,
  full budgets, record only, `[E33 PROBE]` lines), then ~25 min of the 12-task chain run. Rerun / E28EVAL120: no probe.
- **Outputs:** one [LIVE] line per finished stage; `q27x/e28_k100_report.json` keys `e29` + `e33`; every thinking trace saved
  (Duck bundle + `Q27X_E33_TRACES.zip`).
"""


def e33_cells():
    """F.cells_for(E29_DFLASH_D3S) with W1-W4 swapped in for the E29 DIRECT texts (restored after), then W5 / W6."""
    keep = {k: getattr(F, k) for k in ("E29_DIRECT_SWITCH", "E29_DIRECT_CELL1", "E29_DIRECT_CELL3", "E29_EXTRAS_FN")}
    assert F.E29_DIRECT_CELL3.count('CONFIG["e29_direct_sched"]') == 1 and "e29_direct_setup" in F.E29_EXTRAS_FN
    try:
        F.E29_DIRECT_SWITCH, F.E29_DIRECT_CELL1, F.E29_DIRECT_CELL3, F.E29_EXTRAS_FN = W1_SWITCH, W2_CELL1, W3_CELL3, W4_EXTRAS
        cl = list(F.cells_for(VARIANT, CFG))
    finally:
        for k, v in keep.items():
            setattr(F, k, v)
    assert cl[0][0] == "markdown"
    cl[0] = ("markdown", MD0)
    cl[8] = ("code", once(cl[8][1], W5_ANCHOR, W5_REPORT, "W5 report"))
    for i, (k, t) in enumerate(cl):                      # no E29 DIRECT text left in the E33 cells (engine block unpacked here)
        assert "E29_DIRECT2" not in t and "E29D_FILES" not in t and "e29_direct_setup" not in t, (i, "E29 DIRECT text left")
    assert cl[1][1] == KQ.E18R2_CELL0 and cl[1] == E29_CELLS[1], "CELL 0 must stay E18 R2's / E29's"
    for i in (4, 6, 7):
        assert cl[i] == E29_CELLS[i], ("E33 changes no other cell", i)
    return cl


def undefined_in_nbx(cl):
    """pyflakes over the notebook's code cells + e33_nbx.py as ONE module (nbx runs in the notebook namespace)."""
    from pyflakes import api as _pf_api, reporter as _pf_rep
    import io
    out, err = io.StringIO(), io.StringIO()
    _pf_api.check("\n\n".join(F.B.norm(t) for k, t in cl if k == "code") + "\n\n" + NBX, "<e33 cells + nbx>", _pf_rep.Reporter(out, err))
    bad = [ln for ln in out.getvalue().splitlines() if "undefined name" in ln]
    assert not bad and not err.getvalue().strip(), ("undefined names (cells + e33_nbx)", bad[:20], err.getvalue()[:500])
    return True


def smoke_bytes(data):
    """= F.write_smoke's rule: the smoke copy = the production notebook with ONE line changed (the CELL 1 profile line)."""
    a = json.dumps(F.PROFILE_LINE.format(F.PROD_PROFILE))[1:-1].encode("utf-8")
    b = json.dumps(F.PROFILE_LINE.format(F.SMOKE_PROFILE))[1:-1].encode("utf-8")
    assert data.count(a) == 1, data.count(a)
    sm = data.replace(a, b)
    s0 = ["".join(c_["source"]) for c_ in json.loads(data)["cells"]]
    s1 = ["".join(c_["source"]) for c_ in json.loads(sm)["cells"]]
    diff = [(i, x, y) for i, (c0, c1) in enumerate(zip(s0, s1)) for x, y in zip(c0.splitlines(), c1.splitlines()) if x != y]
    assert len(s0) == len(s1) and len(diff) == 1 and all(len(c0.splitlines()) == len(c1.splitlines()) for c0, c1 in zip(s0, s1)), diff
    assert len(sm) < 400_000
    return sm, list(diff[0])


def kernel_metadata(slug, title):
    km = dict(F.kernel_metadata(VARIANT, CFG), id=f"{OWNER}/{slug}", title=title, code_file=f"{slug}.ipynb")
    km["dataset_sources"] = list(km["dataset_sources"]) + [DS_ID]
    assert set(F.kernel_metadata(VARIANT, CFG)["dataset_sources"]) < set(km["dataset_sources"])
    assert km["model_sources"] == F.kernel_metadata(VARIANT, CFG)["model_sources"], "never strip a model"
    return km


def write_push(slug, title, data):
    push = PUSH_ROOT / f"{slug}__{OWNER}"
    km = kernel_metadata(slug, title)
    write_lf(push / f"{slug}.ipynb", data)
    write_lf(push / "kernel-metadata.json", json.dumps(km, indent=2) + "\n")
    return {"push": str(push), "slug": f"{OWNER}/{slug}", "sha256": sha(data), "bytes": len(data), "kernel_metadata": km}


def write_dataset():
    """= F.write_dataset's layout: the files + SHA256SUMS + dataset-metadata.json (read back from disk)."""
    files = {n: b for n, b in PAYLOAD.values()}
    if DS_DIR.exists():
        for f in DS_DIR.iterdir():                       # only this builder writes here: stale files never ship
            if f.is_file() and f.name not in files and f.name not in ("SHA256SUMS", "dataset-metadata.json"):
                f.unlink()
    for n, b in files.items():
        write_lf(DS_DIR / n, b)
    for k, (n, s) in E33D_FILES.items():
        assert sha((DS_DIR / n).read_bytes()) == s, (k, n)
    sums = "".join(f"{sha(b)}  {n}\n" for n, b in sorted(files.items()))
    write_lf(DS_DIR / "SHA256SUMS", sums)
    meta = {"title": DS_SLUG, "id": DS_ID, "licenses": [{"name": "CC0-1.0"}], "isPrivate": True}
    write_lf(DS_DIR / "dataset-metadata.json", json.dumps(meta, indent=1) + "\n")
    return {"dir": str(DS_DIR), "id": DS_ID, "files": {n: {"bytes": len(b), "sha256": sha(b)} for n, b in sorted(files.items())},
            "sha256sums": sha(sums)}


def cell_map(data):
    nb = json.loads(data)
    return [{"i": i, "type": c["cell_type"], "bytes": len("".join(c["source"]).encode()), "head": "".join(c["source"])[:90]}
            for i, c in enumerate(nb["cells"])]


def build():
    cl = e33_cells()
    undefined_in_nbx(cl)
    data, packs = notebook_data(cl)
    assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))
    nb = json.loads(data.decode("utf-8"))
    assert "".join(nb["cells"][1]["source"]).rstrip("\n") == KQ.E18R2_CELL0.rstrip("\n")
    assert nb["cells"][1] == json.loads(E29_DATA.decode("utf-8"))["cells"][1], "CELL 0 byte-identical to E29's"
    sm, diff = smoke_bytes(data)
    for k, (n, s) in E33D_FILES.items():                 # the payload pins in the notebook == the payload files
        assert json.dumps([n, s]) in W2_CELL1 and s == (sha(PAYLOAD[k][1]) if PAYLOAD[k][1] is not None else "0" * 64), k
    rec = {"notebook_sha256": sha(data), "bytes": len(data), "smoke_sha256": sha(sm), "smoke_bytes": len(sm), "smoke_diff": diff,
           "cell_map": cell_map(data), "payload_pins": E33D_FILES, **packs}
    if DRAFT:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        write_lf(SCRATCH / ("DRAFT_" + NB_NAME), data)
        write_lf(SCRATCH / "DRAFT_e33_nbx.py", NBX)
        print("DRAFT (placeholder pins for", [n for k, (n, b) in PAYLOAD.items() if b is None], "):", json.dumps(
            {k: rec[k] for k in ("notebook_sha256", "bytes", "smoke_sha256", "smoke_diff")}))
        return rec
    out = NBDIR / NB_NAME
    if out.exists() and sha(out.read_bytes()) in PUSHED:
        out = out.with_name(out.stem + "_REBUILD.ipynb")
    write_lf(out, data)
    write_lf(NBDIR / NB_NAME.replace(".ipynb", "_SMOKE.ipynb"), sm)
    write_lf(HERE / "e33_nbx.py", NBX)
    export = HERE / "e33_chain_code_export.py"
    write_lf(export, "\n\n".join(F.B.norm(t) for k, t in cl if k == "code"))
    rec.update(notebook=str(out), smoke=str(NBDIR / NB_NAME.replace(".ipynb", "_SMOKE.ipynb")), export=str(export),
               push=write_push(SLUG, TITLE, data), push_smoke=write_push(SLUG + "-smoke", TITLE + " Smoke", sm),
               push_submit=write_push(SLUG + "-submit", TITLE + " Submit", sm), dataset=write_dataset())
    assert rec["push_submit"]["sha256"] == rec["smoke_sha256"], "SUBMIT = the smoke bytes"
    after = tree_shas(*E29_TREES)
    assert after == E29_BEFORE, ("E29 files changed", sorted(set(after.items()) ^ set(E29_BEFORE.items()))[:10])
    rec.update(e29={"notebook_sha256": E29_NB_SHA, "smoke_sha256": E29_SMOKE_SHA, "rebuilt_in_memory_identical": True,
                    "files_checked": len(after), "tree_sha256": sha(json.dumps(after, sort_keys=True)), "unchanged": True},
               switch_line=E33_SWITCH.rstrip("\n"), chain_default=E33L.check_chain(E33L.DEFAULT_CHAIN), lib_version=E33L.LIB_VERSION,
               edits=["W1 switch", "W1b smoke literal", "W2 cell1", "W3 cell3", "W4 extras", "W5 report", "W6 markdown"], variant=VARIANT,
               smoke={"budget_min": SMOKE_BUDGET_MIN, "window_min": SMOKE_WINDOW_MIN, "probe_min": PROBE_MIN,
                      "probe_sched_min": PROBE_SCHED_MIN, "sched_min": SMOKE_SCHED_MIN, "boot_min": BOOT_MIN, "probe": PROBE_TASKS,
                      "line": E33_SMOKE_LINE.rstrip("\n")},
               e29_builder_sha256=sha((E29B / "build_e29_final.py").read_bytes()), builder_sha256=sha((HERE / "build_e33.py").read_bytes()),
               nbx_src_sha256=sha(NBX_SRC), agent_files={k: (n, E33D_FILES[k][1]) for k, n in AGENT_FILES.items()})
    write_lf(HERE / "e33_chain_build_facts.json", json.dumps(rec, indent=1, default=str) + "\n")
    print("wrote", out.name, "sha256", rec["notebook_sha256"], "|", rec["bytes"], "bytes | smoke", rec["smoke_sha256"],
          "| one-line diff:", diff[1].split("   #")[0], "->", diff[2].split("   #")[0])
    print("payload", json.dumps({n: v["sha256"][:12] for n, v in rec["dataset"]["files"].items()}), "| E29 unchanged:", len(after), "files")
    return rec


if __name__ == "__main__":
    build()
