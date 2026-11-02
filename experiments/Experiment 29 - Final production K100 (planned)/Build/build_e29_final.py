"""E29 FINAL production (2026-11-01): the switchable build of the E28 K100 EVAL120 notebook (the production config running on
GPU now). Every FINAL_CFG flag OFF = E28 K100 exactly (the same levers / scheduler / boot; the E29 code paths are inert:
no E29 lever is on, the E28 scheduler is applied, the E28 ladder runs). The decision after tonight's GPU results = flags.

  python build_e29_final.py [VARIANT ...]      (default: the three notebooks below; ALL_OFF = every flag off, a check build)
    E29_DEFAULT     = E28 K100 + attempts_supply (the E29 scheduler S-1, e29_sched_patch.py)
    E29_DFLASH_D3S  = E29_DEFAULT + drafter dflash_d3s + embed_host (rev 2 2026-11-02: indexer_fp8 removed, seam 341d859b)
    E29_MTP_V64     = E29_DEFAULT + draft_vocab 64k + indexer_fp8
  rev 2 (2026-11-02): DEFAULT / MTP_V64 / ALL_OFF are asserted byte-identical to the audited builds (AUDITED); only the DFlash
  notebook changes: e29_lib 1.1 in its modules literal and the E10 edits (engine_df2: GRD read-back, warm-up / pre-phase
  death gates, OOM-at-any-stage D rung step, the in-run OOM relaunch plan).
  -> ../Experiment Notebooks/Q27-XHIGH-<VARIANT>.ipynb, Build/<variant>_code_export.py, Build/e29_build_facts.json and the GPU
     push folders D:/ARC AGI 2/_scratch/push_e25/<slug>__foysalemonshanto/ (nothing is pushed), the payload dataset folder
     D:/ARC AGI 2/_scratch/push_e25/ds_q27-e29-final-payload/ (dflash_seam.py + the two draft vocab tables)

Built on the E28 builder (Experiment 28 Build/build_e28_k100.py, imported, its build() never called; it asserts the K100Q
base 4506a71f byte for byte). The E29 edits (marked "# E29") are exact-once text replacements on E28's cells:
  E1  (CELL 1)  the modules literal + e29_sched_patch.py (scheduler S-1), e29_lib.py (flag helpers), e29_nbdf.py (verbatim
                nbdf_lib pieces: drafter pins / check, dflash spec, captures, SeamSeq); E29_CFG; the config record
  E2  (CELL 1)  workers (top_rung_seqs) and, for 28 / 32, Merlin's capture_sizes_for(workers, 3) (24 = K100_CAPS unchanged)
  E3  (CELL 3)  the scheduler apply: E29 (supply) first when asked, any failure -> revert -> the E28 scheduler (E28's rule)
  E4  (engine)  the E29 lever code: setup (vocab / drafter / seam from the dataset), per-try env + argv, the drop rule, the
                DFlash C-11 + seam read-back (ENGAGED) + the zero-accept canary, the in-run canary (-> MTP relaunch)
  E5  (engine)  nb5_boot: the D tries first (NBDF-K100 pin ladder), the E29 drop rule BEFORE the E28 I8 rule, the E18
                3-try cap counts non-E29 tries only; nb5_boot_try: a launch-prep raise with an E29 lever on = a failed try
  E6  (engine)  the sampler records the spec-decode counters (non-default builds)
  E7  (runner)  e29_setup() after k100_setup / the I8 static check
  E8  (report)  e28_k100_report.json["e29"] (flags, drops, canaries, hourly AL)
  E9  (markdown) the E29 header
"""
import ast
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
E28B_DIR = Path(r"D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build")
E27B_DIR = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
VOCAB_DIR = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Results/prep_0gpu/draft_vocab")
PUSH_ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
OWNER = "foysalemonshanto"
DS_SLUG = "q27-e29-final-payload"
DS_ID = f"{OWNER}/{DS_SLUG}"
DS_DIR = PUSH_ROOT / f"ds_{DS_SLUG}"
DRAFTER_MODEL = "foysalemonshanto/qwen38-flash-next-dflash2v2/transformers/bf16-embed/1"
PUSHED: set = set()          # sha256 of every E29 notebook that ran on Kaggle (never overwrite one: rebuild -> *_REBUILD)

_argv = sys.argv
sys.argv = [_argv[0]]                       # the E28 builder reads sys.argv (--interim): import it without our args
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(E28B_DIR))
import build_e28_k100 as E28B   # noqa: E402
sys.argv = _argv
import e29_lib as E29L          # noqa: E402
import e29_sched_patch as E29S  # noqa: E402
import e29_direct_build as E29DB   # noqa: E402  rev 2: DIRECT (e29_direct.py + the DIRECT scheduler diff)

KQ, B, N6 = E28B.KQ, E28B.B, E28B.N6
sha = E28B.sha
assert E28B.E28S is not None, "the E28 builder must embed e28_sched_patch.py"
E28_NB_SHA = sha(E28B.OUT_NB.read_bytes())

FINAL_CFG = E29L.default_cfg()      # every flag OFF = E28 K100 exactly
VARIANTS = {
    "E29_DEFAULT": dict(FINAL_CFG, attempts_supply=True),
    # rev 2 (2026-11-02): EMBED_HOST from the first D try (NB-S1: without it the first request OOMs on PP1); indexer_fp8 OFF
    # (user decision, REMOVED below); the seam 341d859b (GRD edit)
    "E29_DFLASH_D3S": dict(FINAL_CFG, attempts_supply=True, drafter="dflash_d3s", embed_host=True),
    "E29_MTP_V64": dict(FINAL_CFG, attempts_supply=True, draft_vocab="64k", indexer_fp8=True),
    "ALL_OFF": dict(FINAL_CFG),
}
for _v in VARIANTS.values():
    E29L.check_cfg(_v)


# ------------------------------------------------------------------ the E29 modules
def _cut(src, names, label):
    """Top-level defs / assignments / classes of `src` by name, verbatim (ast.get_source_segment), in source order."""
    t = ast.parse(src)
    seg, got = [], []
    for n in t.body:
        nm = getattr(n, "name", None) or (n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else None)
        if nm in names:
            seg.append(ast.get_source_segment(src, n))
            got.append(nm)
    miss = [x for x in names if x not in got]
    assert not miss, (label, miss)
    return "\n\n\n".join(seg) + "\n", got


NBDF_LIB_SRC = (E27B_DIR / "nbdf_lib.py").read_text(encoding="utf-8").replace("\r\n", "\n")
NBDF_NAMES = ("DRAFTER_MODEL", "DRAFTER_BASE", "DRAFTER_REL", "DRAFTER_VARIATION", "DRAFTER_PINS", "TARGET_LAYER_IDS",
              "MASK_TOKEN_ID", "DRAFT_HIDDEN", "DRAFT_LAYERS", "DRAFT_BLOCK", "DRAFT_KV_HEADS", "DRAFT_HEAD_DIM", "VOCAB",
              "DRAFT_ARCH", "MAX_NUM_BATCHED", "DRAFT_SAMPLE", "SW_BLOCK", "SMOKE_MIN_ACCEPT", "SMOKE_MIN_DRAFTS",
              "capture_sizes", "dflash_spec", "spec_json", "compilation_json", "tree_state", "SeamSeq", "safetensors_keys",
              "drafter_check", "bf16_row_norm", "drafter_pins_check")
_nbdf_cut, _ = _cut(NBDF_LIB_SRC, NBDF_NAMES, "nbdf_lib")
# E29 rev 2 (2026-11-02): E27's nbdf_lib.py / nbdf_k100_cell7_body.py moved on after the audited build (nbdf_lib cd2c7281 ->
# b0147751); the pieces E29 cuts are byte-identical (asserted), so e29_nbdf.py stays the audited module (its header names the
# nbdf_lib version it was cut from) and the DEFAULT / MTP_V64 notebooks stay byte-identical.
NBDF_AT_BUILD = "cd2c72813fea"                       # nbdf_lib.py sha256[:12] when e29_nbdf.py was cut (audited build 2026-11-01)
E29_NBDF_SHA = "68e975c6b25f71d8d191848e79ef5f516f135f0aaa63940e1beff2d57ec544bd"   # the audited e29_nbdf.py (all notebooks)
E29_NBDF = ('"""E29: verbatim pieces of E27 nbdf_lib.py (sha ' + NBDF_AT_BUILD + '; NBDF-K100\'s DFlash2 boot facts):\n'
            "the drafter Model pins, the drafter check, the dflash spec / captures / compilation JSON, the seam apply sequence."
            '"""\nimport json\nimport math\n\n\n' + _nbdf_cut)
assert sha(E29_NBDF) == E29_NBDF_SHA, "nbdf_lib's E29 pieces changed (review them before rebuilding)"
compile(E29_NBDF, "e29_nbdf.py", "exec")
# the seam (rev 2): pinned by its own sha = E27's CURRENT dflash_seam.py (341d859b: + the GRD edit of Merlin e2_guards.py,
# query_cudagraph_manager = the B1 fix, proven on GPU by E25 NB-S1 2026-11-01/02); no longer tied to the NBDF-K100 facts
assert E29L.SEAM_FILE[1] == "341d859be5b2cc619247ff7b645304b9473d09f07995358ead2ee02be116c60b"
SEAM_BYTES = (E27B_DIR / "dflash_seam.py").read_bytes()
assert sha(SEAM_BYTES) == E29L.SEAM_FILE[1] and b"\r" not in SEAM_BYTES, ("E27 dflash_seam.py != the E29 pin", sha(SEAM_BYTES))
_seam_ns = {"__name__": "dflash_seam_e29b"}
exec(compile(SEAM_BYTES.decode("utf-8"), "dflash_seam.py", "exec"), _seam_ns)
assert (_seam_ns["GRD_REL"] == E29L.GRD_REL and E29L.GRD_REL in _seam_ns["CORE_RELS"] and _seam_ns["GRD_POST"] == E29L.GRD_POST
        and _seam_ns["PATCHES"][E29L.GRD_REL][1] == E29L.GRD_POST and "query_cudagraph_manager" in _seam_ns["GRD_PATCH"])
assert set(_seam_ns["server_env"](mask_row_sha="ab", embed_host=True, draft_sw_block=1024)) == {
    "DFLASH_SEAM", "DFLASH_MASK_ROW_SHA", "DFLASH_DRAFT_EMBED_HOST", "DFLASH_DRAFT_SW_BLOCK"}
NBDFK_SEAM_NOW = json.loads((E27B_DIR / "nbdf_k100_build_facts.json").read_text(encoding="utf-8")).get("seam_sha256")   # recorded
# e29_lib: DEFAULT / MTP_V64 / ALL_OFF keep the audited 1.0 (Build/e29_lib_v1_0.py); the DFlash notebook gets 1.1 (e29_lib.py)
LIB10_TEXT = (HERE / "e29_lib_v1_0.py").read_bytes().decode("utf-8")
assert sha(LIB10_TEXT) == "d9fba9076aa6f2aa6cadae01210273a279486a5f2a4d91cbd193c8201364b5fb" and "\r" not in LIB10_TEXT
AUDITED = {"E29_DEFAULT": "5c74abf869bee204de3da97f26cfc0d9abe3fd9d5eb7cef69ae285925b871af9",   # must rebuild byte-identical
           "E29_MTP_V64": "13419f696db4054a508551f97ba7711a050a01df595b45e416704c12d1292a27",
           "ALL_OFF": "d7c391200bd9ab344b1fd2b4db196d13b3c79a70b2f1033372dfc7f8457f8381"}
SUPERSEDED = {"E29_DFLASH_D3S": "c775f653e2353c8d1957fb2f101995388491b79619546a23466cba05f1d70210"}   # rev 1 (seam 7356d473, idx on)
REMOVED = {"E29_DFLASH_D3S": {"indexer_fp8": "user 2026-11-02: kit = 0 extra KV blocks on the DFlash D-SW layout, never run on "
                                             "sm_89 GPU; one unproven change per run (may return after NB-M1L's IDX arm)"}}
_nbdf_ns = {}
exec(compile(E29_NBDF, "e29_nbdf.py", "exec"), _nbdf_ns)
assert _nbdf_ns["DRAFTER_MODEL"] == DRAFTER_MODEL and _nbdf_ns["SW_BLOCK"] == 1024
assert _nbdf_ns["capture_sizes"](24, 3)[-1] == 96 and _nbdf_ns["capture_sizes"](24, 4)[-1] == 120
for _k, (_f, _s) in E29L.VOCAB_FILES.items():
    assert sha((VOCAB_DIR / _f).read_bytes()) == _s, _f

# the NBDF-K100 drafter gate (notebook functions, verbatim from its CELL 7 body; they read MD / N / hashlib / time)
NBDF_BODY_SRC = (E27B_DIR / "nbdf_k100_cell7_body.py").read_text(encoding="utf-8").replace("\r\n", "\n")
DRAFTER_FNS, _ = _cut(NBDF_BODY_SRC, ("drafter_resolve", "_sha_file", "drafter_facts"), "nbdf_k100_cell7_body")
assert sha(DRAFTER_FNS) == "d60a80f373a043d85d94a1dcf26a419ec49e79a9a66a6747d241f3409d757e59", "the NBDF-K100 drafter gate changed"

LIB_TEXT = (HERE / "e29_lib.py").read_bytes().decode("utf-8")
SCHED_TEXT = (HERE / "e29_sched_patch.py").read_bytes().decode("utf-8")
for _n, _t in (("e29_lib.py", LIB_TEXT), ("e29_sched_patch.py", SCHED_TEXT), ("e29_nbdf.py", E29_NBDF), ("drafter_fns", DRAFTER_FNS)):
    assert "\r" not in _t, _n
assert E29L.LIB_VERSION.startswith("e29_lib 1.1") and "e29_lib 1.0" in LIB10_TEXT
assert E29S.NOTEBOOK_EDITS is E29S.E28S.NOTEBOOK_EDITS and E29S.E28S.E28_SCHED_SHA == E28B.E28S.E28_SCHED_SHA
_duck = E28B.DUCK_SRC.read_bytes().decode("utf-8")
_p = E29S.apply_e29_sched(_duck)
assert _p["ok"] and sha(_p["text"].encode()) == E29S.E29_SCHED_SHA["post"]
assert E29S.revert_e29_sched(_p["text"])["text"] == _duck
assert E29S.server_env(supply=False) == E28B.E28S.server_env() and E29S.server_env(supply=True)["E29_SUPPLY"] == "1"

MODULES = dict(E28B.MODULES, **{"e29_lib.py": LIB10_TEXT, "e29_sched_patch.py": SCHED_TEXT, "e29_nbdf.py": E29_NBDF})   # 1.0 set
MODULES_JSON = json.dumps(MODULES, sort_keys=True, separators=(",", ":"))
MODULES_DF = dict(MODULES, **{"e29_lib.py": LIB_TEXT})                                  # rev 2: the DFlash notebook (e29_lib 1.1)
REGISTRY_CAPS = {str(w): KQ.capture_sizes_for(w, 3) for w in (24, 28, 32)}


def is_df(cfg):
    return cfg["drafter"] != "mtp"


def modules_for(cfg):
    return MODULES_DF if is_df(cfg) else MODULES


def once(text, old, new, label):
    assert text.count(old) == 1, f"E29 {label}: x{text.count(old)}"
    return text.replace(old, new)


# ------------------------------------------------------------------ E1 / E2: CELL 1
E29_MODS = r'''# ---- E29 FINAL: the switchable flags (Build/build_e29_final.py FINAL_CFG; every flag OFF = E28 K100 exactly) + the E29
# modules from the same sha-pinned literal: e29_lib.py, e29_sched_patch.py (scheduler S-1 over e28_sched_patch), e29_nbdf.py
import e29_lib as E29L
import e29_nbdf as N
E29S, E29S_ERROR = None, None
try:
    import e29_sched_patch as E29S
except Exception as _e:
    E29S_ERROR = f"{type(_e).__name__}: {_e}"[:400]
    print("!!!!!!!!!! E29 SCHED MODULE NOT LOADED:", E29S_ERROR, flush=True)
E29_CFG = E29L.check_cfg(json.loads(r"""__CFG__"""))
E29_VARIANT = "__VARIANT__"
print("E29 FINAL", E29_VARIANT, json.dumps(E29_CFG), flush=True)
'''


DIRECT_FILES = {}        # rev 2: {"direct": (name, sha), "sched": (name, sha)} (set in build_direct())


def cell1(c, cfg, variant):
    mods = E29_MODS.replace("__CFG__", json.dumps(cfg, sort_keys=True)).replace("__VARIANT__", variant)
    if is_df(cfg):                                       # E29 rev 2: DIRECT (the switch line on top of CELL 1)
        c = E29_DIRECT_SWITCH + c
        mods += E29_DIRECT_CELL1.replace("__FILES__", json.dumps({k: list(v) for k, v in sorted(DIRECT_FILES.items())}))
    c = once(c, "PPK, PPK_ERROR = None, None   # E26 K100 Q-defer:", mods + "PPK, PPK_ERROR = None, None   # E26 K100 Q-defer:", "E1 modules")
    if is_df(cfg):                                       # E29 rev 2e: the E29SMOKE profile row (the production default unchanged)
        for _a, _b in E29_SMOKE_CELL1:
            c = once(c, _a, _b, "E12 smoke")
    c = once(c, '                        "lib": E28L.LIB_VERSION}   # E28 K100\n',
             '                        "lib": E28L.LIB_VERSION}   # E28 K100\n'
             'CONFIG["nb5"]["e29"] = {"variant": E29_VARIANT, "cfg": E29_CFG, "lib": E29L.LIB_VERSION, "sched_error": E29S_ERROR,\n'
             '                        "sched_sha": getattr(E29S, "E29_SCHED_SHA", None)}   # E29\n', "E1 config")
    w = int(cfg["workers"])
    if w != 24:
        c = once(c, "                  top_rung_seqs=24)   # user 2026-11-01: 24 workers / --max-num-seqs 24 on the TOP rung only\n",
                 f"                  top_rung_seqs={w})   # E29: workers {w} / --max-num-seqs {w} on the TOP rung only\n", "E2 workers")
        line = next(l for l in c.splitlines() if l.startswith("K100_CAPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 84]"))
        caps = E29L.mtp_captures_for(w, [], REGISTRY_CAPS)
        assert -(-w * 4 // 2) in caps, (w, caps)
        c = once(c, line + "\n", f"K100_CAPS = {json.dumps(caps)}   # E29: Merlin capture_sizes_for({w}, K=3) (workers {w})\n", "E2 caps")
    return c


# ------------------------------------------------------------------ E3: CELL 3 (the scheduler)
E29_SCHED_APPLY = r'''# ---- E29 FINAL: the scheduler (Build/E28_SCHED_API.md; e29_sched_patch.py docstring). attempts_supply -> the E29 scheduler
# (E28 A / B / C + S-1: idle workers start extra attempts once the A / B / C queues are empty); any failure -> reverted to the
# original file -> the E28 scheduler (proven). The E28 path keeps its rule: commit = NO_GO_E28_SCHED (raised), rerun =
# reverted, the original scheduler runs.
E28_SCHED_REC = {"want": bool(P.get("e28_sched")), "applied": False, "e29_supply_want": bool(E29_CFG["attempts_supply"]), "failures": []}
if E28_SCHED_REC["want"]:
    _e29_paths = (["E29"] if E29_CFG["attempts_supply"] else []) + ["E28"]
    for _lab in _e29_paths:
        try:
            if _lab == "E29":
                if E29S is None:
                    raise RuntimeError("E29_SCHED_MODULE_MISSING: " + str(E29S_ERROR))
                _ap, _vf = E29S.apply_e29_sched, E29S.verify_e29_sched
            else:
                if E28S is None:
                    raise RuntimeError("E28_SCHED_MODULE_MISSING: " + str(E28S_ERROR))
                _ap, _vf = E28S.apply_e28_sched, E28S.verify_e28_sched
            E28_SCHED_REC.update({k: v for k, v in _ap(E12_DIR).items() if k != "text"})
            _ok, _why = _vf(E12_DIR)
            E28_SCHED_REC.update(verify_ok=_ok, verify_why=_why, state=(E29S if _lab == "E29" else E28S).tree_state(E12_DIR))
            if not _ok:
                raise RuntimeError(f"{_lab}_SCHED_VERIFY_FAILED: " + str(_why))
            _e28env = dict(E29S.server_env(supply=True) if _lab == "E29" else E28S.server_env())
            if E11_HARNESS:              # local / CPU-kit harness only: the fake run's short attempts (NB5_ATTEMPT_B0 / E28H_B2_TOKENS)
                _e28env.update(E28_T1=str(NB5_ATTEMPT["b0"]), E28_B2_TOKENS=os.environ.get("E28H_B2_TOKENS", _e28env["E28_B2_TOKENS"]))
            _e28env["E28_B2_REQUEST_TIMEOUT"] = str(E28_B2_TIMEOUT)   # E28 K100 (user 2026-11-01): B2 requests 9,000 s; T1 7,200 s
            if int(_e28env["E28_T1"]) != int(NB5_ATTEMPT["b0"]):
                E28_SCHED_REC["warning"] = f"E28_T1 {_e28env['E28_T1']} != NB5_ATTEMPT b0 {NB5_ATTEMPT['b0']}"
            os.environ.update(_e28env)
            E28_SCHED_REC.update(applied=True, env=_e28env, path=_lab, error=None)
            print(f"{_lab} SCHED: applied", json.dumps({k: E28_SCHED_REC.get(k) for k in ("applied", "path", "before", "after", "sha_after", "verify_why")}), flush=True)
            break
        except Exception as _e:
            E28_SCHED_REC.update(applied=False, error=f"{_lab}: {type(_e).__name__}: {_e}"[:800])
            E28_SCHED_REC["failures"].append(E28_SCHED_REC["error"])
            for _k in set((getattr(E29S, "ENV_NAMES", None) or ()) + (getattr(E28S, "ENV_NAMES", None) or ("E28_SCHED",))):
                os.environ.pop(_k, None)
            try:                        # back to the original arc2_duck.py (E29 revert handles E28-patched and E29-patched)
                E28_SCHED_REC.setdefault("reverts", []).append({k: v for k, v in (E29S.revert_e29_sched(E12_DIR) if E29S is not None
                                                                                   else E28S.revert_e28_sched(E12_DIR)).items() if k != "text"})
            except Exception as _e2:
                E28_SCHED_REC["revert_error"] = f"{type(_e2).__name__}: {_e2}"[:400]
            print(f"!!!!!!!!!! NO_GO_{_lab}_SCHED:", E28_SCHED_REC["error"], "->", "the E28 scheduler" if _lab == "E29" else
                  ("RERUN: the ORIGINAL scheduler runs" if RERUN else "commit: no solve (CELL 0's fallback)"), flush=True)
            if _lab == "E28" and not RERUN:
                raise RuntimeError("NO_GO_E28_SCHED: " + E28_SCHED_REC["error"])
CONFIG["e28_sched"] = E28_SCHED_REC
'''


def cell3(c, df=False):
    c = once(c, E28B.E28_SCHED_APPLY, E29_SCHED_APPLY, "E3 scheduler")
    if df:                                               # E29 rev 2: the DIRECT diff right after the scheduler apply
        c = once(c, 'CONFIG["e28_sched"] = E28_SCHED_REC\n', 'CONFIG["e28_sched"] = E28_SCHED_REC\n' + E29_DIRECT_CELL3, "E10 direct sched")
    return c


# ------------------------------------------------------------------ E4 - E6: the engine cell
E29_ENGINE = r'''


# ==================================================================== E29 FINAL: the switchable levers (e29_lib, build_e29_final.py)
# Real-failure gates only. A boot try with an E29 lever on that fails before any phase (launch / OOM / not ready / probes /
# C-11 / the DFlash read-back / the zero-accept canary) drops ONE lever (e29_lib.lever_to_drop: the one the log names, else
# DFlash, then indexer, vocab, GDN) and the next try runs; DFlash -> the proven MTP K3 config at the top rung (I8 restored).
# With no E29 lever left the E28 ladder (and its I8 rule) runs unchanged.
E29_RT = {"cfg": E29_CFG, "on": [], "dropped": [], "df": None, "vocab": None, "idx": None, "gdn": None,
          "supply": None, "tries": [], "canary": [], "inrun": None, "hourly_al": None, "seam": None, "drafter": None,
          "errors": [], "df_try": False, "df_live": False}
DS, SSEQ = None, None


__DRAFTER_FNS__


def _e29_tree_py():
    eng = merlin_install()
    return Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"


def e29_drop(lever, why, try_name=None):
    """Drops one E29 lever (recorded); DFlash: the seam reverted (exact Merlin files), I8 restored (E28's MTP config)."""
    if lever in E29_RT["on"]:
        E29_RT["on"].remove(lever)
    E29_RT["dropped"].append({"lever": lever, "why": str(why)[:600], "try": try_name, "t_sec": round(time.perf_counter() - NOTEBOOK_T0)})
    log("!!!!!!!!!! E29 lever dropped:", lever, "|", str(why)[:300], "| try", try_name, "| still on", E29_RT["on"])
    if lever == "df":
        E29_RT["df_try"] = E29_RT["df_live"] = False
        if SSEQ is not None and not FAKE:
            try:
                E29_RT["seam"] = dict(E29_RT.get("seam") or {}, finish=SSEQ.finish())
            except Exception as e:
                E29_RT["errors"].append(f"seam finish: {e!r}"[:300])
        if E28_I8.get("want") and not E28_I8.get("dropped"):
            E28_I8["on"] = True                   # E28's proven MTP config (I8 on; its own I8 rule applies from here)
        E28_I8.pop("suspended", None)


def e29_df_setup():
    """NBDF-K100's DFlash2 boot path (E27 Build): the drafter Model gate (pinned shas: model.safetensors d2e0464d, config,
    embed tokens, mask row 248077), dflash_seam.py (sha 7356d473, from the E29 payload dataset) applied with SeamSeq
    (ring first, then core; draft_block on), the seam env (DFLASH_SEAM=1, DFLASH_MASK_ROW_SHA, DFLASH_DRAFT_SW_BLOCK=1024),
    the spec {"method": "dflash", BF16 drafter KV "auto", FLASH_ATTN, probabilistic} and the captures (1 + K) x 1..workers."""
    global DS, SSEQ
    k = E29L.K_OF[E29_CFG["drafter"]]
    rec = E29_RT["df"] = {"want": E29_CFG["drafter"], "k": k, "ok": False}
    try:
        drafter = (drafter_facts() if not FAKE else
                   {"ok": True, "fake": True, "dir": str(TMP / "fake_drafter"), "check": {"embed_in_checkpoint": True}})
        E29_RT["drafter"] = {k_: drafter.get(k_) for k_ in ("ok", "status", "why", "dir", "pins_ok", "pin_mismatches", "check", "sec", "fake")}
        if not drafter.get("ok"):
            raise RuntimeError("DRAFTER: " + str(drafter.get("why")))
        if not (drafter.get("check") or {}).get("embed_in_checkpoint"):
            raise RuntimeError("DRAFTER_NO_EMBED_TOKENS (the bf16-embed variation is required)")
        src = q27x_payload(*E29L.SEAM_FILE)
        (TMP / E29L.SEAM_FILE[0]).write_bytes(src)
        import dflash_seam as _DS
        DS = _DS
        if not FAKE:
            tree, py = _e29_tree_py()
            SSEQ = N.SeamSeq(lambda ring: DS.apply_dflash_patch(tree, py, ring=ring),
                             lambda ring: DS.verify_dflash_patch(tree, ring=ring),
                             lambda ring: DS.revert_dflash_patch(tree, py, ring=ring))
            ok, why = SSEQ.apply()
            E29_RT["seam"] = {"apply": why, "ring": SSEQ.ring, "state": SSEQ.state,
                              "ppack_after": (PPK.verify_ppack_patch(tree)["state"] if PPK is not None else None)}
            if not ok:
                raise RuntimeError("SEAM: " + str(why))
        rec.update(dir=drafter["dir"], spec=N.dflash_spec(drafter["dir"], k, "auto", "FLASH_ATTN"),
                   captures=N.capture_sizes(int(E29_CFG["workers"]), k),
                   mask_row_sha=N.DRAFTER_PINS["mask_row"]["sha256"], sw_block=N.SW_BLOCK)
        rec["tail"] = E29L.df_spec_argv(N.spec_json(rec["spec"]), N.compilation_json(rec["captures"]))
        rec["ok"] = True
        return True
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:800]
        E29_RT["dropped"].append({"lever": "df", "why": "setup: " + rec["error"], "try": None})
        log("!!!!!!!!!! E29 DFlash setup failed -> MTP K3 (E28):", rec["error"][:400])
        if SSEQ is not None and SSEQ.state in ("applied", "failed"):
            try:
                E29_RT["seam"] = dict(E29_RT.get("seam") or {}, finish=SSEQ.finish())
            except Exception as e2:
                E29_RT["errors"].append(f"seam finish: {e2!r}"[:300])
        return False


def e29_setup():
    """Before the first boot (after k100_setup and the I8 static check): which E29 levers are on. A setup miss = that lever
    off (recorded); never raises."""
    c, on = E29_CFG, []
    try:
        E29_RT["supply"] = {"want": bool(c["attempts_supply"]), "path": E28_SCHED_REC.get("path"), "applied": E28_SCHED_REC.get("applied")}
        if c["indexer_fp8"]:
            on.append("idx")
            E29_RT["idx"] = {"want": True, "args": E29L.IDX_ARGS}
        for g_ in ("gdn_fastpath", "gdn_align"):
            if c[g_]:
                on.append(g_)
        E29_RT["gdn"] = {g_: bool(c[g_]) for g_ in ("gdn_fastpath", "gdn_align")}
        if c["draft_vocab"]:
            nm, sh = E29L.VOCAB_FILES[c["draft_vocab"]]
            try:
                q27x_payload(nm, sh)
                E29_RT["vocab"] = {"want": c["draft_vocab"], "file": Q27X_PAYLOAD_USED[nm]["path"], "sha256": sh}
                on.append("vocab")
            except Exception as e:
                E29_RT["vocab"] = {"want": c["draft_vocab"], "error": f"{type(e).__name__}: {e}"[:400]}
                E29_RT["dropped"].append({"lever": "vocab", "why": "setup: " + E29_RT["vocab"]["error"], "try": None})
        if c["drafter"] != "mtp" and e29_df_setup():
            on.insert(0, "df")
            E28_I8.update(on=False, suspended="N/A_DFLASH")   # checklist 2: DRAFT_HEAD never reaches a dflash spec
    except BaseException as e:
        E29_RT["errors"].append("setup: " + traceback.format_exc()[-1200:])
    E29_RT["on"] = on
    NB5_REPORT["e29"] = E29_RT
    log("E29 setup", json.dumps({"on": on, "dropped": E29_RT["dropped"], "i8_on": E28_I8.get("on")}, default=str)[:900])


def e29_env(df):
    """The server env of this try's E29 levers (into NB5_BOOT_LEVERS: the server env, replayed by a relaunch)."""
    env = {}
    if df:
        env.update({str(a): str(b) for a, b in DS.server_env(mask_row_sha=E29_RT["df"]["mask_row_sha"], embed_host=bool(df.get("embed_host")),
                                                             draft_sw_block=E29_RT["df"]["sw_block"]).items()})
    elif "vocab" in E29_RT["on"]:
        env.update(VLLM_DRAFT_VOCAB_FILE=E29_RT["vocab"]["file"], VLLM_DRAFT_VOCAB_SHA256=E29_RT["vocab"]["sha256"])
    for g_, k_ in E29L.GDN_ENV.items():
        if g_ in E29_RT["on"]:
            env[k_] = "1"
    return env


def e29_extra(df):
    """The trailing argv of this try's E29 levers (after --max-num-seqs; the DFlash spec + captures come last: last wins)."""
    return (list(E29L.IDX_ARGS) if "idx" in E29_RT["on"] else []) + (list(E29_RT["df"]["tail"]) if df else [])


def e29_ntries(brec):
    """E18 R2's 3-try cap counts the tries without an E29 lever (the E28 ladder keeps its 3 tries)."""
    return sum(1 for t_ in brec["tries"] if not t_.get("e29_on"))


def e29_df_readback(s):
    """dflash_seam.readback_label on the boot log (NBDF dflash_rb): ENGAGED = aux taps on 4 ranks, embed own + mask row sha,
    head shared, draft_load_check per drafter worker, ingraph_meta consistent."""
    if FAKE:
        return {"label": "FAKE_SERVER"}
    try:
        lab, det = DS.readback_label(s.log() or "", mask_row_sha=E29_RT["df"]["mask_row_sha"],
                                     embed_host=(getattr(s, "env", None) or {}).get("DFLASH_DRAFT_EMBED_HOST") == "1")
    except Exception as e:
        lab, det = "NOT_INTEGRATED", {"error": repr(e)[:300]}
    return {"label": lab, "detail": det, "arch_line": E29L.DF_ARCH in (s.log() or "")}


def e29_counters():
    try:
        return E29L.spec_counters(e21_prom())
    except Exception as e:
        return {"error": repr(e)[:200]}


def e29_inrun_check(rows):
    """The in-run zero-accept canary (checklist 8): DFlash live, >= 200 draft tokens since the phase start and < 2 % accepted
    -> the booted config is switched to MTP K3 (trailing DFlash args + seam env removed) and the server is stopped; the
    E15B watchdog relaunches it with that config (e21_relaunch replays NB5_ACTIVE_BOOT). Fires once."""
    if not E29_RT.get("df_live") or E29_RT.get("inrun_fired"):
        return
    c = E29L.inrun_zero(rows, E29_RT["df"]["k"])
    E29_RT["inrun"] = c
    if not c.get("hit") or FAKE:
        return
    E29_RT["inrun_fired"] = True
    try:
        NB5_ACTIVE_BOOT["extra"] = E29L.strip_df_argv(NB5_ACTIVE_BOOT.get("extra"), E29_RT["df"]["tail"])
        NB5_ACTIVE_BOOT["levers"] = {k_: v_ for k_, v_ in dict(NB5_ACTIVE_BOOT.get("levers") or {}).items() if k_ not in DS.ALL_ENVS}
        e29_drop("df", "IN_RUN DFLASH_ZERO_ACCEPT " + json.dumps(c, default=str)[:300], NB5_ACTIVE_BOOT.get("name"))
        E28_I8["on"] = False                      # the relaunch replays the booted levers (BF16 head: no I8 read-back gate there)
        log("!!!!!!!!!! E29 in-run DFlash zero accept -> the server stops; the watchdog relaunches with MTP K3")
        if E11_SERVER is not None:
            E11_SERVER.kill()
    except Exception as e:
        E29_RT["errors"].append(f"inrun fallback: {e!r}"[:300])
'''


E29_ENGINE_DF2 = r'''

def e29_mtp_boot(why):
    """rev 2c: the relaunch config after DFlash is dropped in-run = the E28 production top rung: MTP K3 argv (the dflash tail
    removed), the top pin, the booted 24 seqs, the seam / QSA env removed, the INT8 head when E28's I8 is on (e29_drop
    restores it)."""
    lv = {k_: v_ for k_, v_ in dict(NB5_ACTIVE_BOOT.get("levers") or {}).items() if k_ not in DS.ALL_ENVS + E29L.QSA_KEYS}
    NB5_ACTIVE_BOOT.update(extra=E29L.strip_df_argv(NB5_ACTIVE_BOOT.get("extra"), E29_RT["df"]["tail"]), kv_bytes=E29L.D_PIN0,
                           pin_gib=K100_PIN_STEPS_GIB[0])
    e29_drop("df", why, NB5_ACTIVE_BOOT.get("name"))
    NB5_ACTIVE_BOOT["levers"] = dict(lv, **(E28_I8_ENV if E28_I8.get("on") else {}))
    E29_RT["mtp_relaunch"] = True           # rev 2d F4: the relaunch runs the E28 boot gates


def e29_qsa_rb(s, want=None):
    """rev 2c / 2d F3: the QSA env of the 4 workers (/proc environ; the worker RPC nb4s_env when /proc gives < 4) + the logged
    granule + the backing bound; want = the env this try asked for."""
    try:
        w_ = E29L.proc_env_readback(s.proc.pid, E29L.QSA_KEYS)
        src = "proc"
        if len(w_) < 4:
            r_ = rpc(s.port, "nb4s_env", ",".join(E29L.QSA_KEYS))
            w_ = [x for x in (r_.get("results") or []) if isinstance(x, dict) and isinstance(x.get("env"), dict)] or w_
            src = "rpc"
        return dict(E29L.qsa_readback(w_, E29L.qsa_env(E29_QSA) if want is None else want, s.log() or ""), source=src)
    except Exception as e:
        return {"error": repr(e)[:200], "env_ok": False, "workers": 0, "label": "UNVERIFIED_QSA"}


def e29_ledger(port):
    """rev 2d F6 / F13 (record only): host cgroup memory + the KV ledger (server gauges, the NB5 reservation, F, the pool)."""
    o = {"host": {k_: v_ for k_, v_ in host_mem().items() if k_ in ("cgroup_cur_mib", "cgroup_max_mib", "avail_mib")}}
    E29_RT.update(host_peak_mib=max(int(E29_RT.get("host_peak_mib") or 0), int(o["host"].get("cgroup_cur_mib") or 0)),
                  host_max_mib=o["host"].get("cgroup_max_mib"))   # rev 2e
    try:
        rr = e21_prom()
        g = lambda n_: sum(float(x.value) for x in rr if x.name == n_)
        o.update(kv_usage=round(max([float(x.value) for x in rr if x.name == "vllm:kv_cache_usage_perc"] or [0.0]), 4),
                 running=int(g("vllm:num_requests_running")), waiting=int(g("vllm:num_requests_waiting")),
                 preemptions=int(g("vllm:num_preemptions_total")))
    except Exception as e:
        o["prom_error"] = repr(e)[:120]
    r_ = globals().get("NB5_RES")
    if r_ is not None:
        o.update(res_held_blocks=r_.used, res_reclaimable=r_.cap - r_.used, res_attempts=len(r_.held), res_waiting=r_.waiting)
    o.update(f=(globals().get("NB5_BLOCKS") or {}).get("f"), pool_blocks=(globals().get("NB5_POOL_CAP") or {}).get("blocks"))
    return o


def e29_prefill_probe(s, trec):
    """rev 2d F2 (every DFlash try, before the phases): ONE forced max-size first prefill (E29L.PROBE_TOKENS, through the
    batched-token chunks), then every worker's own free memory (nb1_cache_info = torch.cuda.mem_get_info in the worker) and
    the ledger. -> None (pass), E29_DFLASH_PROBE_FAILED (dead / error / no token; an OOM line -> the D ladder steps), or
    E29_DFLASH_LOW_FREE (a worker below E29L.PROBE_FREE_MIB -> one ladder step). No RPC = recorded, not gated."""
    o = trec["e29_probe"] = {"tokens": E29L.PROBE_TOKENS}
    try:
        h_, t_ = R["prompt_render"]["head_tokens"], R["prompt_render"]["tail_tokens"]
        p_ = PROMPTS["40k"]
        body = [x for v in range(4) for x in p_[v % len(p_)][h_:len(p_[v % len(p_)]) - t_]]
        ids = p_[0][:h_] + body[:E29L.PROBE_TOKENS - h_ - t_] + p_[0][len(p_[0]) - t_:]
        g = L.Stream(s.port, ids, 8, temperature=1.0, seed=4242)
        t0 = time.time()
        g.start()
        g.join(max(60, min(1800, left() - 600)))
        if g.is_alive():
            g.stop()
            g.join(10)
        o.update(prompt=len(ids), sec=round(time.time() - t0, 1), out_tokens=len(g.tokens), error=g.err)
    except Exception as e:
        o["error"] = f"{type(e).__name__}: {e}"[:300]
    o["alive"] = s.alive()
    if not o["alive"] or E29L.is_oom(s.log() or "") or o.get("error") or not o.get("out_tokens"):
        return "E29_DFLASH_PROBE_FAILED"
    o["workers_free_mib"] = E29L.worker_free_mib(rpc(s.port, "nb1_cache_info"))
    o["ledger"] = e29_ledger(s.port)
    log("E29 DFlash first-prefill probe", json.dumps({k_: o.get(k_) for k_ in ("prompt", "sec", "workers_free_mib")}))
    return "E29_DFLASH_LOW_FREE" if o["workers_free_mib"] and min(o["workers_free_mib"]) < E29L.PROBE_FREE_MIB else None


E29_SEAM_BYTES = {}           # rev 2d F11: the pre-apply bytes of the seam-touched files


def e29_seam_snapshot(tree):
    """rev 2d F11: the pristine bytes + sha256 of every seam-touched file (core + ring + draft-block), before SeamSeq.apply."""
    rels = DS._rels(True, True)
    E29_SEAM_BYTES.clear()
    E29_SEAM_BYTES.update({r_: (Path(tree) / r_).read_bytes() for r_ in rels})
    E29_RT["seam_tree"] = str(tree)
    E29_RT["seam_pristine"] = {r_: hashlib.sha256(b_).hexdigest() for r_, b_ in E29_SEAM_BYTES.items()}


def e29_seam_restore():
    """rev 2d F11 after SSEQ.finish() (also when it raised): every seam-touched file must be back at its pre-apply sha; else
    rewritten from the snapshot and verified again; still not pristine -> seam_unverified (the next MTP boot then needs the
    q_seam_inert read-back: inert -> continue, recorded; not inert -> a real failure of that try). Never raises."""
    snap, tree = E29_RT.get("seam_pristine"), E29_RT.get("seam_tree")
    if not snap or not tree:
        return
    bad = E29L.seam_dirty(tree, snap)
    rec = E29_RT["seam_restore"] = {"dirty_after_finish": bad}
    for r_ in bad:
        try:
            (Path(tree) / r_).write_bytes(E29_SEAM_BYTES[r_])
            DS._drop_pyc(Path(tree) / r_)
        except Exception as e:
            rec.setdefault("errors", []).append(f"{r_}: {e!r}"[:200])
    if bad:
        rec["dirty_after_restore"] = bad = E29L.seam_dirty(tree, snap)
    E29_RT["seam_unverified"] = bool(bad)
    rec["label"] = "SEAM_REVERT_UNVERIFIED" if bad else "SEAM_PRISTINE"


def e29_relaunch_gates(srv, why):
    """rev 2d F4: the in-run DFlash -> MTP relaunch gets the E28 boot gates on the NEW server: C-11, the method map, the I8
    read-back. A failed C-11 / method map = this relaunch failed (the watchdog retries once); the INT8 head not engaged ->
    E28's own I8 rule: the same pin with the BF16 head (the retry boots it). -> (ok, why)."""
    E29_RT["mtp_relaunch"] = False
    try:
        lv = dict(NB5_ACTIVE_BOOT.get("levers") or {})
        cr = NB5L.config_readback(srv.log(), srv.argv, NB5_ACTIVE_BOOT.get("kv") or NB5_BOOT_KV, True)
        kg = k100_boot_gates(srv, True)
        i8 = E28L.i8_readback(srv.log(), srv.argv, lv.get("DRAFT_HEAD"))
        r_ = {"c11": cr.get("ok"), "method_map": kg.get("ok"), "fatal": kg.get("fatal"), "i8": i8.get("label"),
              "draft_head": lv.get("DRAFT_HEAD"), "kv_bytes": NB5_ACTIVE_BOOT.get("kv_bytes")}
        E29_RT.setdefault("mtp_relaunch_gates", []).append(r_)
        if not cr.get("ok") or kg.get("fatal"):
            srv.kill()
            E29_RT["mtp_relaunch"] = True
            return False, f"E29 MTP relaunch gates: C-11 {cr.get('ok')}, method map {kg.get('fatal')}"
        if lv.get("DRAFT_HEAD") == "int8" and i8.get("label") != "I8_ENGAGED":
            E28_I8.update(on=False, dropped={"kind": "READBACK_UNVERIFIED", "try_name": "in-run MTP relaunch", "label": i8.get("label")})
            NB5_ACTIVE_BOOT["levers"] = {k_: v_ for k_, v_ in lv.items() if k_ not in E28_I8_ENV}
            srv.kill()
            E29_RT["mtp_relaunch"] = True
            return False, "E29 MTP relaunch: INT8 head not engaged -> the same pin, BF16 head"
        return True, why
    except Exception as e:
        E29_RT["errors"].append(f"relaunch gates: {e!r}"[:300])
        return True, why


def e29_relaunch_plan(old):
    """E29 rev 2 (in-run; e21_relaunch): a DFlash server that died of an OOM (not a device fault) is relaunched on the next
    D rung (EMBED_HOST kept); none left -> the proven MTP K3 config, top rung (DFlash dropped, seam reverted, BF16 head as
    the in-run canary path). Any other death: E15B's relaunch of the same config."""
    if not E29_RT.get("df_live") or "df" not in E29_RT["on"] or not NB5_ACTIVE_BOOT or FAKE:
        return
    try:
        lt = old.log() or ""
        if E29L.is_device_fault(lt):                    # rev 2d F10: recorded GPU_DEVICE_FAULT -> MTP
            E29_RT.setdefault("gpu_device_fault", []).append({"try": NB5_ACTIVE_BOOT.get("name"), "t_sec": round(time.perf_counter() - NOTEBOOK_T0),
                                                              "lines": [l_[:200] for l_ in lt.splitlines() if E29L.XID.search(l_)][:4]})
        p = E29L.df_relaunch_plan(E29_RT.get("df_cur"), lt, E29L.df_tries(bool(E29_CFG["embed_host"])))
        E29_RT.setdefault("relaunch_plans", []).append({"from": NB5_ACTIVE_BOOT.get("name"), "plan": p if p in (None, "mtp") else p["df"]})
        if p is None:
            return
        lv = {k_: v_ for k_, v_ in dict(NB5_ACTIVE_BOOT.get("levers") or {}).items() if k_ not in DS.ALL_ENVS + E29L.QSA_KEYS}
        if p == "mtp":
            e29_mtp_boot("IN_RUN GPU_DEVICE_FAULT" if E29L.is_device_fault(lt) else "IN_RUN OOM: no D rung left")
        else:
            NB5_ACTIVE_BOOT.update(kv_bytes=int(p["kv_bytes"]), pin_gib=p["gib"], levers=dict(lv, **e29_env(p["df"])))
            E29_RT["df_cur"] = p["df"]
        log("!!!!!!!!!! E29 in-run DFlash OOM -> relaunch", json.dumps(E29_RT["relaunch_plans"][-1], default=str))
    except Exception as e:
        E29_RT["errors"].append(f"relaunch plan: {e!r}"[:300])
'''


E29_LIVE_FNS = r"""


# ==================================================================== E29 rev 2b: LIVE progress lines (print only, user 2026-11-02)
def e29_live_setup():
    # E12A.solve_all_r6 wrapped (looked up at call time by e21_cell6): on_branch_done ALSO prints one LIVE line per finished
    # attempt (e29_lib.LiveLog; after the original callback, every LIVE exception caught), and a [LIVE 15-min] line from the
    # server counters during the solve. Records, submission, scheduler and solver are untouched.
    if getattr(E12A, "_e29_live_orig", None) is not None:
        return
    orig = E12A._e29_live_orig = E12A.solve_all_r6
    E29_RT["live"] = {"lines": 0, "errors": 0}

    def live_solve(tasks, *a, on_branch_done=None, **k):
        lv, stop = None, threading.Event()
        try:
            lv = LiveLog(tasks, globals().get("SOLUTIONS"), lambda t_, task_, rs_: E12A.build_submission({t_: task_}, {t_: rs_})[t_],
                              getattr(E12A, "_e28_cand", None), NOTEBOOK_T0)
        except Exception:
            E29_RT["live"]["errors"] += 1

        def cb(rec):
            if on_branch_done is not None:
                on_branch_done(rec)
            try:                                        # rev 2d F12: the file after the writer callback (record only)
                E29_RT["submission_check"] = E29L.submission_check(globals().get("SUBMISSION_PATH"), globals().get("test_challenges") or tasks)   # rev 2f
            except Exception:
                E29_RT["live"]["errors"] += 1
            try:
                print(lv.on_rec(rec), flush=True)
                E29_RT["live"]["lines"] += 1
            except Exception:
                E29_RT["live"]["errors"] += 1

        def tick():
            g0 = None
            while not stop.wait(900):
                try:
                    rr = e21_prom()
                    g = lambda n_: sum(float(x.value) for x in rr if x.name == n_)
                    g1 = (time.perf_counter(), g("vllm:generation_tokens_total"))
                    ts = (g1[1] - g0[1]) / max(1e-6, g1[0] - g0[0]) if g0 else None
                    g0 = g1
                    sc_ = E29_RT.get("submission_check") or {}
                    print(lv.summary(ts, int(g("vllm:num_requests_running")), " | algo " + json.dumps(E29L.effective(E29_RT, globals().get("RERUN")))
                                     + " | ledger " + json.dumps(e29_ledger(int(NB5_PORT))) + f" | submission ok {sc_.get('ok')} missing "
                                     f"{sc_.get('n_missing')}"), flush=True)
                except Exception:
                    E29_RT["live"]["errors"] += 1
        if lv is not None:
            try:
                print("[LIVE start] algo", json.dumps(E29L.effective(E29_RT, globals().get("RERUN"))), flush=True)   # rev 2d F5
            except Exception:
                E29_RT["live"]["errors"] += 1
            threading.Thread(target=tick, daemon=True, name="e29_live").start()
        try:
            return orig(tasks, *a, on_branch_done=(cb if lv is not None else on_branch_done), **k)
        finally:
            stop.set()
    E12A.solve_all_r6 = live_solve
"""

E29_EXTRAS_FN = r"""


def e29_extras():
    # E29 rev 2c (the 400-kB commit limit): the DIRECT + LIVE notebook functions = e29_nbx.py of the payload dataset (name +
    # sha, E29D_FILES), executed into this namespace, then e29_direct_setup() / e29_live_setup(). A miss = DIRECT off (phase C
    # = the E28 Duck attempt 2) and no LIVE lines, recorded; never raises.
    try:
        exec(compile(q27x_payload(*E29D_FILES["nbx"]).decode("utf-8"), "e29_nbx.py", "exec"), globals())
        globals()["e29_direct_setup"]()
        globals()["e29_live_setup"]()
    except Exception as e:
        os.environ.pop("E29_DIRECT", None)
        try:
            E12A.E29D_RUN = E12A.E29D_CAN_START = None
        except Exception:
            pass
        E29_RT["extras_error"] = f"{type(e).__name__}: {e}"[:600]
        log("!!!!!!!!!! E29 extras (DIRECT / LIVE) off:", E29_RT["extras_error"][:300])
"""

E29_DIRECT_FNS = r'''


# ==================================================================== E29 rev 2: DIRECT = the second branch (attempt_2), user 2026-11-02
# The CELL 1 switch E29_DIRECT2 ('on' False = E29_DFLASH_D3S without DIRECT: the DIRECT diff is never applied). CELL 3
# applied the DIRECT diff on arc2_duck.py (e29d_sched_patch.py, payload dataset); here: the solver e29_direct.py (= E31's
# e31_direct.py + N1 + task_index, payload dataset, name + sha) and the hook E12A.E29D_RUN: each phase-C item of the E28
# scheduler (C's order unchanged) = ONE DIRECT request on THIS server in that worker slot, recorded as branch 1. Any miss =
# DIRECT off (phase C = the E28 Duck attempt 2, today's attempt_2 rule), recorded, never raises.
E29_DIRECT_RT = {"cfg": E29_DIRECT2, "state": None, "hook": False, "calls": 0, "errors": [], "tokenizer": None, "tasks": {}}
E29D, _E29D_TOK, _E29D_INDEX = None, {}, {}


def e29d_tok():
    """= E31 CELL 7b e31_tok (cached): CELL 8's NB5_LOCAL_TOK (pinned klee100 tokenizer / template hashes verified), ORDER2's
    render (enable_thinking True, reasoning_effort xhigh) and decode; EOS ids from the checkpoint's generation_config.json."""
    if _E29D_TOK:
        return _E29D_TOK
    if not NB5_LOCAL_TOK.get("verified"):
        raise RuntimeError("E29 DIRECT: the Flash-Next tokenizer / template are not the pinned ones: " + str(NB5_LOCAL_TOK.get("why")))
    _hf, _tp = NB5_LOCAL_TOK["hf"], NB5_LOCAL_TOK["template"]
    _gp = next((p_ for p_ in (CKPT / "generation_config.json", Path(str(globals().get("K100_DIR") or "/nonexistent")) / "generation_config.json")
                if p_.is_file()), None)
    _eos = (json.loads(_gp.read_text(encoding="utf-8")).get("eos_token_id") if _gp is not None else None) or []
    t_ = {"render": lambda m_: _hf.apply_chat_template(m_, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                                      reasoning_effort="xhigh", chat_template=_tp),
          "encode": lambda s_: TOK.encode(s_, add_special_tokens=False).ids,
          "decode": lambda i_, k_: TOK.decode(list(i_), skip_special_tokens=bool(k_)),
          "eos_ids": _eos if isinstance(_eos, list) else [_eos]}
    E29_DIRECT_RT["tokenizer"] = {"generation_config": str(_gp), "eos_ids": t_["eos_ids"],
                                  "stop_ids_equal_spec": t_["encode"](E29D.STOP_THINKING_TEXT) == E29D.FN_STOP_IDS,
                                  "think_close_ids": t_["encode"]("</think>")}
    _E29D_TOK.update(t_)
    return _E29D_TOK


def e29_direct_run(tid, task, plan, deadline, base_url, model_id, out_dir):
    """E12A.E29D_RUN (arc2_duck's E29 DIRECT block, one call per phase-C item): e29_direct.solve_all_direct on this ONE task
    (workers 1, the scheduler's deadline: no admission after T - drain_min, drain, the hard stop at T - 1 min), the client
    on THIS booted server only (SGL_URL; the E15B watchdog owns the server); -> its record as branch 1, or None."""
    _srv = str(getattr(E11_SERVER, "url", "") or "").rstrip("/")
    assert SGL_URL and SGL_URL.rstrip("/") == _srv and str(base_url).rstrip("/") == SGL_URL.rstrip("/") + "/v1", (
        "E29 DIRECT: base_url != the booted server", SGL_URL, _srv, base_url)
    E29_DIRECT_RT["calls"] += 1
    if not _E29D_INDEX:
        _E29D_INDEX.update({t_: i_ for i_, t_ in enumerate(sorted(globals().get("TASK_IDS") or [tid]))})
    _cfg = {k_: E29_DIRECT2[k_] for k_ in ("think", "answer", "temperature", "drain_min")}
    out = E29D.solve_all_direct({tid: task}, [tid], out_dir=out_dir, deadline=deadline, base_url=SGL_URL, model_id=model_id,
                                workers=1, cfg=dict(_cfg, workers=1), tok=e29d_tok(), solutions=globals().get("SOLUTIONS"),
                                report_dir=Path(out_dir) / "reports" / tid, server_down=E12A.SERVER_DOWN, t0_pc=NOTEBOOK_T0,
                                harness=bool(E11_HARNESS), task_index=_E29D_INDEX)
    rs = out.get(tid) or []
    x = (rs[-1].get("e31") if rs else None) or {}
    E29_DIRECT_RT["tasks"][tid] = dict({k_: x.get(k_) for k_ in ("mode", "end_reason", "think_tokens", "p2_tokens", "p2_kept",
                                                              "mapping_source", "has_grid", "score", "p1_cap")},
                                       record=bool(rs), c_priority=plan.get("c_priority"))
    return dict(rs[-1], branch=1, e29_direct=True) if rs else None


def e29_direct_setup():
    """After the E29 setup: DIRECT on only when the switch is on AND CELL 3's DIRECT diff is applied; then the solver from
    the payload dataset and the hook. Else DIRECT off (E29_DIRECT env removed), recorded."""
    global E29D
    r_ = E29_DIRECT_RT
    E29_RT["direct"] = r_
    r_["sched"] = {k_: v_ for k_, v_ in (globals().get("E29D_SCHED_REC") or {}).items() if k_ != "text"}
    try:
        os.environ.pop("E29_DIRECT_PHASE", None)
        if not E29_DIRECT2.get("on"):
            E12A.E29D_RUN = E12A.E29D_CAN_START = None      # rev 2d F1: OFF = no hook
            r_["state"] = "OFF_SWITCH" + (" (invalid: " + str(E29_DIRECT2.get("invalid")) + ")" if E29_DIRECT2.get("invalid") else "")
            return
        if not r_["sched"].get("ok"):
            raise RuntimeError("the DIRECT scheduler diff is not applied: " + str(r_["sched"].get("error")))
        (TMP / E29D_FILES["direct"][0]).write_bytes(q27x_payload(*E29D_FILES["direct"]))
        import e29_direct as _d
        E29D = _d
        E12A.E29D_RUN = e29_direct_run
        E12A.E29D_CAN_START = lambda deadline: time.monotonic() + 60.0 + float(E29_DIRECT2["drain_min"]) * 60.0 < deadline   # rev 2d F1
        os.environ["E29_DIRECT_PHASE"] = str(E29_DIRECT2.get("phase") or "C1")   # rev 2f: the DIRECT scope (C1 = train-fail C only)
        r_.update(state="ON", hook=True, lib=_d.LIB_VERSION, env=os.environ.get("E29_DIRECT"), phase=os.environ["E29_DIRECT_PHASE"])
    except Exception as e:
        os.environ.pop("E29_DIRECT", None)
        try:
            E12A.E29D_RUN = E12A.E29D_CAN_START = None
        except Exception:
            pass
        r_.update(state="OFF_SETUP", error=f"{type(e).__name__}: {e}"[:600])
        log("!!!!!!!!!! E29 DIRECT off (phase C = the E28 Duck attempt 2):", r_["error"][:300])
    log("E29 DIRECT", json.dumps({k_: r_.get(k_) for k_ in ("state", "hook", "lib", "env", "phase")}, default=str))
'''

PROD_PROFILE, SMOKE_PROFILE = "E28EVAL120", "E29SMOKE"
PROFILE_LINE = 'E29_PROFILE = "{}"   # E29 rev 2e: the commit profile (E28EVAL120 = production 6 h EVAL120; E29SMOKE = the ~70-min GPU smoke)\n'
SMOKE_TASKS = ["1818057f", "2ba387bc", "45a5af55", "bf45cf4b",       # E28 K100 solved, single test, short A (3.9k-9.3k tokens)
               "446ef5d2", "b6f77b65", "7b5033c1", "800d221b",       # K100 failed: 2 / 3 tests (B2 cap cuts) + 2 single (accepted, wrong)
               "31f7f899", "d59b0160", "409aa875", "a47bf94d"]       # mixed single test (solved, solved, failed, CONTEXT_EXHAUSTED -> B2)
SMOKE_WINDOW_MIN, SMOKE_BUDGET_MIN, SMOKE_HARD_MARGIN_MIN = 40, 70, 5
# build-time window math (the E28 fixed-window rule: room = HARD_STOP - E12_STOP_BEFORE_HARD 3 min - 1 min - the boot end):
# a boot that ends by 21 min keeps the full 40-min window; 10 min stay for the reports before HARD_STOP (65), the deadman at 70
assert SMOKE_BUDGET_MIN - SMOKE_HARD_MARGIN_MIN - 3 - 1 - SMOKE_WINDOW_MIN >= 18 and len(SMOKE_TASKS) == len(set(SMOKE_TASKS)) == 12
E29_SMOKE_LINE = ("E29_SMOKE = " + repr({"tasks": sorted(SMOKE_TASKS), "window_min": SMOKE_WINDOW_MIN, "budget_min": SMOKE_BUDGET_MIN})
                  + "   # E29 rev 2e: the E29SMOKE profile (eval, 12 tasks, plain sorted; HARD_STOP = budget - 5, deadman = budget)\n")
E29_SMOKE_CELL1 = [
    ("K100_DEADMAN_SEC = 24900   # E28 K100: HARD_STOP (410 min) + 5 min\n",
     "K100_DEADMAN_SEC = 24900   # E28 K100: HARD_STOP (410 min) + 5 min\n"
     "K100_DEADMAN_SEC = int(E29_SMOKE['budget_min'] * 60) if E29_PROFILE == 'E29SMOKE' else K100_DEADMAN_SEC   # E29 rev 2e\n"),
    ('NB5_DEFAULT_PROFILE = "E28EVAL120"   # E28 K100\n',
     'NB5_DEFAULT_PROFILE = "E28EVAL120"   # E28 K100\n'
     'NB5_DEFAULT_PROFILE = E29_PROFILE   # E29 rev 2e: the CELL 1 profile line\n'),
    ('K100_PROFILES = ("E28EVAL120", "RERUN_PROD")   # E28 K100\n',
     'K100_PROFILES = ("E28EVAL120", "RERUN_PROD", "E29SMOKE")   # E28 K100; E29 rev 2e: + the GPU smoke\n'),
    ('PROFILE_TABLE["RERUN_PROD"] = dict(PROFILE_TABLE["RERUN_PROD"], **E28_LEVERS)',
     'PROFILE_TABLE["E29SMOKE"] = dict(PROFILE_TABLE["E28EVAL120"], time_budget_sec=E29_SMOKE["budget_min"] * 60, n_tasks=len(E29_SMOKE["tasks"]),   # E29 rev 2e\n'
     '                                 task_list=list(E29_SMOKE["tasks"]), fixed_window_sec=E29_SMOKE["window_min"] * 60)\n'
     'assert E29_SMOKE["budget_min"] - 5 - 3 - 1 - E29_SMOKE["window_min"] >= 18   # E29 rev 2e: a boot ending by 21 min keeps the full window\n'
     'PROFILE_TABLE["RERUN_PROD"] = dict(PROFILE_TABLE["RERUN_PROD"], **E28_LEVERS)'),
]
E29_DIRECT_SWITCH = ("E29_DIRECT2 = {'on': True, 'think': 104000, 'answer': 8192, 'temperature': 1.0, 'drain_min': 26, 'phase': 'C1'}"
                     "   # E29 rev 2: DIRECT = the 2nd branch (attempt_2); 'on': False = no DIRECT; 'phase': 'C1' = DIRECT only on"
                     " train-fail C items (Duck C on train-pass), 'C' = every C item (edit freely, no rebuild)\n"
                     "E29_QSA = {'max_logits_mb': 128, 'granule_mb': 64}   # E29 rev 2c: QSA prefill workspace, DFlash tries (Merlin 256/256)\n"
                     + PROFILE_LINE.format(PROD_PROFILE) + E29_SMOKE_LINE)
E29_DIRECT_CELL1 = r'''E29D_FILES = __FILES__   # E29 rev 2: DIRECT files of the payload dataset (name, sha256)
E29_DIRECT2 = E29L.direct2_cfg(E29_DIRECT2, os.environ.get("E29_DIRECT2_JSON") if os.environ.get("E11_HARNESS") == "1" else None)
E29_QSA = E29L.qsa_cfg(E29_QSA)
print("E29 DIRECT2", json.dumps(E29_DIRECT2), "| QSA", json.dumps(E29_QSA), flush=True)
'''
E29_DIRECT_CELL3 = r"""# ---- E29 rev 2 DIRECT (switch E29_DIRECT2): the DIRECT diff on arc2_duck.py (e29d_sched_patch.cell3; a miss = no DIRECT)
E29D_SCHED_REC = {"want": bool(E29_DIRECT2.get("on")), "ok": False, "path": E28_SCHED_REC.get("path")}
os.environ.pop("E29_DIRECT", None)
try:
    if E29D_SCHED_REC["want"]:
        (TMP / E29D_FILES["sched"][0]).write_bytes(q27x_payload(*E29D_FILES["sched"]))
        import e29d_sched_patch as E29DS
        E29D_SCHED_REC = E29DS.cell3(E12_DIR, E28_SCHED_REC, E29D_SCHED_REC)
except Exception as _e:
    E29D_SCHED_REC["error"] = f"{type(_e).__name__}: {_e}"[:600]
CONFIG["e29_direct_sched"] = E29D_SCHED_REC
print("E29 DIRECT sched", json.dumps({k: E29D_SCHED_REC.get(k) for k in ("want", "ok", "path", "error")}), flush=True)
"""


def engine_df2(c):
    c = once(c, '            ok, why = SSEQ.apply()\n', '            e29_seam_snapshot(tree)   # E29 rev 2d F11\n            ok, why = SSEQ.apply()\n', "F11 snap")
    c = once(c, '                E29_RT["errors"].append(f"seam finish: {e!r}"[:300])\n        if E28_I8.get("want") and not E28_I8.get("dropped"):\n',
             '                E29_RT["errors"].append(f"seam finish: {e!r}"[:300])\n            e29_seam_restore()   # E29 rev 2d F11\n'
             '        if E28_I8.get("want") and not E28_I8.get("dropped"):\n', "F11 restore")
    c = once(c, '                E29_RT["errors"].append(f"seam finish: {e2!r}"[:300])\n',
             '                E29_RT["errors"].append(f"seam finish: {e2!r}"[:300])\n            e29_seam_restore()   # E29 rev 2d F11\n', "F11 restore setup")
    c = once(c, '                trec["status"] = "E29_DFLASH_NOT_ENGAGED"\n',
             '                trec["status"] = "E29_QSA_ENV_MISMATCH" if trec["e29_df_readback"]["label"] == "QSA_ENV_MISMATCH" else "E29_DFLASH_NOT_ENGAGED"\n', "F3 status")
    c = once(c, '        trec["i8_readback"] = ({"label": "FAKE_SERVER"',
             '        if E29_RT.get("seam_unverified") and not E29_RT.get("df_try") and not FAKE:   # E29 rev 2d F11\n'
             '            trec["e29_seam_inert"] = E29L.seam_inert(s.log() or "")\n'
             '            if not trec["e29_seam_inert"]["ok"]:\n'
             '                trec["status"] = "E29_SEAM_NOT_INERT"\n'
             '                return {}\n'
             '        trec["i8_readback"] = ({"label": "FAKE_SERVER"', "F11 inert")
    c = once(c, '            E29_RT["df_live"] = True\n',
             '            E29_RT["df_live"] = True\n'
             '            _e29p = None if FAKE else e29_prefill_probe(s, trec)   # E29 rev 2d F2\n'
             '            if _e29p:\n'
             '                trec["status"] = _e29p\n'
             '                return {}\n', "F2 probe")
    c = once(c, '        trec["ladder"] = fail\n',
             '        fail["failed"] = fail["failed"] or str(trec.get("status")) == "E29_SEAM_NOT_INERT"   # E29 rev 2d F11\n'
             '        trec["ladder"] = fail\n', "F11 fail")
    c = once(c, '        ok = bool(ok and sc and sc.get("ok"))\n',
             '        ok = bool(ok and sc and sc.get("ok"))\n'
             '        if ok and E29_RT.get("mtp_relaunch"):   # E29 rev 2d F4\n'
             '            ok, why = e29_relaunch_gates(srv, why)\n'
             '            if not ok:\n'
             '                NB5_SERVERS[NB5_SERVERS.index(srv)] = old\n', "F4 gates")
    c = once(c, '        env.update({str(a): str(b) for a, b in DS.server_env(mask_row_sha=E29_RT["df"]["mask_row_sha"], embed_host=bool(df.get("embed_host")),\n'
                '                                                             draft_sw_block=E29_RT["df"]["sw_block"]).items()})\n',
             '        env.update({str(a): str(b) for a, b in DS.server_env(mask_row_sha=E29_RT["df"]["mask_row_sha"], embed_host=bool(df.get("embed_host")),\n'
             '                                                             draft_sw_block=E29_RT["df"]["sw_block"]).items()})\n'
             '        env.update({} if E29_RT.get("qsa_off") else E29L.qsa_env(E29_QSA))   # E29 rev 2c / 2d: the QSA workspace (DFlash tries)\n', "E11 qsa env")
    c = once(c, '        NB5_ACTIVE_BOOT["extra"] = E29L.strip_df_argv(NB5_ACTIVE_BOOT.get("extra"), E29_RT["df"]["tail"])\n'
                '        NB5_ACTIVE_BOOT["levers"] = {k_: v_ for k_, v_ in dict(NB5_ACTIVE_BOOT.get("levers") or {}).items() if k_ not in DS.ALL_ENVS}\n'
                '        e29_drop("df", "IN_RUN DFLASH_ZERO_ACCEPT " + json.dumps(c, default=str)[:300], NB5_ACTIVE_BOOT.get("name"))\n'
                '        E28_I8["on"] = False                      # the relaunch replays the booted levers (BF16 head: no I8 read-back gate there)\n',
             '        e29_mtp_boot("IN_RUN DFLASH_ZERO_ACCEPT " + json.dumps(c, default=str)[:300])   # E29 rev 2c: the E28 top rung + I8\n',
             "E11 inrun mtp")
    c = once(c, '                    e29_drop("df", f"{_e29s} (oom {_e29o})", name)   # -> the proven MTP K3 config, the top rung\n',
             '                    e29_drop("df", f"{_e29s} (oom {_e29o})", name)   # -> the proven MTP K3 config, the top rung\n'
             '                    _e29n = E29L.mtp_top(tries, _e29n, K100_PIN_STEPS_GIB[0])   # E29 rev 2c: the E28 top rung NEXT\n', "E11 top")
    """E10 (rev 2, the DFlash notebook only; the E25 NB-S1 GPU facts): the seam pin text, the GRD check after the seam apply
    and in the boot read-back, the booted D rung, the warm-up gate + the pre-phase death gate, an OOM at any stage of a D try
    (not XID) -> the next D rung before any lever naming, the in-run OOM relaunch plan."""
    c = once(c, "    embed tokens, mask row 248077), dflash_seam.py (sha 7356d473, from the E29 payload dataset) applied with SeamSeq\n",
             "    embed tokens, mask row 248077), dflash_seam.py (E29 rev 2: sha 341d859b + the GRD edit, E29 payload) applied with SeamSeq\n",
             "E10 doc")
    c = once(c, '            if not ok:\n                raise RuntimeError("SEAM: " + str(why))\n',
             '            if not ok:\n                raise RuntimeError("SEAM: " + str(why))\n'
             '            rec["tree"], E29_RT["seam"]["grd"] = str(tree), E29L.grd_check(DS.verify_dflash_patch(tree, ring=SSEQ.ring))   # E29 rev 2\n'
             '            if not E29_RT["seam"]["grd"]:          # E29 rev 2\n'
             '                raise RuntimeError("SEAM_GRD_NOT_APPLIED: " + E29L.GRD_REL)\n', "E10 grd setup")
    c = once(c, '        lab, det = "NOT_INTEGRATED", {"error": repr(e)[:300]}\n    return {"label": lab, "detail": det, "arch_line": E29L.DF_ARCH in (s.log() or "")}\n',
             '        lab, det = "NOT_INTEGRATED", {"error": repr(e)[:300]}\n'
             '    try:                                     # E29 rev 2: + the GRD file + SeamSeq\n'
             '        g_ = E29L.grd_check(DS.verify_dflash_patch(E29_RT["df"]["tree"], ring=SSEQ.ring)) and SSEQ.check(s.name)[0]\n'
             '    except Exception as e:\n'
             '        g_, det = False, dict(det, grd_error=repr(e)[:300])\n'
             '    if lab == "ENGAGED" and not g_:\n'
             '        lab = "NOT_INTEGRATED_GRD"\n'
             '    want = {} if E29_RT.get("qsa_off") else E29L.qsa_env(E29_QSA)   # E29 rev 2d F3: required before the first prefill\n'
             '    q_ = e29_qsa_rb(s, want)\n'
             '    E29_RT["qsa_state"] = "QSA_OFF_MERLIN_256_256" if not want else q_.get("label")   # F14\n'
             '    if want and lab == "ENGAGED" and not q_.get("env_ok"):\n'
             '        lab = "QSA_ENV_MISMATCH"\n'
             '    return {"label": lab, "detail": det, "arch_line": E29L.DF_ARCH in (s.log() or ""), "grd": g_,\n'
             '            "grd_log_line": E29L.GRD_LOG in (s.log() or ""), "qsa": q_, "ledger": e29_ledger(s.port)}   # E29 rev 2 / 2d\n', "E10 grd readback")
    c = once(c, '        E29_RT["df_try"] = bool(_e29d)\n',
             '        E29_RT["df_try"], E29_RT["df_cur"] = bool(_e29d), _e29d   # E29 rev 2\n', "E10 df_cur")
    c = once(c, '            _e29l = E29L.lever_to_drop(trec["e29_on"], _e28_log, _e29s)\n            if _e29l == "df":\n'
                '                _e29o = E29L.is_oom(_e28_log) and _e29s not in ("E29_DFLASH_NOT_ENGAGED", "E29_DFLASH_ZERO_ACCEPT", "E29_DFLASH_CONFIG_MISMATCH")\n',
             '            if _e29s == "E29_QSA_ENV_MISMATCH" and not E29_RT.get("qsa_off"):   # E29 rev 2d F3: same rung, Merlin 256/256\n'
             '                E29_RT["qsa_off"] = True\n'
             '                continue\n'
             '            _e29o = bool(trec.get("e29_df")) and (E29L.df_oom_step(_e29s, _e28_log) or _e29s == "E29_DFLASH_LOW_FREE")   # rev 2 / 2d F2\n'
             '            _e29l = "df" if _e29o else E29L.lever_to_drop(trec["e29_on"], _e28_log, _e29s)\n            if _e29l == "df":\n', "E10 oom step")
    c = once(c, '        trec["warmup"] = nb5_warmup(s.port, W)\n        if E29_RT.get("df_try"):\n',
             '        trec["warmup"] = nb5_warmup(s.port, W)\n        if E29_RT.get("df_try"):\n'
             '            trec["e29_warmup_gate"] = E29L.df_warmup_failure(trec["warmup"], s.alive())   # E29 rev 2\n'
             '            trec["e29_qsa"] = e29_qsa_rb(s)   # E29 rev 2c (record)\n'
             '            if trec["e29_warmup_gate"] and not FAKE:\n'
             '                trec["status"] = "E29_DFLASH_WARMUP_FAILED"\n'
             '                return {}\n', "E10 warmup gate")
    c = once(c, '        trec["counters_after_warmup"] = e21_counters()\n        trec["phases_started"] = True\n',
             '        if E29_RT.get("df_try") and not s.alive() and not FAKE:   # E29 rev 2\n'
             '            trec["status"] = "E29_DFLASH_DIED_BEFORE_PHASES"\n'
             '            return {}\n'
             '        trec["counters_after_warmup"] = e21_counters()\n        trec["phases_started"] = True\n', "E10 pre-phase death")
    c = once(c, "        global NB5_BOOT_EXTRA, NB5_BOOT_LEVERS\n        _ab = dict(NB5_ACTIVE_BOOT)",
             "        global NB5_BOOT_EXTRA, NB5_BOOT_LEVERS\n        e29_relaunch_plan(old)   # E29 rev 2: an in-run DFlash OOM -> the next D rung / MTP K3\n"
             "        _ab = dict(NB5_ACTIVE_BOOT)", "E10 relaunch hook")
    return c


def engine_block(df2):
    """The E29 engine block inserted after E28's engine anchor; rev 2 (the DFlash notebook): + the DF2 / DIRECT functions and
    e29_setup -> e29_direct_setup() (this block is packed in that notebook: build() / pack_engine)."""
    blk = E29_ENGINE.replace("__DRAFTER_FNS__", DRAFTER_FNS.rstrip("\n"))
    if df2:
        blk = once(blk, '    NB5_REPORT["e29"] = E29_RT\n',
                   '    NB5_REPORT["e29"] = E29_RT\n    e29_extras()   # E29 rev 2c: DIRECT + LIVE (payload e29_nbx.py); a miss = both off\n'
                   '    E29_RT["effective_algorithm"] = E29L.effective(E29_RT, RERUN)   # E29 rev 2d F5\n'
                   '    print("E29 EFFECTIVE ALGORITHM", json.dumps(E29_RT["effective_algorithm"]), flush=True)\n', "E10 setup")
        blk += E29_ENGINE_DF2 + E29_EXTRAS_FN
    return blk


ENGINE_PACK_HEAD = "\n\n\n# ---- E29 FINAL rev 2 (size: the 400-kB commit limit): the E29 engine block"
ENGINE_PACK = ENGINE_PACK_HEAD + r''' (levers, DFlash gates, DIRECT branch) as ONE sha-pinned
# lzma + base85 literal executed into this namespace (= the e21_cell6 / e21_cell7 technique, the same definitions in the same
# place); the readable text is in Build/e29_dflash_d3s_code_export.py.
exec(compile(__PACKED__, "<E29: the E29 engine block>", "exec"), globals())
'''


BLOCK_START = "\n\n\n# ==================================================================== E29 FINAL: the switchable levers"
BLOCK_END = '        log("!!!!!!!!!! E29 extras (DIRECT / LIVE) off:", E29_RT["extras_error"][:300])\n'


def block_of(c):
    """The final (all edits applied) E29 engine block of a DFlash engine cell."""
    assert c.count(BLOCK_START) == 1 and c.count(BLOCK_END) == 1
    i = c.index(BLOCK_START)
    return c[i:c.index(BLOCK_END) + len(BLOCK_END)]


def pack_engine(c, blk=None):
    """The DFlash notebook: the E29 engine block -> ONE packed literal (exact inverse: unpack_engine)."""
    blk = blk if blk is not None else block_of(c)
    assert c.count(blk) == 1
    out = c.replace(blk, ENGINE_PACK.replace("__PACKED__", E28B.packed_lzma(blk, "e29_engine")))
    out, pd = pack_donor(out)                            # rev 2d F9: + the E21 donor block
    assert unpack_engine(out) == c
    return out, {"sha256": sha(blk), "bytes": len(blk.encode()), "donor": pd}


def unpack_engine(c):
    """The readable engine cell (kits / tests use the same rule): EVERY packed E29 block back in place (rev 2d: 2 blocks)."""
    while ENGINE_PACK_HEAD in c:
        h = c.index(ENGINE_PACK_HEAD)
        x = c.index('exec(compile(', h)
        e = c.index("\n", x) + 1
        call = next(n for n in ast.walk(ast.parse(c[x:e])) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "exec")
        expr = ast.get_source_segment(c[x:e], call.args[0].args[0])
        c = c[:h] + eval(compile(ast.Expression(ast.parse(expr, mode="eval").body), "<e29 packed>", "eval")) + c[e:]
    return c


DONOR_START, DONOR_END = "\n\n\n\n# ---- E21 engine functions\n", "# ==================================================================== NB5 glue (new)"
ENGINE_PACK2 = ENGINE_PACK_HEAD + r''' (part 2, rev 2d: the E21 donor functions, verbatim) as ONE sha-pinned
# lzma + base85 literal executed into this namespace; the readable text is in Build/e29_dflash_d3s_code_export.py.
exec(compile(__PACKED__, "<E29: the E21 donor block>", "exec"), globals())
'''


def pack_donor(c):
    """rev 2d (F9, the 400-kB limit): the verbatim E21 donor functions (no test anchor) -> a second packed literal."""
    i, j = c.index(DONOR_START), c.index(DONOR_END)
    assert c.count(DONOR_START) == 1 and c.count(DONOR_END) == 1 and i < j
    blk = c[i:j]
    out = c[:i] + ENGINE_PACK2.replace("__PACKED__", E28B.packed_lzma(blk, "e21_donor")) + c[j:]
    return out, {"sha256": sha(blk), "bytes": len(blk.encode())}


def engine(c, df2=False):
    anchor = E28B.E28_ENGINE
    assert c.count(anchor) == 1
    c = c.replace(anchor, anchor + engine_block(df2))
    # E5: nb5_boot_try
    c = once(c, "    srv = merlin_fn_server(name, NB5_BOOT_KV, NB5_PORT, kv_bytes)\n",
             "    try:\n"
             "        srv = merlin_fn_server(name, NB5_BOOT_KV, NB5_PORT, kv_bytes)\n"
             "    except Exception as _e29e:                         # E29: with an E29 lever on, a launch-prep raise (e.g. FAIL_CLI_FLAGS)\n"
             "        if not trec.get(\"e29_on\"):                    # = a failed try (the lever is dropped); without one: E28 (raise)\n"
             "            raise\n"
             "        trec.update(status=\"E29_LAUNCH_PREP_FAILED\", e29_error=f\"{type(_e29e).__name__}: {_e29e}\"[:800],\n"
             "                    log_text_for_ladder=f\"{type(_e29e).__name__}: {_e29e}\")\n"
             "        return trec\n", "E5 launch prep")
    c = once(c, '        trec["config_readback"] = NB5L.config_readback(s.log(), s.argv, NB5_BOOT_KV, mtp_on)    # C-11\n',
             '        trec["config_readback"] = NB5L.config_readback(s.log(), s.argv, NB5_BOOT_KV, mtp_on)    # C-11\n'
             '        E29_RT["df_live"] = False\n'
             '        if E29_RT.get("df_try"):                             # E29 DFlash: C-11 without the MTP checks + the DFlash facts,\n'
             '            trec["config_readback"] = E29L.df_config_check(trec["config_readback"], s.log(), E29_RT["df"]["k"])\n'
             '            trec["e29_df_readback"] = e29_df_readback(s)     # then the seam read-back (ENGAGED required, NBDF)\n'
             '            if not trec["config_readback"]["ok"] and not FAKE:\n'
             '                trec["status"] = "E29_DFLASH_CONFIG_MISMATCH"\n'
             '                return {}\n'
             '            if trec["e29_df_readback"]["label"] not in ("ENGAGED", "FAKE_SERVER"):\n'
             '                trec["status"] = "E29_DFLASH_NOT_ENGAGED"\n'
             '                return {}\n', "E5 df readback")
    c = once(c, '        trec["k100_gates"] = k100_boot_gates(s, mtp_on)      # E26 K100',
             '        trec["k100_gates"] = k100_boot_gates(s, mtp_on and not E29_RT.get("df_try"))      # E29: no MTP experts on DFlash; E26 K100',
             "E5 gates")
    c = once(c, '        trec["warmup"] = nb5_warmup(s.port, W)\n',
             '        _e29_c0 = e29_counters() if E29_RT.get("df_try") else None   # E29 DFlash: the zero-accept canary over the warm-up\n'
             '        trec["warmup"] = nb5_warmup(s.port, W)\n'
             '        if E29_RT.get("df_try"):\n'
             '            trec["e29_canary"] = E29L.canary(_e29_c0, e29_counters(), E29_RT["df"]["k"])\n'
             '            E29_RT["canary"].append(dict(trec["e29_canary"], try_name=name))\n'
             '            log("E29 DFlash canary", json.dumps(trec["e29_canary"]))\n'
             '            if trec["e29_canary"]["hit"] and not FAKE:\n'
             '                trec["status"] = "E29_DFLASH_ZERO_ACCEPT"\n'
             '                return {}\n'
             '            E29_RT["df_live"] = True\n', "E5 canary")
    # E5: nb5_boot (the D tries, per-try env / argv, the drop rule before the I8 rule)
    c = once(c, "    levers = dict(NB5_PIN[\"levers\"] if NB5_PIN[\"levers\"] is not None else NB5_LEVER_ENV)\n",
             "    levers = dict(NB5_PIN[\"levers\"] if NB5_PIN[\"levers\"] is not None else NB5_LEVER_ENV)\n"
             "    if \"df\" in E29_RT[\"on\"] and mtp_on:            # E29 DFlash: the NBDF-K100 D pin ladder first, then the E28 ladder (MTP)\n"
             "        tries = E29L.df_tries(bool(E29_CFG[\"embed_host\"])) + list(tries)\n", "E5 df tries")
    c = once(c, "        NB5_BOOT_LEVERS = dict(levers, **(E28_I8_ENV if E28_I8[\"on\"] else {}))   # E28 K100 I8: DRAFT_HEAD=int8 into build()\n",
             "        _e29d = tr.get(\"df\") if \"df\" in E29_RT[\"on\"] else None   # E29: this try's DFlash plan (None = MTP)\n"
             "        if tr.get(\"df\") and _e29d is None:\n"
             "            i += 1\n"
             "            continue\n"
             "        E29_RT[\"df_try\"] = bool(_e29d)\n"
             "        NB5_BOOT_LEVERS = dict(levers, **(E28_I8_ENV if E28_I8[\"on\"] and not _e29d else {}))   # E28 K100 I8: DRAFT_HEAD=int8 into build()\n"
             "        NB5_BOOT_LEVERS.update(e29_env(_e29d))     # E29: seam env / draft vocab / GDN switches\n", "E5 levers")
    c = once(c, "        seqs = NB5L.seqs_for(NB5_BOOT_KV, tr[\"gib\"]) if NB5_SEQS_RULE == \"table\" else",
             "        seqs = int(E29_CFG[\"workers\"]) if _e29d else NB5L.seqs_for(NB5_BOOT_KV, tr[\"gib\"]) if NB5_SEQS_RULE == \"table\" else",
             "E5 df seqs")
    c = once(c, "        NB5_BOOT_EXTRA = base_extra + ([\"--max-num-seqs\", str(seqs)] if seqs else [])\n",
             "        NB5_BOOT_EXTRA = base_extra + ([\"--max-num-seqs\", str(seqs)] if seqs else []) + e29_extra(_e29d)   # E29: indexer / DFlash last\n",
             "E5 extra")
    c = once(c, '                + ("_i8" if E28_I8["on"] else ""))   # E28 K100\n',
             '                + ("_i8" if E28_I8["on"] and not _e29d else "") + (f"_df{E29_RT[\'df\'][\'k\']}r{_e29d[\'rung\']}" if _e29d else "")\n'
             '                + "".join("_" + x for x in E29_RT["on"] if x != "df"))   # E28 K100; E29\n', "E5 name")
    c = once(c, '                "e28_i8": bool(E28_I8["on"])}   # E28 K100\n',
             '                "e28_i8": bool(E28_I8["on"]) and not _e29d,   # E28 K100\n'
             '                "e29_on": [x for x in E29_RT["on"] if x != "df" or _e29d], "e29_df": _e29d}   # E29\n', "E5 trec")
    c = once(c, '        if (E28_I8["on"] and not ABORT.get("why") and not trec.get("phases_started") and not (P.get("e18_ladder") and len(brec["tries"]) >= 3)\n',
             '        E29_RT["tries"].append({k_: trec.get(k_) for k_ in ("name", "status", "kv_bytes", "max_num_seqs_launch", "e29_on", "e29_df")})\n'
             '        _e29s = str(trec.get("status"))\n'
             '        if (trec.get("e29_on") and not trec.get("phases_started") and not ABORT.get("why") and _e29s != "DONE"\n'
             '                and _e29s not in ("GPU_NOT_EMPTY", "HOST_RAM_GATE_FAILED")):\n'
             '            # E29: a pre-phase failure with an E29 lever on -> drop ONE lever (DFlash: an OOM -> the next D rung first)\n'
             '            _e29l = E29L.lever_to_drop(trec["e29_on"], _e28_log, _e29s)\n'
             '            if _e29l == "df":\n'
             '                _e29o = E29L.is_oom(_e28_log) and _e29s not in ("E29_DFLASH_NOT_ENGAGED", "E29_DFLASH_ZERO_ACCEPT", "E29_DFLASH_CONFIG_MISMATCH")\n'
             '                _e29n = E29L.next_try_index(tries, i, _e29o, False)\n'
             '                if not (_e29n < len(tries) and tries[_e29n].get("df")):\n'
             '                    e29_drop("df", f"{_e29s} (oom {_e29o})", name)   # -> the proven MTP K3 config, the top rung\n'
             '                else:\n'
             '                    log("E29 DFlash OOM -> the next D rung", tries[_e29n])\n'
             '                i = _e29n\n'
             '                continue\n'
             '            if _e29l:\n'
             '                e29_drop(_e29l, _e29s, name)\n'
             '                continue                           # the same pin without that lever\n'
             '        if (E28_I8["on"] and not ABORT.get("why") and not trec.get("phases_started") and not (P.get("e18_ladder") and e29_ntries(brec) >= 3)\n',
             "E5 drop rule")
    c = once(c, '        if not fail["failed"] or not P.get("pin_ladder") or ABORT.get("why") or (P.get("e18_ladder") and len(brec["tries"]) >= 3):\n',
             '        if not fail["failed"] or not P.get("pin_ladder") or ABORT.get("why") or (P.get("e18_ladder") and e29_ntries(brec) >= 3):\n',
             "E5 e18 cap")
    # E6: the sampler
    c = once(c, '                 "gen": g("vllm:generation_tokens_total"), "preempt": g("vllm:num_preemptions_total")}\n',
             '                 "gen": g("vllm:generation_tokens_total"), "preempt": g("vllm:num_preemptions_total")}\n'
             '            if not E29L.is_default(E29_CFG):           # E29: the spec-decode counters (hourly AL, the in-run canary)\n'
             '                s.update(sd_drafts=g(E29L.SD_DRAFTS), sd_draft_tok=g(E29L.SD_DRAFT_TOK), sd_acc=g(E29L.SD_ACC))\n', "E6 sampler")
    c = once(c, "        rows.append(s)\n",
             "        rows.append(s)\n"
             "        if E29_RT.get(\"df_live\"):                       # E29 DFlash: the in-run zero-accept canary (fires once)\n"
             "            try:\n"
             "                e29_inrun_check(rows)\n"
             "            except Exception as _e:\n"
             "                E29_RT[\"errors\"].append(f\"inrun check: {_e!r}\"[:200])\n", "E6 inrun")
    return engine_df2(c) if df2 else c


E29_SETUP = r'''    e29_setup()   # E29 FINAL: the switchable levers (vocab / DFlash drafter + seam / indexer / GDN); a miss = that lever off
'''


def runner(c):
    a = '    log("E28 K100 I8", json.dumps({k: E28_I8.get(k) for k in ("want", "on", "dropped")}, default=str)[:600], "| static",\n'
    i = c.index(a)
    j = c.index("\n", c.index("\n", i) + 1) + 1
    return c[:j] + E29_SETUP + c[j:]


def report(c):
    return once(c, '    write_json_atomic(BASE_OUT / "e28_k100_report.json", E28_OUT)\n',
                '    try:                                     # E29: flags, drops, canaries, hourly AL (record only)\n'
                '        E29_RT["hourly_al"] = E29L.hourly_al(globals().get("_ksamp"))\n'
                '        E28_OUT["e29"] = dict(E29L.summary(E29_RT), variant=E29_VARIANT, sched=E28_SCHED_REC.get("path"))\n'
                '    except BaseException:\n'
                '        E28_OUT["e29_error"] = traceback.format_exc()[-1500:]\n'
                '    write_json_atomic(BASE_OUT / "e28_k100_report.json", E28_OUT)\n'
                '    print("E29 FINAL:", json.dumps(E28_OUT.get("e29") or E28_OUT.get("e29_error"), default=str)[:6000], flush=True)\n', "E8 report")


def md0(variant, cfg):
    on = {k: v for k, v in cfg.items() if v != E29L.default_cfg()[k]}
    df = ("- **rev 2 (2026-11-02):** DFlash2 seam 341d859b (GRD edit), EMBED_HOST from the first D try, an OOM at any stage steps"
          " the D ladder (then MTP K3), real-failure-only gates, indexer fp8 off. **DIRECT = the 2nd branch (attempt_2):** edit"
          " `E29_DIRECT2` on top of CELL 1 (`'on': False` = no DIRECT); phase C runs one DIRECT request per task (C's order),"
          " attempt_2 = the DIRECT grid when valid and != attempt_1.\n") if is_df(cfg) else ""
    return (f"# E29 FINAL production · {variant} · klee100 3bpw-MTP (E28 K100 base) · E28EVAL120\n\n" + df +
            f"- **Flags on:** `{json.dumps(on, sort_keys=True)}` (every flag off = E28 K100 exactly; Build/build_e29_final.py).\n"
            "- **Base = E28 K100** (klee100 3bpw-MTP on Merlin TP2 x PP2, MTP K3 + Q-defer + I8, 24 workers, the A / B / C scheduler,"
            " T1 104k / B2 120k / max-model-len 131,072).\n"
            "- **Fallbacks (real failures only):** scheduler S-1 -> E28 scheduler; DFlash2 boot / read-back / canary -> MTP K3 at the same"
            " pin; indexer fp8 / draft vocab / GDN -> the same try without it.\n"
            "- **Dual mode (proven):** commit = EVAL120 scored; rerun = every hidden task, CELL 0 = E18 R2 verbatim, the E15B writer,"
            " 12 h budget. Report: `q27x/e28_k100_report.json` (`e29` key) + `q27x/nb5_report.json`.\n")


def report_df(c):
    return once(c, '        E28_OUT["e29"] = dict(E29L.summary(E29_RT), variant=E29_VARIANT, sched=E28_SCHED_REC.get("path"))\n',
                '        E29_RT.update(effective_algorithm=E29L.effective(E29_RT, RERUN),   # E29 rev 2d F5 / F12 (record only)\n'
                '                      submission_check=E29L.submission_check(SUBMISSION_PATH, globals().get("test_challenges") or globals().get("CHALLENGES") or {}))   # rev 2f\n'
                '        E28_OUT["e29"] = dict(E29L.summary(E29_RT), variant=E29_VARIANT, sched=E28_SCHED_REC.get("path"))\n'
                '        _sm = E28_OUT["e29"]["run_summary"] = E29L.run_summary(E29_RT, E28_OUT, getattr(E12A, "SCHED_LAST", None),   # rev 2e\n'
                '                                                               globals().get("_ksamp"), globals().get("NB5_BOOTS"), PROFILE)\n'
                '        print("[SMOKE SUMMARY]" if PROFILE == "E29SMOKE" else "[E29 RUN SUMMARY]", "\\n" + E29L.summary_text(_sm), flush=True)\n'
                '        print(json.dumps(_sm, default=str), flush=True)\n', "F5 report")


def cells_for(variant, cfg):
    E28B.MODULES_JSON = json.dumps(modules_for(cfg), sort_keys=True, separators=(",", ":"))   # E1: + the E29 modules (rev 2: per variant)
    base = E28B.cells()
    out = [("markdown", md0(variant, cfg)), base[1], ("code", cell1(base[2][1], cfg, variant)), ("code", cell3(base[3][1], is_df(cfg))),
           base[4], ("code", engine(base[5][1], df2=is_df(cfg))), base[6], ("code", runner(base[7][1])), ("code", report_df(report(base[8][1])) if is_df(cfg) else report(base[8][1]))]
    assert out[1][1] == KQ.E18R2_CELL0, "CELL 0 must stay E18 R2's (dual mode + fallback submission)"
    return out


CELL6_HEAD = "def e21_cell6():\n"
CELL6_PACK = r'''# ---- E29 FINAL (size: the 400-kB commit limit): e21_cell6() (E15B CELL 6 as E28 K100 runs it, the E28 scheduler N3 included,
# UNCHANGED) as ONE sha-pinned lzma + base85 literal executed into this namespace (= E28's e21_cell7 technique). The readable
# text is in the build's code export.
exec(compile(__PACKED__, "<E29: e21_cell6 = E15B CELL 6>", "exec"), globals())
'''


def pack_cell6(c):
    i = c.index(CELL6_HEAD)
    j = c.index("# ---- E28 K100 (size: the 400-kB commit limit): e21_cell7()")
    assert c.count(CELL6_HEAD) == 1 and i < j
    body = c[i:j]
    return c[:i] + CELL6_PACK.replace("__PACKED__", E28B.packed_lzma(body, "e21_cell6")) + "\n\n" + c[j:], {"sha256": sha(body), "bytes": len(body.encode())}


def slug_of(variant):
    return "q27-xhigh-" + variant.lower().replace("_", "-")


def kernel_metadata(variant, cfg):
    ds = list(KQ.DATASETS) + ([DS_ID] if (cfg["drafter"] != "mtp" or cfg["draft_vocab"]) else [])
    models = list(KQ.MODELS) + ([DRAFTER_MODEL] if cfg["drafter"] != "mtp" else [])
    return {"id": f"{OWNER}/{slug_of(variant)}", "title": "Q27 Xhigh " + variant.replace("_", " ").title(),
            "code_file": f"{slug_of(variant)}.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [], "dataset_sources": ds,
            "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": models,
            "docker_image": KQ.DOCKER, "machine_shape": "NvidiaL4"}


def write_dataset():
    DS_DIR.mkdir(parents=True, exist_ok=True)
    files = {E29L.SEAM_FILE[0]: SEAM_BYTES}
    for n_ in ("e29_direct.py", "e29d_sched_patch.py", "e29_nbx.py"):      # rev 2 / 2c: the DIRECT + LIVE files
        files[n_] = build_direct()[n_].encode("utf-8")
    files[FIXTURE.name] = FIXTURE.read_bytes()             # rev 2: the NB-S1 GPU log fixture of the kits' gate tests (kit size)
    for _k, (_f, _s) in E29L.VOCAB_FILES.items():
        files[_f] = (VOCAB_DIR / _f).read_bytes()
    for n, b in files.items():
        (DS_DIR / n).write_bytes(b)
    for n, b in files.items():                          # rev 2: read back what is on disk
        assert sha((DS_DIR / n).read_bytes()) == sha(b), n
    assert sha((DS_DIR / E29L.SEAM_FILE[0]).read_bytes()) == E29L.SEAM_FILE[1]
    _all = dict(files, **{f.name: f.read_bytes() for f in DS_DIR.iterdir() if f.is_file() and f.name not in files
                          and f.name not in ("SHA256SUMS", "dataset-metadata.json")})   # rev 2b: + the kits' K100Q fixture
    sums = "".join(f"{sha(b)}  {n}\n" for n, b in sorted(_all.items()))
    (DS_DIR / "SHA256SUMS").write_bytes(sums.encode("utf-8"))
    meta = {"title": "q27-e29-final-payload", "id": DS_ID, "licenses": [{"name": "CC0-1.0"}], "isPrivate": True}
    (DS_DIR / "dataset-metadata.json").write_bytes((json.dumps(meta, indent=1) + "\n").encode("utf-8"))
    return {"dir": str(DS_DIR), "id": DS_ID, "files": {n: {"bytes": len(b), "sha256": sha(b)} for n, b in files.items()},
            "sha256sums": sha(sums), "seam_prev_sha256": E29L.SEAM_PREV_SHA}


DIRECT_TEXTS = {}
SCHED_PIN = "b88efa86246623871e74a984a5b613b019705beabf24da57035f1040011c8ee2"   # rev 2g: + early DIRECT on train-fail tasks; rev 2f: + the C1 scope (E29_DIRECT_PHASE); rev 2d: F1 (E29D_CAN_START)
FIXTURE = HERE / "e29_nbs1_log_fixture.json"           # rev 2: server-log lines of the E25 NB-S1 GPU runs (kit gate tests only)


def build_direct():
    """rev 2: e29_direct.py (E31's solver + N1 + task_index) and e29d_sched_patch.py (the DIRECT diff on the E29 post
    arc2_duck.py), checked with the E28 applier both ways; written to Build/ and the payload dataset folder."""
    if DIRECT_TEXTS:
        return DIRECT_TEXTS
    d = E29DB.make_direct()
    post, patch, mod = E29DB.make_dsched(_p["text"], E29S.E29_SCHED_SHA["post"])
    assert sha(mod) == SCHED_PIN, ("e29d_sched_patch.py != the SCHED agent's rev 2d build (SCHED_DONE.json)", sha(mod))
    assert E29S.apply_unified(_p["text"], patch) == post and E29S.apply_unified(post, E29S.reverse_unified(patch)) == _p["text"]
    _ns = {"__name__": "e29d_sched_patch_b"}
    exec(compile(mod, "e29d_sched_patch.py", "exec"), _ns)
    assert _ns["apply_direct"](_p["text"])["text"] == post and _ns["revert_direct"](post)["text"] == _p["text"]
    nbx = ('"""E29 rev 2c: the DIRECT + LIVE notebook functions of E29_DFLASH_D3S (generated by Build/build_e29_final.py; the notebook\n'
           'executes this file into its own namespace via e29_extras(): it reads the notebook\'s globals). = Build/e29_live.py +\n'
           'E29_LIVE_FNS + E29_DIRECT_FNS of the builder."""\n' + (HERE / "e29_live.py").read_text(encoding="utf-8").split('"""', 2)[2]
           + E29_LIVE_FNS + E29_DIRECT_FNS)
    compile(nbx, "e29_nbx.py", "exec")
    DIRECT_TEXTS.update({"e29_direct.py": d, "e29d_sched_patch.py": mod, "e29_nbx.py": nbx, "post_sha256": sha(post), "patch_sha256": sha(patch)})
    DIRECT_FILES.update(direct=("e29_direct.py", sha(d)), sched=("e29d_sched_patch.py", sha(mod)), nbx=("e29_nbx.py", sha(nbx)))
    for n_ in ("e29_direct.py", "e29d_sched_patch.py", "e29_nbx.py"):
        (HERE / n_).write_bytes(DIRECT_TEXTS[n_].encode("utf-8"))
    return DIRECT_TEXTS


def write_smoke(variant, cfg, data):
    """rev 2e: the GPU smoke notebook = the production notebook with ONE line changed (the CELL 1 profile line)."""
    a = json.dumps(PROFILE_LINE.format(PROD_PROFILE))[1:-1].encode("utf-8")
    b = json.dumps(PROFILE_LINE.format(SMOKE_PROFILE))[1:-1].encode("utf-8")
    assert data.count(a) == 1, data.count(a)
    sm = data.replace(a, b)
    s0 = ["".join(c_["source"]) for c_ in json.loads(data)["cells"]]
    s1 = ["".join(c_["source"]) for c_ in json.loads(sm)["cells"]]
    diff = [(i, x, y) for i, (c0, c1) in enumerate(zip(s0, s1)) for x, y in zip(c0.splitlines(), c1.splitlines()) if x != y]
    assert len(s0) == len(s1) and len(diff) == 1 and all(len(c0.splitlines()) == len(c1.splitlines()) for c0, c1 in zip(s0, s1)), diff
    assert len(sm) < 400_000
    out = ROOT / "Experiment Notebooks" / f"Q27-XHIGH-{variant}_SMOKE.ipynb"
    out.write_bytes(sm)
    slug = slug_of(variant) + "-smoke"
    push = PUSH_ROOT / f"{slug}__{OWNER}"
    push.mkdir(parents=True, exist_ok=True)
    (push / f"{slug}.ipynb").write_bytes(sm)
    km = dict(kernel_metadata(variant, cfg), id=f"{OWNER}/{slug}", title="Q27 Xhigh E29 Dflash D3S Smoke", code_file=f"{slug}.ipynb")
    (push / "kernel-metadata.json").write_bytes((json.dumps(km, indent=2) + "\n").encode("utf-8"))
    print("wrote", out.name, "sha256", sha(sm), "|", len(sm), "bytes | the one-line diff:", diff[0][1].split("   #")[0], "->", diff[0][2].split("   #")[0])
    return {"notebook": str(out), "sha256": sha(sm), "bytes": len(sm), "push": str(push), "slug": f"{OWNER}/{slug}", "diff": list(diff[0]),
            "tasks": sorted(SMOKE_TASKS), "window_min": SMOKE_WINDOW_MIN, "budget_min": SMOKE_BUDGET_MIN, "kernel_metadata": km}


def write_submit(variant, cfg, smoke_rec):
    """rev 2f (user 2026-11-02): the SUBMISSION notebook = the E15B SUBMIT pattern (commit = a short GPU smoke; the competition
    rerun = RERUN_PROD on every hidden task, 12 h - 3 min, chosen by RERUN alone). Bytes = the smoke copy (profile E29SMOKE);
    only the kernel slug / title differ."""
    sm = Path(smoke_rec["notebook"]).read_bytes()
    assert sha(sm) == smoke_rec["sha256"] and len(sm) < 400_000
    slug = slug_of(variant) + "-submit"
    push = PUSH_ROOT / f"{slug}__{OWNER}"
    push.mkdir(parents=True, exist_ok=True)
    (push / f"{slug}.ipynb").write_bytes(sm)
    km = dict(kernel_metadata(variant, cfg), id=f"{OWNER}/{slug}", title="Q27 Xhigh E29 Dflash D3S Submit", code_file=f"{slug}.ipynb")
    (push / "kernel-metadata.json").write_bytes((json.dumps(km, indent=2) + "\n").encode("utf-8"))
    print("wrote the SUBMIT push folder", push.name, "sha256", sha(sm), "(= the smoke copy)")
    return {"sha256": sha(sm), "push": str(push), "slug": f"{OWNER}/{slug}", "kernel_metadata": km}


def build(variant):
    cfg = E29L.check_cfg(VARIANTS[variant])
    if is_df(cfg):
        build_direct()
    cl = cells_for(variant, cfg)
    B.undefined_names_check(cl)
    ncl, pack7 = E28B.notebook_cells(cl)
    packe = None
    if is_df(cfg):                                       # rev 2: the E29 engine block packed (the 400-kB commit limit)
        assert unpack_engine(ncl[5][1]) == ncl[5][1]
        _c5, packe = pack_engine(ncl[5][1])
        ncl[5] = ("code", _c5)
    data, n, fmt = E28B.notebook_bytes(ncl)
    pack6 = None
    if len(data) >= 395_000:
        c6, pack6 = pack_cell6(ncl[6][1])
        ncl[6] = ("code", c6)
        data, n, fmt = E28B.notebook_bytes(ncl)
    assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
    assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))
    nb = json.loads(data.decode("utf-8"))
    assert "".join(nb["cells"][1]["source"]) == B.norm(KQ.E18R2_CELL0) or "".join(nb["cells"][1]["source"]).rstrip("\n") == KQ.E18R2_CELL0.rstrip("\n")
    if variant in AUDITED:                               # rev 2: the audited notebooks never change: rebuilt + compared, nothing written
        assert sha(data) == AUDITED[variant], (variant, "not byte-identical to the audited build", sha(data))
        on_disk = {"notebook": ROOT / "Experiment Notebooks" / f"Q27-XHIGH-{variant}.ipynb"}
        if variant != "ALL_OFF":
            on_disk["push"] = PUSH_ROOT / f"{slug_of(variant)}__{OWNER}" / f"{slug_of(variant)}.ipynb"
        got = {k: sha(p.read_bytes()) for k, p in on_disk.items()}
        assert all(v == AUDITED[variant] for v in got.values()), (variant, got)
        print("verified", variant, "byte-identical to the audited build", sha(data), "| on disk:", sorted(got))
        return {"variant": variant, "sha256": sha(data), "bytes": len(data), "rev2_rebuilt_identical": True, "on_disk_identical": sorted(got)}
    out = ROOT / "Experiment Notebooks" / f"Q27-XHIGH-{variant}.ipynb"
    if out.exists() and sha(out.read_bytes()) in PUSHED:
        out = out.with_name(out.stem + "_REBUILD.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    slug = slug_of(variant)
    push = PUSH_ROOT / f"{slug}__{OWNER}"
    push.mkdir(parents=True, exist_ok=True)
    (push / f"{slug}.ipynb").write_bytes(data)
    (push / "kernel-metadata.json").write_bytes((json.dumps(kernel_metadata(variant, cfg), indent=2) + "\n").encode("utf-8"))
    export = HERE / f"{variant.lower()}_code_export.py"
    export.write_bytes("\n\n".join(B.norm(t) for k, t in cl if k == "code").encode("utf-8"))
    smoke_rec = write_smoke(variant, cfg, data) if is_df(cfg) else None   # rev 2e: the GPU smoke copy (one line differs)
    submit_rec = write_submit(variant, cfg, smoke_rec) if smoke_rec else None   # rev 2f: the SUBMIT push folder (= the smoke bytes)
    rec = {"variant": variant, "cfg": cfg, "notebook": str(out), "sha256": sha(data), "bytes": len(data), "cells": n, "smoke": smoke_rec, "submit": submit_rec,
           "push": str(push), "slug": f"{OWNER}/{slug}", "export": str(export), "e21_cell7_packed": pack7, "e21_cell6_packed": pack6,
           "e29_engine_packed": packe,
           "kernel_metadata": kernel_metadata(variant, cfg)}
    # the per-variant facts in the E28 facts format (the E28 kit chain reads that format), E29 keys added
    f28 = json.loads(E28B.FACTS.read_text(encoding="utf-8"))
    f28.update(notebook=str(out), notebook_sha256=rec["sha256"], bytes=len(data), cells=n, json_format=fmt, e21_cell7_packed=pack7,
               e21_cell6_packed=pack6, export=str(export), push=str(push), kernel_metadata=rec["kernel_metadata"],
               modules={nm: sha(t) for nm, t in modules_for(cfg).items()},
               modules_json_sha256=sha(json.dumps(modules_for(cfg), sort_keys=True, separators=(",", ":"))),
               e29={"variant": variant, "cfg": cfg, "sched_sha": E29S.E29_SCHED_SHA, "seam": list(E29L.SEAM_FILE),
                    "vocab_files": E29L.VOCAB_FILES, "dataset": DS_ID, "drafter_model": DRAFTER_MODEL,
                    "e28_notebook_sha256": E28_NB_SHA,
                    "rev": 2 if is_df(cfg) else 1, "lib_version": (LIB_TEXT if is_df(cfg) else LIB10_TEXT).split('LIB_VERSION = "')[1].split('"')[0],
                    "removed_levers": REMOVED.get(variant, {}), "superseded_sha256": SUPERSEDED.get(variant),
                    "grd": {"rel": E29L.GRD_REL, "post": E29L.GRD_POST} if is_df(cfg) else None,
                    "seam_prev_sha256": E29L.SEAM_PREV_SHA if is_df(cfg) else None, "e29_engine_packed": packe,
                    "nbs1_fixture_sha256": sha(FIXTURE.read_bytes()) if is_df(cfg) else None,
                    "smoke": ({k: smoke_rec[k] for k in ("sha256", "bytes", "slug", "diff", "tasks", "window_min", "budget_min")} if smoke_rec else None),
                    "qsa_switch": {"max_logits_mb": 128, "granule_mb": 64} if is_df(cfg) else None,
                    "qsa_env": E29L.qsa_env({"max_logits_mb": 128, "granule_mb": 64}) if is_df(cfg) else None,
                    "direct": ({"files": {k: list(v) for k, v in DIRECT_FILES.items()}, "switch_line": E29_DIRECT_SWITCH.rstrip(chr(10)),
                                "duck_post_sha256": DIRECT_TEXTS.get("post_sha256"), "patch_sha256": DIRECT_TEXTS.get("patch_sha256"),
                                "e31_direct_sha256": E29DB.E31_DIRECT_SHA} if is_df(cfg) else None)})
    vf = HERE / f"{variant.lower()}_build_facts.json"
    vf.write_bytes((json.dumps(f28, indent=1, default=str) + "\n").encode("utf-8"))
    rec["facts"] = str(vf)
    print("wrote", out.name, "sha256", rec["sha256"], "|", len(data), "bytes |", n, "cells")
    return rec


if __name__ == "__main__":
    want = sys.argv[1:] or ["E29_DEFAULT", "E29_DFLASH_D3S", "E29_MTP_V64", "ALL_OFF"]
    recs = {v: build(v) for v in want}
    facts_f = HERE / "e29_build_facts.json"
    facts = json.loads(facts_f.read_text(encoding="utf-8")) if facts_f.exists() else {}
    for v, r in recs.items():                           # rev 2: an audited variant keeps its record (+ the identity check)
        if r.get("rev2_rebuilt_identical") and v in facts.get("variants", {}):
            facts["variants"][v].update(rev2_rebuilt_identical=True, rev2_on_disk_identical=r["on_disk_identical"])
        else:
            facts.setdefault("variants", {})[v] = r
    facts.update(e28_notebook_sha256=E28_NB_SHA, e28_builder_sha256=sha((E28B_DIR / "build_e28_k100.py").read_bytes()),
                 modules={k: sha(v) for k, v in MODULES.items()}, modules_json_sha256=sha(MODULES_JSON),
                 modules_df={k: sha(v) for k, v in MODULES_DF.items()}, e29_lib_v1_0_sha256=sha(LIB10_TEXT),
                 e29_sched_sha=E29S.E29_SCHED_SHA, nbdf_lib_sha256=sha(NBDF_LIB_SRC), nbdf_lib_at_cut=NBDF_AT_BUILD,
                 e29_nbdf_sha256=sha(E29_NBDF), seam_sha256=E29L.SEAM_FILE[1], seam_prev_sha256=E29L.SEAM_PREV_SHA,
                 nbdf_k100_facts_seam_sha256_now=NBDFK_SEAM_NOW, grd={"rel": E29L.GRD_REL, "post": E29L.GRD_POST},
                 drafter_fns_sha256=sha(DRAFTER_FNS), registry_caps=REGISTRY_CAPS, dataset=write_dataset(),
                 final_cfg_default=FINAL_CFG, variants_cfg=VARIANTS, audited=AUDITED, superseded=SUPERSEDED, removed_levers=REMOVED)
    facts_f.write_bytes((json.dumps(facts, indent=1, default=str) + "\n").encode("utf-8"))
