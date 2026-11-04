"""E25 NB-M1 (2026-11-02): MTP lever arms on the PROVEN production base Q = NBDF-K100's Q arm verbatim (E28 K100 production).

  python build_e25_nbm1.py [--kv-gate-report FILE] [--test-out DIR]
      -> ../Q27-XHIGH-E25_NBM1.ipynb (every arm on: the kit notebook), nbm1_code_export.py, nbm1_cell7_body.py,
         nbm1_build_facts.json, the three copies ../Q27-XHIGH-E25_NBM1_{A,B,C}.ipynb and their GPU push folders
         D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nbm1-{a,b,c}__<owner>/ (private; nothing is pushed)

Built by the NBDF builder itself, exactly as NBDF-K100 and NB-S1 are (build_e27_nbdf.py with NBDF_VARIANT=k100 and the K100
module's edits), with the K100 module's names re-pointed IN MEMORY to NB-M1's outputs and its edit_cells / edit_body /
edit_meta / gate wrapped: the K100 edits run first, then the NB-M1 edits below (exact-once text replacements). No NBDF /
K100 / NB-S1 / NB-S2 file is written or changed (asserted: every Exp 27 file + the NB-S1 / NB-S2 files byte-identical).

NB-M1 edits (nbm1_lib.py embedded verbatim, sha-checked at run time):
  - CELL 1: the plain switch line `NBM1_ON = {...}` at the top (the three copies differ ONLY in this line), the hard stop
    BUDGET_MIN (= the max over the copies' planned walks, rounded up) and the deadman 5 min later;
  - arms: Q (NBDF-K100's Q, budget fields only) > V65 > V32 > I4 > IDX > S28 > S32 > LP > G1 > G2 > K4 > DYN, each = Q + one
    change (nbm1_lib.check_arms); PROF = two 30-s profiled windows inside Q's boot (no change); a switched-off arm is
    recorded SKIPPED_SWITCH_OFF and never booted;
  - nbdf_main: the D loop -> the NB-M1 loop (switch, the kit gate + the vocab table, budget, nb4_ppack before every arm);
    Q's switch; PROF's record;
  - arm_body: an MTP lever reads the seam-inert check (not the DFlash read-back), stops on a FAIL smoke (hard-only labels),
    reads back its own change (nbm1_lib.readback: a real failure = NOT_INTEGRATED for that arm only), skips the
    record-only Q items (ladder, prefill probe, long oracle, prefix test, profiles; TF only on IDX / LP) and runs its own
    A/A-free window set (90k and 116k always); the M1 env keys in the worker env read-back;
  - arm_boot: the lever's draft head (Q's int8 / BF16 per Q_I8, I4 int4) and the GPU-v1 OOM fix
    `oom = bool(o["oom_in_log"]) and not o["xid_in_log"]` (an OOM at any stage, incl. the first request after ready,
    steps the Q pin ladder);
  - tree_check: an MTP arm accepts the seam ORIGINAL or FULLY_PATCHED (Q's rule).

  python build_e25_nbm1.py --light [--kv-gate-report FILE] [--test-out DIR]      (E25 NB-M1L, user decision 2026-11-02)
      -> ../Q27-XHIGH-E25_NBM1L.ipynb (every arm on: the light kit notebook), nbm1l_code_export.py, nbm1l_cell7_body.py,
         nbm1l_build_facts.json, the two light copies ../Q27-XHIGH-E25_NBM1L_{B,C}.ipynb and their push folders
         D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nbm1l-{b,c}__<owner>/ (B hosen42, C foysalemonshanto; every kernel
         source of NB-M1 kept). nbm1_lib.py embedded unchanged + nbm1l_overlay.py (exec'd into its namespace, sha-checked):
         Q = facts + smoke (no ladder) + prefill probe + warmup + its windows (no oracle / oracle A/A / long oracle / prefix /
         TF / profiles; PROF stays); levers without TF and without the oracle vs Q; arm_boot's evidence gate without an
         oracle when the plan has none (probes, sampled, windowed garbage: the hard-only gates); due-aware body caps; light
         budget / deadman. Without --light the build is byte-identical to NB-M1 (closed 65050260, gated 9e052435).
"""
import hashlib
import json
import os
import runpy
import shutil
import sys
import textwrap
from pathlib import Path

LIGHT = "--light" in sys.argv                 # E25 NB-M1L: the light build (new names; the NB-M1 run files never touched)
if LIGHT:
    sys.argv = [x for x in sys.argv if x != "--light"]
HERE = Path(__file__).resolve().parent
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
NBDIR = HERE.parent
TEST_OUT = Path(sys.argv[sys.argv.index("--test-out") + 1]) if "--test-out" in sys.argv else None
OUT_DIR = TEST_OUT or HERE
TAG = "NBM1L" if LIGHT else "NBM1"
PFX = "nbm1l" if LIGHT else "nbm1"
OUT_NB = (TEST_OUT or NBDIR) / f"Q27-XHIGH-E25_{TAG}.ipynb"
SLUG = f"q27-xhigh-e25-{PFX}"
PUSH_ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
LIB = HERE / "nbm1_lib.py"
OVERLAY = HERE / "nbm1l_overlay.py"
PUSHED = set()        # shas of NB-M1 versions that ran on Kaggle (never overwrite one: rebuilds then go to *_REBUILD)
if OUT_NB.exists() and hashlib.sha256(OUT_NB.read_bytes()).hexdigest() in PUSHED:
    OUT_NB = OUT_NB.with_name(OUT_NB.stem + "_REBUILD.ipynb")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


# NB-M1 versions RUNNING / run on Kaggle (2026-11-02: gated base 9e052435, A 8d183071 foysalemonshanto, C 48996e78 hosen42; the
# B copy 56fce34d): a default (non --light) build outside --test-out would overwrite them -> refused (never rebuild run notebooks)
RUN_SHAS = {"9e052435eb7862be4e7d2ee02b3e22ebf2c35a591914689b5d9b10ea0a091d72", "8d183071e0c7a245a1b056f5b1cd6049dff942a8c3ac9a563dfd43cbbbee2d5e",
            "56fce34d205c52ad3ba252184c8fdf03958c9586f474109c90982e660f53a046", "48996e782f2a932bb7f44b1cefb7e2a22516bab8b5e95ad43b06138b966c17cb"}
NBM1_RUN_FILES = sorted(NBDIR.glob("Q27-XHIGH-E25_NBM1.ipynb")) + sorted(NBDIR.glob("Q27-XHIGH-E25_NBM1_?.ipynb"))
if not LIGHT and not TEST_OUT and any(sha(p.read_bytes()) in RUN_SHAS for p in NBM1_RUN_FILES):
    raise SystemExit("NB-M1 notebooks that run on Kaggle are on disk (RUN_SHAS): a default rebuild would overwrite them. "
                     "Use --test-out DIR (reproduction) or --light (NB-M1L, new names).")


# the frozen files: every Exp 27 file (read-only) + NB-S1 / NB-S2 (not this builder's), byte-identical before and after
FROZEN = sorted(p for p in E27B.glob("*") if p.is_file()) + sorted(p for p in E27B.parent.glob("*.ipynb"))
FROZEN += [HERE / n for n in ("build_e25_nbs1.py", "nbs1_lib.py", "nbs1_unit_tests.py", "build_e25_nbs1_cpu_kit.py",
                              "build_e25_nbs1_units_kit.py", "nbs1_build_facts.json", "nbs1_cell7_body.py", "nbs1_code_export.py",
                              "build_e25_nbs2.py", "nbs2_lib.py", "nbs2_unit_tests.py", "nbs2_body_add.py", "nbs2_build_facts.json",
                              "nbs2_cell7_body.py", "nbs2_code_export.py", "build_e25_nbs2_cpu_kit.py", "build_e25_nbs2_units_kit.py",
                              "nb4t_lib.py", "nb4_ppack.py", "nb4_timeline.py", "k100q_ref27b.json")]
FROZEN += [NBDIR / "Q27-XHIGH-E25_NBS1.ipynb", NBDIR / "Q27-XHIGH-E25_NBS2_CAPACITY.ipynb"]
if LIGHT:                                     # NB-M1L: every NB-M1 file (the run notebooks, their build files, push folders) untouched
    FROZEN += NBM1_RUN_FILES + [HERE / n for n in ("nbm1_lib.py", "build_e25_nbm1_units_kit.py", "build_e25_nbm1_cpu_kit.py",
                                                    "nbm1_unit_tests.py", "nbm1_build_facts.json", "nbm1_code_export.py",
                                                    "nbm1_cell7_body.py")]
    FROZEN += sorted(p for d in PUSH_ROOT.glob("q27-xhigh-e25-nbm1-*") for p in d.glob("*") if p.is_file())
    FROZEN += sorted(p for d in PUSH_ROOT.glob("q27-e25-nbm1-kit-*") for p in d.glob("*") if p.is_file())
FROZEN = [p for p in FROZEN if p.is_file()]
FROZEN_SHA = {str(p): sha(p.read_bytes()) for p in FROZEN}

LIB_SRC = LIB.read_bytes().decode("utf-8")
assert "\r" not in LIB_SRC and "'''" not in LIB_SRC and '"""__' not in LIB_SRC, "nbm1_lib.py: LF only, no triple single quotes"
compile(LIB_SRC, "nbm1_lib.py", "exec")
LIB_SHA = sha(LIB_SRC)
OVL_SRC = OVERLAY.read_bytes().decode("utf-8") if LIGHT else None
if LIGHT:
    assert "\r" not in OVL_SRC and "'''" not in OVL_SRC and '"""__' not in OVL_SRC, "nbm1l_overlay.py: LF only, no triple single quotes"
    compile(OVL_SRC, "nbm1l_overlay.py", "exec")
OVL_SHA = sha(OVL_SRC) if LIGHT else None

os.environ["NBDF_VARIANT"] = "k100"
sys.dont_write_bytecode = True
sys.path.insert(0, str(E27B))
sys.path.insert(0, str(HERE))
import build_e27_nbdf_k100 as KX  # noqa: E402  (module import only: its build runs under __main__)
import nbm1_lib as M1  # noqa: E402

if LIGHT:                                     # NB-M1L: the overlay into nbm1_lib's namespace (as the notebook does it)
    exec(compile(OVL_SRC, str(OVERLAY), "exec"), M1.__dict__)
    assert M1.LIGHT is True and set(M1.COPIES) == {"b", "c"} and M1.OWNERS == {"b": "hosen42", "c": "foysalemonshanto"}
assert M1.LIB_VERSION and set(M1.COPIES) == ({"b", "c"} if LIGHT else {"a", "b", "c"})
NBM1_CFG = {"lib_sha256": LIB_SHA, "budget_min": M1.BUDGET_MIN, "deadman_min": M1.DEADMAN_MIN, "copies": M1.COPIES,
            "prof_windows": [list(x) for x in M1.PROF_WINDOWS], "prof_secs": M1.PROF_SECS}
if LIGHT:
    NBM1_CFG.update(light=True, overlay_sha256=OVL_SHA, owners=M1.OWNERS)
NBM1_ON_DEFAULT = dict(M1.SWITCH_DEFAULT)            # every arm on (the kit notebook); each copy = its own line
SWITCH_LINE = "NBM1_ON = " + repr(NBM1_ON_DEFAULT) + "\n"

# ---------------------------------------------------------------- the K100 module, re-pointed in memory (never on disk)
assert KX.K100_CFG["q_i8"] is True                   # Q = E28 K100 production: the INT8 draft head stays ON
KX.OUT_NB_NAME = str(OUT_NB)
KX.SLUG = SLUG
KX.EXPORT_NAME, KX.BODY_NAME, KX.FACTS_NAME = (str(OUT_DIR / f"{PFX}_code_export.py"), str(OUT_DIR / f"{PFX}_cell7_body.py"),
                                               str(OUT_DIR / f"{PFX}_build_facts.json"))
KX.TITLE = f"Q27 Xhigh E25 {TAG}"
KX.DATASETS = list(KX.DATASETS) + [M1.VOCAB_DS]     # the kit notebook + kits mount the vocab tables (copy A needs them)
_k100_edit_body, _k100_edit_meta, _k100_gate, _k100_edit_cells = KX.edit_body, KX.edit_meta, KX.k100_gate_extra, KX.edit_cells
NBM1_KIT_KEYS = ("nbm1_arm_plan_ok", "nbm1_launch_path_all_arms_ok", "nbm1_help_flag_gate_real_ok", "nbm1_argparse_all_arms_ok",
                 "nbm1_kv_probes_ok", "nbm1_vocab_tables_ok")


def once(text, old, new, where):
    n = text.count(old)
    assert n == 1, f"NB-M1 anchor {where}: {n} matches"
    return text.replace(old, new)


def after_line(text, needle, add, where):
    assert text.count(needle) == 1, f"NB-M1 anchor {where}: {text.count(needle)} matches"
    i = text.index(needle)
    j = text.index("\n", i) + 1
    return text[:j] + add + text[j:]


M1_BLOCK_HEAD = '''# ---- E25 NB-M1: MTP levers on Q = NBDF-K100's Q verbatim (nbm1_lib, embedded); CELL 1's NBM1_ON switches the arms
NBM1_CFG = json.loads(r"""__CFG__""")
NBM1_LIB_SRC = r'''
M1_BLOCK_TAIL = '''
if hashlib.sha256(NBM1_LIB_SRC.encode("utf-8")).hexdigest() != NBM1_CFG["lib_sha256"]:
    raise RuntimeError("NB-M1: the embedded nbm1_lib does not match its build sha")
import types as _nbm1_types
M1 = _nbm1_types.ModuleType("nbm1_lib")
exec(compile(NBM1_LIB_SRC, "nbm1_lib.py", "exec"), M1.__dict__)
_nbm1_on = globals().get("NBM1_ON")          # CELL 1 arm switches (E25 NB-M1); absent -> every arm on
M1_SWITCH = M1.switch_map(_nbm1_on)
M1_VOCAB = M1.resolve_vocab(INP, dict(os.environ))   # the E29 payload dataset (sha-pinned tables); missing -> V65 / V32 NOT_INTEGRATED
_nbm1_q0 = ARMS["Q"]                          # NBDF-K100's Q (m1_arms deep-copies it)
ARMS = M1.m1_arms(N, _nbm1_q0, M1_SWITCH, M1_VOCAB)
M1.install(N, ARMS)
M1.check_arms(N, T, ARMS, _nbm1_q0)          # raises: no boot
NBDF["nbm1"] = {"lib": M1.LIB_VERSION, "cfg": NBM1_CFG, "arms": {k: a["title"] for k, a in ARMS.items()},
                "switch": M1.switch_record(_nbm1_on), "vocab": M1_VOCAB, "walk": M1.copy_estimate(ARMS, M1_SWITCH),
                "time": {k: {"est_min": a["est_min"], "cap_min": a["cap_min"], "model": a.get("time_model")} for k, a in ARMS.items()}}
print("E25 NB-M1 arm switches:", M1_SWITCH, "| walk:", json.dumps(NBDF["nbm1"]["walk"])[:600], flush=True)
'''
if LIGHT:                                     # NB-M1L: the overlay right after nbm1_lib (sha-checked), its records after the walk
    _lib_exec = 'exec(compile(NBM1_LIB_SRC, "nbm1_lib.py", "exec"), M1.__dict__)\n'
    M1_BLOCK_TAIL = once(M1_BLOCK_TAIL, _lib_exec, _lib_exec + "NBM1L_OVL_SRC = r'''" + OVL_SRC + "'''\n" +
                         'if hashlib.sha256(NBM1L_OVL_SRC.encode("utf-8")).hexdigest() != NBM1_CFG["overlay_sha256"]:\n'
                         '    raise RuntimeError("NB-M1L: the embedded nbm1l overlay does not match its build sha")\n'
                         'exec(compile(NBM1L_OVL_SRC, "nbm1l_overlay.py", "exec"), M1.__dict__)   # E25 NB-M1L (light): item plans + time\n',
                         "light overlay exec")
    assert M1_BLOCK_TAIL.count("NBM1L_OVL_SRC = r'''") == 1
    M1_BLOCK_TAIL += ('NBDF["nbm1"]["light"] = {"version": M1.LIB_VERSION, "q_items": M1.Q_ITEMS, "lever_items": M1.LEVER_ITEMS,\n'
                      '                         "need_min": M1.copy_need_min(ARMS, M1_SWITCH), "budget_min": M1.BUDGET_MIN}\n')


SWITCH_BLOCK = ("# ==================================================================== E25 NB-M1 ARM SWITCHES (edit before the commit)\n"
                "# True = the arm runs, False = it is not booted (recorded SKIPPED_SWITCH_OFF; the other arms are unaffected).\n"
                "# Q = the base (NBDF-K100's Q = E28 K100 production); PROF = the 30-s profiles inside Q's boot. A key left out = off.\n"
                "# Copy A: Q V65 V32 I4 | copy B: Q IDX S28 S32 LP | copy C: Q G1 G2 K4 DYN PROF. No rebuild / new kit needed.\n"
                + SWITCH_LINE)
if LIGHT:
    SWITCH_BLOCK = ("# ==================================================================== E25 NB-M1L (LIGHT) ARM SWITCHES (edit before the commit)\n"
                    "# True = the arm runs, False = it is not booted (recorded SKIPPED_SWITCH_OFF; the other arms are unaffected).\n"
                    "# Q = the base (NBDF-K100's Q, light: facts + smoke + windows); PROF = the 30-s profiles inside Q's boot. A key left out = off.\n"
                    "# Light copy B: Q IDX S28 S32 LP | light copy C: Q G1 G2 K4 DYN PROF. No rebuild / new kit needed.\n"
                    + SWITCH_LINE)
BUDGET_OLD = 'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 150 * 60)     # E27 NBDF: hard stop 150 min (user: Q + D4 + D3 guaranteed; D4t if time)'
BUDGET_NEW = (f'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or {M1.BUDGET_MIN} * 60)     # E25 NB-M1: hard stop {M1.BUDGET_MIN} min '
              f'(the max over the three copies\' planned walks; each copy\'s runtime <= {M1.COPY_RUNTIME_MAX} min)')
if LIGHT:
    BUDGET_NEW = (f'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or {M1.BUDGET_MIN} * 60)     # E25 NB-M1L: hard stop {M1.BUDGET_MIN} min '
                  f'(the max over the two light copies\' planned walks with every arm\'s full body cap; each copy\'s runtime <= {M1.COPY_RUNTIME_MAX} min)')
WHAT_NEW = ('R = {"what": "E25 NB-M1: MTP levers on Q = NBDF-K100 Q (E28 K100 production), 8k-116k (Flash-Next klee100 3bpw-MTP on Merlin)",   # E25 NB-M1'
            if not LIGHT else
            'R = {"what": "E25 NB-M1L (light): MTP levers on Q = NBDF-K100 Q (E28 K100 production), windows 8k-116k (Flash-Next klee100 3bpw-MTP on Merlin)",   # E25 NB-M1L')


def edit_cells(cell1, cell2, cell3_6, b85x):
    out = _k100_edit_cells(cell1, cell2, cell3_6, b85x)
    c1 = out[0]
    c1 = once(c1, BUDGET_OLD, BUDGET_NEW, "c1 budget")
    c1 = once(c1, "NBDF_DEADMAN_SEC = 9300\n", f"NBDF_DEADMAN_SEC = {M1.DEADMAN_MIN * 60}   # E25 NB-M1: 5 min after the hard stop\n", "c1 deadman")
    c1 = once(c1, 'R = {"what": "E27 NBDF-K100: DFlash2 vs MTP K3 on the Q-defer base, 8k-116k (Flash-Next klee100 3bpw-MTP on Merlin)",   # E27 K100',
              WHAT_NEW, "c1 what")
    return (SWITCH_BLOCK + c1,) + tuple(out[1:])


PROF_BLOCK = '''        if is_q and M1_SWITCH.get("PROF", True):   # E25 NB-M1 PROF: the stage-1 profiler on Q (no change), NB-S1's 30-s windows
            for band, lvb in M1.PROF_WINDOWS:
                pk = f"profile_prof_{band}@{lvb}"
                if due(win_need("profile_" + band, lvb)) and s.alive() and not o.get("admission_deadlock"):
                    CUR["item"] = pk
                    try:
                        o["steady"][pk] = profile_window(p, dict(arm, key="PROF"), band, lvb, secs=float(M1.PROF_SECS))
                    except Exception as e:
                        o["steady"][pk] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:400]}
                else:
                    o["steady"][pk] = {"status": "SKIPPED_BOOT_BUDGET"}
                    o["items_skipped"].append(pk)
            _live(o, "steady", o["steady"])
'''

M1_LOOP = '''        for k in M1.LEVERS:                        # E25 NB-M1: every lever = Q + one change, in order (no cross-arm skip)
            a = ARMS[k]
            if not M1_SWITCH.get(k, True):          # CELL 1 arm switch off -> not booted (recorded, no other arm affected)
                skip_arm(k, "switched off in CELL 1 (NBM1_ON)", "SKIPPED_SWITCH_OFF")
                continue
            g_ = M1.pre_gate(a, KV_KIT_GATE)        # the real-kit gate (closed build: SKIPPED_KV_KIT_GATE) + the vocab table
            if g_:
                skip_arm(k, g_[1], g_[0], not_integrated=g_[0] == "NOT_INTEGRATED")
                continue
            if not N.arm_fits(left(), a):
                skip_arm(k, f"left {left() / 60:.1f} min < its full minimum {a['est_min']} + {N.START_RESERVE_SEC // 60} min "
                            "(priority = the NB-M1 order)", "SKIPPED_BUDGET")
                continue
            ok_p, why_p = pseq.before_arm(a)
            NBDF["patch"] = _patch_rec(pseq)
            if not ok_p:
                skip_arm(k, why_p, "TREE_INVALID_MIXED" if "INVALID" in why_p else "NOT_INTEGRATED", not_integrated=True)
                continue
            r = step(k, a["name"], a["est_min"], lambda k=k: runner(k))
            if N.needs_isolation(r) and not str(r.get("smoke_gate", "")).endswith("ZERO_ACCEPT"):
                PENDING.append(("iso", k))              # the K100 deferred Q-defer-off rerun of a died arm (budget permitting)
        _k100_run_pending(runner)
'''


def edit_body(body):
    b = _k100_edit_body(body)
    blk = M1_BLOCK_HEAD.replace("__CFG__", json.dumps(NBM1_CFG)) + "'''" + LIB_SRC + "'''" + M1_BLOCK_TAIL
    b = after_line(b, 'N.check_arm_plan(ARMS)                      # never-do', blk, "arm plan")
    # arm_body
    b = once(b, '''        is_q, is_d = arm["key"] == "Q", arm.get("method") == "dflash"
''', '''        is_q, is_d = arm["key"] == "Q", arm.get("method") == "dflash"
        is_lv = bool(arm.get("m1_lever"))           # E25 NB-M1: an MTP lever arm (Q + one change)
''', "is_lv")
    b = once(b, '''        if is_q:
            _live(o, "seam_inert", N.q_seam_inert(t))
        else:
            _live(o, "dflash_readback", dflash_rb(arm, t))''', '''        if not is_d:                              # E25 NB-M1: every MTP arm (Q and the levers) -> the seam-inert check
            _live(o, "seam_inert", N.q_seam_inert(t))
        else:
            _live(o, "dflash_readback", dflash_rb(arm, t))''', "readback mtp")
    b = once(b, '''        _live(o, "k100_gates", k100_boot_gates(''', '''        if is_lv and not FAKE:                    # E25 NB-M1: the lever's own change, read back (a real failure stops this arm only)
            o["i8_readback"] = i8_readback(t, list(getattr(s, "argv", None) or []), arm.get("draft_head"))
            _live(o, "m1_readback", M1.readback(arm, t, o.get("env_readback_raw"), o.get("config_readback"), o.get("workers"), o["i8_readback"]))
            if o["m1_readback"].get("real_failure"):
                o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="NB-M1 read-back: " + str(o["m1_readback"].get("why"))[:400])
                return o
        _live(o, "k100_gates", k100_boot_gates(''', "m1 readback")
    b = once(b, '''    for c in N.CONC_LADDER:''', '''    for c in (() if arm.get("m1_no_ladder") else N.CONC_LADDER):   # E25 NB-M1: a lever skips the ladder (Q's A/A keeps it)''',
             "smoke ladder")
    b = once(b, '''            if is_d and o["smoke_gate"].startswith("FAIL"):       # errors / garbage / short / zero accept / contamination''',
             '''            if (is_d or is_lv) and o["smoke_gate"].startswith("FAIL"):   # E25 NB-M1: levers too (FAIL_* = hard labels only)''',
             "smoke stop")
    b = once(b, '''        if due(300):
            CUR["item"] = "prefill_probe"''', '''        if due(300) and M1.item_on(arm, "prefill_probe"):   # E25 NB-M1: Q only (a lever reuses Q's PREFILL)
            CUR["item"] = "prefill_probe"''', "prefill probe")
    b = once(b, '''        if due(300) and not o.get("admission_deadlock"):
            CUR["item"] = "long_oracle"''', '''        if due(300) and not o.get("admission_deadlock") and M1.item_on(arm, "long_oracle"):   # E25 NB-M1: Q only
            CUR["item"] = "long_oracle"''', "long oracle")
    b = once(b, '''            o["long_gate"] = ("INCONCLUSIVE_ADMISSION_DEADLOCK" if o.get("admission_deadlock") else "INCONCLUSIVE_LONG_ORACLE_SKIPPED_BUDGET")''',
             '''            o["long_gate"] = ("INCONCLUSIVE_ADMISSION_DEADLOCK" if o.get("admission_deadlock") else
                              "NOT_IN_PLAN_NB_M1" if not M1.item_on(arm, "long_oracle") else "INCONCLUSIVE_LONG_ORACLE_SKIPPED_BUDGET")''',
             "long oracle label")
    b = once(b, '''        if due(240) and not o.get("admission_deadlock"):
            CUR["item"] = "prefix_cache"''', '''        if due(240) and not o.get("admission_deadlock") and M1.item_on(arm, "prefix"):   # E25 NB-M1: Q only (record-only)
            CUR["item"] = "prefix_cache"''', "prefix")
    b = once(b, '''            o["prefix_gate"] = "INCONCLUSIVE_PREFIX_SKIPPED_BUDGET"''',
             '''            o["prefix_gate"] = "NOT_IN_PLAN_NB_M1" if not M1.item_on(arm, "prefix") else "INCONCLUSIVE_PREFIX_SKIPPED_BUDGET"''', "prefix label")
    b = once(b, '''        for band, lvw, tag in K100_CFG["windows"]:''',
             '''        for band, lvw, tag in (arm.get("windows_fixed") or K100_CFG["windows"]):   # E25 NB-M1: a lever's A/A-free set (90k + 116k always)''',
             "windows")
    b = once(b, '''        for band, lvb in (("8k", int(K100_CFG["profile_level"])),) + ((("116k", lv116),) if long_ok and lv116 > 0 else ()):''',
             '''        for band, lvb in (((("8k", int(K100_CFG["profile_level"])),) + ((("116k", lv116),) if long_ok and lv116 > 0 else ()))
                          if M1.item_on(arm, "profiles") else ()):   # E25 NB-M1: Q's own profiles (verbatim); none on a lever''', "profiles")
    b = once(b, '''        if due(240) and s.alive():
            CUR["item"] = "tf_k100"''', '''        if due(240) and s.alive() and M1.item_on(arm, "tf"):   # E25 NB-M1: Q; a lever only when its plan has TF (IDX / LP)
            CUR["item"] = "tf_k100"''', "tf")
    b = once(b, '''            o["items_skipped"].append("tf_k100")
        CUR["item"] = "end"
''', '''            o["items_skipped"].append("tf_k100")
            if not M1.item_on(arm, "tf"):
                o["tf_plan"] = "NOT_IN_PLAN_NB_M1"
''' + PROF_BLOCK + '''        CUR["item"] = "end"
''', "tf else + prof")
    b = once(b, '''        if not is_q:
            o["dflash_readback"] = dflash_rb(arm, t)''', '''        if is_d:                                  # E25 NB-M1: every MTP arm -> the seam-inert check
            o["dflash_readback"] = dflash_rb(arm, t)''', "end readback")
    b = once(b, '''    if N.kv_bound(r) and n > 1 and band in ("40k",) + N.LONG_BANDS and due(win_need(band, n - 1)):''',
             '''    if N.kv_bound(r) and n > 1 and (band in ("40k",) + N.LONG_BANDS or arm.get("m1_lever")) and due(win_need(band, n - 1)):   # E25 NB-M1: a lever steps down on any band''',
             "kv-bound step down")
    b = once(b, '''RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P"''',
             '''RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P," + ",".join(M1.M1_ENV_KEYS)   # E25 NB-M1: + the GDN / vocab keys''', "rpc keys")
    b = once(b, '''def profile_window(port, arm, band="8k", n=None):''', '''def profile_window(port, arm, band="8k", n=None, secs=6.0):   # E25 NB-M1: secs (Q's own 6 s; PROF 30 s)''',
             "profile secs arg")
    b = once(b, "secs=6.0, max_new=", "secs=float(secs), max_new=", "profile secs")
    b = once(b, '"secs": 6.0', '"secs": float(secs)', "profile secs rec")
    b = once(b, '''    ok = pp == "FULLY_PATCHED" and (sm in (None, "FULLY_PATCHED")) and (arm["key"] == "Q" or sm == "FULLY_PATCHED")''',
             '''    ok = pp == "FULLY_PATCHED" and (sm in (None, "FULLY_PATCHED")) and (arm.get("method") != "dflash" or sm == "FULLY_PATCHED")   # E25 NB-M1''',
             "tree check")
    # arm_boot
    b = once(b, '''            NB4T_BUILD_ENV[lname] = {"DRAFT_HEAD": "int8"} if Q_I8["on"] else {}
''', '''            NB4T_BUILD_ENV[lname] = {"DRAFT_HEAD": "int8"} if Q_I8["on"] else {}
        elif arm.get("m1_lever"):                 # E25 NB-M1: the lever's draft head (Q's int8 / BF16 per Q_I8; I4 int4)
            M1.apply_head(N, arm, Q_I8["on"])
            NB4T_BUILD_ENV[lname] = {"DRAFT_HEAD": arm["draft_head"]} if arm["draft_head"] else {}
''', "lever head")
    b = once(b, '''        oom = o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]''',
             '''        oom = bool(o["oom_in_log"]) and not o["xid_in_log"]   # E25 NB-M1 (NB-S1 GPU v1: OOM at the first request after ready) -> the ladder''',
             "oom any stage")
    # nbdf_main
    b = once(b, '''        q = (step("Q", ARMS["Q"]["name"], ARMS["Q"]["est_min"], lambda: runner("Q")) if q_tree_ok
             else R["sections"]["Q"][ARMS["Q"]["name"]])''', '''        if q_tree_ok and not M1_SWITCH.get("Q", True):   # E25 NB-M1: CELL 1 switch (Q off: the levers still run)
            skip_arm("Q", "switched off in CELL 1 (NBM1_ON)", "SKIPPED_SWITCH_OFF")
        q = (step("Q", ARMS["Q"]["name"], ARMS["Q"]["est_min"], lambda: runner("Q")) if q_tree_ok and M1_SWITCH.get("Q", True)
             else R["sections"]["Q"][ARMS["Q"]["name"]])
        R["sections"].setdefault("PROF", {})["nbm1_prof"] = M1.prof_record(q, M1_SWITCH)   # E25 NB-M1 PROF (inside Q's boot)
        save()''', "Q switch + PROF")
    i = b.index("        stop_d, isos = None, 0")
    j = b.index("        _k100_run_pending(runner)\n", i) + len("        _k100_run_pending(runner)\n")
    assert b.count("        stop_d, isos = None, 0") == 1 and "for k in N.D_ARMS:" in b[i:j]
    b = b[:i] + M1_LOOP + b[j:]
    b = once(b, '''    v["minutes"] = round((time.time() - T_START) / 60, 1)''', '''    v["minutes"] = round((time.time() - T_START) / 60, 1)
    try:                                      # E25 NB-M1 records (raw numbers per arm, every lever vs Q per band; record only)
        v["nbm1_records"] = {k: M1.arm_records(T, _arm_rec(k), k, _arm_rec("Q"), N.LONG_THR_FLOOR) for k in list(ARMS) + list(ISO)}
    except Exception as e:
        v["nbm1_records"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    v["nbm1_prof"] = (R["sections"].get("PROF") or {}).get("nbm1_prof")
    v["nbm1_switch"] = M1_SWITCH''', "finalize records")
    assert "for k in N.D_ARMS:" not in b[b.index("def nbdf_main("):]
    if LIGHT:
        b = edit_body_light(b)
    return b


ORACLE_HEAD = '''        if not due(240):
            o["items_skipped"].append("oracle")
            o["status"] = "SKIPPED_BOOT_BUDGET_ORACLE"
            return o
'''
ORACLE_END = '''        if arm.get("short_only"):'''


def edit_body_light(b):
    """E25 NB-M1L (light) body edits, after every NB-M1 edit: (1) the greedy oracle block (Q's reference + Q's oracle A/A,
    a lever's oracle vs Q and its quality gate) runs only when the arm's plan has it (M1.item_on(arm, "oracle"); light:
    never) -> recorded NOT_IN_PLAN_NB_M1L; (2) arm_boot's evidence gate: an arm without an oracle in its plan is gated on the
    hard-only evidence (probe request errors, the sampled first request, hard garbage in a window) instead of a missing
    oracle (INCONCLUSIVE_NO_ORACLE would replace its DONE status)."""
    assert b.count(ORACLE_HEAD) == 1 and b.count(ORACLE_END) == 1
    i = b.index(ORACLE_HEAD)
    j = b.index(ORACLE_END, i)
    blk = b[i:j]
    assert 'CUR["item"] = "oracle_aa"' in blk and 'o["quality_gate"] = quality_gate(o, ref)' in blk and "'''" not in blk and '"""' not in blk
    b = b[:i] + ('        if M1.item_on(arm, "oracle"):           # E25 NB-M1L: the oracle only when the arm\'s plan has it (light: never)\n'
                 + textwrap.indent(blk, "    ") +
                 '        else:\n'
                 '            o["items_skipped"].append("oracle")\n'
                 '            o["oracle_vs"], o["quality_gate"] = None, "NOT_IN_PLAN_NB_M1L"   # recorded only (verdict: INCONCLUSIVE_ORACLE)\n') + b[j:]
    b = once(b, '''    return gated(o, need=(ref,), expected_rows=4)
''', '''    if not M1.item_on(arm, "oracle"):         # E25 NB-M1L: no oracle in the plan -> the hard-only evidence (probes, sampled, windows)
        return gated(o, need=(), expected_rows=None)
    return gated(o, need=(ref,), expected_rows=4)
''', "light gated")
    return b


def edit_meta(md0, flags):
    _, f = _k100_edit_meta(md0, flags)
    md = ("# E25 NB-M1: MTP levers on the proven base Q = NBDF-K100 Q = E28 K100 production (Qwen3.8-Flash-Next klee100 3bpw-MTP, "
          "Merlin v3.0.0, 4 x L4), 8k to 116k\n\n"
          "Q (MTP K3, I8 int8 head, Q-defer, seqs 24, pin 7,933,526,016; A/A, TF, oracle, profiles) > V65 / V32 (reduced draft vocab) > "
          "I4 (int4 head) > IDX (fp8 indexer cache) > S28 / S32 (seqs) > LP (long prefill) > G1 / G2 (GDN switches) > K4 > DYN "
          "(K4 at <= 8 running, K3 at 9-24); PROF = two 30-s profiles inside Q's boot. Each lever = Q + one change. Arm switches: "
          f"NBM1_ON at the top of CELL 1 (copy A: Q V65 V32 I4, copy B: Q IDX S28 S32 LP, copy C: Q G1 G2 K4 DYN PROF). Hard stop "
          f"{M1.BUDGET_MIN} min. Readable code: nbm1_cell7_body.py. Commit run only; a competition rerun writes CELL 0's identity "
          "submission (E18 R2 format).\n")
    if LIGHT:
        md = ("# E25 NB-M1L (LIGHT): MTP levers on the proven base Q = NBDF-K100 Q = E28 K100 production (Qwen3.8-Flash-Next klee100 "
              "3bpw-MTP, Merlin v3.0.0, 4 x L4), windows 8k to 116k\n\n"
              "Q (MTP K3, I8 int8 head, Q-defer, seqs 24, pin 7,933,526,016): facts, smoke (no ladder), prefill probe, warmup, windows "
              "8k@16 x 2 / 20k@24 / 40k@16 / 90k@8 / 116k@6 (no oracle, A/A, long oracle, prefix, TF, profiles). Each lever = Q + one "
              "change: facts, the smoke hard gate, the read-back of its change, warmup, its windows (90k and 116k always; no TF, no "
              "oracle). PROF = two 30-s profiles inside Q's boot. Arm switches: NBM1_ON at the top of CELL 1 (light copy B: Q IDX S28 "
              f"S32 LP, light copy C: Q G1 G2 K4 DYN PROF). Hard stop {M1.BUDGET_MIN} min. Readable code: nbm1l_cell7_body.py. Commit "
              "run only; a competition rerun writes CELL 0's identity submission (E18 R2 format).\n")
    f["NBM1"] = {"lib": M1.LIB_VERSION, "lib_sha256": LIB_SHA, "source_refs": "nbm1_build_facts.json (size: the notebook stays < 400 kB)"}
    if LIGHT:
        f["NBM1"].update(overlay_sha256=OVL_SHA, source_refs="nbm1l_build_facts.json (size: the notebook stays < 400 kB)")
    MERLIN_REFS.update({"vocab": ["vllm/models/qwen4_exp/nvidia/draft_head.py:32-74 (VLLM_DRAFT_VOCAB_FILE / VLLM_DRAFT_VOCAB_SHA256, "
                                      "load + validate, N % (128 x TP))", "mtp.py:525-600 (the N-row head on the last PP stage, "
                                      "has_own_lm_head)", "mtp.py:697-752 (select_head_rows, 'REDUCED DRAFT VOCAB ACTIVE' per TP rank)",
                                      "v1/worker/gpu/spec_decode/speculator.py:547-603 (draft_lm_head_quant wraps the draft's own "
                                      "ParallelLMHead: the reduced head is quantized; N/TP % 64)",
                                      "tools/merlin/launch.py:424-425 (build() pops the two env vars: NB1_ENV is applied after build())"],
                            "i4": ["tools/merlin/launch.py:319-322 (DRAFT_HEAD bf16 / int8 / int4 -> draft_lm_head_quant)",
                                   "vllm/config/speculative.py:610-622"],
                            "idx": ["vllm/engine/arg_utils.py:1856 --attention-config (JSON)", "vllm/config/attention.py indexer_kv_dtype"],
                            "lp": ["vllm/engine/arg_utils.py:600-602, 1736-1740, 2632-2634"],
                            "gdn": ["vllm/merlin/envs.py:544-555 (strict switches), :822-823 (not in the compile hash)"],
                            "dyn": ["vllm/config/vllm.py:1069-1105 (PIECEWISE override only without the V2 runner; DP disable)",
                                    "vllm/v1/worker/gpu/model_runner.py:2786-2803 (propose(num_speculative_tokens=K), drafts[:, K:] = -1)"]})
    return md, f


MERLIN_REFS = {}


def gate_extra(gate, rep):
    """NBDF-K100's gate (its real-kit checks incl. the D3s / D4s per-K KV successes of the template's R9 runs) + every
    NB-M1 real-kit check (never set by hand: it can only close the gate)."""
    g = _k100_gate(gate, rep)
    chk = ((rep or {}).get("verdict") or {}).get("checks") or {}
    miss = [k for k in NBM1_KIT_KEYS if chk.get(k) is not True]
    if miss:
        g["passed"] = False
        g["why"] = list(g.get("why") or []) + [f"NB-M1 real-kit checks not True: {miss}"]
    g["nbm1_kit_checks"] = {k: chk.get(k) for k in NBM1_KIT_KEYS}
    return g


KX.edit_body, KX.edit_meta, KX.k100_gate_extra, KX.edit_cells = edit_body, edit_meta, gate_extra, edit_cells

# ---------------------------------------------------------------- the time plan (build-time assertion)
import importlib.util as _ilu  # noqa: E402
_spec0 = _ilu.spec_from_file_location("nbdf_lib_nbm1_plan", str(E27B / "nbdf_lib.py"))   # a PRIVATE copy: configure() below
_N0 = _ilu.module_from_spec(_spec0)                                                       # must not touch the module the NBDF
_spec0.loader.exec_module(_N0)                                                            # builder imports (its own asserts)
import nb4_ppack as _P0  # noqa: E402
_N0.configure(SEQS=24, CONC_LADDER=(2, 8, 23), Q_CAPTURES=list(KX.K100_CFG["q_captures"]),
              Q_TAIL=["--compilation-config", _N0.compilation_json(KX.K100_CFG["q_captures"])], WKEYS=tuple(KX.K100_CFG["wkeys"]))
_arms0 = _N0.arm_plan(["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}",
                       "--max-num-seqs", "24"], {"VLLM_SERVER_DEV_MODE": "1", "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, 7933526016,
                      "/kaggle/input/models/foysalemonshanto/" + _N0.DRAFTER_REL, q_env=_P0.server_env("Q-defer"))
_vfake = {k: {"path": f"/kaggle/input/datasets/{M1.VOCAB_DS}/{v[0]}", "found": True} for k, v in M1.VOCAB_TABLES.items()}
PLAN_ARMS = M1.m1_arms(_N0, _arms0["Q"], None, _vfake)
import nb4t_lib as _T0  # noqa: E402
M1.check_arms(_N0, _T0, PLAN_ARMS, _arms0["Q"])
WALKS = {x: M1.copy_estimate(PLAN_ARMS, sw) for x, sw in M1.COPIES.items()}
for x, w in WALKS.items():
    assert w["all_run"], (x, w)
    assert w["runtime_min"] <= M1.COPY_RUNTIME_MAX, (x, w["runtime_min"])
    assert w["runtime_min"] + M1.START_RESERVE_MIN <= M1.BUDGET_MIN, (x, w)
if not LIGHT:
    assert M1.BUDGET_MIN == 5 * -(-max(w["runtime_min"] + M1.START_RESERVE_MIN for w in WALKS.values()) // 5), (
        "BUDGET_MIN must be the copies' max planned walk (+ the 5-min start reserve) rounded up to 5 min",
        {x: w["runtime_min"] for x, w in WALKS.items()})
NEEDS = {}
if LIGHT:                                     # NB-M1L: the hard stop also holds every arm's FULL body cap (+ the 5-min reserve)
    NEEDS = {x: M1.copy_need_min(PLAN_ARMS, sw) for x, sw in M1.COPIES.items()}
    _top = max(max(w["runtime_min"] for w in WALKS.values()), max(NEEDS.values())) + M1.START_RESERVE_MIN
    assert M1.BUDGET_MIN == 5 * -(-_top // 5) and M1.DEADMAN_MIN == M1.BUDGET_MIN + 5, (
        "light BUDGET_MIN = max(runtime, full-cap end) + 5 min rounded up to 5", {x: w["runtime_min"] for x, w in WALKS.items()}, NEEDS)
    _q = PLAN_ARMS["Q"]
    assert _q["m1_items"] == M1.Q_ITEMS and _q["m1_no_ladder"] is True and not any(
        M1.item_on(_q, x) for x in ("ladder", "oracle", "oracle_aa", "long_oracle", "prefix", "profiles", "tf")) and M1.item_on(_q, "prefill_probe")
    for k_ in M1.LEVERS:
        assert not M1.item_on(PLAN_ARMS[k_], "tf") and not M1.item_on(PLAN_ARMS[k_], "oracle"), k_
    for k_, a_ in PLAN_ARMS.items():               # the body's due() never starves a window inside the arm's own cap
        assert a_["cap_min"] >= a_["time_model"]["body"] + 1, (k_, a_["cap_min"], a_["time_model"]["body"])

# ---------------------------------------------------------------- run the NBDF builder (as NBDF-K100 / NB-S1 are built)
argv0 = list(sys.argv)
sys.argv = [str(E27B / "build_e27_nbdf.py")] + argv0[1:]
try:
    runpy.run_path(str(E27B / "build_e27_nbdf.py"), run_name="__main__")
finally:
    sys.argv = argv0

for p in FROZEN:
    assert sha(p.read_bytes()) == FROZEN_SHA[str(p)], f"FROZEN FILE CHANGED: {p}"

# ---------------------------------------------------------------- the three copies (only the NBM1_ON line differs)
facts_p = OUT_DIR / f"{PFX}_build_facts.json"
facts = json.loads(facts_p.read_text(encoding="utf-8"))
base_push = Path(facts["push_dir"])
data = OUT_NB.read_bytes()
assert sha(data) == facts["sha256"]
nb = json.loads(data.decode("utf-8"))
export = (OUT_DIR / f"{PFX}_code_export.py").read_text(encoding="utf-8")
assert export.count(SWITCH_LINE) == 1 and export.startswith("# ==") and "NBM1_ON =" not in (OUT_DIR / f"{PFX}_cell7_body.py").read_text(encoding="utf-8")
c1 = nb["cells"][2]["source"]
assert c1.count(SWITCH_LINE) == 1, "the switch line must be one source line of CELL 1"
base_meta = json.loads((base_push / "kernel-metadata.json").read_text(encoding="utf-8"))
COPY_INFO = {}
for x, sw in M1.COPIES.items():
    owner = M1.OWNERS[x]
    slug_x = f"{SLUG}-{x}"
    line = "NBM1_ON = " + repr(sw) + "\n"
    nbx = json.loads(data.decode("utf-8"))
    src = nbx["cells"][2]["source"]
    src[src.index(SWITCH_LINE)] = line
    dx = json.dumps(nbx, separators=(",", ":")).encode("utf-8")
    assert len(dx) < 400_000 and b"\r" not in dx.replace(b"\\r", b"")
    code_x = "\n\n".join("".join(c["source"]) for c in nbx["cells"] if c["cell_type"] == "code")
    assert code_x == export.replace(SWITCH_LINE, line), f"copy {x}: differs from the base beyond the NBM1_ON line"
    if not TEST_OUT:
        (NBDIR / f"Q27-XHIGH-E25_{TAG}_{x.upper()}.ipynb").write_bytes(dx)
    dst = (TEST_OUT / f"push_{x}") if TEST_OUT else PUSH_ROOT / f"{slug_x}__{owner}"
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.glob("*.ipynb"):
        f.unlink()
    (dst / f"{slug_x}.ipynb").write_bytes(dx)
    if LIGHT:                                 # NB-M1L: every kernel source of NB-M1 kept (the e29 payload, the dflash2v2 model)
        ds = list(base_meta["dataset_sources"])
        assert M1.VOCAB_DS in ds and any("dflash2v2" in m_ for m_ in base_meta["model_sources"]), base_meta
    else:
        ds = [d for d in base_meta["dataset_sources"] if d != M1.VOCAB_DS or any(sw.get(k) for k in M1.VOCAB_TABLES)]
    meta = dict(base_meta, id=f"{owner}/{slug_x}", title=f"Q27 Xhigh E25 {TAG} {x.upper()}", code_file=f"{slug_x}.ipynb", dataset_sources=ds)
    (dst / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    COPY_INFO[x] = {"owner": owner, "slug": slug_x, "push_dir": str(dst), "sha256": sha(dx), "bytes": len(dx), "switch": sw,
                    "switch_line": line.rstrip("\n"), "dataset_sources": ds, "walk": WALKS[x],
                    "per_arm_est_min": {w_["arm"]: w_["est_min"] for w_ in WALKS[x]["walk"]}}
    if LIGHT:
        COPY_INFO[x].update(model_sources=list(base_meta["model_sources"]), need_min=NEEDS[x],
                            per_arm_cap_min={k_: (M1.q_time(sw["PROF"])[1] if k_ == "Q" else PLAN_ARMS[k_]["cap_min"])
                                             for k_ in M1.ARM_KEYS if sw.get(k_)})
if not TEST_OUT:
    assert base_push == PUSH_ROOT / f"{SLUG}__foysalemonshanto", base_push
    shutil.rmtree(base_push)                  # the all-on notebook is the kit notebook only; the copies are the pushes
facts.update(owner_copies={x: v["owner"] for x, v in COPY_INFO.items()}, copies=COPY_INFO, base_push_removed=str(base_push),
             nbm1={"lib_sha256": LIB_SHA, "cfg": NBM1_CFG, "arms": list(M1.ARM_KEYS), "switch_default": NBM1_ON_DEFAULT,
                   "switch_line": SWITCH_LINE.rstrip("\n"), "kit_keys": list(NBM1_KIT_KEYS), "frozen_files_unchanged": len(FROZEN),
                   "plan": {k: {x: a.get(x) for x in ("title", "port", "spec", "captures", "streams", "windows_fixed", "est_min", "cap_min",
                                                      "m1_change", "m1_items", "m1_head", "time_model")} for k, a in PLAN_ARMS.items()},
                   "budget_min": M1.BUDGET_MIN, "deadman_min": M1.DEADMAN_MIN, "merlin_refs": MERLIN_REFS,
                   "vocab": {k: list(v) for k, v in M1.VOCAB_TABLES.items()}, "vocab_dataset": M1.VOCAB_DS})
if LIGHT:
    facts["nbm1"].update(light=True, overlay_sha256=OVL_SHA, overlay_version=M1.LIB_VERSION, q_items=M1.Q_ITEMS, lever_items=M1.LEVER_ITEMS,
                         needs_min=NEEDS, copy_runtime_max=M1.COPY_RUNTIME_MAX,
                         q_time={"b": M1.q_time(False), "c": M1.q_time(True)})
facts_p.write_bytes((json.dumps(facts, indent=1, default=str) + "\n").encode("utf-8"))
for p in FROZEN:                              # again after the copies (NB-M1L: no NB-M1 run file / push folder touched)
    assert sha(p.read_bytes()) == FROZEN_SHA[str(p)], f"FROZEN FILE CHANGED: {p}"
print(json.dumps({"notebook": str(OUT_NB), "sha256": facts["sha256"], "bytes": facts["bytes"], "gpu_ready": facts["gpu_ready"],
                  "lib_sha256": LIB_SHA, "copies": {x: {k: v[k] for k in ("push_dir", "sha256", "bytes")} | {"runtime_min": v["walk"]["runtime_min"]}
                                                    for x, v in COPY_INFO.items()}}, indent=1))
