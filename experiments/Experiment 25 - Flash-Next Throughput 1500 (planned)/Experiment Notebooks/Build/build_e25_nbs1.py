"""E25 NB-S1 (coordinator / user 2026-11-01: DFlash2 ONLY): levers on top of B = NBDF-K100's D3s.

  python build_e25_nbs1.py [--kv-gate-report FILE] [--test-out DIR]
      -> ../Q27-XHIGH-E25_NBS1.ipynb, nbs1_code_export.py, nbs1_cell7_body.py, nbs1_build_facts.json and the GPU push folder
         D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nbs1__foysalemonshanto/ (private; nothing is pushed)

Built by the NBDF builder itself, exactly as NBDF-K100 is (build_e27_nbdf.py with NBDF_VARIANT=k100 and the K100 module's
edits), with the K100 module's names re-pointed IN MEMORY to NB-S1's outputs and its edit_body / edit_meta / gate wrapped:
the K100 edits run first, then the NB-S1 edits below (exact-once text replacements). No NBDF / K100 file is written or
changed (nbdf_lib.py, build_e27_nbdf*.py, dflash_seam.py, the K100 bodies and kits stay byte-identical: asserted).

NB-S1 edits (nbs1_lib.py embedded verbatim, sha-checked at run time):
  - arms: B (= NBDF-K100 D3s, under nbdf_lib's base key Q: the A/A, the TF envelope and k100_rel are B's), G1, G2, EH, F8,
    DYN (nbs1_lib.s1_arms; check_arms: each = B + one change); Q_I8 off (S4: no draft head on DFlash2);
  - nbdf_main: B gets the D-arm prerequisites (drafter, kit gate, seam) before its launch; every other line is NBDF-K100's
    (the seam apply / verify / revert, the D loop with each arm's own checks, the pending reruns, finalize);
  - arm_body: a DFlash2 base arm reads its DFlash read-back (not the MTP seam-inert check); the NB-S1 read-back of each
    arm's own change (a real failure stops that arm only); the GDN keys in the worker env read-back;
  - the profiled window: 20k@24 on B only, 30 s of load (the trace stays bounded at 40 iterations).
"""
import hashlib
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
NBDIR = HERE.parent
TEST_OUT = Path(sys.argv[sys.argv.index("--test-out") + 1]) if "--test-out" in sys.argv else None
OUT_DIR = TEST_OUT or HERE
OUT_NB = (TEST_OUT or NBDIR) / "Q27-XHIGH-E25_NBS1.ipynb"
SLUG = "q27-xhigh-e25-nbs1"
OWNER = "foysalemonshanto"                           # coordinator 2026-11-01: foysal owns every source (2 GPU commits / account)
PUSH_ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
LIB = HERE / "nbs1_lib.py"


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


# the frozen NBDF / K100 files: byte-identical before and after this build
FROZEN = [E27B / n for n in ("nbdf_lib.py", "build_e27_nbdf.py", "build_e27_nbdf_k100.py", "build_e27_nbdf_cpu_kit.py",
                             "build_e27_nbdf_k100_cpu_kit.py", "dflash_seam.py", "nbdf_k100_cell7_body.py", "nbdf_cell7_body.py",
                             "nbdf_k100_code_export.py", "nbdf_k100_build_facts.json", "nbdf_build_facts.json")]
FROZEN += [E27B.parent / "Q27-XHIGH-E27_NBDF_K100.ipynb", E27B.parent / "Q27-XHIGH-E27_NBDF_DFLASH2.ipynb"]
FROZEN_SHA = {str(p): sha(p.read_bytes()) for p in FROZEN}

LIB_SRC = LIB.read_text(encoding="utf-8")
assert "\r" not in LIB_SRC and "'''" not in LIB_SRC, "nbs1_lib.py: LF only, no triple single quotes (embedded as r''' ''')"
LIB_SHA = sha(LIB_SRC)
NBS1_CFG = {"profile": ["20k", 24], "profile_secs": 30.0, "lib_sha256": LIB_SHA}

os.environ["NBDF_VARIANT"] = "k100"
sys.dont_write_bytecode = True
sys.path.insert(0, str(E27B))
sys.path.insert(0, str(HERE))
import build_e27_nbdf_k100 as KX  # noqa: E402  (module import only: its build runs under __main__)
import nbs1_lib as S1  # noqa: E402

# ---------------------------------------------------------------- the K100 module, re-pointed in memory (never on disk)
KX.K100_CFG["q_i8"] = False                          # S4: B is DFlash2 (no DRAFT_HEAD / draft_lm_head_quant)
KX.K100_CFG["wkeys"] = ["8k@16#1", "8k@16#2", "profile_20k@24", "40k@16"]
KX.OUT_NB_NAME = str(OUT_NB)                         # NBDIR / <absolute path> = the absolute path
KX.SLUG = SLUG
KX.EXPORT_NAME, KX.BODY_NAME, KX.FACTS_NAME = (str(OUT_DIR / "nbs1_code_export.py"), str(OUT_DIR / "nbs1_cell7_body.py"),
                                               str(OUT_DIR / "nbs1_build_facts.json"))
KX.TITLE = "Q27 Xhigh E25 NBS1"
_k100_edit_body, _k100_edit_meta, _k100_gate = KX.edit_body, KX.edit_meta, KX.k100_gate_extra
NBS1_KIT_KEYS = ("nbs1_arm_plan_ok", "nbs1_launch_path_all_arms_ok", "nbs1_help_flag_gate_real_ok")


def once(text, old, new, where):
    n = text.count(old)
    assert n == 1, f"NB-S1 anchor {where}: {n} matches"
    return text.replace(old, new)


def after_line(text, needle, add, where):
    assert text.count(needle) == 1, f"NB-S1 anchor {where}: {text.count(needle)} matches"
    i = text.index(needle)
    j = text.index("\n", i) + 1
    return text[:j] + add + text[j:]


S1_BLOCK = '''# ---- E25 NB-S1 (DFlash2 only): B = NBDF-K100 D3s under the base key Q + G1 / G2 / EH / F8 / DYN (nbs1_lib, embedded)
NBS1_CFG = json.loads(r"""__CFG__""")
NBS1_LIB_SRC = r\'\'\'__LIB__\'\'\'
if hashlib.sha256(NBS1_LIB_SRC.encode("utf-8")).hexdigest() != NBS1_CFG["lib_sha256"]:
    raise RuntimeError("NB-S1: the embedded nbs1_lib does not match its build sha")
import types as _nbs1_types
S1 = _nbs1_types.ModuleType("nbs1_lib")
exec(compile(NBS1_LIB_SRC, "nbs1_lib.py", "exec"), S1.__dict__)
ARMS = S1.s1_arms(N, ARMS["D3s"], ARMS["Q"])
S1.install(N, ARMS)
S1.check_arms(N, T, ARMS)                   # raises: no boot
NBDF["nbs1"] = {"lib": S1.LIB_VERSION, "cfg": NBS1_CFG, "arms": {k: a["title"] for k, a in ARMS.items()}, "not_built": S1.NOT_BUILT,
                "f8_source_check": S1.F8_SOURCE_CHECK, "dyn_source_check": S1.DYN_SOURCE_CHECK}
_nbs1_on = globals().get("NBS1_ON")          # CELL 1 arm switches (E25 NB-S1); absent -> every arm on
NBS1_SWITCH = {k: True if _nbs1_on is None else bool(_nbs1_on.get(k, False)) for k in ARMS}
NBDF["nbs1"]["switch"] = {"NBS1_ON": _nbs1_on, "effective": NBS1_SWITCH, "unknown_keys": sorted(set(_nbs1_on or {}) - set(ARMS))}
print("E25 NB-S1 arm switches:", NBS1_SWITCH, flush=True)
'''


NBS1_ON_DEFAULT = {"Q": True, "F8": True, "G1": False, "G2": False, "EH": False, "DYN": False}   # user 2026-11-01: B + F8
SWITCH_BLOCK = ("# ==================================================================== E25 NB-S1 ARM SWITCHES (edit before the commit)\n"
                "# True = the arm runs, False = it is not booted (recorded SKIPPED_SWITCH_OFF; the other arms are unaffected).\n"
                "# Q = B (the D3s base). B only: \"F8\": False. Editing this line needs no rebuild and no new kit (every arm is kit-checked).\n"
                "NBS1_ON = " + repr(NBS1_ON_DEFAULT) + "\n")
_k100_edit_cells = KX.edit_cells


def edit_cells(cell1, cell2, cell3_6, b85x):
    out = _k100_edit_cells(cell1, cell2, cell3_6, b85x)
    return (SWITCH_BLOCK + out[0],) + tuple(out[1:])


def edit_body(body):
    b = _k100_edit_body(body)
    blk = S1_BLOCK.replace("__CFG__", json.dumps(NBS1_CFG)).replace("__LIB__", LIB_SRC)
    b = after_line(b, 'N.check_arm_plan(ARMS)                      # never-do', blk, "arm plan")
    b = once(b, '''        if is_q:
            _live(o, "seam_inert", N.q_seam_inert(t))
        else:
            _live(o, "dflash_readback", dflash_rb(arm, t))''', '''        if is_q and not is_d:                     # E25 NB-S1: B (key Q) is DFlash2 -> its DFlash read-back
            _live(o, "seam_inert", N.q_seam_inert(t))
        else:
            _live(o, "dflash_readback", dflash_rb(arm, t))''', "readback B")
    b = once(b, '''        if not is_q:
            o["dflash_readback"] = dflash_rb(arm, t)''', '''        if is_d:                                  # E25 NB-S1: B (key Q) is DFlash2
            o["dflash_readback"] = dflash_rb(arm, t)''', "end readback B")
    b = once(b, '''        _live(o, "k100_gates", k100_boot_gates(''', '''        if not FAKE:                              # E25 NB-S1: each arm's own change, read back (a real failure stops this arm only)
            _live(o, "s1_readback", S1.readback(arm, t, o.get("env_readback"), o.get("dflash_readback")))
            if o["s1_readback"].get("real_failure"):
                o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="NB-S1 read-back: " + str(o["s1_readback"].get("why"))[:400])
                return o
        _live(o, "k100_gates", k100_boot_gates(''', "s1 readback")
    b = once(b, '''RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P"''',
             '''RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P," + ",".join(S1.S1_ENV_KEYS)   # E25 NB-S1: + the GDN switches''', "rpc keys")
    b = once(b, '''        lv116 = (o["long_levels"].get("116k") or {}).get("level") or 0
        for band, lvb in (("8k", int(K100_CFG["profile_level"])),) + ((("116k", lv116),) if long_ok and lv116 > 0 else ()):''',
             '''        for band, lvb in S1.profile_plan(arm, NBS1_CFG):   # E25 NB-S1: 20k@24 on B only''', "profile plan")
    b = once(b, "secs=6.0", 'secs=float(NBS1_CFG["profile_secs"])', "profile secs")
    b = once(b, '"secs": 6.0', '"secs": float(NBS1_CFG["profile_secs"])', "profile secs rec")
    b = once(b, '''        q = (step("Q", ARMS["Q"]["name"], ARMS["Q"]["est_min"], lambda: runner("Q")) if q_tree_ok
             else R["sections"]["Q"][ARMS["Q"]["name"]])''', '''        b_gate = S1.base_gate(q_tree_ok, DRAFTER, KV_KIT_GATE, seam_ok, seam_why)   # E25 NB-S1: B is a DFlash2 arm
        if not b_gate and not NBS1_SWITCH.get("Q", True):
            b_gate = ("SKIPPED_SWITCH_OFF", "switched off in CELL 1 (NBS1_ON)")
        if b_gate:
            skip_arm("Q", b_gate[1], b_gate[0], not_integrated=b_gate[0] == "NOT_INTEGRATED")
        q = (step("Q", ARMS["Q"]["name"], ARMS["Q"]["est_min"], lambda: runner("Q")) if q_tree_ok and not b_gate
             else R["sections"]["Q"][ARMS["Q"]["name"]])''', "B gate")
    b = once(b, '''        for k in N.D_ARMS:
            a = ARMS[k]
''', '''        for k in N.D_ARMS:
            a = ARMS[k]
            if not NBS1_SWITCH.get(k, True):         # E25 NB-S1: CELL 1 arm switch off -> not booted (recorded, no other arm affected)
                skip_arm(k, "switched off in CELL 1 (NBS1_ON)", "SKIPPED_SWITCH_OFF")
                continue
''', "arm switch")
    b = once(b, '''        oom = o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]''',
             '''        oom = bool(o["oom_in_log"]) and not o["xid_in_log"]   # E25 NB-S1 (GPU v1: OOM at the first request after ready) -> the ladder''',
             "oom any stage")
    b = once(b, '''    v["minutes"] = round((time.time() - T_START) / 60, 1)''', '''    v["minutes"] = round((time.time() - T_START) / 60, 1)
    v["nbs1_records"] = {k: S1.arm_records(_arm_rec(k), k) for k in ARMS}      # E25 NB-S1''', "finalize records")
    return b


def edit_meta(md0, flags):
    _, f = _k100_edit_meta(md0, flags)
    md = ("# E25 NB-S1: DFlash2 levers on B = NBDF-K100 D3s (Qwen3.8-Flash-Next klee100 3bpw-MTP, Merlin v3.0.0, 4 x L4), 8k to 116k\n\n"
          "Arms: B (D3s: seam, SW-1024, BF16 drafter KV, FLASH_ATTN, no draft head; A/A) > G1 (VLLM_GDN_DECODE_FASTPATH) > G2 "
          "(VLLM_GDN_SHARED_ALIGN_INDEX) > EH (DFLASH_DRAFT_EMBED_HOST) > F8 (FP8 drafter KV, TRITON_ATTN, SW 2048) > DYN (K4 at <= 8 "
          "running, K3 at 9-24). Seqs 24, pin 7,933,526,016. Readable code: nbs1_cell7_body.py. Arm switches: NBS1_ON at the top of CELL 1 (default B + F8). Commit run only; a competition "
          "rerun writes CELL 0's identity submission (E18 R2 format).\n")
    f["NBS1"] = {"cfg": NBS1_CFG, "arms": S1.TITLES, "not_built": S1.NOT_BUILT, "lib": S1.LIB_VERSION}
    return md, f


def gate_extra(gate, rep):
    """NBDF-K100's gate (the K100 real kit: D3s per-K success included) + every NB-S1 real-kit check (never set by hand)."""
    g = _k100_gate(gate, rep)
    chk = ((rep or {}).get("verdict") or {}).get("checks") or {}
    miss = [k for k in NBS1_KIT_KEYS if chk.get(k) is not True]
    if miss:
        g["passed"] = False
        g["why"] = list(g.get("why") or []) + [f"NB-S1 real-kit checks not True: {miss}"]
    g["nbs1_kit_checks"] = {k: chk.get(k) for k in NBS1_KIT_KEYS}
    return g


KX.edit_body, KX.edit_meta, KX.k100_gate_extra, KX.edit_cells = edit_body, edit_meta, gate_extra, edit_cells

# ---------------------------------------------------------------- run the NBDF builder (as NBDF-K100 is built)
argv0 = list(sys.argv)
sys.argv = [str(E27B / "build_e27_nbdf.py")] + argv0[1:]
try:
    runpy.run_path(str(E27B / "build_e27_nbdf.py"), run_name="__main__")
finally:
    sys.argv = argv0

for p in FROZEN:
    assert sha(p.read_bytes()) == FROZEN_SHA[str(p)], f"FROZEN FILE CHANGED: {p}"

# ---------------------------------------------------------------- the push folder (the NBDF builder writes <slug>__foysalemonshanto)
facts_p = OUT_DIR / "nbs1_build_facts.json"
facts = json.loads(facts_p.read_text(encoding="utf-8"))
dst = Path(facts["push_dir"])
if not TEST_OUT:
    assert dst == PUSH_ROOT / f"{SLUG}__{OWNER}", dst
    shutil.rmtree(PUSH_ROOT / f"{SLUG}__hosen42", ignore_errors=True)    # the earlier hamid-account folder (retargeted to foysal)
meta = json.loads((dst / "kernel-metadata.json").read_text(encoding="utf-8"))
assert meta["id"] == f"{OWNER}/{SLUG}" and meta["is_private"] is True, meta["id"]
assert sha((dst / f"{SLUG}.ipynb").read_bytes()) == facts["sha256"] == sha(OUT_NB.read_bytes())
facts.update(push_dir=str(dst), owner=OWNER, nbs1={"lib_sha256": LIB_SHA, "cfg": NBS1_CFG, "arms": list(S1.ARM_KEYS),
                                                   "not_built": S1.NOT_BUILT, "kit_keys": list(NBS1_KIT_KEYS), "switch_default": NBS1_ON_DEFAULT,
                                                   "frozen_files_unchanged": len(FROZEN)})
facts_p.write_bytes((json.dumps(facts, indent=1) + "\n").encode("utf-8"))
print(json.dumps({"notebook": str(OUT_NB), "sha256": facts["sha256"], "bytes": facts["bytes"], "gpu_ready": facts["gpu_ready"],
                  "push_dir": str(dst), "lib_sha256": LIB_SHA}, indent=1))
