"""E25 NB-D1 (2026-11-02): the 1500+ speed-lever measurement notebook on the DFlash2 PRODUCTION base B = the E29_DFLASH config
(NBDF-K100's D3s: DFlash2 K3, SW-1024, BF16 drafter KV, FLASH_ATTN, seam 341d859b with GRD, seqs 24, pin 7,933,526,016,
Q-defer + DFLASH_DRAFT_EMBED_HOST=1 from the FIRST try on every arm + the QSA workspace env 128 / 64 after Merlin's build(),
read back in all 4 workers + the OOM-any-stage ladder fix). Qwen3.8-Flash-Next klee100 3bpw-MTP on Merlin v3.0.0, Kaggle 4 x L4.

  python build_e25_nbd1.py [--kv-gate-report FILE] [--test-out DIR]
      -> ../Q27-XHIGH-E25_NBD1.ipynb (every arm on: the kit notebook), ../Q27-XHIGH-E25_NBD1_{A,B,C}.ipynb (the copies: only the
         NBD1_ON line differs), nbd1_code_export.py, nbd1_cell7_body.py, nbd1_build_facts.json, the payload folder
         D:/ARC AGI 2/_scratch/push_e25/ds_e25_nbd1_payload/ (dataset foysalemonshanto/q27-e25-nbd1-payload: NOT uploaded) and
         the GPU push folders q27-xhigh-e25-nbd1-{a,c}__foysalemonshanto, q27-xhigh-e25-nbd1-b__hosen42 (nothing is pushed).
      Without --kv-gate-report the build is CLOSED (KV_KIT_GATE passed False: every arm SKIPPED_KV_KIT_GATE at run time,
      gpu_ready false). With the NB-D1 real kit's report (nbd1_kit_real_report.json, ALL OK on THIS closed build) -> gated.
      --test-out DIR: every output under DIR (a reproduction; the real outputs are not touched).

How it is built (NB-S2's pattern; NBDF / NB-S1 / NB-S2 files are read-only, asserted byte-identical before and after):
  1. The NBDF builder (Exp 27 build_e27_nbdf.py, NBDF_VARIANT=k100) runs IN THIS PROCESS twice with --test-out (outputs only
     in a scratch folder): closed, then gated with NBDF-K100's real-kit report (kits_wave3, ALL OK on closed e2198285). The
     closed output must be byte-identical to Exp 27's NBDF-K100 e2198285 and the gated one to c20eba7f (asserted): B's boot
     path IS that notebook's (B1 seam 341d859b GRD, B2 hard-only gates nbdf_lib b0147751, klee100 LOAD / K100Q merge, K6,
     sha gate, k100_setup + scales sidecar, the --help flag gate, the launcher, Q-defer nb4_ppack, the seam apply / verify /
     revert, the drafter Model gate, the D pin ladder, seqs 24, pin 7,933,526,016).
  2. Its cells are taken from that run; CELL 0 = E18 R2 verbatim (asserted). NB-D1 edits are exact-once text replacements
     on CELL 1 (switch line, title / budget / deadman / out dir), the CELL 7 body (nbd1_body_add.py blocks + the marked edits
     below) and CELL 8; the body and nbd1_lib.py live in the payload dataset (sha-pinned, checked at run time).
  3. The plain line `NBD1_ON = {...}` at the top of CELL 1 (outside the payload): absent -> every key on, a missing key ->
     off; an arm switched off is recorded SKIPPED_SWITCH_OFF and never booted. Copy A = B BAA PROF Q256 D4s DYN, copy B = B
     G1 G2 IDX LP, copy C = B S28 S32 SPC (BAA = B's A/A 8k pair + 40k; off = B light: the lever window set).
Body edits (light design): no ladder / oracle / long oracle / prefix test / TF on any arm; the prefill probe on B (a lever
reuses B's rate); SPC's sibling read-back (recorded); PROF = two 30-s profiles 8k@16 / 20k@24 in B's boot; the NB-D1 read-back
(QSA / GDN env in 4 workers, EMBED_HOST, indexer dtype, logged seqs / LP / SPC / K / schedule, the DYN fallback lines) = a real
failure of that arm only; `oom = bool(o["oom_in_log"]) and not o["xid_in_log"]` (an OOM at any stage steps the D ladder); the D
ladder without the same-pin EMBED_HOST rung (EH is on from the first try); a lever's KV-bound window steps down on any band;
the RPC env read-back + the arm's expect carry the QSA / GDN keys; a [LIVE] line after every window.
"""
import ast
import base64
import builtins
import hashlib
import json
import lzma
import os
import runpy
import shutil
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
NBDF_BUILDER = E27B / "build_e27_nbdf.py"
K100_REPORT = "D:/ARC AGI 2/_scratch/kits_wave3/q27-e27-nbdf-k100-kit-real/nbdf_kit_real_report.json"   # NB-S2's exact string
NBDF_K100_SHA = "c20eba7f00b90b8d8e08308c48d57f09b404214039d6c4fa46c059fea92e2004"          # the gated NBDF-K100
NBDF_K100_CLOSED_SHA = "e21982855d6430616f277784f162cad27cac94f7bcd8488fed97f123645ba730"   # Exp 27 nbdf_k100_build_facts sha256
SEAM_B1 = "341d859be5b2cc619247ff7b645304b9473d09f07995358ead2ee02be116c60b"
NBDF_LIB_B2 = "b014775162fb55401826fa8c498f7009b5d47ad15cc5cb365ab23e5ed37b1929"
TEST_OUT = Path(sys.argv[sys.argv.index("--test-out") + 1]) if "--test-out" in sys.argv else None
KV_GATE_REPORT = sys.argv[sys.argv.index("--kv-gate-report") + 1] if "--kv-gate-report" in sys.argv else None
OUT_BUILD = TEST_OUT or HERE
OUT_NBDIR = TEST_OUT or NBDIR
SCRATCH = (TEST_OUT / "_nbdf_ref") if TEST_OUT else Path(r"D:/ARC AGI 2/_scratch/nbd1_build_ref")
PUSH_ROOT = (TEST_OUT / "push_e25") if TEST_OUT else Path(r"D:/ARC AGI 2/_scratch/push_e25")
OUT_NB = OUT_NBDIR / "Q27-XHIGH-E25_NBD1.ipynb"
EXPORT, BODY_FILE, FACTS = OUT_BUILD / "nbd1_code_export.py", OUT_BUILD / "nbd1_cell7_body.py", OUT_BUILD / "nbd1_build_facts.json"
SLUG = "q27-xhigh-e25-nbd1"
TITLE = "Q27 Xhigh E25 NBD1"
LIB = HERE / "nbd1_lib.py"
DS_SLUG = "foysalemonshanto/q27-e25-nbd1-payload"      # NEW dataset (the coordinator creates it after the gated rebuild)
DS_DIR = PUSH_ROOT / "ds_e25_nbd1_payload"
PUSHED = set()        # shas of NB-D1 versions that ran on Kaggle (never overwrite one: rebuilds then go to *_REBUILD)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def once(text, old, new, label):
    assert text.count(old) == 1, f"NB-D1 {label}: x{text.count(old)}"
    return text.replace(old, new)


if not TEST_OUT:
    for _p in [OUT_NB] + [OUT_NB.with_name(f"Q27-XHIGH-E25_NBD1_{x}.ipynb") for x in "ABC"]:
        if _p.exists() and sha(_p.read_bytes()) in PUSHED:
            raise SystemExit(f"{_p.name} ran on Kaggle (PUSHED): never rebuild a run notebook; use --test-out")

# the frozen files: Exp 27 (read-only) + NB-S1 / NB-S2 (not this builder's) + the shared libs, byte-identical before and after
FROZEN = sorted(p for p in E27B.glob("*") if p.is_file()) + sorted(p for p in E27B.parent.glob("*.ipynb"))
FROZEN += [HERE / n for n in ("build_e25_nbs1.py", "nbs1_lib.py", "nbs1_unit_tests.py", "build_e25_nbs1_cpu_kit.py",
                              "build_e25_nbs1_units_kit.py", "nbs1_build_facts.json", "nbs1_cell7_body.py", "nbs1_code_export.py",
                              "build_e25_nbs2.py", "nbs2_lib.py", "nbs2_unit_tests.py", "nbs2_body_add.py", "nbs2_build_facts.json",
                              "nbs2_cell7_body.py", "nbs2_code_export.py", "build_e25_nbs2_cpu_kit.py", "build_e25_nbs2_units_kit.py",
                              "nb4t_lib.py", "nb4_ppack.py", "nb4_timeline.py", "k100q_ref27b.json")]
FROZEN += [NBDIR / "Q27-XHIGH-E25_NBS1.ipynb", NBDIR / "Q27-XHIGH-E25_NBS2_CAPACITY.ipynb"]
FROZEN = [p for p in FROZEN if p.is_file()]
FROZEN_SHA = {str(p): sha(p.read_bytes()) for p in FROZEN}


# ---------------------------------------------------------------- 1. the NBDF-K100 build, reproduced in-process (scratch outputs)
def nbdf_run(out_dir, report=None):
    os.environ["NBDF_VARIANT"] = "k100"
    argv0 = sys.argv[:]
    sys.argv = [str(NBDF_BUILDER), "--test-out", str(out_dir)] + (["--kv-gate-report", str(report)] if report else [])
    try:
        return runpy.run_path(str(NBDF_BUILDER), run_name="nbdf_build_for_nbd1")
    finally:
        sys.argv = argv0


sys.dont_write_bytecode = True
SCRATCH.mkdir(parents=True, exist_ok=True)
for f in SCRATCH.glob("*"):
    if f.is_file():
        f.unlink()
_g0 = nbdf_run(SCRATCH)                                   # closed (writes SCRATCH/Q27-XHIGH-E27_NBDF_K100.ipynb)
assert sha(_g0["data"]) == NBDF_K100_CLOSED_SHA, ("the closed NBDF-K100 rebuild != e2198285", sha(_g0["data"]))
G = nbdf_run(SCRATCH, K100_REPORT)                        # gated; PRE_SHA = the closed file above (as the original rebuild)
assert sha(G["data"]) == NBDF_K100_SHA, ("the gated NBDF-K100 rebuild != c20eba7f", sha(G["data"]))
assert G["KV_GATE"]["passed"] is True and G["KV_GATE"]["per_k_success"]["D3s"]["success"] is True
assert G["KV_GATE"]["per_k_success"]["D4s"]["success"] is True                 # K4 (D4s / DYN): the K4_swblock1024 KV run
K100_GATE = G["KV_GATE"]
CELL0, CELL1, CELL2, CELL2B, CELL3_6, CELL7_T, STUB_T, CELL8 = (G["CELL0"], G["CELL1"], G["CELL2"], G["CELL2B"], list(G["CELL3_6"]),
                                                                 G["CELL7"], G["CELL7_STUB"], G["CELL8"])
assert sha(CELL0.encode()) == G["E18_CELL0_SHA"]
KX = sys.modules["build_e27_nbdf_k100"]
MODELS, DATASETS = list(G["MODELS"]), list(KX.DATASETS) + [DS_SLUG]
assert json.loads((E27B / "nbdf_k100_build_facts.json").read_text())["sha256"] == NBDF_K100_CLOSED_SHA, "Exp 27's NBDF-K100 is not e2198285"
assert G["SEAM_SHA"] == SEAM_B1, ("B1 seam", G["SEAM_SHA"])
assert G["LIB_SHA"] == NBDF_LIB_B2, ("B2 nbdf_lib", G["LIB_SHA"])

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(E27B))
import nbd1_body_add as ADD  # noqa: E402
import nbd1_lib as D1  # noqa: E402
LIB_SRC = LIB.read_bytes().decode("utf-8")
assert "\r" not in LIB_SRC
compile(LIB_SRC, "nbd1_lib.py", "exec")
LIB_SHA = sha(LIB_SRC)

NBD1_ON_DEFAULT = dict(D1.SWITCH_DEFAULT)                # every key on (the kit notebook)
assert list(NBD1_ON_DEFAULT) == list(D1.SWITCH_KEYS) and all(v is True for v in NBD1_ON_DEFAULT.values())


def switch_line(sw):
    return "NBD1_ON = " + repr(dict(sw)) + "\n"


SWITCH_LINE = switch_line(NBD1_ON_DEFAULT)
SWITCH_BLOCK_HEAD = ("# ==================================================================== E25 NB-D1 ARM SWITCHES (edit before the commit)\n"
                     "# True = the arm runs, False = it is not booted (recorded SKIPPED_SWITCH_OFF; the other arms are unaffected). A key left out = off.\n"
                     "# B = the DFlash2 production base (E29_DFLASH: NBDF-K100 D3s + EMBED_HOST + QSA 128/64); BAA = B's 8k A/A pair + 40k; PROF = B's 30-s profiles.\n"
                     "# Copy A: B BAA PROF Q256 D4s DYN | copy B: B G1 G2 IDX LP | copy C: B S28 S32 SPC. No rebuild / new kit needed (every arm is kit-checked).\n")
D1_CFG = {"base": "E29_DFLASH = NBDF-K100 D3s + DFLASH_DRAFT_EMBED_HOST=1 (first try) + QSA 128/64", "base_key": D1.BASE_KEY,
          "arms": list(D1.ARM_KEYS), "switch_keys": list(D1.SWITCH_KEYS), "copies": D1.COPIES, "owners": D1.OWNERS,
          "budget_min": D1.BUDGET_MIN, "deadman_min": D1.DEADMAN_MIN, "sibling": {"prefix_tokens": 6900, "suffix_tokens": 800, "gen": 16},
          "qsa_b": D1.QSA_B, "qsa_merlin": D1.QSA_MERLIN, "prof_windows": [list(x) for x in D1.PROF_WINDOWS], "prof_secs": D1.PROF_SECS,
          "lib_sha256": LIB_SHA}

# ---------------------------------------------------------------- the arm plan + the time plan at build time (nbdf_lib as the body)
_N = G["_N"]
_N.configure(SEQS=24, CONC_LADDER=(2, 8, 23), Q_CAPTURES=list(KX.K100_CFG["q_captures"]),
             Q_TAIL=["--compilation-config", _N.compilation_json(KX.K100_CFG["q_captures"])], WKEYS=tuple(KX.K100_CFG["wkeys"]))
_q_extra = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}", "--max-num-seqs", "24"]
_seam = {"DFLASH_SEAM": "1", "DFLASH_MASK_ROW_SHA": _N.DRAFTER_PINS["mask_row"]["sha256"], "DFLASH_DRAFT_SW_BLOCK": "1024"}
_arms0 = _N.arm_plan(_q_extra, {"VLLM_SERVER_DEV_MODE": "1", "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, 7933526016,
                     "/kaggle/input/models/foysalemonshanto/" + _N.DRAFTER_REL, q_env=G["_P"].server_env("Q-defer"),
                     seam_envs={"D3s": dict(_seam), "D4s": dict(_seam)})
_N.restrict_arms(("Q", "D3s"))
_T = sys.modules["nb4t_lib"]
PLANS = {}
for _x, _sw in [("all", NBD1_ON_DEFAULT)] + list(D1.COPIES.items()):
    _a = D1.arm_plan(_N, _arms0["D3s"], seam_env_k4=dict(_seam), switch=_sw, pin=7933526016, seqs=24)
    D1.check_arm_plan(_N, _T, _a, _arms0["D3s"])
    PLANS[_x] = _a
WALKS = {x: D1.copy_walk(PLANS[x], sw) for x, sw in D1.COPIES.items()}
NEEDS = {x: D1.copy_need_min(PLANS[x], sw) for x, sw in D1.COPIES.items()}
for x, w in WALKS.items():
    assert w["all_run"], (x, w)
    assert w["runtime_min"] <= D1.COPY_RUNTIME_MAX, (x, w["runtime_min"], "each copy within ~120-150 min (user)")
_top = max(max(w["runtime_min"] for w in WALKS.values()), max(NEEDS.values())) + D1.START_RESERVE_MIN
assert D1.BUDGET_MIN == 5 * -(-_top // 5) and D1.DEADMAN_MIN == D1.BUDGET_MIN + 5, (
    "BUDGET_MIN = max(runtime, full-cap end) + 5 min rounded up to 5; deadman 5 min later", {x: w["runtime_min"] for x, w in WALKS.items()}, NEEDS)
for _x, _a in PLANS.items():
    for _k, _arm in _a.items():
        assert _arm["cap_min"] >= _arm["time_model"]["body"] + 1, (_x, _k, "the body cap holds the body")
BUDGET_MIN, DEADMAN_MIN = D1.BUDGET_MIN, D1.DEADMAN_MIN

# ---------------------------------------------------------------- 2. CELL 1 (switch line, title, budget, deadman, out dir, results file)
c1 = CELL1
c1 = once(c1, 'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 150 * 60)     # E27 NBDF: hard stop 150 min (user: Q + D4 + D3 guaranteed; D4t if time)',
          f'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or {BUDGET_MIN} * 60)     # E25 NB-D1: hard stop {BUDGET_MIN} min '
          f'(max over the copies of (planned runtime, full-cap end) + 5; each copy <= {D1.COPY_RUNTIME_MAX} min)', "c1 budget")
c1 = once(c1, f"NBDF_DEADMAN_SEC = {G['_N'].DEADMAN_MIN * 60}\n", f"NBDF_DEADMAN_SEC = {DEADMAN_MIN * 60}   # E25 NB-D1: 5 min after the hard stop\n", "c1 deadman")
c1 = once(c1, '"/kaggle/working/nbdf"))   # E27 NBDF (E27 K100: the same folder name)', '"/kaggle/working/nbd1"))   # E25 NB-D1', "c1 out")
c1 = once(c1, 'R = {"what": "E27 NBDF-K100: DFlash2 vs MTP K3 on the Q-defer base, 8k-116k (Flash-Next klee100 3bpw-MTP on Merlin)",   # E27 K100',
          'R = {"what": "E25 NB-D1: 1500+ speed levers (QSA 256, K4, dynamic K, GDN, indexer fp8, long prefill, seqs 28 / 32, shared-prefix checkpoint, profiles) on the DFlash2 production base B (E29_DFLASH), 8k-116k",   # E25 NB-D1',
          "c1 what")
c1 = once(c1, '        tmp = OUT / "nbdf_results.json.tmp"                      # E27 NBDF\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "nbdf_results.json")                   # E27 NBDF',
          '        tmp = OUT / "nbd1_results.json.tmp"                      # E25 NB-D1\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "nbd1_results.json")                   # E25 NB-D1', "c1 save")
c1 = once(c1, '''    print("E27 NBDF DFlash2: competition rerun -> CELL 0's identity fallback submission.json only (E18 R2 format)")   # E27 NBDF''',
          '''    print("E25 NB-D1 speed levers: competition rerun -> CELL 0's identity fallback submission.json only (E18 R2 format)")   # E25 NB-D1''', "c1 rerun")
c1 = once(c1, "(RC2 verbatim + E25 NB1 / NB4S / NB4T / E27 NBDF / E27 K100 marked edits)",
          "(RC2 verbatim + E25 NB1 / NB4S / NB4T / E27 NBDF / E27 K100 / E25 NB-D1 marked edits)", "c1 title")
CELL1_D1 = c1                                # the switch block is prepended per notebook (assemble)

# ---------------------------------------------------------------- 3. the CELL 7 body (NBDF-K100's, the marked NB-D1 edits)
ORACLE_HEAD = '''        if not due(240):
            o["items_skipped"].append("oracle")
            o["status"] = "SKIPPED_BOOT_BUDGET_ORACLE"
            return o
'''
ORACLE_END = '''        if arm.get("short_only"):'''
b = CELL7_T
b = once(b, "N.check_arm_plan(ARMS)                      # never-do list + \"each D arm = Q + exactly the DFlash change\" (raises: no boot)\n",
         "N.check_arm_plan(ARMS)                      # never-do list + \"each D arm = Q + exactly the DFlash change\" (raises: no boot)\n"
         + ADD.D1_SETUP.replace("__D1LIBSHA__", LIB_SHA).replace("__D1CFG__", json.dumps(D1_CFG)), "b setup")
b = once(b, 'TF_P = {b: PROMPTS[b][5] for b in K100_CFG["tf_bands"]}     # teacher-forced 2k / 8k / 40k (RC2\'s TF variant 5)\n',
         'TF_P = {b: PROMPTS[b][5] for b in K100_CFG["tf_bands"]}     # teacher-forced 2k / 8k / 40k (RC2\'s TF variant 5)\n' + ADD.D1_PROMPTS, "b prompts")
b = once(b, 'NBDF["order"] = list(N.ORDER)', 'NBDF["order"] = list(ARMS)                  # E25 NB-D1', "b order")
b = once(b, '''                  "priority": "Q > D4 > D3 > D7 > D4b (an arm starts only with est_min x 60 + 300 s left; D7 / D4b dropped first)",
                  "planned_walk": N.budget_walk({}, BUDGET / 60, 8.0, ARMS)}''',
         '''                  "priority": " > ".join(ARMS) + " (switched-on arms; an arm starts only with est_min x 60 + 300 s left)",   # E25 NB-D1
                  "planned_walk": D1.copy_walk(ARMS, NBD1_SWITCH, BUDGET / 60)}''', "b budget")
b = once(b, '''    if is_d:
        plan = K100_CFG["d_retry"]
''', '''    if is_d:
        plan = D1.retry_plan(K100_CFG["d_retry"], int(K100_CFG["q_pin"]), bool(arm.get("d1_eh_first")))   # E25 NB-D1: EMBED_HOST from the first try
''', "b retry eh first")
b = once(b, '''        is_q, is_d = arm["key"] == "Q", arm.get("method") == "dflash"''',
         '''        is_q, is_d = arm["key"] == D1.BASE_KEY, arm.get("method") == "dflash"   # E25 NB-D1: B (a D arm) is the reference arm''', "b is_q")
b = once(b, '''        if is_q:
            _live(o, "seam_inert", N.q_seam_inert(t))
        else:
            _live(o, "dflash_readback", dflash_rb(arm, t))''', '''        if not is_d:                               # E25 NB-D1: every arm is a D arm (B included)
            _live(o, "seam_inert", N.q_seam_inert(t))
        else:
            _live(o, "dflash_readback", dflash_rb(arm, t))''', "b readback")
b = once(b, '''            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="SPEC_MISMATCH: " + sm_[1][:400])
            return o
''', '''            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="SPEC_MISMATCH: " + sm_[1][:400])
            return o
        try:                                       # E25 NB-D1: the arm's planned settings read back (a real failure stops this arm only)
            _d1f = d1_readback_failure(o, arm, t)
        except Exception as e:                     # a read-back code error is recorded, never a stop
            _d1f, o["d1_readback_error"] = None, f"{type(e).__name__}: {e}"[:300]
        if _d1f:
            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop=_d1f)
            return o
''', "b d1 readback")
b = once(b, '''        if is_q:                                  # E28 K100 I8: INT8 requested''',
         '''        if is_q and not is_d:                     # E28 K100 I8 (never on a D arm): INT8 requested''', "b i8")
b = once(b, '''            if is_d and o["smoke_gate"].startswith("FAIL"):       # errors / garbage / short / zero accept / contamination''',
         '''            if is_q and o["smoke_gate"] in ("FAIL_SMOKE_CONCURRENCY_CONTAMINATION", "INCONCLUSIVE_SMOKE_CONCURRENCY_CONTAMINATION"):   # E25 NB-D1 (NB-S2): recorded
                o["smoke_gate_raw"], o["smoke_gate"] = o["smoke_gate"], "BASE_LADDER_NONDET_RECORDED"
            if is_d and o["smoke_gate"].startswith("FAIL"):       # errors / garbage / short / zero accept / contamination''', "b smoke")
b = once(b, '''    for c in N.CONC_LADDER:''', '''    for c in (N.CONC_LADDER if D1.item_on(arm, "ladder") else ()):   # E25 NB-D1 light: no contamination ladder''', "b ladder")
b = once(b, '''        if due(300):
            CUR["item"] = "prefill_probe"''', '''        if due(300) and (D1.item_on(arm, "prefill_probe") or not PREFILL.get("measured")):   # E25 NB-D1: B (a lever reuses B's rate)
            CUR["item"] = "prefill_probe"''', "b prefill")
assert b.count(ORACLE_HEAD) == 1 and b.count(ORACLE_END) == 1
_i = b.index(ORACLE_HEAD)
_j = b.index(ORACLE_END, _i)
_blk = b[_i:_j]
assert 'CUR["item"] = "oracle_aa"' in _blk and 'o["quality_gate"] = quality_gate(o, ref)' in _blk and "'''" not in _blk and '"""' not in _blk
b = b[:_i] + ('        if D1.item_on(arm, "oracle"):           # E25 NB-D1 light: the oracle only when the arm\'s plan has it (never)\n'
              + textwrap.indent(_blk, "    ") +
              '        else:\n'
              '            o["items_skipped"].append("oracle")\n'
              '            o["oracle_vs"], o["quality_gate"] = None, "NOT_IN_PLAN_NB_D1"   # recorded only\n') + b[_j:]
b = once(b, '''        if due(300) and not o.get("admission_deadlock"):
            CUR["item"] = "long_oracle"''', '''        if due(300) and not o.get("admission_deadlock") and D1.item_on(arm, "long_oracle"):   # E25 NB-D1 light: off
            CUR["item"] = "long_oracle"''', "b long oracle")
b = once(b, '''            o["long_gate"] = ("INCONCLUSIVE_ADMISSION_DEADLOCK" if o.get("admission_deadlock") else "INCONCLUSIVE_LONG_ORACLE_SKIPPED_BUDGET")''',
         '''            o["long_gate"] = ("INCONCLUSIVE_ADMISSION_DEADLOCK" if o.get("admission_deadlock") else
                              "NOT_IN_PLAN_NB_D1" if not D1.item_on(arm, "long_oracle") else "INCONCLUSIVE_LONG_ORACLE_SKIPPED_BUDGET")''',
         "b long label")
b = once(b, '''        if due(240) and not o.get("admission_deadlock"):
            CUR["item"] = "prefix_cache"''', '''        if due(240) and not o.get("admission_deadlock") and D1.item_on(arm, "prefix"):   # E25 NB-D1 light: off
            CUR["item"] = "prefix_cache"''', "b prefix")
b = once(b, '''            o["items_skipped"].append("prefix_cache")
            o["prefix_gate"] = "INCONCLUSIVE_PREFIX_SKIPPED_BUDGET"
''', '''            o["items_skipped"].append("prefix_cache")
            o["prefix_gate"] = "NOT_IN_PLAN_NB_D1" if not D1.item_on(arm, "prefix") else "INCONCLUSIVE_PREFIX_SKIPPED_BUDGET"
        if D1.item_on(arm, "sibling") and due(240) and not o.get("admission_deadlock"):   # E25 NB-D1: SPC's activation read-back (recorded)
            CUR["item"] = "sibling_probe"
            try:
                _live(o, "sibling_probe", sibling_probe(p, o))
            except Exception as e:
                o["sibling_probe"] = {"error": f"{type(e).__name__}: {e}"[:300]}
''', "b sibling")
b = once(b, '''        for band, lvw, tag in K100_CFG["windows"]:''',
         '''        for band, lvw, tag in (arm.get("windows_fixed") or K100_CFG["windows"]):   # E25 NB-D1: the arm's light window set''', "b windows")
b = once(b, '''            if band in ("40k",) + N.LONG_BANDS:
                lv, why = band_level(o, band)''', '''            if band in D1.CAPPED_BANDS:            # E25 NB-D1: 20k too (S32 x 20k can exceed the resident bound)
                lv, why = band_level(o, band)''', "b capped")
b = once(b, '''    return N.level_for_band(band, BANDS[band], pool, cap.get("max_concurrency_131072"), int(blk or N.BLOCK_TOKENS), o.get("kv_layout"),
                            PREFILL["tok_s"], WIN)''', '''    return N.level_for_band(band, BANDS[band], pool, cap.get("max_concurrency_131072"), int(blk or N.BLOCK_TOKENS), o.get("kv_layout"),
                            PREFILL["tok_s"], WIN, seqs=int(o.get("streams") or N.SEQS))   # E25 NB-D1: the arm's own max_num_seqs''', "b band seqs")
b = once(b, '''        for band, lvb in (("8k", int(K100_CFG["profile_level"])),) + ((("116k", lv116),) if long_ok and lv116 > 0 else ()):''',
         '''        for band, lvb in D1.prof_plan(arm, NBD1_SWITCH):   # E25 NB-D1 PROF: 8k@16 / 20k@24 inside B's boot, 30 s each''', "b profile plan")
b = once(b, '''                    o["steady"][pk] = profile_window(p, arm, band, lvb)''',
         '''                    o["steady"][pk] = profile_window(p, arm, band, lvb, secs=float(D1.PROF_SECS))   # E25 NB-D1''', "b profile call")
b = once(b, '''def profile_window(port, arm, band="8k", n=None):''', '''def profile_window(port, arm, band="8k", n=None, secs=6.0):   # E25 NB-D1: secs (PROF 30 s)''',
         "b profile secs arg")
b = once(b, "secs=6.0, max_new=", "secs=float(secs), max_new=", "b profile secs")
b = once(b, '"secs": 6.0', '"secs": float(secs)', "b profile secs rec")
b = once(b, '''        if due(240) and s.alive():
            CUR["item"] = "tf_k100"''', '''        if due(240) and s.alive() and D1.item_on(arm, "tf"):   # E25 NB-D1 light: no TF
            CUR["item"] = "tf_k100"''', "b tf")
b = once(b, '''            o["items_skipped"].append("tf_k100")
        CUR["item"] = "end"
''', '''            o["items_skipped"].append("tf_k100")
            if not D1.item_on(arm, "tf"):
                o["tf_plan"] = "NOT_IN_PLAN_NB_D1"
        CUR["item"] = "end"
''', "b tf label")
b = once(b, '''        if not is_q:
            o["dflash_readback"] = dflash_rb(arm, t)               # the whole log: the SpecDecoding lines now exist''',
         '''        if is_d:                                   # E25 NB-D1: B included
            o["dflash_readback"] = dflash_rb(arm, t)               # the whole log: the SpecDecoding lines now exist''', "b end readback")
b = once(b, '''        if not is_d:
            o["ingraph_meta"] = None
        else:
            o["ingraph_meta"] = N.ingraph_meta(t)
        return o
    return body''', '''        if not is_d:
            o["ingraph_meta"] = None
        else:
            o["ingraph_meta"] = N.ingraph_meta(t)
        try:
            o["d1_records"] = d1_records(o, arm, t)     # E25 NB-D1: K_eff, KV ledgers, QSA over the whole log (recorded)
        except Exception as e:                     # records only: never fails the arm
            o["d1_records"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        return o
    return body''', "b d1 records")
b = once(b, '''    ref = K100_CFG["q_ref"] if arm["key"] == "Q" else "q"   # E27 K100''',
         '''    ref = K100_CFG["q_ref"] if arm["key"] == D1.BASE_KEY else "q"   # E27 K100 / E25 NB-D1: B''', "b ref1")
b = once(b, '''    o = arm_boot(arm, arm_body(arm, K100_CFG["q_ref"] if key == "Q" else "q", "q" if key == "Q" else None))   # E27 K100''',
         '''    o = arm_boot(arm, arm_body(arm, K100_CFG["q_ref"] if key == D1.BASE_KEY else "q", "q" if key == D1.BASE_KEY else None))   # E25 NB-D1''', "b ref2")
b = once(b, '''    try:
        r = window(port, s, band, n)
    finally:
        rows, RESP["rows"] = RESP["rows"] or [], None''', '''    try:
        with D1Ledger(port) as _d1l:              # E25 NB-D1: the window's KV ledger samples (records only)
            r = window(port, s, band, n)
        r["_d1_samples"] = _d1l.samples
    finally:
        rows, RESP["rows"] = RESP["rows"] or [], None''', "b ledger")
b = once(b, '''            r = hist_window(s.port, s, band, n, arm)
''', '''            r = hist_window(s.port, s, band, n, arm)
        try:
            r["kv_ledger"] = d1_window_ledger(o, r, band, n)   # E25 NB-D1 (records only)
        except Exception as e:                     # records only: never voids the measured window
            r.pop("_d1_samples", None)
            r["kv_ledger"] = {"error": f"{type(e).__name__}: {e}"[:300]}
''', "b ledger summary")
b = once(b, '''    o["steady"][key] = r
    _live(o, "steady", o["steady"])
''', '''    o["steady"][key] = r
    _live(o, "steady", o["steady"])
    d1_live(o, arm, key, r)                       # E25 NB-D1: [LIVE <min>] <arm> <window> tok/s | AL | running | preempt | vs B (print only)
''', "b live")
b = once(b, '''        o["steady"][key] = {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need), "boot_left_sec": round(boot_left())}
        o["items_skipped"].append(key)
''', '''        o["steady"][key] = {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need), "boot_left_sec": round(boot_left())}
        o["items_skipped"].append(key)
        d1_live(o, arm, key, o["steady"][key])     # E25 NB-D1 (print only)
''', "b live skipped")
b = once(b, '''    if N.kv_bound(r) and n > 1 and band in ("40k",) + N.LONG_BANDS and due(win_need(band, n - 1)):''',
         '''    if N.kv_bound(r) and n > 1 and (band in ("40k",) + N.LONG_BANDS or arm.get("lever")) and due(win_need(band, n - 1)):   # E25 NB-D1: a lever steps down on any band (never skipped)''',
         "b step down")
b = once(b, '''RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P"''',
         '''RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P," + ",".join(D1.D1_ENV_KEYS)   # E25 NB-D1: + the QSA / GDN keys (4-worker read-back)''',
         "b rpc keys")
b = once(b, '''        oom = o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]''',
         '''        oom = bool(o["oom_in_log"]) and not o["xid_in_log"]   # E25 NB-D1 (= NB-S1 GPU v1 fix: an OOM at ANY stage, not a device fault) -> the ladder''',
         "b oom any stage")
b = once(b, '''                arm["expect"] = {w: arm["env"].get(w, N.WATCHED_DEFAULT.get(w)) for w in N.WATCHED}''',
         '''                arm["expect"] = dict(arm["expect"], **{w: arm["env"].get(w, N.WATCHED_DEFAULT.get(w)) for w in N.WATCHED})   # E25 NB-D1: keep the QSA / GDN keys''',
         "b expect keep")
b = once(b, '''    return gated(o, need=(ref,), expected_rows=4)
''', '''    if not D1.item_on(arm, "oracle"):         # E25 NB-D1 light: no oracle in the plan -> the hard-only evidence (probes, sampled, windows)
        return gated(o, need=(), expected_rows=None)
    return gated(o, need=(ref,), expected_rows=4)
''', "b light gated")
b = once(b, "\ndef _gpu_util():\n", "\n" + ADD.D1_MAIN + "def _gpu_util():\n", "b main block")
b = once(b, '''    nbdf_main()
    _gpu_util_stop.set()''', '''    nbd1_main()                                  # E25 NB-D1
    _gpu_util_stop.set()''', "b main call")
CELL7_D1 = b

c8 = once(CELL8, '''    print("NBDF THROUGHPUT WINNER:", _v.get("best_arm"), _v.get("best_verdict"), _v.get("best_gain_8k"), "|", _v.get("why"))''',
          '''    print("NB-D1 RECORDS (labels record-only; read the raw numbers):", _v.get("verdict"), "|", _v.get("why"))   # E25 NB-D1''', "c8 print")
c8 = once(c8, '''    print("NBDF TABLE (tok/s, tick, AL per arm):", json.dumps(_v.get("table"), default=str)[:8000])''',
          '''    print("NB-D1 TABLE (tok/s, AL, preemptions per arm and window):", json.dumps(_v.get("table"), default=str)[:8000])
    print("NB-D1 GAINS vs B (gain, label per band):", json.dumps(_v.get("gains"), default=str)[:6000])
    print("NB-D1 PROF:", json.dumps({k_: (_v.get("prof") or {}).get(k_) for k_ in ("status", "why", "windows")}, default=str)[:3000])''', "c8 table")
CELL8_D1 = c8

FLAGS = json.loads(json.dumps(G["FLAGS"]))
FLAGS["NBD1"] = {
    "base": "B = E29_DFLASH production = NBDF-K100 D3s (nbdf_lib.arm_plan D3s after configure(SEQS=24, ...)) + DFLASH_DRAFT_EMBED_HOST=1 "
            "from the first try (dflash_seam host embed; the D ladder then skips its same-pin EH rung) + QSA 128 / 64",
    "QSA": ["tools/merlin/launch.py:419-420 (build() sets VLLM_SPARSE_INDEXER_MAX_LOGITS_MB / VLLM_E2_QSA_LOGITS_GRANULE_MB = 256)",
            "the notebook's merlin_fn_server: env.update(NB1_ENV) after build() (NB1_ENV wins)", "qsa_indexer.py:797 'QSA prefill logits "
            "backing: granule <n> MiB' (logged at the first prefill)", "E29_DFLASH_D3S rev 2c report (the 128 / 64 decision)"],
    "D4s_DYN": ["vllm/config/speculative.py:493-498 num_speculative_tokens_per_batch_size", "v1/spec_decode/dynamic/utils.py:7-150",
                "config/vllm.py:1068-1086 (PIECEWISE override only without the V2 runner)", "v1/worker/gpu/model_runner.py:2786-2803"],
    "GDN": ["vllm/merlin/envs.py:544-555 (strict switches), :822-823 (not in the compile hash)"],
    "IDX": ["vllm/config/attention.py:88-92", "vllm/models/qwen4_exp/nvidia/indexer_qsa.py:154-166", "vllm/engine/arg_utils.py:1856 --attention-config (JSON)"],
    "LP": ["vllm/config/scheduler.py:79-90", "vllm/engine/arg_utils.py:1735-1740"],
    "S28_S32": ["launch.py:453 Merlin's --max-num-seqs (the trailing one wins)", "registry.py:83-89 (multiples of 1 + K here)"],
    "SPC": ["vllm/config/cache.py:198-202", "vllm/v1/core/kv_cache_manager.py:185-198"],
    "dyn_source_check": D1.DYN_SOURCE_CHECK}

MD0 = ("# E25 NB-D1: 1500+ speed levers on the DFlash2 production base (Qwen3.8-Flash-Next klee100 3bpw-MTP, Merlin v3.0.0, 4 x L4), 8k to 116k\n\n"
       "B = E29_DFLASH: NBDF-K100's D3s (DFlash2 K3, SW-1024, BF16 drafter KV, FLASH_ATTN, seam 341d859b + GRD, Q-defer, seqs 24, pin "
       "7,933,526,016) + DFLASH_DRAFT_EMBED_HOST=1 from the first try + QSA 128 / 64. Levers (each = B + one change, LIGHT: boot, smoke "
       "hard gate, the read-back of its change, windows 8k@16 / 20k@24 / 90k@8 / 116k@6): Q256, D4s, DYN, G1, G2, IDX, LP, S28 / S32 "
       "(20k at 28 / 32), SPC. PROF = two 30-s profiles 8k@16 / 20k@24 in B's boot; BAA = B's 8k A/A pair + 40k. Arm switches: NBD1_ON "
       f"at the top of CELL 1 (copy A: B BAA PROF Q256 D4s DYN, copy B: B G1 G2 IDX LP, copy C: B S28 S32 SPC). Hard stop {BUDGET_MIN} "
       "min. A [LIVE] line after every window. Readable code: nbd1_cell7_body.py. Commit run only; a competition rerun writes CELL 0's "
       "identity submission (E18 R2 format).\n")

DS_LOADER = r'''def _nbd1_ds(name, want):                     # E25 NB-D1: a sha-pinned file of the payload dataset (notebook < 400 kB)
    for d in (INP / "datasets/foysalemonshanto/q27-e25-nbd1-payload", INP / "q27-e25-nbd1-payload",
              Path(os.environ.get("NBD1_PAYLOAD_DIR") or "/nonexistent")):
        f = d / name
        if f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest() == want:
            return f.read_bytes().decode("utf-8")
    raise RuntimeError(f"NBD1_PAYLOAD missing or sha mismatch: {name} (dataset foysalemonshanto/q27-e25-nbd1-payload)")
NBDF_BODY_SRC = _nbd1_ds("nbd1_cell7_body.py", NBDF_BODY_SHA)
'''


# ---------------------------------------------------------------- 4. assemble (= the NBDF builder's assemble with the NB-D1 cells)
def assemble(kv_gate, sw_line=SWITCH_LINE):
    subs = {"__NB4TSHA__": G["NB4T_SHA"], "__E18C0SHA__": G["E18_CELL0_SHA"], "__FLAGS__": json.dumps(FLAGS), "__KVGATE__": json.dumps(kv_gate)}
    body = CELL7_D1
    for k, v in subs.items():
        body = body.replace(k, v)
    assert '"""__' not in body and "__KVGATE" not in body and "__FLAGS" not in body and "__D1" not in body
    compile(body, "nbd1_cell7_body.py", "exec")
    st = STUB_T.replace("E27 NBDF: arms Q > D4 > D3 > D7 > D4b (DFlash2 vs MTP K3), long bands (NEW code)",
                        "E25 NB-D1: B > Q256 > D4s > DYN > G1 > G2 > IDX > LP > S28 > S32 > SPC (speed levers on DFlash2 production), long bands")
    st = once(st, 'NBDF_BODY_SRC = _unx("__BODYB85__")' + chr(10), DS_LOADER, "stub body from the dataset")
    for k, v in dict(G["_BLOBS"], __BODYSHA__=sha(body)).items():
        st = st.replace(k, v)
    assert "__LIB" not in st and "__SEAM" not in st and "__BODY" not in st
    c1_ = SWITCH_BLOCK_HEAD + sw_line + CELL1_D1
    cs = ([CELL0, c1_, CELL2, "if not RERUN:\n" + textwrap.indent(CELL2B, "    ")] + list(CELL3_6) +
          ["if not RERUN:\n" + textwrap.indent(c, "    ") for c in (st, CELL8_D1)])
    nb_ = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD0.splitlines(True)}] +
                    [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(True)} for c in cs],
           "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
           "nbformat": 4, "nbformat_minor": 5}
    return body, st, cs, nb_, json.dumps(nb_, separators=(",", ":")).encode("utf-8")


def static_names(cells, body):
    check_src = cells[:-1] + ["if not RERUN:\n" + textwrap.indent(body, "    "), cells[-1]]
    for c in cells + check_src:
        compile(c, "<nbd1 cell>", "exec")
    defined, loaded = set(dir(builtins)) | {"__file__", "__name__"}, set()
    for c in check_src:
        for node in ast.walk(ast.parse(c)):
            if isinstance(node, ast.Name):
                (loaded if isinstance(node.ctx, ast.Load) else defined).add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if not isinstance(node, ast.Lambda):
                    defined.add(node.name)
                defined.update(a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs)
                if node.args.vararg:
                    defined.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined.add(node.args.kwarg.arg)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                defined.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
    return sorted(loaded - defined)


# the closed build first (its sha = what the NB-D1 real kit runs), then the gated one when a report is given
GATE_CLOSED = D1.closed_gate(K100_GATE)
body_c, stub_c, cells_c, nb_c, data_c = assemble(GATE_CLOSED)
SHA_CLOSED = sha(data_c)
KV_GATE, REPORT_INFO = GATE_CLOSED, None
if KV_GATE_REPORT:
    _rep_b = Path(KV_GATE_REPORT).read_bytes()
    _rep = json.loads(_rep_b.decode("utf-8"))
    KV_GATE = D1.gate_from_report(K100_GATE, _rep, SHA_CLOSED)
    REPORT_INFO = {"report": str(KV_GATE_REPORT), "report_sha256": sha(_rep_b), "passed": KV_GATE["passed"], "why": KV_GATE["why"]}
    KV_GATE.update(nbd1_report=str(KV_GATE_REPORT), nbd1_report_sha256=sha(_rep_b))
body, stub, cells_src, nb, data = assemble(KV_GATE)
_undef = static_names(cells_src, body)
assert not _undef, f"names loaded but never defined in the notebook: {_undef}"
assert cells_src[0] == CELL0 and sha(cells_src[0].encode()) == G["E18_CELL0_SHA"], "CELL 0 != E18 R2 cell 1"
assert cells_src[1].startswith(SWITCH_BLOCK_HEAD + SWITCH_LINE) and cells_src[1].count(SWITCH_LINE) == 1, "the CELL 1 switch line"
assert "NBD1_ON =" not in body and body.count('globals().get("NBD1_ON")') == 1, "the switch line must stay out of the payload body"
assert all(c.startswith("if not RERUN:\n") for c in cells_src[2:])
assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))

# the copies: only the NBD1_ON line differs (asserted on the cells and on the bytes)
COPY_NB = {}
for x, sw in D1.COPIES.items():
    sl = switch_line(sw)
    b_x, st_x, cs_x, nb_x, data_x = assemble(KV_GATE, sl)
    assert b_x == body and st_x == stub and cs_x[0] == cells_src[0] and cs_x[2:] == cells_src[2:], x
    assert cs_x[1] == cells_src[1].replace(SWITCH_LINE, sl) and cs_x[1].count(sl) == 1, x
    assert len(data_x) < 400_000, (x, len(data_x))
    COPY_NB[x] = {"switch": sw, "switch_line": sl.rstrip("\n"), "data": data_x, "sha256": sha(data_x), "owner": D1.OWNERS[x],
                  "walk": WALKS[x], "need_min": NEEDS[x], "notebook": f"Q27-XHIGH-E25_NBD1_{x.upper()}.ipynb"}

# ---------------------------------------------------------------- 5. write (notebooks, export, body, payload, push folders, facts)
OUT_NBDIR.mkdir(parents=True, exist_ok=True)
OUT_BUILD.mkdir(parents=True, exist_ok=True)
OUT_NB.write_bytes(data)
for x, c in COPY_NB.items():
    (OUT_NBDIR / c["notebook"]).write_bytes(c["data"])
EXPORT.write_bytes("\n\n".join(cells_src).encode("utf-8"))
BODY_FILE.write_bytes(body.encode("utf-8"))
DS_DIR.mkdir(parents=True, exist_ok=True)
for f in DS_DIR.glob("*"):
    f.unlink()
(DS_DIR / "nbd1_cell7_body.py").write_bytes(body.encode("utf-8"))
(DS_DIR / "nbd1_lib.py").write_bytes(LIB_SRC.encode("utf-8"))
(DS_DIR / "dataset-metadata.json").write_bytes((json.dumps({"title": "q27-e25-nbd1-payload", "id": DS_SLUG, "licenses": [{"name": "CC0-1.0"}]},
                                                           indent=2) + chr(10)).encode("utf-8"))
(DS_DIR / "SHA256SUMS.json").write_bytes((json.dumps({"nbd1_cell7_body.py": sha(body), "nbd1_lib.py": LIB_SHA}, indent=1) + chr(10)).encode("utf-8"))
META = {"title": TITLE, "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": [], "dataset_sources": DATASETS, "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": MODELS, "docker_image": G["IMAGE"], "machine_shape": "NvidiaL4"}
PUSH_DIRS = {}
for x, c in COPY_NB.items():
    slug_x = f"{SLUG}-{x}"
    d = PUSH_ROOT / f"{slug_x}__{c['owner']}"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*"):
        if f.is_file():
            f.unlink()
    (d / f"{slug_x}.ipynb").write_bytes(c["data"])
    m = dict(META, id=f"{c['owner']}/{slug_x}", title=f"{TITLE} {x.upper()}", code_file=f"{slug_x}.ipynb")
    (d / "kernel-metadata.json").write_bytes((json.dumps(m, indent=2) + "\n").encode("utf-8"))
    PUSH_DIRS[x] = str(d)
    c["push_dir"], c["dataset_sources"], c["model_sources"] = str(d), list(DATASETS), list(MODELS)

for p in FROZEN:
    assert sha(p.read_bytes()) == FROZEN_SHA[str(p)], f"FROZEN FILE CHANGED: {p}"

CELL_BYTES = [len(c.encode()) for c in cells_src]
facts = {"notebook": OUT_NB.name, "sha256": sha(data), "bytes": len(data), "cells": len(nb["cells"]), "cell_bytes": CELL_BYTES,
         "closed_build_sha256": SHA_CLOSED, "gpu_ready": bool(KV_GATE.get("passed")), "kv_gate_report": REPORT_INFO,
         "nbd1_lib_sha256": LIB_SHA, "nbdf_k100_reproduced_sha256": NBDF_K100_SHA, "nbdf_k100_closed_sha256": NBDF_K100_CLOSED_SHA,
         "kv_kit_gate": KV_GATE, "d1_cfg": D1_CFG, "budget_min": BUDGET_MIN, "deadman_min": DEADMAN_MIN,
         "copies": {x: {k: v for k, v in c.items() if k != "data"} for x, c in COPY_NB.items()},
         "e18_r2_cell0_sha256": G["E18_CELL0_SHA"], "export_sha256": sha(EXPORT.read_bytes()), "body_file_sha256": sha(body),
         "payload_dataset": {"slug": DS_SLUG, "dir": str(DS_DIR), "files": {"nbd1_cell7_body.py": sha(body), "nbd1_lib.py": LIB_SHA},
                             "uploaded": False},
         "model_sources": MODELS, "dataset_sources": DATASETS, "nbdf_lib_sha256": G["LIB_SHA"], "seam_sha256": G["SEAM_SHA"],
         "ppack_sha256": G["PP_SHA"], "image": G["IMAGE"],
         "arms": {k: {x_: a.get(x_) for x_ in ("title", "port", "spec", "captures", "streams", "windows_fixed", "prof_windows", "est_min",
                                                "cap_min", "time_model", "d1_change", "d1_items", "env")} for k, a in PLANS["all"].items()},
         "arms_by_copy": {x: {k: {"est_min": a["est_min"], "cap_min": a["cap_min"], "windows": a["windows_fixed"], "prof": a["prof_windows"]}
                              for k, a in PLANS[x].items() if D1.COPIES[x].get(k)} for x in D1.COPIES},
         "walks": WALKS, "need_min": NEEDS, "frozen_files_unchanged": len(FROZEN), "flags": FLAGS["NBD1"],
         "switch": {"line": SWITCH_LINE.rstrip("\n"), "default": NBD1_ON_DEFAULT, "cell": 1, "in_payload": False},
         "k100_report": K100_REPORT, "push_dirs": PUSH_DIRS, "test_out": str(TEST_OUT) if TEST_OUT else None}
FACTS.write_bytes((json.dumps(facts, indent=1, default=str) + "\n").encode("utf-8"))
print(json.dumps({k: v for k, v in facts.items() if k not in ("kv_kit_gate", "arms", "flags", "d1_cfg", "arms_by_copy", "copies")}, indent=1, default=str))
print(json.dumps({x: {"sha256": c["sha256"], "owner": c["owner"], "runtime_min": c["walk"]["runtime_min"], "need_min": c["need_min"],
                      "runs": c["walk"]["runs"]} for x, c in COPY_NB.items()}, indent=1))
