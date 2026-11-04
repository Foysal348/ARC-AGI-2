"""E25 NB-S2 "capacity" (user 2026-11-01): capacity levers on the DFlash2 base B = NBDF-K100's D3s (Qwen3.8-Flash-Next
klee100 3bpw-MTP on Merlin v3.0.0, Kaggle 4 x L4). DFlash2 only (user): no MTP base, no MTP-only arm.

  python build_e25_nbs2.py [--budget-min N]
      -> ../Q27-XHIGH-E25_NBS2_CAPACITY.ipynb, nbs2_code_export.py, nbs2_cell7_body.py, nbs2_build_facts.json and the GPU
         push folder D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nbs2-capacity__esfersami50/ (nothing is pushed)

How it is built (NBDF files are read-only; nothing in Exp 27 is written):
  1. The NBDF builder (Exp 27 build_e27_nbdf.py, NBDF_VARIANT=k100) runs IN THIS PROCESS twice with --test-out (outputs only
     in a scratch folder): closed, then gated with NBDF-K100's real-kit report (kits_wave3, ALL OK on closed e2198285). The
     closed output must be byte-identical to Exp 27's closed NBDF-K100 e2198285 and the gated one to c20eba7f (asserted):
     B's boot path IS that notebook's (B1 dflash_seam 341d859b GRD edit, B2 hard-only gates nbdf_lib b0147751, klee100 LOAD rev 3 / K100Q merge,
     K6, sha gate, k100_setup + scales sidecar, the --help flag gate, the launcher, Q-defer nb4_ppack, dflash_seam apply /
     verify / revert, the drafter Model gate, the D pin ladder (EMBED_HOST first, then 0.25-GiB rungs), seqs 24, pin
     7,933,526,016, the embedded KV kit gate (passed)).
  2. Its cells are taken from that run; CELL 0 = E18 R2 verbatim (asserted again). NB-S2 edits are exact-once text
     replacements on CELL 1 (title / budget / deadman / out dir), the CELL 7 body (nbs2_body_add.py blocks + the marked
     edits below) and CELL 8; the body and nbs2_lib.py live in the payload dataset (sha-pinned, checked at run time).
  3. Arm switches: the plain line `NBS2_ON = {...}` (a Python literal) at the top of CELL 1, in the notebook itself (outside
     the sha-pinned payload): edit it before the commit; absent -> every arm on, a missing key -> that arm off; an arm
     switched off is recorded SKIPPED_SWITCH_OFF and never booted (B off: the levers still run).
Arms (one change each vs B; budget priority): B > IDX > S28 > LP > S32 > SPC (nbs2_lib docstring). K4 / PP4: N/A_DFLASH; DYN: NB-S1.
"""
import ast
import base64
import builtins
import hashlib
import json
import lzma
import os
import runpy
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
NBDF_BUILDER = E27B / "build_e27_nbdf.py"
K100_REPORT = "D:/ARC AGI 2/_scratch/kits_wave3/q27-e27-nbdf-k100-kit-real/nbdf_kit_real_report.json"   # the exact string of the gated build (real kit ALL OK on e2198285)
NBDF_K100_SHA = "c20eba7f00b90b8d8e08308c48d57f09b404214039d6c4fa46c059fea92e2004"     # the gated NBDF-K100 (closed e2198285 + the kits_wave3 real-kit ALL OK report)
NBDF_K100_CLOSED_SHA = "e21982855d6430616f277784f162cad27cac94f7bcd8488fed97f123645ba730"   # = Exp 27 nbdf_k100_build_facts.json sha256 (B1 seam 341d859b, B2 lib b0147751)
SCRATCH = Path(r"D:/ARC AGI 2/_scratch/nbs2_build_ref")
OUT_NB = NBDIR / "Q27-XHIGH-E25_NBS2_CAPACITY.ipynb"
EXPORT, BODY_FILE, FACTS = HERE / "nbs2_code_export.py", HERE / "nbs2_cell7_body.py", HERE / "nbs2_build_facts.json"
SLUG, OWNER = "q27-xhigh-e25-nbs2-capacity", "esfersami50"
PUSH = Path(r"D:/ARC AGI 2/_scratch/push_e25") / f"{SLUG}__{OWNER}"
TITLE = "Q27 Xhigh E25 NBS2 Capacity"
LIB = HERE / "nbs2_lib.py"
DS_SLUG = "foysalemonshanto/q27-e25-nbs2-payload"      # NEW dataset (user creates it): the CELL 7 body + nbs2_lib.py, sha-pinned
DS_DIR = Path(r"D:/ARC AGI 2/_scratch/push_e25/ds_e25_nbs2_payload")
BUDGET_MIN = int(sys.argv[sys.argv.index("--budget-min") + 1]) if "--budget-min" in sys.argv else 175
DEADMAN_MIN = BUDGET_MIN + 5
PUSHED = set()        # shas of NB-S2 versions that ran on Kaggle (never overwrite one: rebuilds then go to *_REBUILD)
if OUT_NB.exists() and hashlib.sha256(OUT_NB.read_bytes()).hexdigest() in PUSHED:
    OUT_NB = OUT_NB.with_name(OUT_NB.stem + "_REBUILD.ipynb")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def once(text, old, new, label):
    assert text.count(old) == 1, f"NB-S2 {label}: x{text.count(old)}"
    return text.replace(old, new)


def b85x(text):
    return base64.b85encode(lzma.compress(text.encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


E27_SHAS_BEFORE = {p.name: sha(p.read_bytes()) for p in sorted(E27B.glob("*.py")) + sorted(E27B.glob("*.json")) + sorted(E27B.parent.glob("*.ipynb"))}


# ---------------------------------------------------------------- 1. the NBDF-K100 build, reproduced in-process (scratch outputs)
def nbdf_run(out_dir, report=None):
    os.environ["NBDF_VARIANT"] = "k100"
    argv0 = sys.argv[:]
    sys.argv = [str(NBDF_BUILDER), "--test-out", str(out_dir)] + (["--kv-gate-report", str(report)] if report else [])
    try:
        return runpy.run_path(str(NBDF_BUILDER), run_name="nbdf_build_for_nbs2")
    finally:
        sys.argv = argv0


SCRATCH.mkdir(parents=True, exist_ok=True)
for f in SCRATCH.glob("*"):
    if f.is_file():
        f.unlink()
_g0 = nbdf_run(SCRATCH)                                   # closed (writes SCRATCH/Q27-XHIGH-E27_NBDF_K100.ipynb)
assert sha(_g0["data"]) == NBDF_K100_CLOSED_SHA, ("the closed NBDF-K100 rebuild != e2198285", sha(_g0["data"]))
G = nbdf_run(SCRATCH, K100_REPORT)                        # gated; PRE_SHA = the closed file above (as the original rebuild)
assert sha(G["data"]) == NBDF_K100_SHA, ("the gated NBDF-K100 rebuild != c20eba7f", sha(G["data"]))
assert G["KV_GATE"]["passed"] is True and G["KV_GATE"]["per_k_success"]["D3s"]["success"] is True
KV_GATE = G["KV_GATE"]
CELL0, CELL1, CELL2, CELL2B, CELL3_6, CELL7_T, STUB_T, CELL8 = (G["CELL0"], G["CELL1"], G["CELL2"], G["CELL2B"], list(G["CELL3_6"]),
                                                                 G["CELL7"], G["CELL7_STUB"], G["CELL8"])
assert sha(CELL0.encode()) == G["E18_CELL0_SHA"]
KX = sys.modules["build_e27_nbdf_k100"]
MODELS, DATASETS = list(G["MODELS"]), list(KX.DATASETS) + [DS_SLUG]
_EX = (E27B / "nbdf_k100_code_export.py").read_text(encoding="utf-8")
assert sha(_EX) == json.loads((E27B / "nbdf_k100_build_facts.json").read_text())["export_sha256"]
assert json.loads((E27B / "nbdf_k100_build_facts.json").read_text())["sha256"] == NBDF_K100_CLOSED_SHA, "Exp 27's NBDF-K100 is not e2198285"
assert G["SEAM_SHA"] == "341d859be5b2cc619247ff7b645304b9473d09f07995358ead2ee02be116c60b", ("B1 seam", G["SEAM_SHA"])
assert G["LIB_SHA"] == "b014775162fb55401826fa8c498f7009b5d47ad15cc5cb365ab23e5ed37b1929", ("B2 nbdf_lib", G["LIB_SHA"])

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(E27B))
import nbs2_body_add as ADD  # noqa: E402
import nbs2_lib as S2  # noqa: E402
LIB_SRC = LIB.read_bytes().decode("utf-8")
assert "\r" not in LIB_SRC
compile(LIB_SRC, "nbs2_lib.py", "exec")
LIB_SHA = sha(LIB_SRC)

NBS2_ON_DEFAULT = dict(S2.SWITCH_DEFAULT)                # every arm on
assert list(NBS2_ON_DEFAULT) == ["B"] + list(S2.LEVERS) and all(v is True for v in NBS2_ON_DEFAULT.values())
SWITCH_LINE = "NBS2_ON = " + repr(NBS2_ON_DEFAULT) + "\n"
SWITCH_BLOCK = ("# ==================================================================== E25 NB-S2 ARM SWITCHES (edit before the commit)\n"
                "# True = the arm runs, False = it is not booted (recorded SKIPPED_SWITCH_OFF; the other arms are unaffected).\n"
                "# B = the DFlash2 base (NBDF-K100 D3s); B off: the levers still run. A key left out = that arm off.\n"
                "# Editing this line needs no rebuild and no new kit (it is outside the sha-pinned payload; every arm is kit-checked).\n"
                + SWITCH_LINE)
S2_CFG = {"base": "dflash_d3s", "base_key": "B", "arms": ["B"] + list(S2.LEVERS), "dropped": S2.DROPPED,
          "budget_min": BUDGET_MIN, "deadman_min": DEADMAN_MIN,
          "sibling": {"prefix_tokens": 6900, "suffix_tokens": 800, "gen": 16},
          "stall": {"decoders": 8, "inject": 2, "inject_tokens": 40000, "poll_ms": 20, "baseline_sec": 5},
          "windows_std": S2.STD_WINDOWS, "dyn_schedule": S2.DYN_SCHEDULE, "lp_args": S2.LP_ARGS, "idx": S2.IDX_CFG,
          "primary": S2.PRIMARY, "est_min": S2.EST, "cap_min": S2.CAP}

# ---------------------------------------------------------------- 2. CELL 1 (title, budget, deadman, out dir, results file)
c1 = CELL1
c1 = once(c1, 'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 150 * 60)     # E27 NBDF: hard stop 150 min (user: Q + D4 + D3 guaranteed; D4t if time)',
          f'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or {BUDGET_MIN} * 60)     # E25 NB-S2: hard stop {BUDGET_MIN} min (150 GPU-min of arms + boot / setup)', "c1 budget")
c1 = once(c1, f"NBDF_DEADMAN_SEC = {G['_N'].DEADMAN_MIN * 60}\n", f"NBDF_DEADMAN_SEC = {DEADMAN_MIN * 60}   # E25 NB-S2: 5 min after the hard stop\n", "c1 deadman")
c1 = once(c1, '"/kaggle/working/nbdf"))   # E27 NBDF (E27 K100: the same folder name)', '"/kaggle/working/nbs2"))   # E25 NB-S2', "c1 out")
c1 = once(c1, 'R = {"what": "E27 NBDF-K100: DFlash2 vs MTP K3 on the Q-defer base, 8k-116k (Flash-Next klee100 3bpw-MTP on Merlin)",   # E27 K100',
          'R = {"what": "E25 NB-S2 CAPACITY: indexer fp8 / seqs 28 / 32 / D4-D3 dynamic K / long-prefill / shared-prefix checkpoint on the DFlash2 base B (NBDF-K100 D3s), 8k-116k",   # E25 NB-S2', "c1 what")
c1 = once(c1, '        tmp = OUT / "nbdf_results.json.tmp"                      # E27 NBDF\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "nbdf_results.json")                   # E27 NBDF',
          '        tmp = OUT / "nbs2_results.json.tmp"                      # E25 NB-S2\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "nbs2_results.json")                   # E25 NB-S2', "c1 save")
c1 = once(c1, '''    print("E27 NBDF DFlash2: competition rerun -> CELL 0's identity fallback submission.json only (E18 R2 format)")   # E27 NBDF''',
          '''    print("E25 NB-S2 capacity: competition rerun -> CELL 0's identity fallback submission.json only (E18 R2 format)")   # E25 NB-S2''', "c1 rerun")
c1 = once(c1, "(RC2 verbatim + E25 NB1 / NB4S / NB4T / E27 NBDF / E27 K100 marked edits)",
          "(RC2 verbatim + E25 NB1 / NB4S / NB4T / E27 NBDF / E27 K100 / E25 NB-S2 marked edits)", "c1 title")
CELL1_S2 = SWITCH_BLOCK + c1                 # E25 NB-S2: the arm switches first (a plain editable line)

# ---------------------------------------------------------------- 3. the CELL 7 body (NBDF-K100's, the marked NB-S2 edits)
b = CELL7_T
b = once(b, "N.check_arm_plan(ARMS)                      # never-do list + \"each D arm = Q + exactly the DFlash change\" (raises: no boot)\n",
         "N.check_arm_plan(ARMS)                      # never-do list + \"each D arm = Q + exactly the DFlash change\" (raises: no boot)\n"
         + ADD.S2_SETUP.replace("__S2LIBSHA__", LIB_SHA).replace("__S2CFG__", json.dumps(S2_CFG)), "b setup")
b = once(b, 'TF_P = {b: PROMPTS[b][5] for b in K100_CFG["tf_bands"]}     # teacher-forced 2k / 8k / 40k (RC2\'s TF variant 5)\n',
         'TF_P = {b: PROMPTS[b][5] for b in K100_CFG["tf_bands"]}     # teacher-forced 2k / 8k / 40k (RC2\'s TF variant 5)\n' + ADD.S2_PROMPTS, "b prompts")
b = once(b, 'NBDF["order"] = list(N.ORDER)', 'NBDF["order"] = list(ARMS)                  # E25 NB-S2', "b order")
b = once(b, '''                  "priority": "Q > D4 > D3 > D7 > D4b (an arm starts only with est_min x 60 + 300 s left; D7 / D4b dropped first)",
                  "planned_walk": N.budget_walk({}, BUDGET / 60, 8.0, ARMS)}''',
         '''                  "priority": " > ".join(ARMS) + " (an arm starts only with est_min x 60 + 300 s left)",   # E25 NB-S2
                  "planned_walk": S2.budget_walk({k_: a_ for k_, a_ in ARMS.items() if NBS2_SWITCH.get(k_, True)}, BUDGET / 60)}''', "b budget")
b = once(b, '''        is_q, is_d = arm["key"] == "Q", arm.get("method") == "dflash"''',
         '''        is_q, is_d = arm["key"] == S2.BASE_KEY, arm.get("method") == "dflash"   # E25 NB-S2: B (a D arm) is the reference arm''', "b is_q")
b = once(b, '''        if is_q:
            _live(o, "seam_inert", N.q_seam_inert(t))
        else:
            _live(o, "dflash_readback", dflash_rb(arm, t))''', '''        if not is_d:                               # E25 NB-S2: every arm is a D arm (B included)
            _live(o, "seam_inert", N.q_seam_inert(t))
        else:
            _live(o, "dflash_readback", dflash_rb(arm, t))''', "b readback")
b = once(b, '''            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="SPEC_MISMATCH: " + sm_[1][:400])
            return o
''', '''            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="SPEC_MISMATCH: " + sm_[1][:400])
            return o
        try:                                       # E25 NB-S2: IDX not fp8 / DYN graphs not FULL (this arm only)
            _s2f = s2_readback_failure(o, arm, t)
        except Exception as e:                     # a read-back code error is recorded, never a stop
            _s2f, o["s2_readback_error"] = None, f"{type(e).__name__}: {e}"[:300]
        if _s2f:
            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop=_s2f)
            return o
''', "b s2 readback")
b = once(b, '''        if is_q:                                  # E28 K100 I8: INT8 requested''',
         '''        if is_q and not is_d:                     # E28 K100 I8 (never on a D arm): INT8 requested''', "b i8")
b = once(b, '''            if is_d and o["smoke_gate"].startswith("FAIL"):       # errors / garbage / short / zero accept / contamination''',
         '''            if is_q and o["smoke_gate"] in ("FAIL_SMOKE_CONCURRENCY_CONTAMINATION", "INCONCLUSIVE_SMOKE_CONCURRENCY_CONTAMINATION"):   # E25 NB-S2: B's ladder = its A/A (recorded)
                o["smoke_gate_raw"], o["smoke_gate"] = o["smoke_gate"], "BASE_LADDER_NONDET_RECORDED"
            if is_d and o["smoke_gate"].startswith("FAIL"):       # errors / garbage / short / zero accept / contamination''', "b smoke")
b = once(b, '''        if due(240) and not o.get("admission_deadlock"):
            CUR["item"] = "prefix_cache"''', '''        if due(240) and not o.get("admission_deadlock") and (arm.get("s2") or {}).get("prefix_test", True):   # E25 NB-S2: B / SPC
            CUR["item"] = "prefix_cache"''', "b prefix")
b = once(b, '''            o["items_skipped"].append("prefix_cache")
            o["prefix_gate"] = "INCONCLUSIVE_PREFIX_SKIPPED_BUDGET"
''', '''            o["items_skipped"].append("prefix_cache")
            o["prefix_gate"] = ("INCONCLUSIVE_PREFIX_SKIPPED_BUDGET" if (arm.get("s2") or {}).get("prefix_test", True)
                                else "NOT_IN_PLAN_NB_S2")
        if (arm.get("s2") or {}).get("sibling") and due(240) and not o.get("admission_deadlock"):   # E25 NB-S2: SPC read-back (B = control)
            CUR["item"] = "sibling_probe"
            try:
                _live(o, "sibling_probe", sibling_probe(p, o))
            except Exception as e:
                o["sibling_probe"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        if (arm.get("s2") or {}).get("stall") and due(420) and not o.get("admission_deadlock"):   # E25 NB-S2: LP's metric (B = control)
            CUR["item"] = "stall_probe"
            try:
                _live(o, "stall_probe", stall_probe(p, arm))
            except Exception as e:
                o["stall_probe"] = {"error": f"{type(e).__name__}: {e}"[:300]}
''', "b probes")
b = once(b, '''        for band, lvw, tag in K100_CFG["windows"]:''',
         '''        for band, lvw, tag in (arm.get("windows_fixed") or K100_CFG["windows"]):   # E25 NB-S2: S28 / S32 at their level''', "b windows")
b = once(b, '''            if band in ("40k",) + N.LONG_BANDS:
                lv, why = band_level(o, band)''', '''            if band in S2.CAPPED_BANDS:            # E25 NB-S2: 20k too (S32 x 20k can exceed the resident bound)
                lv, why = band_level(o, band)''', "b capped")
b = once(b, '''    return N.level_for_band(band, BANDS[band], pool, cap.get("max_concurrency_131072"), int(blk or N.BLOCK_TOKENS), o.get("kv_layout"),
                            PREFILL["tok_s"], WIN)''', '''    return N.level_for_band(band, BANDS[band], pool, cap.get("max_concurrency_131072"), int(blk or N.BLOCK_TOKENS), o.get("kv_layout"),
                            PREFILL["tok_s"], WIN, seqs=int(o.get("streams") or N.SEQS))   # E25 NB-S2: the arm's own max_num_seqs''', "b band seqs")
b = once(b, '''        for band, lvb in (("8k", int(K100_CFG["profile_level"])),) + ((("116k", lv116),) if long_ok and lv116 > 0 else ()):''',
         '''        for band, lvb in (((("8k", int(K100_CFG["profile_level"])),) + ((("116k", lv116),) if long_ok and lv116 > 0 else ()))
                          if (arm.get("s2") or {}).get("profile") else ()):   # E25 NB-S2: no profiles (NB-S1 profiles)''', "b profile")
b = once(b, '''        if not is_q:
            o["dflash_readback"] = dflash_rb(arm, t)               # the whole log: the SpecDecoding lines now exist''',
         '''        if is_d:                                   # E25 NB-S2: B included
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
            o["s2_records"] = s2_records(o, arm, t)     # E25 NB-S2: K_eff, tiers, KV ledgers (recorded)
        except Exception as e:                     # records only: never fails the arm
            o["s2_records"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        return o
    return body''', "b s2 records")
b = once(b, '''    ref = K100_CFG["q_ref"] if arm["key"] == "Q" else "q"   # E27 K100''',
         '''    ref = K100_CFG["q_ref"] if arm["key"] == S2.BASE_KEY else "q"   # E27 K100 / E25 NB-S2: B''', "b ref1")
b = once(b, '''    o = arm_boot(arm, arm_body(arm, K100_CFG["q_ref"] if key == "Q" else "q", "q" if key == "Q" else None))   # E27 K100''',
         '''    o = arm_boot(arm, arm_body(arm, K100_CFG["q_ref"] if key == S2.BASE_KEY else "q", "q" if key == S2.BASE_KEY else None))   # E25 NB-S2''', "b ref2")
b = once(b, '''    try:
        r = window(port, s, band, n)
    finally:
        rows, RESP["rows"] = RESP["rows"] or [], None''', '''    try:
        with S2Ledger(port) as _s2l:              # E25 NB-S2: the window's KV ledger samples
            r = window(port, s, band, n)
        r["_s2_samples"] = _s2l.samples
    finally:
        rows, RESP["rows"] = RESP["rows"] or [], None''', "b ledger")
b = once(b, '''            r = hist_window(s.port, s, band, n, arm)
''', '''            r = hist_window(s.port, s, band, n, arm)
        try:
            r["kv_ledger"] = s2_window_ledger(o, r, band, n)   # E25 NB-S2 (records only)
        except Exception as e:                     # records only: never voids the measured window
            r.pop("_s2_samples", None)
            r["kv_ledger"] = {"error": f"{type(e).__name__}: {e}"[:300]}
''', "b ledger summary")
b = once(b, "\ndef _gpu_util():\n", "\n" + ADD.S2_MAIN + "def _gpu_util():\n", "b main block")
b = once(b, '''    nbdf_main()
    _gpu_util_stop.set()''', '''    nbs2_main()                                  # E25 NB-S2
    _gpu_util_stop.set()''', "b main call")
CELL7_S2 = b

c8 = once(CELL8, '''    print("NBDF THROUGHPUT WINNER:", _v.get("best_arm"), _v.get("best_verdict"), _v.get("best_gain_8k"), "|", _v.get("why"))''',
          '''    print("NB-S2 CAPACITY WINNERS:", _v.get("winners"), _v.get("best_arm"), _v.get("best_verdict"), "|", _v.get("why"))   # E25 NB-S2''', "c8 print")
c8 = once(c8, '''    print("NBDF TABLE (tok/s, tick, AL per arm):", json.dumps(_v.get("table"), default=str)[:8000])''',
          '''    print("NB-S2 TABLE (tok/s, tick, AL per arm):", json.dumps(_v.get("table"), default=str)[:8000])
    print("NB-S2 BANDS:", json.dumps({k: {b_: (c_.get("status"), c_.get("gain"), c_.get("x_level")) for b_, c_ in (v_.get("bands") or {}).items()}
                                      for k, v_ in (_v.get("arms") or {}).items()}, default=str)[:6000])''', "c8 table")
CELL8_S2 = c8

FLAGS = json.loads(json.dumps(G["FLAGS"]))
FLAGS["NBS2"] = {
    "base": "B = NBDF-K100 D3s verbatim (nbdf_lib.arm_plan D3s after configure(SEQS=24, ...)); the reference arm (A/A twice)",
    "IDX": ["vllm/config/attention.py:88-92 indexer_kv_dtype ('auto' = the model default)", ":163-167 resolve_indexer_kv_dtype",
            "vllm/models/qwen4_exp/nvidia/indexer_qsa.py:154-166 (bf16 / fp8 only; fp8 = a plain Triton fp8 dot, sm_89 compile unverified)",
            "vllm/engine/arg_utils.py:1856 --attention-config (JSON; the dotted --attention-config.indexer_kv_dtype is equivalent but the "
            "--help flag gate checks the literal flag string, so the JSON form is used)", "launch.py:440-475: the flash-next argv has no --attention-config"],
    "S28_S32": ["launch.py:453 Merlin's --max-num-seqs (the trailing one wins)", "registry.py:83-89 capture rule (multiples of 1 + K here, nbdf_lib)"],
    "DYN": ["vllm/config/speculative.py:493-498 num_speculative_tokens_per_batch_size [(start, end, K)]",
            "vllm/v1/spec_decode/dynamic/utils.py:7-150 validation + the dense lookup (clamped to num_speculative_tokens = 4)",
            "vllm/v1/core/sched/scheduler.py:346-352, :1779-1783 K per step", "vllm/v1/worker/gpu/model_runner.py:2785-2803 propose(..., "
            "num_speculative_tokens=K) + drafts[:, K:] = -1 (method-generic)", "vllm/config/vllm.py:1069-1086 PIECEWISE override only without the "
            "V2 runner (launch.py sets VLLM_USE_V2_MODEL_RUNNER=1)", "vllm/v1/worker/gpu/cudagraph_utils.py:285-320 per-tier decode query lengths",
            "vllm/v1/spec_decode/metrics.py:33-50 per-position arrays sized num_speculative_tokens (4)"],
    "LP": ["vllm/config/scheduler.py:79-90 long_prefill_token_threshold + _adaptive", "vllm/engine/arg_utils.py:1735-1740",
           "launch.py:329 / 463-464 Merlin's CHUNK / LPT knobs (unset on flash-next: chunk 1024, no threshold)"],
    "SPC": ["vllm/config/cache.py:198-202", "vllm/v1/core/kv_cache_manager.py:185-198 (eagle group + partial hash hits + no re-prefillable)",
            "vllm/config/speculative.py:2007-2015 use_eagle includes dflash", "vllm/v1/core/kv_cache_coordinator.py:129-135 (no flagged group -> all groups)"],
    "dropped": S2.DROPPED}

MD0 = ("# E25 NB-S2 CAPACITY: capacity levers on the DFlash2 base (Qwen3.8-Flash-Next klee100 3bpw-MTP, Merlin v3.0.0, 4 x L4), 8k to 116k\n\n"
       "Arms: " + " > ".join(S2_CFG["arms"]) + " (one change each vs B = NBDF-K100's D3s: DFlash2 K3, BF16 drafter KV, DFLASH_DRAFT_SW_BLOCK=1024, "
       "Q-defer, seqs 24, pin 7,933,526,016). K4 / PP4: N/A_DFLASH; DYN moved to NB-S1. Boot path = the gated NBDF-K100 notebook c20eba7f (closed "
       "e2198285; reproduced byte for byte at build time). Arm switches: NBS2_ON at the top of CELL 1 (default: every arm on). "
       "Readable code: nbs2_cell7_body.py. Commit run only; a competition rerun writes CELL 0's identity submission (E18 R2 format).\n")


DS_LOADER = r'''def _nbs2_ds(name, want):                     # E25 NB-S2: a sha-pinned file of the payload dataset (notebook < 400 kB)
    for d in (INP / "datasets/foysalemonshanto/q27-e25-nbs2-payload", INP / "q27-e25-nbs2-payload",
              Path(os.environ.get("NBS2_PAYLOAD_DIR") or "/nonexistent")):
        f = d / name
        if f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest() == want:
            return f.read_bytes().decode("utf-8")
    raise RuntimeError(f"NBS2_PAYLOAD missing or sha mismatch: {name} (dataset foysalemonshanto/q27-e25-nbs2-payload)")
NBDF_BODY_SRC = _nbs2_ds("nbs2_cell7_body.py", NBDF_BODY_SHA)
'''


# ---------------------------------------------------------------- 4. assemble (= the NBDF builder's assemble with the NB-S2 cells)
def assemble():
    subs = {"__NB4TSHA__": G["NB4T_SHA"], "__E18C0SHA__": G["E18_CELL0_SHA"], "__FLAGS__": json.dumps(FLAGS), "__KVGATE__": json.dumps(KV_GATE)}
    body = CELL7_S2
    for k, v in subs.items():
        body = body.replace(k, v)
    assert '"""__' not in body and "__KVGATE" not in body and "__FLAGS" not in body and "__S2" not in body
    compile(body, "nbs2_cell7_body.py", "exec")
    st = STUB_T.replace("E27 NBDF: arms Q > D4 > D3 > D7 > D4b (DFlash2 vs MTP K3), long bands (NEW code)",
                        "E25 NB-S2: B > IDX > S28 > LP > S32 > SPC (capacity levers on DFlash2 D3s), long bands")
    st = once(st, 'NBDF_BODY_SRC = _unx("__BODYB85__")' + chr(10), DS_LOADER, "stub body from the dataset")
    for k, v in dict(G["_BLOBS"], __BODYSHA__=sha(body)).items():
        st = st.replace(k, v)
    assert "__LIB" not in st and "__SEAM" not in st and "__BODY" not in st
    cs = ([CELL0, CELL1_S2, CELL2, "if not RERUN:\n" + textwrap.indent(CELL2B, "    ")] + list(CELL3_6) +
          ["if not RERUN:\n" + textwrap.indent(c, "    ") for c in (st, CELL8_S2)])
    nb_ = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD0.splitlines(True)}] +
                    [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(True)} for c in cs],
           "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
           "nbformat": 4, "nbformat_minor": 5}
    return body, st, cs, nb_, json.dumps(nb_, separators=(",", ":")).encode("utf-8")


body, stub, cells_src, nb, data = assemble()
check_src = cells_src[:-1] + ["if not RERUN:\n" + textwrap.indent(body, "    "), cells_src[-1]]
for c in cells_src + check_src:
    compile(c, "<nbs2 cell>", "exec")
assert cells_src[0] == CELL0 and sha(cells_src[0].encode()) == G["E18_CELL0_SHA"], "CELL 0 != E18 R2 cell 1"
assert cells_src[1].startswith(SWITCH_BLOCK) and cells_src[1].count(SWITCH_LINE) == 1, "the CELL 1 switch line"
assert "NBS2_ON =" not in body and body.count('globals().get("NBS2_ON")') == 1, "the switch line must stay out of the payload body"
assert all(c.startswith("if not RERUN:\n") for c in cells_src[2:])
_defined, _loaded = set(dir(builtins)) | {"__file__", "__name__"}, set()
for c in check_src:
    for node in ast.walk(ast.parse(c)):
        if isinstance(node, ast.Name):
            (_loaded if isinstance(node.ctx, ast.Load) else _defined).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(node, ast.Lambda):
                _defined.add(node.name)
            _defined.update(a.arg for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs)
            if node.args.vararg:
                _defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                _defined.add(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            _defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            _defined.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            _defined.add(node.name)
        elif isinstance(node, ast.arg):
            _defined.add(node.arg)
_undef = sorted(_loaded - _defined)
assert not _undef, f"names loaded but never defined in the notebook: {_undef}"

# the arm plan at build time: NBDF-K100's D3s from nbdf_lib (configure as the K100 body does) -> nbs2 arm plan + check
_N = G["_N"]
_N.configure(SEQS=24, CONC_LADDER=(2, 8, 23), Q_CAPTURES=list(KX.K100_CFG["q_captures"]),
             Q_TAIL=["--compilation-config", _N.compilation_json(KX.K100_CFG["q_captures"])], WKEYS=tuple(KX.K100_CFG["wkeys"]))
_q_extra = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}", "--max-num-seqs", "24"]
_seam = {"DFLASH_SEAM": "1", "DFLASH_MASK_ROW_SHA": _N.DRAFTER_PINS["mask_row"]["sha256"], "DFLASH_DRAFT_SW_BLOCK": "1024"}
_arms0 = _N.arm_plan(_q_extra, {"VLLM_SERVER_DEV_MODE": "1", "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, 7933526016,
                     "/kaggle/input/models/foysalemonshanto/" + _N.DRAFTER_REL, q_env=G["_P"].server_env("Q-defer"),
                     seam_envs={"D3s": dict(_seam), "D4s": dict(_seam)})
_s2arms = S2.arm_plan(_arms0["D3s"], seam_env_k4=dict(_seam), pin=7933526016, seqs=24)
S2.check_arm_plan(_s2arms, _arms0["D3s"])
WALK = S2.budget_walk(_s2arms, BUDGET_MIN)

assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))
OUT_NB.write_bytes(data)
EXPORT.write_bytes("\n\n".join(cells_src).encode("utf-8"))
BODY_FILE.write_bytes(body.encode("utf-8"))
DS_DIR.mkdir(parents=True, exist_ok=True)
for f in DS_DIR.glob("*"):
    f.unlink()
(DS_DIR / "nbs2_cell7_body.py").write_bytes(body.encode("utf-8"))
(DS_DIR / "nbs2_lib.py").write_bytes(LIB_SRC.encode("utf-8"))
(DS_DIR / "dataset-metadata.json").write_bytes((json.dumps({"title": "q27-e25-nbs2-payload", "id": DS_SLUG, "licenses": [{"name": "CC0-1.0"}]},
                                                           indent=2) + chr(10)).encode("utf-8"))
(DS_DIR / "SHA256SUMS.json").write_bytes((json.dumps({"nbs2_cell7_body.py": sha(body), "nbs2_lib.py": LIB_SHA}, indent=1) + chr(10)).encode("utf-8"))
PUSH.mkdir(parents=True, exist_ok=True)
for f in PUSH.glob("*.ipynb"):
    f.unlink()
(PUSH / f"{SLUG}.ipynb").write_bytes(data)
meta = {"id": f"{OWNER}/{SLUG}", "title": TITLE, "code_file": f"{SLUG}.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": [],
        "dataset_sources": DATASETS, "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": MODELS,
        "docker_image": G["IMAGE"], "machine_shape": "NvidiaL4"}
(PUSH / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
PUSH_F = Path(r"D:/ARC AGI 2/_scratch/push_e25") / f"{SLUG}__foysalemonshanto"     # the same notebook for the foysal account
PUSH_F.mkdir(parents=True, exist_ok=True)
for f in PUSH_F.glob("*.ipynb"):
    f.unlink()
(PUSH_F / f"{SLUG}.ipynb").write_bytes(data)
(PUSH_F / "kernel-metadata.json").write_bytes((json.dumps(dict(meta, id=f"foysalemonshanto/{SLUG}"), indent=2) + "\n").encode("utf-8"))
E27_SHAS_AFTER = {p.name: sha(p.read_bytes()) for p in sorted(E27B.glob("*.py")) + sorted(E27B.glob("*.json")) + sorted(E27B.parent.glob("*.ipynb"))}
_e27_changed = sorted(k for k in set(E27_SHAS_BEFORE) | set(E27_SHAS_AFTER) if E27_SHAS_BEFORE.get(k) != E27_SHAS_AFTER.get(k))
facts = {"notebook": OUT_NB.name, "sha256": sha(data), "bytes": len(data), "cells": len(nb["cells"]),
         "cell_bytes": [len(c.encode()) for c in cells_src], "nbs2_lib_sha256": LIB_SHA, "nbdf_k100_reproduced_sha256": NBDF_K100_SHA,
         "nbdf_k100_closed_sha256": NBDF_K100_CLOSED_SHA, "kv_kit_gate": KV_GATE, "s2_cfg": S2_CFG, "budget_min": BUDGET_MIN,
         "deadman_min": DEADMAN_MIN, "planned_walk": WALK, "e18_r2_cell0_sha256": G["E18_CELL0_SHA"], "export_sha256": sha(EXPORT.read_bytes()),
         "body_file_sha256": sha(body), "push_dir": str(PUSH), "payload_dataset": {"slug": DS_SLUG, "dir": str(DS_DIR),
         "files": {"nbs2_cell7_body.py": sha(body), "nbs2_lib.py": LIB_SHA}}, "model_sources": MODELS, "dataset_sources": DATASETS,
         "nbdf_lib_sha256": G["LIB_SHA"], "seam_sha256": G["SEAM_SHA"], "ppack_sha256": G["PP_SHA"],
         "arms": {k: {x: a.get(x) for x in ("title", "port", "spec", "captures", "streams", "windows_fixed", "est_min", "cap_min", "change", "s2")}
                  for k, a in _s2arms.items()},
         "e27_files_changed_by_this_build": _e27_changed, "flags": FLAGS["NBS2"],
         "switch": {"line": SWITCH_LINE.rstrip("\n"), "default": NBS2_ON_DEFAULT, "cell": 1, "in_payload": False},
         "k100_report": K100_REPORT}
FACTS.write_bytes((json.dumps(facts, indent=1) + "\n").encode("utf-8"))
print(json.dumps({k: v for k, v in facts.items() if k not in ("kv_kit_gate", "arms", "flags", "s2_cfg")}, indent=1))
if _e27_changed:
    raise SystemExit(f"Exp 27 files changed during the build: {_e27_changed}")
