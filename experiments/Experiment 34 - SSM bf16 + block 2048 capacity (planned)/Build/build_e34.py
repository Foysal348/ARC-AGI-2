"""E34 (2026-11-03): GDN SSM state bf16 instead of fp32 with KV block 2048 instead of 4096, on the DFlash2 production engine
(NB-D1's base B = E29_DFLASH: NBDF-K100 D3s + DFLASH_DRAFT_EMBED_HOST=1 first try + QSA 128 / 64; Qwen3.8-Flash-Next klee100
3bpw-MTP on Merlin v3.0.0 ad42e37, Kaggle 4 x L4, TP2 x PP2 + EP). Arms B / S16 / S16B2 / S16B2W (e34_lib.py).

  python -B build_e34.py [--kv-gate-report FILE] [--test-out DIR]
      -> ../Experiment Notebooks/Q27-XHIGH-E34_FULL.ipynb (every arm; the kit notebook; the closed build)
         ../Experiment Notebooks/Q27-XHIGH-E34_SMOKE.ipynb (S16B2's smoke plan; only the CELL 1 copy block differs)
         e34_code_export.py, e34_cell7_body.py, e34_build_facts.json,
         push/ds_e34_payload/ (dataset foysalemonshanto/q27-e34-payload: body + nbd1_lib + e34_lib; NOT uploaded),
         push/ds_e34_tf_long/ (dataset foysalemonshanto/q27-e34-tf-long: the recorded traces; NOT uploaded),
         push/q27-xhigh-e34-{full,smoke}__foysalemonshanto/ (GPU push folders; nothing is pushed)
      Without --kv-gate-report the build is CLOSED (every arm SKIPPED_KV_KIT_GATE at run time, gpu_ready false); with the
      E34 real kit's report (e34_kit_real_report.json, ALL OK on THIS closed FULL build) -> gated.
      --test-out DIR: every output under DIR (a reproduction; the real outputs are not touched).

How it is built (NB-D1's pattern, one level up; NB-D1 / NBDF / E29 files are read-only, asserted byte-identical before and after):
  1. NB-D1's builder (Exp 25 build_e25_nbd1.py) runs IN THIS PROCESS with --test-out (outputs only under Build/_ref_nbd1):
     its closed notebook must be byte-identical to NB-D1's abbf21df (asserted), i.e. B's boot path IS NB-D1's (which is
     NBDF-K100's e2198285 / c20eba7f, asserted inside NB-D1's builder).
  2. Its cells are taken from that run (CELL 0 = E18 R2 verbatim, asserted). E34 edits are exact-once text replacements on
     CELL 1 (the copy block E34_ON + E34_BUDGET_MIN at the top, budget / deadman / out dir / title), the CELL 7 body
     (e34_body_add.py blocks + the marked edits below) and CELL 8; the body, nbd1_lib and e34_lib live in the payload dataset
     (sha-pinned loader _e34_ds, aliased _nbd1_ds); the recorded TF traces in their own dataset (sha-pinned).
  3. Body edits: E34's arms on NB-D1's B (E34.arm_plan / check_arm_plan), NBDF's oracle / long oracle / prefix / TF items ON
     (NB-D1's own code, its item switches), the TF set (8k / 40k + 3 recorded traces), the never-do lift of exactly
     `--mamba-ssm-cache-dtype bfloat16` on the SSM arms (E34.launch_check), the dtype / block / SW read-back (a real failure
     of that arm), the boot capacity record + [LIVE] boot line, a [LIVE] line after every test, the W re-plan of S16B2W from
     S16B2's boot, an empty window set = no window (the smoke plan), the E34 records in the verdict.
"""
import ast
import builtins
import copy
import hashlib
import json
import os
import runpy
import shutil
import sys
import textwrap
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E34 = HERE.parent
NBDIR = E34 / "Experiment Notebooks"
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
E25NB = E25B.parent
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
E29 = Path(r"D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)")
E31_DUCK = Path(r"D:/ARC AGI 2/_scratch/e31_hamid_out/q27x/phase_Q_E31_FN_DIRECT30/duck")
NBD1_BUILDER = E25B / "build_e25_nbd1.py"
NBD1_CLOSED_SHA = "abbf21dfa9cda439de7e557e5782526512f9087c7e36a0c3f70625c30d13bb6d"
NBD1_LIB_SHA = "af0f795f685fe63f6169e5171f25f5d35f25bf66cfc6538c1272554d3190bad3"
TEST_OUT = Path(sys.argv[sys.argv.index("--test-out") + 1]) if "--test-out" in sys.argv else None
KV_GATE_REPORT = sys.argv[sys.argv.index("--kv-gate-report") + 1] if "--kv-gate-report" in sys.argv else None
OUT_BUILD = TEST_OUT or HERE
OUT_NBDIR = TEST_OUT or NBDIR
REF = (TEST_OUT / "_ref_nbd1") if TEST_OUT else HERE / "_ref_nbd1"
PUSH_ROOT = (TEST_OUT / "push") if TEST_OUT else HERE / "push"
EXPORT, BODY_FILE, FACTS = OUT_BUILD / "e34_code_export.py", OUT_BUILD / "e34_cell7_body.py", OUT_BUILD / "e34_build_facts.json"
SLUG, TITLE = "q27-xhigh-e34", "Q27 Xhigh E34"
NB_NAME = {"full": "Q27-XHIGH-E34_FULL.ipynb", "smoke": "Q27-XHIGH-E34_SMOKE.ipynb"}
DS_SLUG, TF_SLUG = "foysalemonshanto/q27-e34-payload", "foysalemonshanto/q27-e34-tf-long"
DS_DIR, TF_DIR = PUSH_ROOT / "ds_e34_payload", PUSH_ROOT / "ds_e34_tf_long"
PUSHED = set()        # shas of E34 versions that ran on Kaggle (never overwrite one: rebuild with --test-out)


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def once(text, old, new, label):
    assert text.count(old) == 1, f"E34 {label}: x{text.count(old)}"
    return text.replace(old, new)


if not TEST_OUT:
    for _p in [NBDIR / n for n in NB_NAME.values()]:
        if _p.exists() and sha(_p.read_bytes()) in PUSHED:
            raise SystemExit(f"{_p.name} ran on Kaggle (PUSHED): never rebuild a run notebook; use --test-out")

# the frozen files: NB-D1 (Exp 25 Build + notebooks), NBDF (Exp 27 Build + notebooks), E29 (all), the E31 traces, the capacity port
FROZEN = sorted(p for p in E25B.glob("*") if p.is_file()) + sorted(E25NB.glob("Q27-XHIGH-E25_NBD1*.ipynb"))
FROZEN += sorted(p for p in E27B.glob("*") if p.is_file()) + sorted(E27B.parent.glob("*.ipynb"))
FROZEN += sorted(p for p in E29.rglob("*") if p.is_file())
FROZEN += sorted(p for p in E31_DUCK.rglob("token_ids.json"))
FROZEN += [Path(r"D:/ARC AGI 2/_scratch/capacity_port/measure_capacity.py")]
FROZEN_SHA = {str(p): sha(p.read_bytes()) for p in FROZEN}

# ---------------------------------------------------------------- 1. NB-D1's build, reproduced in-process (outputs under REF)
if REF.exists():
    shutil.rmtree(REF)
REF.mkdir(parents=True)
_argv0 = sys.argv[:]
sys.argv = [str(NBD1_BUILDER), "--test-out", str(REF)]
try:
    G1 = runpy.run_path(str(NBD1_BUILDER), run_name="nbd1_build_for_e34")
finally:
    sys.argv = _argv0
assert sha(G1["data"]) == NBD1_CLOSED_SHA, ("the closed NB-D1 rebuild != abbf21df", sha(G1["data"]))
assert json.loads((E25B / "nbd1_build_facts.json").read_text(encoding="utf-8"))["sha256"] == NBD1_CLOSED_SHA, "Exp 25's NB-D1 is not abbf21df"
assert G1["LIB_SHA"] == NBD1_LIB_SHA
G = G1["G"]                                   # the NBDF-K100 build namespace (inside NB-D1's)
D1, ADD1, KX = G1["D1"], G1["ADD"], G1["KX"]
CELL0, CELL2, CELL2B, CELL3_6 = G1["CELL0"], G1["CELL2"], G1["CELL2B"], list(G1["CELL3_6"])
CELL1_D1, CELL7_D1, CELL8_D1, STUB_T = G1["CELL1_D1"], G1["CELL7_D1"], G1["CELL8_D1"], G1["STUB_T"]
K100_GATE, MODELS = G1["K100_GATE"], list(G1["MODELS"])
NBD1_LIB_SRC = G1["LIB_SRC"]
assert sha(CELL0.encode()) == G["E18_CELL0_SHA"]
DATASETS = list(KX.DATASETS) + [DS_SLUG, TF_SLUG]

sys.path.insert(0, str(HERE))
import e34_body_add as ADD  # noqa: E402
import e34_lib as E  # noqa: E402
LIB_SRC = (HERE / "e34_lib.py").read_bytes().decode("utf-8")
assert "\r" not in LIB_SRC
compile(LIB_SRC, "e34_lib.py", "exec")
LIB_SHA = sha(LIB_SRC)

# ---------------------------------------------------------------- 2. the TF traces (E31 direct recorded thinking, prompt + p1)
_tr = {}
for nm, tid in E.TF_LONG:
    tk = json.loads((E31_DUCK / f"{tid}_b0" / "token_ids.json").read_text(encoding="utf-8"))
    br = json.loads((E31_DUCK / f"{tid}_b0" / "branch_result.json").read_text(encoding="utf-8"))
    _tr[nm] = (tid, tk["prompt"], tk["p1"], {"source": f"E31 FN direct (hamid run) {tid}_b0/token_ids.json", "accepted": br.get("accepted"),
                                             "generated_tokens": br.get("generated_tokens"), "usage": br.get("usage")})
TF_LONG = E.tf_long_payload(_tr)
TF_BYTES = (json.dumps(TF_LONG, separators=(",", ":")) + "\n").encode("utf-8")
TF_SHA = sha(TF_BYTES)
for nm, v in TF_LONG["traces"].items():
    assert 60000 <= v["len"] <= 131072 - 1024, (nm, v["len"])
assert [v["len"] for v in TF_LONG["traces"].values()] == sorted(v["len"] for v in TF_LONG["traces"].values())

# ---------------------------------------------------------------- 3. the arm plans + the time plan at build time (NB-D1's nbdf_lib)
_N, _T, _A0, _SEAM = G1["_N"], G1["_T"], G1["_arms0"], G1["_seam"]
CAP = json.loads((E34 / "Experiment Results" / "e34_capacity_probe.json").read_text(encoding="utf-8"))
_cr = {r["e34_arm"]["name"].split(":")[0]: r for r in CAP["results"]}
assert min(_cr["B"]["num_blocks"]) == E.CPU_PLANNER["B"]["pool"] and min(_cr["S16"]["num_blocks"]) == E.CPU_PLANNER["S16"]["pool"]
assert min(_cr["S16B2"]["num_blocks"]) == E.CPU_PLANNER["S16B2"]["pool"] and _cr["S16B2"]["F_admission"] == E.CPU_PLANNER["S16B2"]["F_adm"]
assert min(_cr["S16B2 (SW1024)"]["num_blocks"]) == E.CPU_PLANNER["S16B2_SW1024"]["pool"]
assert any(r.get("BLOCK_RAISED") for r in CAP["results"]), "the fp32 + block 2048 control must be raised by the platform"
assert E.W_PRED == (E.CPU_PLANNER["S16B2"]["pool"] - 2) // (-(-E.W_BAND_TOKENS // 2048) + E.CPU_PLANNER["S16B2"]["F_adm"])
for _k, _pool, _mc, _blk, _seqs in (("B", 540, 12.25, 4096, 24), ("S16B2W", 1080, 13.32, 2048, E.W_PRED)):
    for _b, _tok in (("20k", 20000), ("40k", 40000), ("90k", 90000), ("116k", 116000)):
        _lv, _ = _N.level_for_band(_b, _tok, _pool, _mc, _blk if _blk == 4096 else 512, {"block_sizes": {"FullAttentionSpec": [_blk]}},
                                   E.ADMIT_TOK_S, E.WIN_SEC, seqs=_seqs)
        assert E.PRED_LEVELS[_k][_b] == _lv, ("PRED_LEVELS", _k, _b, _lv)
PLANS, WALKS, NEEDS, BUDGETS = {}, {}, {}, {}
for _x, _sw in list(E.COPIES.items()) + [("default", None)]:
    _a = E.arm_plan(D1, _N, _A0["D3s"], seam_env_k4=dict(_SEAM), switch=_sw, pin=7933526016, seqs=24)
    E.check_arm_plan(D1, _N, _T, _a, _A0["D3s"], _sw)
    PLANS[_x] = _a
for _w in range(E.W_MIN, E.W_MAX + 1):        # every W the run-time re-plan can choose is a valid plan
    _a = dict(PLANS["full"])
    _a["S16B2W"] = E.lever_arm(D1, _N, PLANS["full"]["B"], "S16B2W", E.COPIES["full"], _w)
    E.check_arm_plan(D1, _N, _T, _a, _A0["D3s"], E.COPIES["full"])
for _x, _sw in E.COPIES.items():
    BUDGETS[_x] = E.budget_for(PLANS[_x], _sw)
    WALKS[_x] = E.copy_walk(PLANS[_x], _sw, BUDGETS[_x])
    NEEDS[_x] = E.copy_need_min(PLANS[_x], _sw)
    assert WALKS[_x]["all_run"], (_x, WALKS[_x])
    assert WALKS[_x]["runtime_min"] <= 11 * 60, (_x, "a copy over ~11 h must be split (user)")
    for _k, _arm in PLANS[_x].items():
        assert _arm["cap_min"] >= _arm["time_model"]["body"] + 1, (_x, _k, "the body cap holds the body")
assert BUDGETS["smoke"] <= 40, ("the smoke copy is a ~25-min hard-stop run", BUDGETS["smoke"])
E34_CFG = {"what": "E34 SSM bf16 + block 2048", "base": "NB-D1 B (E29_DFLASH)", "arms": list(E.ARM_KEYS), "switch_keys": list(E.SWITCH_KEYS),
           "copies": E.COPIES, "owners": E.OWNERS, "budgets_min": BUDGETS, "plan": E.PLAN, "w_rule": {"band_tokens": E.W_BAND_TOKENS,
           "min": E.W_MIN, "max": E.W_MAX, "pred": E.W_PRED}, "cpu_planner": E.CPU_PLANNER, "tf_long_sha256": TF_SHA,
           "tf_long_file": E.TF_LONG_FILE, "tf_long_dataset": TF_SLUG, "lib_sha256": LIB_SHA, "never_do_lifted": E.LIFTED}

# ---------------------------------------------------------------- 4. CELL 1 (the copy block at the top, budget, deadman, out dir, title)
c1 = CELL1_D1
c1 = once(c1, f'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or {D1.BUDGET_MIN} * 60)',
          'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or E34_BUDGET_MIN * 60)', "c1 budget")
c1 = once(c1, f"NBDF_DEADMAN_SEC = {D1.DEADMAN_MIN * 60}   # E25 NB-D1: 5 min after the hard stop\n",
          "NBDF_DEADMAN_SEC = (E34_BUDGET_MIN + 5) * 60   # E34: 5 min after the copy's hard stop\n", "c1 deadman")
c1 = once(c1, '"/kaggle/working/nbd1"))   # E25 NB-D1', '"/kaggle/working/e34"))   # E34', "c1 out")
c1 = once(c1, 'R = {"what": "E25 NB-D1: 1500+ speed levers (QSA 256, K4, dynamic K, GDN, indexer fp8, long prefill, seqs 28 / 32, shared-prefix checkpoint, profiles) on the DFlash2 production base B (E29_DFLASH), 8k-116k",   # E25 NB-D1',
          'R = {"what": "E34: GDN SSM state bf16 + KV block 2048 (B / S16 / S16B2 / S16B2W) on the DFlash2 production base B (E29_DFLASH), capacity + speed 8k-116k + accuracy (TF on 69k-113k recorded traces, oracle, long oracle)",   # E34',
          "c1 what")
c1 = once(c1, '        tmp = OUT / "nbd1_results.json.tmp"                      # E25 NB-D1\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "nbd1_results.json")                   # E25 NB-D1',
          '        tmp = OUT / "e34_results.json.tmp"                       # E34\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "e34_results.json")                    # E34', "c1 save")
c1 = once(c1, '''    print("E25 NB-D1 speed levers: competition rerun -> CELL 0's identity fallback submission.json only (E18 R2 format)")   # E25 NB-D1''',
          '''    print("E34 SSM bf16 + block 2048: competition rerun -> CELL 0's identity fallback submission.json only (E18 R2 format)")   # E34''', "c1 rerun")
c1 = once(c1, "/ E27 K100 / E25 NB-D1 marked edits)", "/ E27 K100 / E25 NB-D1 / E34 marked edits)", "c1 title")
c1 = once(c1, 'COMP = INP / "competitions" / "arc-prize-2026-arc-agi-2"\n',
          'COMP = next((p_ for p_ in (INP / "competitions" / "arc-prize-2026-arc-agi-2", INP / "arc-prize-2026-arc-agi-2") if p_.exists()),\n'
          '            INP / "competitions" / "arc-prize-2026-arc-agi-2")   # E34: both Kaggle mount layouts (the legacy one has no competitions/)\n',
          "c1 comp layouts")
CELL1_E = c1
COPY_HEAD = ("# ==================================================================== E34 ARM SWITCHES + HARD STOP (edit before the commit)\n"
             "# True = the arm runs, False = it is not booted (recorded SKIPPED_SWITCH_OFF; the other arms are unaffected). A key left out = off.\n"
             "# B = the DFlash2 production base (NB-D1's B = E29_DFLASH); S16 = + SSM bf16; S16B2 = + SSM bf16 + block 2048 + drafter SW 512;\n"
             "# S16B2W = S16B2 + max_num_seqs W from S16B2's measured pool; BMAX = B's max-resident windows; SMOKE = S16B2's smoke plan.\n"
             "# FULL: B S16 S16B2 S16B2W BMAX | SMOKE: S16B2 SMOKE. The budget line below is this copy's hard stop in minutes (deadman 5 min later).\n")


def copy_block(x):
    return COPY_HEAD + "E34_ON = " + repr(dict(E.COPIES[x])) + "\n" + f"E34_BUDGET_MIN = {BUDGETS[x]}\n"


# ---------------------------------------------------------------- 5. the CELL 7 body (NB-D1's, the marked E34 edits)
b = CELL7_D1
b = once(b, 'print("E25 NB-D1 arm switches:", NBD1_SWITCH, "| walk:", json.dumps(NBD1["walk"])[:700], flush=True)\n',
         'print("E25 NB-D1 arm switches:", NBD1_SWITCH, "| walk:", json.dumps(NBD1["walk"])[:700], flush=True)\n'
         + ADD.E34_SETUP.replace("__E34LIBSHA__", LIB_SHA).replace("__E34CFG__", json.dumps(E34_CFG)), "b setup")
b = once(b, '    PROMPTS["20k"] = list(PROMPTS["20k"]) + [prompt(BANDS["20k"], v) for v in range(len(PROMPTS["20k"]), 32)]\n',
         '    PROMPTS["20k"] = list(PROMPTS["20k"]) + [prompt(BANDS["20k"], v) for v in range(len(PROMPTS["20k"]), 32)]\n' + ADD.E34_PROMPTS,
         "b prompts")
b = once(b, '"planned_walk": D1.copy_walk(ARMS, NBD1_SWITCH, BUDGET / 60)}', '"planned_walk": E34.copy_walk(ARMS, NBD1_SWITCH, BUDGET / 60)}   # E34',
         "b walk")
b = once(b, '''        rec.setdefault("live", {})[key] = val
    save()
    return val
''', '''        rec.setdefault("live", {})[key] = val
    e34_item_live(o, key, val)                    # E34: a [LIVE] line after every test (print only)
    save()
    return val
''', "b live item")
b = once(b, '''    bad = N.launch_check(arm, s.env, s.argv, kv_bytes) if arm else [f"launch {name} has no arm"]''',
         '''    bad = E34.launch_check(N, T, arm, s.env, s.argv, kv_bytes) if arm else [f"launch {name} has no arm"]   # E34: NBDF's check minus exactly the lifted SSM bf16 entry + the SSM / block / SW plan''',
         "b launch check")
b = once(b, '''        o["k100_residents"] = _k100_residents(o)              # E27 K100: residents at 40k / 64k / 90k / 120k (block 4096)\n''',
         '''        o["k100_residents"] = _k100_residents(o)              # E27 K100: residents at 40k / 64k / 90k / 120k (block 4096)\n'''
         '''        e34_capacity_rec(o, arm, t)                           # E34: boot capacity + dtype / block / SW read-back (records, printed)\n''',
         "b capacity")
b = once(b, '''        if due(150):
            CUR["item"] = "warmup"
            _live(o, "warmup", warmup(p, n))
        else:
            o["items_skipped"].append("warmup")''', '''        if due(150) and arm.get("windows_fixed") != []:   # E34: no warmup when the arm has no window (the smoke plan)
            CUR["item"] = "warmup"
            _live(o, "warmup", warmup(p, n))
        else:
            o["items_skipped"].append("warmup")''', "b warmup")
b = once(b, '''        for band, lvw, tag in (arm.get("windows_fixed") or K100_CFG["windows"]):   # E25 NB-D1: the arm's light window set''',
         '''        for band, lvw, tag in (arm["windows_fixed"] if arm.get("windows_fixed") is not None else K100_CFG["windows"]):   # E25 NB-D1 / E34: an empty set = no window''',
         "b windows")
b = once(b, '''                if is_q:
                    _live(o, "tf_ref", tforced(p, key="q_tf"))''', '''                if is_q or arm.get("e34_smoke"):    # E34: the smoke plan stores its own TF reference + A/A (the request path itself)
                    _live(o, "tf_ref", tforced(p, key="q_tf"))''', "b tf smoke")
b = once(b, '''    o["d1_readback"] = D1.readback(arm, t, o.get("env_readback_raw"), o.get("config_readback"), o.get("workers"), o.get("dflash_readback"))\n''',
         '''    o["d1_readback"] = D1.readback(arm, t, o.get("env_readback_raw"), o.get("config_readback"), o.get("workers"), o.get("dflash_readback"))\n'''
         '''    o["e34_readback"] = E34.readback(arm, t, o.get("workers"), o.get("dflash_readback"))   # E34: SSM dtype / block / SW per worker\n''',
         "b e34 readback")
b = once(b, '''    if o["d1_readback"].get("real_failure"):
        return "NB-D1 read-back: " + str(o["d1_readback"].get("why"))[:400]''', '''    if (o.get("e34_readback") or {}).get("real_failure"):   # E34: the planned SSM dtype / block / SW not engaged = this arm only
        return "E34 read-back: " + str(o["e34_readback"].get("why"))[:400]
    if o["d1_readback"].get("real_failure"):
        return "NB-D1 read-back: " + str(o["d1_readback"].get("why"))[:400]''', "b e34 readback failure")
b = once(b, '''        print(D1.live_line((time.time() - T_START) / 60.0, arm["key"], key, r, b_rec), flush=True)''',
         '''        print(E34.live_line(D1, (time.time() - T_START) / 60.0, arm["key"], key, r, b_rec), flush=True)   # E34: + prefix hit, W vs B's W''',
         "b live window")
b = once(b, '''            if k not in RUN_KEYS:                # E25 NB-D1: switched off (recorded above, before any other check)
                continue
''', '''            if k not in RUN_KEYS:                # E25 NB-D1: switched off (recorded above, before any other check)
                continue
            if k in E34.W_ARMS:                  # E34: S16B2W's max_num_seqs from S16B2's measured pool (its boot log)
                a = e34_w_replan(k, a)
''', "b w replan")
b = once(b, '''        v = D1.verdict(T, {k: r for k, r in recs.items() if r is not None}, ARMS, NBD1_SWITCH, N.LONG_THR_FLOOR)\n''',
         '''        v = D1.verdict(T, {k: r for k, r in recs.items() if r is not None}, ARMS, NBD1_SWITCH, N.LONG_THR_FLOOR)\n'''
         '''        v["e34"] = dict(E34.records({k: r for k, r in recs.items() if r is not None}, ARMS, NBD1_SWITCH), w=E34R.get("w"), tf=E34R.get("tf"))   # E34\n''',
         "b verdict e34")
b = once(b, '''        (OUT / "nbd1_verdict.json").write_text(json.dumps(v, indent=1, default=str))''',
         '''        (OUT / "nbd1_verdict.json").write_text(json.dumps(v, indent=1, default=str))
        (OUT / "e34_verdict.json").write_text(json.dumps(v.get("e34"), indent=1, default=str))   # E34''', "b verdict file")
b = once(b, "\ndef _gpu_util():\n", "\n" + ADD.E34_MAIN + "def _gpu_util():\n", "b main block")
CELL7_E = b

_i = CELL8_D1.index('print("NB-D1 PROF:", json.dumps({k_: (_v.get("prof") or {}).get(k_) for k_ in ("status", "why", "windows")}, default=str)[:3000])')
_ls = CELL8_D1.rfind("\n", 0, _i) + 1
_ind = CELL8_D1[_ls:_i]
assert _ind.strip() == ""
_j = CELL8_D1.index("\n", _i) + 1
CELL8_E = (CELL8_D1[:_j] + _ind + 'print("E34 RECORDS (raw numbers; every label record-only):", json.dumps(_v.get("e34"), default=str)[:20000])   # E34\n'
           + CELL8_D1[_j:])

FLAGS = json.loads(json.dumps(G1["FLAGS"]))
FLAGS["E34"] = {
    "ssm_bf16": ["vllm/engine/arg_utils.py:1396 --mamba-ssm-cache-dtype (config/cache.py:63 MambaDType incl. bfloat16, :186 default auto)",
                 "model_executor/models/config.py:908-934 Qwen3_5 hook (Qwen4ExpForConditionalGenerationConfig :966-973 inherits): auto -> the "
                 "HF mamba_ssm_dtype (float32); an explicit value wins + the warning 'Qwen3.5 model specifies mamba_ssm_dtype=...' (:928)",
                 "models/qwen4_exp/nvidia/model.py:1016-1027 GDN state dtypes = (conv, ssm) from cache_config + the fp32 replay record (RecoverSSM)",
                 "model_executor/layers/mamba/mamba_utils.py:99-112 ssm dtype = cache_config.mamba_ssm_cache_dtype",
                 "config/vllm.py:3417-3423 RecoverSSM accepts bf16 / fp32 SSM caches (the only refusal: stochastic rounding); :3477-3491 fp32 is "
                 "required only for NemotronH Mamba-2",
                 "model_executor/layers/mamba/gdn/recoverssm_gdn.py:254 only the replay record must be fp32; :440 / :445 the state store "
                 "casts to the cache dtype; :495 every layer the same state dtype"],
    "block_2048": ["platforms/interface.py:940-1140 _align_hybrid_block_size: kernel alignment = max(min backend block, block_size) "
                   "(:1077-1086), attention block = alignment x cdiv(mamba page, alignment x 512 B) (:1087-1093), raised + 'Setting attention "
                   "block size' when smaller (:1095-1101), align mode mamba_block_size = block_size (:1103-1104), the Mamba page padded to "
                   "block x 512 B (:1107-1122 'Padding mamba page size'): bf16 946,560 B <= 1,048,576 B at 2048 (10.78 %); fp32 1,732,992 B "
                   "-> raised to 4096 (the CPU planner control)",
                   "v1/attention/backends/flash_attn.py:354-365 FA2 on sm_89: MultipleOf(16); models/qwen4_exp/nvidia/qsa.py:377-379 QSA MultipleOf(16)",
                   "models/qwen4_exp/common/qsa_cache.py:954-970 QSA ring capacity 4 x ceil((4 + K) / 4) = 8 must divide the block (8 | 2048)",
                   "models/qwen4_exp/common/qsa_cache.py:976-986 the indexer compressed-key cache = an MLAAttentionSpec at the cache block, "
                   "tokens_per_state 4 (2048 / 4 = 512 rows); nvidia/indexer_qsa.py:187-200 page_cap = blocks x block / 4 (= 32,768 rows at "
                   "4096 and at 2048)",
                   "v1/core/kv_cache_utils.py:805-905 scheduler block = LCM of the group blocks (2048), hash unit = prefix_match_unit 64 "
                   "(divides 2048 / 512, :884), the compressed key's 4-token alignment (64 % 4, :904)",
                   "v1/engine/core.py:431-450 the engine block = min over the prefix-cacheable groups (the SW drafter block: 1024 on B, 512 "
                   "on S16B2); config/vllm.py:3289-3318 validate_block_size (align: chunked MM input only)",
                   "v1/kv_cache_interface.py:1077-1083 MambaSpec align: 2 pages a request (running + CoW checkpoint), num_speculative_blocks 0 "
                   "under RecoverSSM", "v1/core/kv_cache_utils.py:1251-1262 + :1304 one pool block = the largest group page of the rank",
                   "dflash_seam ATT_PATCH (Exp 27 dflash_seam.py:1116-1180): DFLASH_DRAFT_SW_BLOCK must divide the attention block (512 | 2048)",
                   "tools/merlin/launch.py:334 BLOCK_SIZE default 4096 -> :474 --block-size; :475 --prefix-match-unit 64; :110-111 the "
                   "notebook's extra args come last (the last --block-size wins)"],
    "sw_512_why": "CPU planner (e34_capacity_probe.json): block 2048 with SW 1024 -> the drafter page 10,485,760 B sets the PP1 stride, pool "
                  "756 x 2048 (mc 9.93 < B's 12.25); SW 512 -> 5,242,880 B, pool 1,080 x 2048 (mc 13.32)",
    "never_do": "nb4t_lib.NEVER_DO_ARG_VALUES {'--mamba-ssm-cache-dtype': ('bfloat16',)} ('C3 SSM bf16: parked (0 capacity + B2 FAIL)', "
                "E25 NB1 R2): lifted for S16 / S16B2 / S16B2W only (E34.launch_check / check_arm_plan drop exactly 'argv "
                "--mamba-ssm-cache-dtype bfloat16')"}

MD0 = ("# E34: GDN SSM state bf16 + KV block 2048 on the DFlash2 production engine (Qwen3.8-Flash-Next klee100 3bpw-MTP, Merlin v3.0.0, 4 x L4)\n\n"
       "B = NB-D1's B (E29_DFLASH: NBDF-K100 D3s, DFlash2 K3, SW-1024 BF16 drafter KV, FLASH_ATTN, seam 341d859b + GRD, Q-defer, seqs 24, "
       "pin 7,933,526,016, DFLASH_DRAFT_EMBED_HOST=1 first try, QSA 128 / 64). S16 = B + --mamba-ssm-cache-dtype bfloat16; S16B2 = + "
       "--block-size 2048 + DFLASH_DRAFT_SW_BLOCK=512; S16B2W = S16B2 + max_num_seqs W from S16B2's measured pool. Every arm: boot "
       "read-backs (dtype / block / SW per worker, pool, F, padding, group sizes), smoke, NBDF's greedy oracle + long oracle vs B, the "
       "prefix multi-turn test, windows 8k@16 / 20k@24 / 40k@16 / 90k@8 / 116k@6 (+ max-resident W windows on B and S16B2W), NBDF's "
       "teacher-forced logprobs vs B on 8k / 40k + three recorded 69k / 94k / 113k thinking traces. Copy switches: E34_ON + E34_BUDGET_MIN "
       "at the top of CELL 1 (FULL / SMOKE). A [LIVE] line after every window and test. Readable code: e34_cell7_body.py. Commit run only; "
       "a competition rerun writes CELL 0's identity submission (E18 R2 format).\n")

DS_LOADER = r'''def _e34_ds(name, want):                      # E34: a sha-pinned file of the payload dataset (notebook < 400 kB)
    for d in (INP / "datasets/foysalemonshanto/q27-e34-payload", INP / "q27-e34-payload",
              Path(os.environ.get("E34_PAYLOAD_DIR") or "/nonexistent")):
        f = d / name
        if f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest() == want:
            return f.read_bytes().decode("utf-8")
    raise RuntimeError(f"E34_PAYLOAD missing or sha mismatch: {name} (dataset foysalemonshanto/q27-e34-payload)")
_nbd1_ds = _e34_ds                           # NB-D1's body loads nbd1_lib.py through this name (the same bytes, af0f795f)
NBDF_BODY_SRC = _e34_ds("e34_cell7_body.py", NBDF_BODY_SHA)
'''


# ---------------------------------------------------------------- 6. assemble (= NB-D1's assemble with the E34 cells)
def assemble(kv_gate, x):
    subs = {"__NB4TSHA__": G["NB4T_SHA"], "__E18C0SHA__": G["E18_CELL0_SHA"], "__FLAGS__": json.dumps(FLAGS), "__KVGATE__": json.dumps(kv_gate)}
    body = CELL7_E
    for k, v in subs.items():
        body = body.replace(k, v)
    assert '"""__' not in body and "__KVGATE" not in body and "__FLAGS" not in body and "__D1" not in body and "__E34" not in body
    compile(body, "e34_cell7_body.py", "exec")
    st = STUB_T.replace("E27 NBDF: arms Q > D4 > D3 > D7 > D4b (DFlash2 vs MTP K3), long bands (NEW code)",
                        "E34: B > S16 > S16B2 > S16B2W (SSM bf16 + block 2048 on DFlash2 production), long bands")
    st = once(st, 'NBDF_BODY_SRC = _unx("__BODYB85__")' + chr(10), DS_LOADER, "stub body from the dataset")
    for k, v in dict(G["_BLOBS"], __BODYSHA__=sha(body)).items():
        st = st.replace(k, v)
    assert "__LIB" not in st and "__SEAM" not in st and "__BODY" not in st
    c1_ = copy_block(x) + CELL1_E
    cs = ([CELL0, c1_, CELL2, "if not RERUN:\n" + textwrap.indent(CELL2B, "    ")] + list(CELL3_6) +
          ["if not RERUN:\n" + textwrap.indent(c, "    ") for c in (st, CELL8_E)])
    nb_ = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD0.splitlines(True)}] +
                    [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(True)} for c in cs],
           "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
           "nbformat": 4, "nbformat_minor": 5}
    return body, st, cs, nb_, json.dumps(nb_, separators=(",", ":")).encode("utf-8")


def static_names(cells, body):
    check_src = cells[:-1] + ["if not RERUN:\n" + textwrap.indent(body, "    "), cells[-1]]
    for c in cells + check_src:
        compile(c, "<e34 cell>", "exec")
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


GATE_CLOSED = E.closed_gate(K100_GATE)
body_c, stub_c, cells_c, nb_c, data_c = assemble(GATE_CLOSED, "full")
SHA_CLOSED = sha(data_c)
KV_GATE, REPORT_INFO = GATE_CLOSED, None
if KV_GATE_REPORT:
    _rep_b = Path(KV_GATE_REPORT).read_bytes()
    _rep = json.loads(_rep_b.decode("utf-8"))
    KV_GATE = E.gate_from_report(K100_GATE, _rep, SHA_CLOSED)
    REPORT_INFO = {"report": str(KV_GATE_REPORT), "report_sha256": sha(_rep_b), "passed": KV_GATE["passed"], "why": KV_GATE["why"]}
    KV_GATE.update(e34_report=str(KV_GATE_REPORT), e34_report_sha256=sha(_rep_b))
COPY_NB = {}
for x in E.COPIES:
    body, stub, cells_src, nb, data = assemble(KV_GATE, x)
    _undef = static_names(cells_src, body)
    assert not _undef, f"names loaded but never defined in the notebook: {_undef}"
    assert cells_src[0] == CELL0 and sha(cells_src[0].encode()) == G["E18_CELL0_SHA"], "CELL 0 != E18 R2 cell 1"
    assert cells_src[1].startswith(copy_block(x)) and cells_src[1].count("E34_ON = ") == 1 and cells_src[1].count("E34_BUDGET_MIN = ") == 1
    assert all("NBD1_ON =" not in c for c in cells_src) and "E34_ON =" not in body and body.count('globals().get("E34_ON")') == 1
    assert all(c.startswith("if not RERUN:\n") for c in cells_src[2:])
    assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
    assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", x, len(data))
    COPY_NB[x] = {"switch": E.COPIES[x], "copy_block": copy_block(x), "data": data, "sha256": sha(data), "owner": E.OWNERS[x],
                  "walk": WALKS[x], "need_min": NEEDS[x], "budget_min": BUDGETS[x], "notebook": NB_NAME[x], "cells": cells_src, "body": body}
assert COPY_NB["full"]["body"] == COPY_NB["smoke"]["body"]
assert COPY_NB["full"]["cells"][0] == COPY_NB["smoke"]["cells"][0] and COPY_NB["full"]["cells"][2:] == COPY_NB["smoke"]["cells"][2:]
assert COPY_NB["smoke"]["cells"][1] == COPY_NB["full"]["cells"][1].replace(copy_block("full"), copy_block("smoke"))
if not KV_GATE_REPORT:
    assert COPY_NB["full"]["sha256"] == SHA_CLOSED
body, cells_src = COPY_NB["full"]["body"], COPY_NB["full"]["cells"]

# ---------------------------------------------------------------- 7. write (notebooks, export, body, payload, TF dataset, push folders, facts)
OUT_NBDIR.mkdir(parents=True, exist_ok=True)
OUT_BUILD.mkdir(parents=True, exist_ok=True)
for x, c in COPY_NB.items():
    (OUT_NBDIR / c["notebook"]).write_bytes(c["data"])
EXPORT.write_bytes("\n\n".join(cells_src).encode("utf-8"))
BODY_FILE.write_bytes(body.encode("utf-8"))
for d_ in (DS_DIR, TF_DIR):
    d_.mkdir(parents=True, exist_ok=True)
    for f in d_.glob("*"):
        f.unlink()
(DS_DIR / "e34_cell7_body.py").write_bytes(body.encode("utf-8"))
(DS_DIR / "nbd1_lib.py").write_bytes(NBD1_LIB_SRC.encode("utf-8"))
(DS_DIR / "e34_lib.py").write_bytes(LIB_SRC.encode("utf-8"))
(DS_DIR / "dataset-metadata.json").write_bytes((json.dumps({"title": "q27-e34-payload", "id": DS_SLUG, "licenses": [{"name": "CC0-1.0"}]},
                                                           indent=2) + "\n").encode("utf-8"))
(DS_DIR / "SHA256SUMS.json").write_bytes((json.dumps({"e34_cell7_body.py": sha(body), "nbd1_lib.py": NBD1_LIB_SHA, "e34_lib.py": LIB_SHA},
                                                     indent=1) + "\n").encode("utf-8"))
(TF_DIR / E.TF_LONG_FILE).write_bytes(TF_BYTES)
(TF_DIR / "dataset-metadata.json").write_bytes((json.dumps({"title": "q27-e34-tf-long", "id": TF_SLUG, "licenses": [{"name": "CC0-1.0"}]},
                                                           indent=2) + "\n").encode("utf-8"))
(TF_DIR / "SHA256SUMS.json").write_bytes((json.dumps({E.TF_LONG_FILE: TF_SHA}, indent=1) + "\n").encode("utf-8"))
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

facts = {"notebook": NB_NAME["full"], "sha256": COPY_NB["full"]["sha256"], "bytes": len(COPY_NB["full"]["data"]),
         "cells": len(cells_src) + 1, "cell_bytes": [len(c.encode()) for c in cells_src], "closed_build_sha256": SHA_CLOSED,
         "gpu_ready": bool(KV_GATE.get("passed")), "kv_gate_report": REPORT_INFO, "e34_lib_sha256": LIB_SHA, "nbd1_lib_sha256": NBD1_LIB_SHA,
         "nbd1_reproduced_sha256": NBD1_CLOSED_SHA, "kv_kit_gate": KV_GATE, "e34_cfg": E34_CFG,
         "copies": {x: {k: v for k, v in c.items() if k not in ("data", "cells", "body")} for x, c in COPY_NB.items()},
         "e18_r2_cell0_sha256": G["E18_CELL0_SHA"], "export_sha256": sha(EXPORT.read_bytes()), "body_file_sha256": sha(body),
         "payload_dataset": {"slug": DS_SLUG, "dir": str(DS_DIR), "files": {"e34_cell7_body.py": sha(body), "nbd1_lib.py": NBD1_LIB_SHA,
                                                                            "e34_lib.py": LIB_SHA}, "uploaded": False},
         "tf_long_dataset": {"slug": TF_SLUG, "dir": str(TF_DIR), "file": E.TF_LONG_FILE, "sha256": TF_SHA, "bytes": len(TF_BYTES), "uploaded": False,
                             "traces": {k: {x: v[x] for x in ("task", "prompt_tokens", "p1_tokens", "len")} for k, v in TF_LONG["traces"].items()}},
         "model_sources": MODELS, "dataset_sources": DATASETS, "nbdf_lib_sha256": G["LIB_SHA"], "seam_sha256": G["SEAM_SHA"],
         "ppack_sha256": G["PP_SHA"], "image": G["IMAGE"],
         "arms": {x: {k: {f: a.get(f) for f in ("title", "port", "spec", "captures", "streams", "windows_fixed", "est_min", "cap_min",
                                                 "time_model", "d1_change", "d1_items", "e34", "e34_smoke", "env", "extra")}
                      for k, a in PLANS[x].items() if E.COPIES[x].get(k)} for x in E.COPIES},
         "walks": WALKS, "need_min": NEEDS, "budgets_min": BUDGETS, "frozen_files_unchanged": len(FROZEN), "flags": FLAGS["E34"],
         "cpu_planner": E.CPU_PLANNER, "pred_levels": E.PRED_LEVELS, "capacity_probe_sha256": sha((E34 / "Experiment Results" / "e34_capacity_probe.json").read_bytes()),
         "push_dirs": PUSH_DIRS, "test_out": str(TEST_OUT) if TEST_OUT else None}
FACTS.write_bytes((json.dumps(facts, indent=1, default=str) + "\n").encode("utf-8"))
shutil.rmtree(REF, ignore_errors=True)       # NB-D1's reproduction (incl. its push folders) is a build input only, never kept
print(json.dumps({k: v for k, v in facts.items() if k not in ("kv_kit_gate", "arms", "flags", "e34_cfg", "copies", "walks")}, indent=1, default=str))
print(json.dumps({x: {"sha256": c["sha256"], "bytes": len(c["data"]), "runtime_min": c["walk"]["runtime_min"], "need_min": c["need_min"],
                      "budget_min": c["budget_min"], "runs": c["walk"]["runs"]} for x, c in COPY_NB.items()}, indent=1))
