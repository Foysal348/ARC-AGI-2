"""E28 W4 unit tests (quick; local once and on the Kaggle CPU units kit, Py3.12; the real kit runs the T6 section on the real
installed Duck payload). Stdlib only (+ psutil when present for one deadman path).

  T1 static     size < 400 kB, no CR, the cell layout = NB6Q's; the review view (e28w4_unpack: the size-only packings undone,
                packed module literals normalized) differs from NB6Q only in hunks that carry an E28 marker and equals the
                shipped review diff exactly; the packed e21_cell7 / engine region = NB6Q's text; the W16 re-encodings decode to
                NB6Q's modules; the W12 packed literal = e28w4_lib.py + e28_sched_patch.py (Build files); placements (AST): the
                deadman after the time block, W12 at the end of CELL 1, the SCHED block between the manifest check and
                `import arc2_duck`, W10 after NB6Q's Q2 / Q3, W8 / W9 inside nb5_boot, W7 inside nb5_capacity, W6 inside
                merlin_fn_server (build env before build(), after the lever env)
  T2 CELL 0+1   executed for real (commit / rerun): E28EVAL120 (eval, 120 tasks, 412 min, HARD_STOP 410 min, window 6 h) /
                RERUN_PROD (test, 12 h - 3 min, no window); both: T1 104,000 (cap 99,904, window 130,560), E15B budgets,
                B2 120,000, server max-model-len 131,072 (NB6Q), C12 seqs / workers 12, I8 pin 0.75, DRAFT_HEAD only
                in the build env, --language-model-only + CAPS16 in the launch args, no ORDER2, scheduler on, NB6's levers;
                the deadman armed (commit, fire = HARD_STOP + 300 s) / not armed (rerun); W12 loaded both modules, no error
  T3 deadman    a real fire in a child python (psutil path + killpg path): exit 0, marker on stdout + file, the child dead
  T4 e28w4_lib  e28_i8_parse on the REAL NB4T GPU log lines (I8 -> ENGAGED: spec int8 + 2 INT8 build lines on PP1; Q / C12 ->
                NOT_ENGAGED; C12 seqs 12 + text-only read; synthetic SPEC_ONLY / CONTRADICTED); e28_readback records into the
                try / NB5_REPORT, flags knobs, never raises (no server, broken log)
  T5 engine     the pin ladder from 0.75 (NB5L.pin_tries: 3,285,145,068 / 3,016,709,612 / 2,479,838,700), captures cover
                12 x 4; the W8 / W9 / W7 statements exec'd on stubs; N1 (_nb5_peak_len: B2 131,072 = MAX_LEN, A / B1 / C 120,384,
                no plan = NB6Q); N2 + W19 (need: B2 59 blocks, A 38); the ORDER2 statement (plain order on no_order2 or
                E28_SCHED=1, ORDER2 otherwise)
  T6 scheduler  the CELL 3 SCHED block exec'd on a real extraction of the Duck payload: APPLIED (post sha, marker, env set
                from the notebook's numbers), not requested (nothing touched, env cleared), foreign arc2_duck.py -> FAILED +
                restored to the manifest sha + env unset (no raise); verify / revert / re-apply round trip
  T7 push       the GPU push folder: id / title / code_file / the notebook bytes, GPU L4, private, NB6Q's sources + image
usage: python e28w4_unit_tests.py   (exit 0 = all PASS; last line @@E28W4_UNITS_OK@@). Env overrides: E28T_NB, E28T_NB6Q,
E28T_FACTS, E28T_DIFF, E28T_LIB, E28T_SCHED, E28T_FIXTURE, E28T_PAYLOAD (the .bin), E28T_PUSH, E28T_SECTIONS (default all)."""
import ast
import base64
import contextlib
import hashlib
import io
import json
import lzma
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e28w4_unpack as U  # noqa: E402

ENV = os.environ.get
NB = Path(ENV("E28T_NB") or HERE.parent / "Q27-XHIGH-E28_W4_EVAL120_6H.ipynb")
NB6Q = Path(ENV("E28T_NB6Q") or "D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/"
            "Q27-XHIGH-E25_NB6Q_SUBMIT.ipynb")
FACTS = json.loads(Path(ENV("E28T_FACTS") or HERE / "e28w4_build_facts.json").read_text(encoding="utf-8"))
DIFF = Path(ENV("E28T_DIFF") or HERE / "e28w4_review_diff_vs_nb6q.diff")
LIB = Path(ENV("E28T_LIB") or HERE / "e28w4_lib.py")
SCHED = Path(ENV("E28T_SCHED") or HERE / "e28_sched_patch.py")
FIXTURE = Path(ENV("E28T_FIXTURE") or HERE / "e28w4_nb4t_log_fixture.json")
PAYLOAD = Path(ENV("E28T_PAYLOAD") or "D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/"
               "Experiment Notebooks/Build/payload/q27x_e15b_duck_9e4b89285136.bin")
PUSH = Path(ENV("E28T_PUSH") or "D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e28-w4-eval120-6h__foysalemonshanto")
NB6Q_SHA = "499a431904fdaf47831ef14529b7b956c45d9e374c6c427b3b3e4698fa8a5c7d"
PAYLOAD_SHA = "9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566"
DUCK_PRE = "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128"
SEC = set((ENV("E28T_SECTIONS") or "T1,T2,T3,T4,T5,T6,T7").split(","))
RESULTS, SKIPPED = [], []


def check(name, cond, detail=""):
    try:
        ok = bool(cond() if callable(cond) else cond)
    except Exception as e:
        ok, detail = False, f"raised {type(e).__name__}: {e} | {detail}"
    RESULTS.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" -> {str(detail)[:1500]}"), flush=True)
    return ok


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def cells_of(p):
    return [(c["cell_type"], "".join(c["source"])) for c in json.loads(Path(p).read_text(encoding="utf-8"))["cells"]]


def fn_src(text, name):
    n = next(x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == name)
    return ast.get_source_segment(text, n)


RAW = NB.read_bytes()
C, CQ = cells_of(NB), cells_of(NB6Q)
UP, _NPK = U.unpack_cells(C)
CODE = [t for k, t in C if k == "code"]
C1, C3, ENG, C67, RUNNER, REPORT = C[2][1], C[3][1], UP[5][1], UP[6][1], C[7][1], C[8][1]
print("E28 W4", NB.name, "sha256", sha(RAW), "|", len(RAW), "bytes", flush=True)

# ============================================================================ T1 static
if "T1" in SEC:
    check("T1 notebook < 400,000 B (Kaggle commit limit) and = the build facts (sha, bytes)", len(RAW) < 400_000
          and sha(RAW) == FACTS["sha256"] and len(RAW) == FACTS["bytes"], (len(RAW), FACTS.get("bytes")))
    check("T1 no CR byte in the notebook", b"\r" not in RAW.replace(b"\\r", b""))
    check("T1 the NB6Q reference is 499a4319 (CPU kits ALL OK)", sha(NB6Q.read_bytes()) == NB6Q_SHA)
    check("T1 the same 9-cell layout as NB6Q (kinds, cell headers)", [k for k, _ in C] == [k for k, _ in CQ]
          and [t.splitlines()[0] for k, t in C if k == "code"] == [t.splitlines()[0] for k, t in CQ if k == "code"])
    check("T1 the size-only packings: exactly one W17 (e21_cell7) and one W18 (engine donors) loader, sha-checked on unpack",
          _NPK == {"W17": 1, "W18": 1}, _NPK)
    _d = U.review_diff(CQ, UP)
    check("T1 the review diff (NB6Q -> E28 W4, unpacked + normalized) == the shipped review diff, byte for byte",
          _d == DIFF.read_text(encoding="utf-8") and sha(_d) == FACTS["review_diff_sha256"], (len(_d), sha(_d)))
    _hk = U.hunks(_d)
    check("T1 every hunk of the review diff carries an E28 marker (no unmarked change)", _hk and all(any("E28" in l for l in h) for h in _hk),
          [h[:3] for h in _hk if not any("E28" in l for l in h)][:3])
    _rm = [l[1:] for h in _hk for l in h if l.startswith("-")]
    check("T1 removed NB6Q lines = exactly the 12 replaced ones (title, profile default, E21 clamp, N1 / N2 / W19, W6 max-model-len "
          "x2 + fake env + plan record, W9 seqs, ORDER2)", len(_rm) == FACTS["diff_lines"]["removed"] and any("NB5_DEFAULT_PROFILE" in l for l in _rm)
          and any("E12_ORDER = sorted" in l for l in _rm) and any("seqs_for" in l for l in _rm), _rm)
    _c7 = fn_src(CQ[6][1], "e21_cell7")
    _uc7 = UP[6][1]
    check("T1 W17: the unpacked e21_cell7 == NB6Q's e21_cell7, byte for byte, and it is the last statement of CELL 6/7",
          _uc7.rstrip("\n").endswith(_c7) and fn_src(_uc7, "e21_cell7") == _c7)
    _a, _b = "\n# ---- RC2 CELL 1 input block, wrapped\n", "# ==================================================================== NB5 glue (new)"
    check("T1 W18: the unpacked engine region (RC2 inputs / prompts / step runner / checks / E21 donors) == NB6Q's text",
          ENG[ENG.index(_a):ENG.index(_b)] == CQ[5][1][CQ[5][1].index(_a):CQ[5][1].index(_b)])
    import re as _re
    _tok = lambda t: _re.findall(r"__PACKED_[0-9a-f]{16}__", U.normalize(t))
    check("T1 W16: the re-encoded module literals decode to NB6Q's: nb5_lib.py (CELL 1) and the e23 lab (engine) = NB6Q's packed "
          "modules, now lzma + base85 (no zlib + base64 left)", _tok(CQ[2][1])[0] in _tok(C1) and _tok(CQ[5][1]) == _tok(C[5][1])[:1]
          and "zlib" not in C1 + C[5][1] and len(_tok(CQ[2][1])) == 1 and len(_tok(CQ[5][1])) == 1,
          (_tok(CQ[2][1]), _tok(C1), _tok(CQ[5][1]), _tok(C[5][1])))
    _m = _re.search(r'E28_MOD_SHAS = (\{[^}]*\})', C1)
    _pk = _re.search(r'b85decode\(\n\s+"([^"]+)"\)\)', C1[C1.index("E28 W4 W12"):])
    PACK = json.loads(lzma.decompress(base64.b85decode(_pk.group(1))).decode("utf-8")) if _pk else {}
    _LIB_DISK = LIB.read_bytes().decode("utf-8").replace("\r\n", "\n")
    _SCH_DISK = SCHED.read_bytes().decode("utf-8").replace("\r\n", "\n")
    check("T1 W12: the packed literal = {e28w4_lib.py, e28_sched_patch.py}, verbatim = the Build files, shas in the cell match",
          sorted(PACK) == ["e28_sched_patch.py", "e28w4_lib.py"] and PACK["e28w4_lib.py"] == _LIB_DISK and PACK["e28_sched_patch.py"] == _SCH_DISK
          and json.loads(_m.group(1)) == {k: sha(v) for k, v in PACK.items()}, sorted(PACK))
    check("T1 W12 sits at the END of CELL 1 (after the profile / time / E28 constants; before CELL 3 needs the scheduler module)",
          C1.index("E28 W4 W12") > C1.index('CONFIG["e28"] = ') > C1.index("E28_B2_ATTEMPT = ") and "def e28_readback(trec, s, stage):" in C1)
    check("T1 W5: the deadman follows the time block's assert (HARD_STOP known), defined before the profile's Duck block",
          C1.index("assert NOTEBOOK_T0 < ADMIT_DEADLINE < DRAIN_DEADLINE < HARD_STOP") < C1.index("E28_DEADMAN_SEC = (HARD_STOP - NOTEBOOK_T0)")
          < C1.index('E28_DEADMAN["thread"].start()') < C1.index("E28 W4 W12"))
    check("T1 SCHED: the CELL 3 block is after the manifest assert and before `import arc2_duck as E12A`",
          C3.index('assert not _bad, ("E12 package hash mismatch", _bad)') < C3.index("E28 W4 SCHED (Build/E28_SCHED_API.md")
          < C3.index("import arc2_duck as E12A"))
    _bt = fn_src(ENG, "nb5_boot_try")
    check("T1 W10: e28_readback right after NB6Q's Q2 (after the warm-up) and Q3 (after the server stopped) in nb5_boot_try",
          'nb6q_readback(trec, s, "after_warmup")   # NB6Q Q2: Q-defer read-back after the warm-up (record only)\n        e28_readback(trec, s, "after_warmup")' in _bt
          and 'nb6q_readback(trec, srv, "final")           # NB6Q Q3: Q-defer read-back, whole log after the server stopped (record only)\n    e28_readback(trec, srv, "final")' in _bt)
    _bo = fn_src(ENG, "nb5_boot")
    check("T1 W8 / W9 inside nb5_boot: the I8 start after NB5_PIN['start_gib'], before pin_tries; seqs_fixed only on the C12 rung "
          "(kv bytes = seqs_fixed_kv_bytes), the rung set before the launch extra, recorded in the try",
          _bo.index('start = NB5_PIN["start_gib"]') < _bo.index('P.get("pin_start_gib")') < _bo.index("tries = NB5L.pin_tries(")
          and 'int(P["seqs_fixed"]) if E28_RUNG["c12"]' in _bo and 'int(P.get("seqs_fixed_kv_bytes") or -1)' in _bo
          and _bo.index('globals()["E28_RUNG"] = {') < _bo.index('NB5_BOOT_EXTRA = base_extra') < _bo.index('trec["e28_rung"] = E28_RUNG'))
    _ca = fn_src(ENG, "nb5_capacity")
    check("T1 W7 inside nb5_capacity: workers_fixed overrides plan['workers'] after the admission-mode blocks (both modes), only on the C12 rung and "
          "when the plan is not NO_GO (C-8)", _ca.index('if NB5_ADMIT_MODE == "slots":') < _ca.index('if _e28c12 and plan["workers"] >= 1:')
          < _ca.index('plan["workers"] = int(P["workers_fixed"])') < _ca.index('if caps.get("max_num_seqs") is None:'))
    _ms = fn_src(C[5][1], "merlin_fn_server")
    check("T1 W6 inside merlin_fn_server: --max-model-len = NB6Q's MAX_LEN (real + fake, 131,072), E28_BUILD_ENV into build() after the "
          "lever env and before build(), NOT in the server env update after build(); recorded in the plan",
          _ms.count('"--max-model-len", str(MAX_LEN)') == 2 and "E28_SERVER_MAX_LEN" not in _ms
          and _ms.index("base.update(NB5_BOOT_LEVERS)") < _ms.index("base.update(E28_BUILD_ENV)") < _ms.index('plan = eng["ML"].build(')
          and "env.update(E28_BUILD_ENV)" not in _ms and '"build_env": dict(E28_BUILD_ENV)' in _ms)
    check("T1 the scheduler env never reaches a server launch: no E28_* name in merlin_fn_server (NB4_PPDEFER stays NB6Q's only)",
          "E28_SCHED" not in _ms and "E28_T1" not in _ms and "E28_B2" not in _ms)

# ============================================================================ T2 CELL 0 + CELL 1 executed for real
_CLR = ("E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN", "Q27X_PROFILE", "NB5_ATTEMPT_B0", "NB5_ATTEMPT_B1", "E12_MAX_GEN", "Q27X_TIME_BUDGET",
        "NB5_EXTRA_ARGS_JSON", "NB5_BASE_ENV_JSON", "E28_B2_ATTEMPT", "E12_B_EXTRA_ATTEMPTS", "E12_B_SKIP_B1_IF_ACCEPTED",
        "E12_B_B1_ACC_MODE", "LOCAL_ANALYZER_SEED", "E28_SCHED", "E28_T1", "E28_B2_TOKENS", "E28_B2_WINDOW")
ARC = Path(tempfile.mkdtemp(prefix="e28w4_arc_"))
json.dump({"aaaa0001": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[0, 1], [1, 0]]}]}},
          open(ARC / "arc-agi_test_challenges.json", "w"))


def run_cells(env, n=2):
    work = Path(tempfile.mkdtemp(prefix="e28w4_w_"))
    old = dict(os.environ)
    for k in _CLR:
        os.environ.pop(k, None)
    os.environ.update(Q27X_ARC_ROOT=str(ARC), Q27X_WORKING=str(work), E23_TMP=str(work), E23_INPUT=str(work / "input"), **env)
    G = {"__name__": "__main__"}
    try:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            for c in CODE[:n]:
                exec(compile(c, "<cell>", "exec"), G)
        G["_ENV"], G["_OUT"] = dict(os.environ), out.getvalue()
    finally:
        os.environ.clear()
        os.environ.update(old)
        sys.modules.pop("nb5_lib", None)
    G["_WORK"] = work
    return G


if "T2" in SEC or "T5" in SEC or "T6" in SEC:
    GC = run_cells({})
    GR = run_cells({"KAGGLE_IS_COMPETITION_RERUN": "1"})
if "T2" in SEC:
    P, PR = GC["P"], GR["P"]
    check("T2 commit: PROFILE == E28EVAL120 (baked default), RERUN False; rerun: RERUN_PROD, RERUN True",
          GC["PROFILE"] == "E28EVAL120" and GC["RERUN"] is False and GR["PROFILE"] == "RERUN_PROD" and GR["RERUN"] is True)
    check("T2 E28EVAL120: source eval, n_tasks 120 (no task_list), 412-min budget, hard margin 2 min, fixed 6-h window, one EVAL phase",
          P["source"] == "eval" and P["n_tasks"] == 120 and "task_list" not in P and P["time_budget_sec"] == 412 * 60
          and P["hard_margin_sec"] == 120 and P["equal_windows"] is True and P["fixed_window_sec"] == 21600
          and P["nb5_phases"] == [["EVAL", "all", None, "on", "A", "fp8"]])
    check("T2 E28EVAL120 timing: HARD_STOP = T0 + 410 min (6 h 50 min); NB5's fracs / margins (_NB5_T)",
          abs(GC["HARD_STOP"] - GC["NOTEBOOK_T0"] - 410 * 60) < 1e-6 and all(P[k] == GC["_NB5_T"][k] for k in GC["_NB5_T"] if k not in
                                                                             ("time_budget_sec", "hard_margin_sec")))
    check("T2 RERUN_PROD: NB6's 12 h - 3 min budget, source test, all tasks, no fixed window, one EVAL phase",
          PR["time_budget_sec"] == 12.0 * 3600 - 3 * 60 and PR["source"] == "test" and PR["n_tasks"] is None
          and not PR.get("fixed_window_sec") and not PR.get("equal_windows") and PR["nb5_phases"] == [["EVAL", "all", None, "on", "A", "fp8"]])
    _lv = GC["E28_LEVERS"]
    check("T2 E28_LEVERS = NB6_LEVERS + exactly the E28 keys (T1 104k, C12, I8, text-only, B2 120k / 9,000 s, no ORDER2, scheduler)",
          {k: v for k, v in _lv.items() if k not in ("nb5_attempt", "seqs_rule", "admit_mode", "nb5_env")} == dict(
              {k: v for k, v in GC["NB6_LEVERS"].items() if k not in ("nb5_attempt", "seqs_rule", "admit_mode", "nb5_env")}, seqs_fixed=12, workers_fixed=12,
              seqs_fixed_kv_bytes=3285145068,
              pin_start_gib=0.75, build_env={"DRAFT_HEAD": "int8"}, e28_extra_args=["--language-model-only"],
              b2_attempt=120000, b2_request_timeout=9000, no_order2=True, e28_sched=True)
          and _lv["nb5_attempt"] == {"b0": 104000, "b1": 104000} and _lv["seqs_rule"] == GC["NB6_LEVERS"]["seqs_rule"] == "table" and _lv["admit_mode"] == "slots"
          and GC["NB6_LEVERS"]["admit_mode"] == "slots" and _lv["nb5_env"] == dict(GC["NB6_LEVERS"]["nb5_env"], E12_DUCK_REQUEST_TIMEOUT="7200"),
          sorted(_lv))
    check("T2 both profiles carry E28_LEVERS (the hidden rerun = the same engine + scheduler as the commit)",
          all(P[k] == v and PR[k] == v for k, v in _lv.items()))
    for lab, G in (("commit", GC), ("rerun", GR)):
        check(f"T2 {lab}: T1 104,000 (b0 = b1), cap 99,904, Duck window 130,560, E15B 104,000 / 104,000, the asserts held",
              G["NB5_ATTEMPT"] == {"b0": 104000, "b1": 104000} and G["NB5_CAP"] == 99904 and G["NB5_WINDOW"] == 130560
              and G["E15B"]["E12_B_ATTEMPT_TOKENS"] == G["E15B"]["E12_B_ATTEMPT_TOKENS_B1"] == 104000 and G["E15B"]["E12_B_RESERVE"] == 4096
              and G["E15B"]["E12_B_WINDOW"] == 130560 and G["E12_MAX_GEN_PER_BRANCH"] == 104000)
        check(f"T2 {lab}: B2 120,000; the server stays NB6Q's (no E28 max-model-len, E21's clamp 131,072)",
              G["E28_B2_ATTEMPT"] == 120000 and "E28_SERVER_MAX_LEN" not in G and G["E21_VLLM_MAX_MODEL_LEN"] == 131072)
        _ea = G["NB5_EXTRA_ARGS"]
        check(f"T2 {lab}: launch args = CAPS16 compilation config (covers 12 x 4 = 48) + --language-model-only once; no other flag",
              _ea == ["--compilation-config", json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": G["NB5_CAPS16"]},
                                                         separators=(",", ":")), "--language-model-only"]
              and max(G["NB5_CAPS16"]) >= 48, _ea)
        check(f"T2 {lab}: DRAFT_HEAD=int8 only in E28_BUILD_ENV (build() input), NB5_BASE_ENV empty (no server-env knob); lever env "
              "MERLIN_SHARE_DRAFT_WORKSPACE=1", G["E28_BUILD_ENV"] == {"DRAFT_HEAD": "int8"} and G["NB5_BASE_ENV"] == {}
              and G["NB5_LEVER_ENV"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"})
        check(f"T2 {lab}: CONFIG['e28'] records the E28 numbers; the E12 / E15B attempt policy = NB6's (no stage 3, no fixed seed)",
              G["CONFIG"]["e28"]["b2_request_timeout"] == 9000 and G["CONFIG"]["e28"]["seqs_fixed"] == 12
              and G["_ENV"].get("E12_B_EXTRA_ATTEMPTS") == "0" and "LOCAL_ANALYZER_SEED" not in G["_ENV"])
        check(f"T2 {lab}: W12 loaded both modules (shas checked), e28_readback / e28_summary from e28w4_lib, no load error",
              sorted(G["E28_SRC"]) == ["e28_sched_patch.py", "e28w4_lib.py"] and "E28_LOAD_ERROR" not in G
              and G["e28_readback"].__doc__.startswith("Record only") and G["E28_LIB_VERSION"].startswith("e28w4_lib 1.0"))
    for lab, G in (("commit", GC), ("rerun", GR)):
        check(f"T2 {lab}: admission = slots (NB5_ADMIT_MODE, NB6Q / K100Q), the request timeout 7,200 s (E12_DUCK_REQUEST_TIMEOUT, also in the "
              "Duck env; the scheduler scales B2 to 12,000 s)", G["NB5_ADMIT_MODE"] == "slots" and G["E12_DUCK_REQUEST_TIMEOUT"] == 7200
              and G["_ENV"].get("E12_DUCK_REQUEST_TIMEOUT") == "7200")
    _dm = GC["E28_DEADMAN"]
    _dmt = _dm.get("thread")
    check("T2 deadman (commit): armed, a daemon thread alive, fire = HARD_STOP + 300 s (T0 + 415 min), marker in Q27X_WORKING",
          _dm["armed"] is True and _dmt is not None and _dmt.daemon and _dmt.is_alive()
          and abs(_dm["fire_pc"] - (GC["HARD_STOP"] + 300)) < 1e-6 and abs(_dm["fire_pc"] - GC["NOTEBOOK_T0"] - 415 * 60) < 1e-6
          and _dm["marker"] == str(GC["WORKING"] / "e28w4_deadman.txt"), {k: v for k, v in _dm.items() if k != "thread"})
    check("T2 deadman (rerun): NOT armed (RERUN_PROD keeps NB6's behaviour, = K100Q)", GR["E28_DEADMAN"]["armed"] is False
          and "thread" not in GR["E28_DEADMAN"])
    check("T2 deadman printed its arming line; the order T0 < ADMIT < DRAIN < HARD_STOP < VERIFY < T0 + budget < deadman holds",
          "E28 W4 DEADMAN armed: fires at +415 min" in GC["_OUT"] and GC["NOTEBOOK_T0"] < GC["ADMIT_DEADLINE"] < GC["DRAIN_DEADLINE"]
          < GC["HARD_STOP"] < GC["VERIFY_DEADLINE"] < GC["NOTEBOOK_T0"] + GC["TOTAL_BUDGET_SEC"] < _dm["fire_pc"])
    GH = run_cells({"E11_HARNESS": "1", "E28_B2_ATTEMPT": "9000", "NB5_ATTEMPT_B0": "6000", "NB5_ATTEMPT_B1": "4000",
                    "Q27X_TIME_BUDGET": "1800"})
    check("T2 harness only: E28_B2_ATTEMPT / NB5_ATTEMPT / the budget follow the env (the kit e2e), the deadman = HARD_STOP + 300",
          GH["E28_B2_ATTEMPT"] == 9000 and GH["NB5_ATTEMPT"] == {"b0": 6000, "b1": 4000} and GH["TOTAL_BUDGET_SEC"] == 1800
          and abs(GH["E28_DEADMAN"]["fire_pc"] - GH["HARD_STOP"] - 300) < 1e-6)

# ============================================================================ T3 deadman fire (subprocess)
if "T3" in SEC:
    _DM_SRC = fn_src(C1, "e28_deadman")

    def _dm_fire(use_psutil):
        d = Path(tempfile.mkdtemp(prefix="e28w4_dm_"))
        scr = d / "fire.py"
        scr.write_text("\n".join([
            "import os, sys, time, threading, subprocess",
            "from pathlib import Path",
            "NOTEBOOK_T0 = time.perf_counter()",
            "E28_DEADMAN = {'pgids': []}",
            _DM_SRC,
            "kw = {'start_new_session': True} if os.name == 'posix' else {}",
            "ch = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], **kw)",
            "E28_DEADMAN['pgids'].append(ch.pid)",
            f"Path(r'{d / 'child.pid'}').write_text(str(ch.pid))",
            f"threading.Thread(target=e28_deadman, args=(time.perf_counter() + 1.5, r'{d / 'e28w4_deadman.txt'}', 0.2, {use_psutil}), daemon=True).start()",
            "time.sleep(60)",
            "print('NOT_FIRED', flush=True)",
            "sys.exit(3)"]) + "\n", encoding="utf-8")
        t0 = time.time()
        cp = subprocess.run([sys.executable, str(scr)], capture_output=True, text=True, timeout=50)
        sec = time.time() - t0
        pid = int((d / "child.pid").read_text())
        time.sleep(1.0)
        try:
            import psutil as _psu
            alive = _psu.pid_exists(pid) and _psu.Process(pid).status() != _psu.STATUS_ZOMBIE
        except Exception:
            alive = None
        out = {"rc": cp.returncode, "sec": round(sec, 1), "stdout": cp.stdout[-200:],
               "marker_file": (d / "e28w4_deadman.txt").read_text() if (d / "e28w4_deadman.txt").exists() else None, "child_alive": alive}
        shutil.rmtree(d, ignore_errors=True)
        return out

    try:
        import psutil  # noqa: F401
        _HAS_PSUTIL = True
    except ImportError:
        _HAS_PSUTIL = False
    if _HAS_PSUTIL:
        _f1 = _dm_fire(True)
        check("T3 deadman fire (psutil): exit 0 within seconds, @@E28W4_DEADMAN_FIRED@@ on stdout + in the marker file, the child killed",
              _f1["rc"] == 0 and _f1["sec"] < 20 and "@@E28W4_DEADMAN_FIRED@@" in _f1["stdout"] and "NOT_FIRED" not in _f1["stdout"]
              and "@@E28W4_DEADMAN_FIRED@@" in (_f1["marker_file"] or "") and _f1["child_alive"] is False, _f1)
    else:
        SKIPPED.append("T3 deadman psutil path: psutil not installed here (the Kaggle image has it)")
    if hasattr(os, "killpg"):
        _f2 = _dm_fire(False)
        check("T3 deadman fire (no psutil): os.killpg on the recorded group (+ pkill -9 -f vllm), exit 0, marker, the child killed",
              _f2["rc"] == 0 and "@@E28W4_DEADMAN_FIRED@@" in (_f2["marker_file"] or "") and _f2["child_alive"] in (False, None), _f2)
    else:
        SKIPPED.append("T3 deadman killpg path: no os.killpg on this OS (the Linux CPU kit runs it)")
    check("T3 the deadman body = E26 K100Q's (names only): the K100Q code sha recorded at build, the same statements",
          FACTS["k100q_deadman_code_sha256"] and "os._exit(0)" in _DM_SRC and 'subprocess.run(["pkill", "-9", "-f", "vllm"], timeout=10)' in _DM_SRC)

# ============================================================================ T4 e28w4_lib
if "T4" in SEC:
    L = {"__name__": "e28w4_lib_t"}
    exec(compile(LIB.read_text(encoding="utf-8"), "e28w4_lib.py", "exec"), L)
    FX = json.loads(FIXTURE.read_text(encoding="utf-8"))["logs"]
    i8 = L["e28_i8_parse"]("\n".join(FX["I8"]["lines"]))
    check("T4 NB4T I8 log (GPU, 3,285,145,068 B): ENGAGED = spec draft_lm_head_quant int8 + 2 INT8 build lines on the PP1 workers "
          "(124160x2560, 307.9 MiB), K3 probabilistic, pin read", i8["label"] == "ENGAGED" and i8["spec_draft_head"] == "int8"
          and i8["draft_head_line_count"] == 2 and {h["worker"] for h in i8["draft_head_lines"]} == {"PP1_TP0", "PP1_TP1"}
          and all(h["n"] == 124160 and h["k"] == 2560 and h["mib"] == 307.9 for h in i8["draft_head_lines"])
          and i8["spec_k"] == 3 and i8["spec_sample"] == "probabilistic" and i8["kv_cache_memory_bytes"] == 3285145068
          and i8["language_model_only"] is True, i8)
    q = L["e28_i8_parse"]("\n".join(FX["Q"]["lines"]))
    c12 = L["e28_i8_parse"]("\n".join(FX["C12"]["lines"]))
    check("T4 NB4T Q / C12 logs: NOT_ENGAGED (no draft head argument, no build line); C12 max_num_seqs 12, text-only, 131072",
          q["label"] == "NOT_ENGAGED" and c12["label"] == "NOT_ENGAGED" and c12["max_num_seqs"] == 12 and c12["language_model_only"] is True
          and c12["max_model_len"] == 131072 and q["max_num_seqs"] == 8, (q["label"], c12))
    _spec_only = FX["I8"]["lines"][0] if "non-default args" in FX["I8"]["lines"][0] else next(l for l in FX["I8"]["lines"] if "non-default" in l)
    check("T4 SPEC_ONLY (argument, no build line yet) / CONTRADICTED (an INT4 build line, or another spec value)",
          L["e28_i8_parse"](_spec_only)["label"] == "SPEC_ONLY"
          and L["e28_i8_parse"](_spec_only + "\n(Worker_PP1_TP0 pid=1) INFO Draft lm_head: INT4 weight-only copy of the 1x2 shard (1.0 MiB")["label"] == "CONTRADICTED"
          and L["e28_i8_parse"](_spec_only.replace("'draft_lm_head_quant': 'int8'", "'draft_lm_head_quant': 'int4'"))["label"] == "CONTRADICTED")

    class _S:
        name = "merlin_fp8_mtp_pin75"

        def __init__(self, t):
            self.t = t

        def log(self):
            if self.t is None:
                raise OSError("no log")
            return self.t
    _txt = "\n".join(FX["I8"]["lines"]).replace("'max_num_seqs': 8", "'max_num_seqs': 12")
    L.update(P={"seqs_fixed": 12, "build_env": {"DRAFT_HEAD": "int8"}, "e28_extra_args": ["--language-model-only"]},
             NB5_REPORT={"phases": {}}, NOTEBOOK_T0=time.perf_counter(), E28_BUILD_ENV={"DRAFT_HEAD": "int8"},
             NB5_CAPACITY={"pool_blocks": 197, "f_eff": 5, "f_log": 5.0, "workers": 12, "caps": {"kv_tokens": 679504}})
    tr = {"kv_bytes": 3285145068, "pin_gib": 0.75}
    with contextlib.redirect_stdout(io.StringIO()) as _o:
        r = L["e28_readback"](tr, _S(_txt), "after_warmup")
    check("T4 e28_readback: ENGAGED + knobs ok (seqs 12, 131072, text-only), pool / F / workers recorded in the try and NB5_REPORT['e28'], "
          "one log line", r["i8_label"] == "ENGAGED" and all(r["knobs_ok"].values()) and tr["e28_readback"]["after_warmup"] is r
          and L["NB5_REPORT"]["e28"]["capacity"][0]["pool_blocks"] == 197 and L["NB5_REPORT"]["e28"]["capacity"][0]["kv_bytes"] == 3285145068
          and "E28 W4 read-back [after_warmup]" in _o.getvalue() and "!!!!" not in _o.getvalue(), r)
    with contextlib.redirect_stdout(io.StringIO()) as _o:
        r2 = L["e28_readback"]({}, _S("\n".join(FX["C12"]["lines"])), "final")
        r3 = L["e28_readback"]({}, _S(None), "final")
        r4 = L["e28_readback"](None, None, "final")
    check("T4 e28_readback: a BF16-head (NB4T C12) log under DRAFT_HEAD=int8 -> NOT_ENGAGED (loud line), 131072 / seqs 12 knobs ok; no log / no server "
          "/ no trec never raise", r2["i8_label"] == "NOT_ENGAGED" and r2["knobs_ok"]["max_model_len"] is True and "!!!!" in _o.getvalue()
          and r3 is not None and r3["i8_label"] == "NOT_ENGAGED" and r4 is not None)
    s_ = L["e28_summary"]()
    check("T4 e28_summary: the last read-back + capacity + the scheduler state (E28_SCHED_REC)", s_["i8_label_last"] in ("NOT_ENGAGED",)
          and s_["pool_blocks"] == 197 and s_["readbacks"] == 4 and set(s_["sched"]) >= {"state", "env"}, s_)

if "T4" in SEC:
    L["E12_RUN"] = {"rev6_scheduler": {"e28": {"config": {"t1": 120000}, "started": {"A": 2, "B": 1, "C": 2}, "done": {"A": 2, "B": 1, "C": 2},
                                                "a_class": {"t1": {"kind": "TRAIN_PASS"}, "t2": {"kind": "B2"}},
                                                "attempts": [{"task": "t1", "branch": 0, "phase": "A", "tokens": 30000, "kind": "TRAIN_PASS", "end_reason": "accepted"},
                                                             {"task": "t2", "branch": 0, "phase": "A", "tokens": 120000, "kind": "B2", "end_reason": "budget"},
                                                             {"task": "t2", "branch": 2, "phase": "B2", "tokens": 150000, "kind": "TRAIN_PASS", "end_reason": "accepted"},
                                                             {"task": "t1", "branch": 1, "phase": "C", "tokens": 20000}, {"task": "t2", "branch": 1, "phase": "C", "tokens": 9000}],
                                                "slots": {"t1": [{"attempt_1": "A", "attempt_2": "C"}], "t2": [{"attempt_1": "B", "attempt_2": "A"}]}}}}
    _td = Path(tempfile.mkdtemp(prefix="e28w4_sr_"))
    (_td / "s.jsonl").write_text("\n".join(json.dumps(x) for x in ({"t": 100.0, "preempt": 0}, {"t": 115.0, "preempt": 2},
                                                                    {"t": 130.0, "preempt": 2})) + "\n", encoding="utf-8")
    L["E12_RECORDS"] = {"t1": [{"branch": 0, "t0": 90.0, "t1": 105.0}], "t2": [{"branch": 0, "t0": 120.0, "t1": 140.0}]}
    L["NB5_REPORT"]["phases"] = {"EVAL": {"counters": {"vllm:num_preemptions_total": 2}}}
    sr = L["e28_sched_report"](_td / "e28_sched_report.json", _td / "s.jsonl")
    check("T4 e28_sched_report: per-phase counts / tokens, the A classes, the B2 rows, the slot sources; the full block written to JSON",
          sr["state"] == "OK" and sr["phases"]["A"]["attempts"] == 2 and sr["phases"]["B2"]["tokens"] == 150000 and len(sr["b2_rows"]) == 1
          and sr["a_class_counts"] == {"TRAIN_PASS": 1, "B2": 1} and sr["slot_sources"]["attempt_1"] == {"A": 1, "B": 1}
          and json.loads((_td / "e28_sched_report.json").read_text(encoding="utf-8"))["e28"]["slots"]["t2"][0]["attempt_1"] == "B", sr)
    check("T4 preemptions (= e28k100_lib): run level from the sampler (2 in 1 interval) + the server counter; per attempt: t1 b0 ran "
          "[90, 105] over the [100, 115] interval -> 2, t2 b0 [120, 140] -> 0", sr["preemptions"]["sampler_total"] == 2
          and sr["preemptions"]["intervals"] == 1 and sr["preemptions"]["server_counter_phase"] == 2
          and sr["preemptions"]["attempts_with_preemption"] == 1, sr["preemptions"])
    L["E12_RUN"] = {}
    check("T4 e28_sched_report without a scheduler record -> NO_E28_RECORD (never raises)", L["e28_sched_report"]()["state"] == "NO_E28_RECORD")

# ============================================================================ T5 engine statements on stubs
if "T5" in SEC:
    NB5L = GC["NB5L"]
    tr_ = NB5L.pin_tries(2479838700, 0.75)
    check("T5 pin ladder from the I8 pin: 0.75 / 0.5 / 0 GiB = 3,285,145,068 / 3,016,709,612 / 2,479,838,700 B (E18's <= 3 attempts)",
          [(t["gib"], t["kv_bytes"]) for t in tr_] == [(0.75, 3285145068), (0.5, 3016709612), (0.0, 2479838700)], tr_)
    check("T5 CAPS16 covers C12: 12 x (1 + K3) = 48 <= 64", NB5L.captures_cover(GC["NB5_CAPS16"], 12) and not NB5L.captures_cover([1, 2, 32], 12))
    _bo = fn_src(ENG, "nb5_boot")
    _w8 = _bo[_bo.index('        if start is None and P.get("pin_start_gib")'):_bo.index("        tries = NB5L.pin_tries(")]
    _w8 = _w8[:_w8.index("        if start is None and P.get(\"patch_60490\")")] if 'if start is None and P.get("patch_60490")' in _w8 else _w8
    import textwrap as _tw

    def _w8_run(p60490, fake, prof=None):
        ns = {"P": dict(GC["P"]) if prof is None else prof, "NB5_PIN": {"start_gib": None, "patch_60490": p60490}, "FAKE": fake, "start": None}
        exec(_tw.dedent(_w8), ns)
        return ns["start"]
    check("T5 W8 (exec'd): start 0.75 when #60490 is verified or FAKE; 0.0 (the RC2 pin) when #60490 failed; None for an NB6 profile",
          _w8_run({"ok": True}, None) == 0.75 and _w8_run({"ok": None, "fake": True}, "fake") == 0.75
          and _w8_run({"ok": False}, None) == 0.0 and _w8_run({"ok": True}, None, dict(GC["PROFILE_TABLE"]["NB6COMMIT"])) is None)
    _w9 = _tw.dedent(_bo[_bo.index('        globals()["E28_RUNG"] = {'):_bo.index("        NB5_BOOT_EXTRA = base_extra")])

    def _s9(prof, gib, kv="fp8"):
        ns = {"P": prof, "NB5L": NB5L, "NB5_BOOT_KV": kv, "E28_RUNG": {"stale": 1},
              "tr": {"gib": gib, "kv_bytes": GC["NB5_KV_BYTES"][kv] + int(round(gib * 2 ** 30))}, "NB5_SEQS_RULE": prof.get("seqs_rule")}
        exec(_w9, ns)
        return ns["seqs"], ns["E28_RUNG"]
    _e75, _e50, _e00 = _s9(GC["P"], 0.75), _s9(GC["P"], 0.5), _s9(GC["P"], 0.0)
    check("T5 W9 (exec'd): the C12 rung = the I8 start pin 3,285,145,068 B -> seqs 12 (commit + rerun); the lower rungs -> NB6Q's "
          "table (0.5 -> 8, 0 = the RC2 pin -> 6); a BF16-KV rung -> its table (0.75 -> 4); NB6COMMIT -> its table (0.5 -> 8); "
          "E28_RUNG reset per try and recorded",
          _e75[0] == 12 and _e75[1] == {"kv_bytes": 3285145068, "gib": 0.75, "c12": True, "seqs": 12} and _s9(GR["P"], 0.75)[0] == 12
          and _e50[0] == 8 and _e50[1]["c12"] is False and _e00[0] == 6 and _e00[1]["c12"] is False and _s9(GR["P"], 0.0)[0] == 6
          and _s9(GC["P"], 0.75, "bf16")[0] == 4 and _s9(GC["PROFILE_TABLE"]["NB6COMMIT"], 0.5)[0] == 8, (_e75, _e50, _e00))
    _w7 = _tw.dedent(_ca[_ca.index('    if P.get("workers_fixed"):'):_ca.index('    if caps.get("max_num_seqs") is None:')])

    def _s7(rung, workers):
        ns = {"P": dict(GC["P"]), "E28_RUNG": dict(rung), "plan": {"workers": workers}}
        exec(_w7, ns)
        return ns["plan"], ns["E28_RUNG"]
    _p75, _r75 = _s7(_e75[1], 6)
    _p50, _r50 = _s7(_e50[1], 8)
    _p00, _ = _s7(_e00[1], 6)
    _png, _ = _s7(_e75[1], 0)
    check("T5 W7 (exec'd): the C12 rung -> workers 12 (rule 6 recorded); the lower rungs keep the slots rule value (0.5: 8, the RC2 "
          "pin: 6), workers_fixed None; C-8 (0 workers) stays 0 on the C12 rung; the rung's workers in E28_RUNG + plan['e28_workers']",
          _p75["workers"] == 12 and _p75["e28_workers"] == {"rung_kv_bytes": 3285145068, "rung_gib": 0.75, "c12_rung": True, "launch_seqs": 12,
                                                             "rule_workers": 6, "workers_fixed": 12, "workers": 12} and _r75["workers"] == 12
          and _p50["workers"] == 8 and _p50["e28_workers"]["workers_fixed"] is None and _p50["e28_workers"]["c12_rung"] is False
          and _r50["workers"] == 8 and _p00["workers"] == 6 and _p00["e28_workers"]["rule_workers"] == 6 and _png["workers"] == 0,
          (_p75, _p50, _p00, _png))
    _pk = fn_src(C3, "_nb5_peak_len")

    class _A:
        def __init__(self, plan=None, branch=0):
            self._e28_plan, self.arc2_branch = plan, branch
    ns5 = {"NB5_PROMPT_ALLOW": 8192, "NB5_TOOL_ALLOW": 8192, "MAX_LEN": 131072,
           "NB5_BLOCKS": {"peak_b0": 131072, "peak_b1": 131072}}
    exec(_pk, ns5)
    _f = ns5["_nb5_peak_len"]
    check("T5 N1 _nb5_peak_len: B2 (120k) -> 131,072 (8,192 + 120,000 + 8,192 capped at MAX_LEN); A / B1 / C (104k) -> 120,384; "
          "no E28 plan -> NB6Q's peak_b0 / peak_b1", _f(_A({"phase": "B2", "budget": 120000})) == 131072
          and _f(_A({"phase": "A", "budget": 104000})) == 120384 and _f(_A({"phase": "C", "budget": 104000}, 1)) == 120384
          and _f(_A({"phase": "B1", "budget": 104000})) == 120384 and _f(_A(None)) == 131072)
    _sr = fn_src(C3, "_nb5_session_run")
    _need = "\n".join(ast.get_source_segment(_sr, n) for n in ast.parse(_sr).body[0].body
                      if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in ("_e28p", "need"))

    class _Sess:
        def __init__(self, plan=None, branch=0):
            self.e28_plan, self.branch = plan, branch

    def _need_of(sess, mode="slots"):
        ns = {"self": sess, "NB5L": NB5L, "MAX_LEN": 131072, "NB5_PROMPT_ALLOW": 8192, "NB5_TOOL_ALLOW": 8192,
              "NB5_ADMIT_MODE": mode, "NB5_BLOCKS": {"block": 4096, "f": 5, "b0": 37, "b1": 37, "start": 7}}
        exec(_need, ns)
        return ns["need"]
    check("T5 N2 reservation need: B2 = cdiv(131,072, 4096) + F 5 = 37; A = cdiv(120,384, 4096) + 5 = 35; no plan = NB6Q's b0",
          _need_of(_Sess({"phase": "B2", "budget": 120000})) == 37 and _need_of(_Sess({"phase": "A", "budget": 104000})) == 35
          and _need_of(_Sess(None)) == 37 and _need_of(_Sess(None), "live") == 7)
    if "def _e28_try_reserve(key, plan):" in C3:
        _ns4 = {"NB5L": NB5L, "MAX_LEN": 131072, "NB5_PROMPT_ALLOW": 8192, "NB5_TOOL_ALLOW": 8192,
                "NB5_ADMIT_MODE": "projected", "NB5_BLOCKS": {"block": 4096, "f": 5, "b0": 37, "b1": 37, "start": 7}}
        _ns4["NB5_RES"] = NB5L.Reservation(100)
        exec(fn_src(C3, "_e28_try_reserve"), _ns4)
        exec(fn_src(C3, "_e28_release"), _ns4)
        _tr4, _R4 = _ns4["_e28_try_reserve"], _ns4["NB5_RES"]
        _g1 = _tr4(("t1", 2), {"phase": "B2", "budget": 120000})
        _u1 = _R4.used
        _g2 = _tr4(("t2", 0), {"phase": "A", "budget": 104000})
        _g3 = _tr4(("t3", 0), {"phase": "A", "budget": 104000})
        _ns4["_e28_release"](("t1", 2))
        check("T5 N4 (scheduler fit check, non-blocking): B2 reserves 37 blocks, A 35; a full pool (100) refuses at once (no wait); "
              "release frees", _g1 is True and _u1 == 37 and _g2 is True and _g3 is False and _R4.used == 35,
              (_g1, _u1, _g2, _g3, _R4.used))
    else:
        SKIPPED.append("T5 N4: the scheduler module offers no N4 edit")
    # ---- the I8 boot gate (W21) + the same-pin BF16 retry (W22 / W20), lead 2026-11-01 (K100 audits)
    _FX = json.loads(FIXTURE.read_text(encoding="utf-8"))["logs"]
    _i8l = "\n".join(_FX["I8"]["lines"])
    _spec_only = next(l for l in _FX["I8"]["lines"] if "non-default" in l)
    _gate = GC["e28_i8_gate"]
    check("T5 I8 gate: the real NB4T I8 log -> ENGAGED (continue); spec-only, a contradiction (INT4 line / int4 spec) and the BF16 "
          "evidence (NB4T Q log) -> I8_READBACK_UNVERIFIED", _gate(_i8l)[0] is True and _gate(_i8l)[1]["verdict"] == "ENGAGED"
          and all(_gate(t)[0] is False and _gate(t)[1]["verdict"] == "I8_READBACK_UNVERIFIED" for t in (
              _spec_only, _spec_only.replace("'draft_lm_head_quant': 'int8'", "'draft_lm_head_quant': 'int4'"),
              _i8l.replace("INT8 weight-only", "INT4 weight-only"), "\n".join(_FX["Q"]["lines"]))))
    _bt = fn_src(ENG, "nb5_boot_try")
    check("T5 W21 placement: the gate runs after the C-11 config read-back and before the probes; it sets I8_READBACK_UNVERIFIED "
          "and returns (with_servers stops the server)", _bt.index('trec["config_readback"] = NB5L.config_readback(') < _bt.index("E28 W4 W21")
          < _bt.index("ok, pr = nb5_probes(s)") and 'trec["status"] = "I8_READBACK_UNVERIFIED"' in _bt)
    _boot_src = fn_src(ENG, "nb5_boot")

    def _boot_run(script):
        calls = []
        ns = {"P": dict(GC["P"]), "NB5_PIN": {"start_gib": None, "levers": None, "patch_60490": {"ok": True}}, "FAKE": None, "NB5L": NB5L,
              "NB5_KV_BYTES": GC["NB5_KV_BYTES"], "NB5_KV_DTYPE": "fp8", "NB5_EXTRA_ARGS": list(GC["NB5_EXTRA_ARGS"]),
              "NB5_MTP_OFF_ARGS": GC["NB5_MTP_OFF_ARGS"], "NB5_LEVER_ENV": dict(GC["NB5_LEVER_ENV"]), "NB5_SEQS_RULE": GC["NB5_SEQS_RULE"],
              "NB5_BOOTS": [], "left": lambda: 1e9, "log": lambda *a: None, "nb5_write_report": lambda: None, "ABORT": {},
              "NB5_READY_TIMEOUT": 1800, "E28_BUILD_ENV": {"DRAFT_HEAD": "int8"}, "json": json, "NB5_BOOT_EXTRA": [],
              "NB5_BOOT_KV": "fp8", "NB5_BOOT_LEVERS": {}, "E28_RUNG": {}}

        def _try(trec, name, kv_bytes, phases, mtp_on):
            st = script[len(calls)]
            seqs = ns["NB5_BOOT_EXTRA"][ns["NB5_BOOT_EXTRA"].index("--max-num-seqs") + 1]
            calls.append({"name": name, "kv_bytes": kv_bytes, "seqs": seqs, "head": ns["E28_BUILD_ENV"].get("DRAFT_HEAD"),
                          "rung": dict(trec.get("e28_rung") or {})})
            trec["status"] = None if st == "DONE" else st
            trec["phases_started"] = st == "DONE"
            trec["log_text_for_ladder"] = "torch.OutOfMemoryError: CUDA out of memory" if st == "LAUNCH_FAILED" else ""
            return trec
        ns["nb5_boot_try"] = _try
        exec(_boot_src, ns)
        b = ns["nb5_boot"]("merlin_fp8_mtp", [["EVAL", "all", None, "on", "A", "fp8"]], True, "fp8")
        return b, calls, ns
    _b1, _c1, _ = _boot_run(["DONE"])
    _b2, _c2, _n2 = _boot_run(["I8_READBACK_UNVERIFIED", "DONE"])
    _b3, _c3, _ = _boot_run(["I8_READBACK_UNVERIFIED", "LAUNCH_FAILED", "DONE"])
    check("T5 W22 (nb5_boot exec'd on stubs): ENGAGED -> one try, the I8 head kept, the eval phase starts",
          len(_c1) == 1 and _c1[0]["head"] == "int8" and _c1[0]["kv_bytes"] == 3285145068 and not _b1.get("i8_fallback"), _c1)
    check("T5 W22: I8_READBACK_UNVERIFIED -> the SAME pin (3,285,145,068 B) with the BF16 head, seqs 12 kept, its own launch name "
          "(_bf16head), recorded; the BF16 retry boots -> the eval starts (2 tries, no ladder step)",
          [c["kv_bytes"] for c in _c2] == [3285145068, 3285145068] and [c["seqs"] for c in _c2] == ["12", "12"]
          and [c["head"] for c in _c2] == ["int8", None] and _c2[1]["name"] == _c2[0]["name"] + "_bf16head"
          and [c["rung"].get("c12") for c in _c2] == [True, True]
          and len(_b2["i8_fallback"]) == 1 and _b2["i8_fallback"][0]["kv_bytes"] == 3285145068 and _b2["pin_used"]["kv_bytes"] == 3285145068
          and _n2["E28_BUILD_ENV"] == {}, (_c2, _b2.get("i8_fallback")))
    check("T5 W22: a failed BF16 retry steps the ladder down (0.75 I8 -> 0.75 BF16 OOM -> 0.5 BF16 boots): seqs 12 on the I8 rung "
          "and its same-pin BF16 retry, NB6Q's table 8 on the 0.5 rung (c12 False), E18's 3-attempt cap",
          [c["kv_bytes"] for c in _c3] == [3285145068, 3285145068, 3016709612]
          and [c["head"] for c in _c3] == ["int8", None, None] and [c["seqs"] for c in _c3] == ["12", "12", "8"]
          and [c["rung"].get("c12") for c in _c3] == [True, True, False] and _b3["pin_used"]["gib"] == 0.5
          and _b3["pin_used"]["max_num_seqs_launch"] == 8, _c3)
    # ---- W23: the relaunch = K100Q fix D (NB5_ACTIVE_BOOT) + the I8 state
    _bt23 = fn_src(ENG, "nb5_boot_try")
    _set23 = _tw.dedent(_bt23[_bt23.index("        NB5_ACTIVE_BOOT.clear()"):_bt23.index("        trec[\"config_readback\"] = NB5L.config_readback(")])
    _rl23 = fn_src(ENG, "e21_relaunch")
    check("T1 W23: NB5_ACTIVE_BOOT defined before NB5_PIN; set right after ready in nb5_boot_try (before the C-11 read-back / I8 gate); "
          "e21_relaunch boots it (K100Q K8b text) + restores the I8 state", ENG.index("NB5_ACTIVE_BOOT = {}") < ENG.index('NB5_PIN = {"start_gib"')
          and _bt23.index('trec["ready_sec"] = ') < _bt23.index("NB5_ACTIVE_BOOT.clear()") < _bt23.index("E28 W4 W21")
          and "NB5_KV_BYTES[NB5_BOOT_KV])\n" not in _rl23.split("_kvb = ")[0] and "E28_BUILD_ENV.update(_ab.get(\"build_env\") or {})" in _rl23)

    class _Srv23:
        def __init__(self, name="", port=8001, env=None):
            self.name, self.port, self.env, self.url = name, port, dict(env or {}), "http://x"
            self.log_path = Path(tempfile.mkdtemp()) / (name + ".log")
            self.log_path.write_text("log", encoding="utf-8")

        def kill(self):
            pass

        def start(self):
            pass

        def wait_ready(self, t):
            return True, "ready"

    def _relaunch23(boot_trec, boot_extra, boot_build_env, stale_extra=None, stale_build_env=None, active=True):
        ns = {"NB5_ACTIVE_BOOT": {}, "NB5_BOOT_KV": "fp8", "NB5_BOOT_EXTRA": list(boot_extra), "NB5_BOOT_LEVERS": {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"},
              "E28_BUILD_ENV": dict(boot_build_env), "trec": dict(boot_trec), "name": boot_trec["name"], "kv_bytes": boot_trec["kv_bytes"],
              "s": _Srv23(boot_trec["name"], env={"NB4_PPDEFER": "1", "X": "y"})}
        if active:
            exec(_set23, ns)
        calls = []
        ns.update(NB5_BOOT_EXTRA=list(stale_extra if stale_extra is not None else boot_extra),
                  E28_BUILD_ENV=dict(stale_build_env if stale_build_env is not None else boot_build_env))

        def _mfs(name, kv, port, kvb):
            calls.append({"name": name, "kv": kv, "kv_bytes": kvb, "extra": list(ns["NB5_BOOT_EXTRA"]),
                          "levers": dict(ns["NB5_BOOT_LEVERS"]), "head": ns["E28_BUILD_ENV"].get("DRAFT_HEAD")})
            return _Srv23(name, port)
        old = _Srv23(boot_trec["name"])
        ns.update(E11_SERVER=old, SGL_URL=old.url, NB5_SERVERS=[old], merlin_fn_server=_mfs, NB5_REPORT={}, NB5_RELAUNCH_MIN_SEC=600,
                  NB5_KV_BYTES=dict(GC["NB5_KV_BYTES"]), E12_RUN={}, time=time, shutil=__import__("shutil"), FAKE="fake",
                  e21_snapshot_counters=lambda: None, nb5_host_gate=lambda n: {"ok": True}, sampled_check=lambda s: {"ok": True},
                  L=type("L", (), {"wait_gpu_release": staticmethod(lambda *a, **k: True)}), NB5_CAPACITY={"workers": boot_trec["_workers"]})
        exec(_rl23, ns)
        ok, why = ns["e21_relaunch"]()
        return ok, calls, ns
    _x12 = ["--language-model-only", "--max-num-seqs", "12"]
    _x8 = ["--language-model-only", "--max-num-seqs", "8"]
    _t12 = {"name": "merlin_fp8_mtp_pin75", "kv_bytes": 3285145068, "pin_gib": 0.75, "max_num_seqs_launch": 12,
            "e28_rung": {"kv_bytes": 3285145068, "gib": 0.75, "c12": True, "seqs": 12}, "_workers": 12}
    _ok1, _k1, _n1 = _relaunch23(_t12, _x12, {"DRAFT_HEAD": "int8"}, stale_extra=_x8, stale_build_env={})
    check("T5 W23 (e21_relaunch exec'd): after a C12-rung I8 boot -> the relaunch boots 3,285,145,068 B with --max-num-seqs 12, the "
          "boot's levers, DRAFT_HEAD=int8, NB4_PPDEFER kept; recorded (pin / seqs / head / c12 / workers 12)",
          _ok1 and len(_k1) == 1 and _k1[0]["kv_bytes"] == 3285145068 and _k1[0]["extra"] == _x12 and _k1[0]["head"] == "int8"
          and _k1[0]["levers"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"} and _n1["E11_SERVER"].env.get("NB4_PPDEFER") == "1"
          and _n1["NB5_REPORT"]["relaunch_boot"] == [{"kv_bytes": 3285145068, "max_num_seqs": 12, "pin_gib": 0.75, "from": "merlin_fp8_mtp_pin75",
                                                     "draft_head": "int8", "c12": True, "workers": 12}], (_k1, _n1["NB5_REPORT"]))
    _tbf = dict(_t12, name="merlin_fp8_mtp_pin75_bf16head")
    _ok2, _k2, _n2 = _relaunch23(_tbf, _x12, {}, stale_build_env={"DRAFT_HEAD": "int8"})
    check("T5 W23: after the same-pin BF16-head retry booted -> the relaunch keeps 3,285,145,068 B / 12 and the BF16 head (no DRAFT_HEAD)",
          _ok2 and _k2[0]["kv_bytes"] == 3285145068 and _k2[0]["extra"] == _x12 and _k2[0]["head"] is None
          and _n2["NB5_REPORT"]["relaunch_boot"][0]["draft_head"] is None, _k2)
    _t8 = {"name": "merlin_fp8_mtp_pin50_bf16head", "kv_bytes": 3016709612, "pin_gib": 0.5, "max_num_seqs_launch": 8,
           "e28_rung": {"kv_bytes": 3016709612, "gib": 0.5, "c12": False, "seqs": 8}, "_workers": 8}
    _ok3, _k3, _n3 = _relaunch23(_t8, _x8, {}, stale_extra=_x12, stale_build_env={"DRAFT_HEAD": "int8"})
    check("T5 W23: after a lower-rung boot (0.5 GiB, NB6Q's table seqs 8, BF16 head) -> the relaunch boots 3,016,709,612 B with seqs 8, "
          "no DRAFT_HEAD (a stale extra / head is replaced); recorded c12 False, workers 8",
          _ok3 and _k3[0]["kv_bytes"] == 3016709612 and _k3[0]["extra"] == _x8 and _k3[0]["head"] is None
          and _n3["NB5_REPORT"]["relaunch_boot"][0]["c12"] is False and _n3["NB5_REPORT"]["relaunch_boot"][0]["workers"] == 8, _k3)
    _ok4, _k4, _n4 = _relaunch23(_t8, _x8, {"DRAFT_HEAD": "int8"}, active=False)
    check("T5 W23: no booted try recorded -> NB6Q's fallback (NB5_KV_BYTES[kv] = the RC2 pin, the current extra / head)",
          _ok4 and _k4[0]["kv_bytes"] == 2479838700 and _k4[0]["head"] == "int8", _k4)
    _c6 = fn_src(C67, "e21_cell6")
    _ord = next(ast.get_source_segment(_c6, n) for n in ast.parse(_c6).body[0].body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "E12_ORDER")

    def _order(p, env):
        ch = {"t3": {"train": [1, 2, 3], "test": [{}]}, "t1": {"train": [1], "test": [{}, {}]}, "t2": {"train": [1, 2, 3], "test": [{}]}}
        ns = {"P": p, "os": types.SimpleNamespace(environ=env), "TASK_IDS": ["t1", "t2", "t3"], "_E12_INDEX": {"t1": 0, "t2": 1, "t3": 2},
              "_e12_order2_score": lambda t: int(len(ch[t]["train"]) >= 3) + int(len(ch[t]["test"]) == 1), "_e12_total_cells": lambda t: 0}
        exec(_ord, ns)
        return ns["E12_ORDER"]
    check("T5 ORDER2 statement (N3 + W11): plain TASK_IDS order with no_order2 or E28_SCHED=1; ORDER2 otherwise",
          _order({"no_order2": True}, {}) == ["t1", "t2", "t3"] and _order({}, {"E28_SCHED": "1"}) == ["t1", "t2", "t3"]
          and _order({}, {}) == ["t2", "t3", "t1"])

# ============================================================================ T6 the CELL 3 scheduler block on a real Duck payload
if "T6" in SEC:
    if not PAYLOAD.is_file():
        SKIPPED.append(f"T6 scheduler block: no Duck payload at {PAYLOAD} (the kits mount it)")
    else:
        _zb = PAYLOAD.read_bytes()
        check("T6 the Duck payload is 9e4b8928 (sha256)", sha(_zb) == PAYLOAD_SHA)
        _blk = C3[C3.index("# ---- E28 W4 SCHED (Build/E28_SCHED_API.md"):C3.index("sys.path[:0] = [str(E12_DIR)")]

        def sched_ns(tag, prof, foreign=False, harness=False, attempt=104000, b2=120000, rerun=True):
            d = Path(tempfile.mkdtemp(prefix=f"e28w4_sched_{tag}_"))
            with zipfile.ZipFile(io.BytesIO(_zb)) as z:
                z.extractall(d)
            if foreign:
                with open(d / "arc2_duck.py", "a", encoding="utf-8", newline="\n") as fh:
                    fh.write("# a foreign edit\n")
            ns = {"P": prof, "E12_DIR": d, "E28_SRC": GC["E28_SRC"], "E28_MOD_SHAS": GC["E28_MOD_SHAS"], "NB5_ATTEMPT": {"b0": attempt, "b1": attempt},
                  "E28_B2_ATTEMPT": b2, "NB5_WINDOW": 130560, "E11_HARNESS": harness, "_E12_ZIP_BYTES": _zb, "RERUN": rerun,
                  "_E12_MANIFEST": {"arc2_duck.py": DUCK_PRE}, "_io": io, "zipfile": zipfile, "hashlib": hashlib, "json": json, "os": os}
            old = dict(os.environ)
            os.environ["E28_SCHED"] = "stale"
            try:
                with contextlib.redirect_stdout(io.StringIO()) as o:
                    try:
                        exec(compile(_blk, "<cell3 sched>", "exec"), ns)
                    except RuntimeError as _ex:
                        ns["_raised"] = str(_ex)
                ns["_env"] = {k: os.environ.get(k) for k in ("E28_SCHED", "E28_T1", "E28_B2_TOKENS", "E28_B2_WINDOW", "E28_B2_REQUEST_TIMEOUT")}
                ns["_out"] = o.getvalue()
            finally:
                os.environ.clear()
                os.environ.update(old)
            ns["_sha"] = sha((d / "arc2_duck.py").read_bytes())
            return ns
        a = sched_ns("apply", dict(GC["P"]))
        M = a["E28S"]
        check("T6 applied: E28_SCHED_REC APPLIED, arc2_duck.py FULLY_PATCHED (post sha + marker), py_compile ok, logged",
              a["E28_SCHED_REC"]["state"] == "APPLIED" and a["_sha"] == M.DUCK_POST and M.tree_state(a["E12_DIR"]) == "FULLY_PATCHED"
              and a["E28_SCHED_REC"]["py_compile"]["rc"] == 0 and "E28 W4 SCHED: A -> B -> C scheduler APPLIED" in a["_out"], a["E28_SCHED_REC"])
        check("T6 applied: the notebook process env = the module's server_env() (E28_SCHED 1, T1 104000, B2 120000, window 130560) + "
              "E28_B2_REQUEST_TIMEOUT 9000 (explicit)", {k: v for k, v in a["_env"].items() if k != "E28_B2_REQUEST_TIMEOUT"} == M.server_env()
              == {"E28_SCHED": "1", "E28_T1": "104000", "E28_B2_TOKENS": "120000", "E28_B2_WINDOW": "130560"}
              and a["_env"]["E28_B2_REQUEST_TIMEOUT"] == "9000" and "_raised" not in a, a["_env"])
        h = sched_ns("harness", dict(GC["P"]), harness=True, attempt=6000, b2=9000)
        check("T6 harness: the env follows the notebook's numbers (T1 = NB5_ATTEMPT b0, B2 = E28_B2_ATTEMPT), APPLIED",
              h["E28_SCHED_REC"]["state"] == "APPLIED" and h["_env"]["E28_T1"] == "6000" and h["_env"]["E28_B2_TOKENS"] == "9000", h["_env"])
        x = sched_ns("mismatch", dict(GC["P"]), harness=False, attempt=6000)
        check("T6 not harness + T1 != the module's 104,000 -> FAILED (E28_SCHED_ENV_MISMATCH), restored, env unset",
              x["E28_SCHED_REC"]["state"] == "FAILED" and "ENV_MISMATCH" in x["E28_SCHED_REC"]["error"] and x["_sha"] == DUCK_PRE
              and x["_env"]["E28_SCHED"] is None, x["E28_SCHED_REC"])
        xc = sched_ns("mismatch_commit", dict(GC["P"]), harness=False, attempt=6000, rerun=False)
        check("T6 COMMIT + a scheduler failure -> NO_GO_E28_SCHED raised before any boot (arc2_duck.py restored, env unset); the rerun "
              "case above falls back instead", "NO_GO_E28_SCHED" in xc.get("_raised", "") and xc["E28_SCHED_REC"]["state"] == "FAILED"
              and xc["_sha"] == DUCK_PRE and xc["_env"]["E28_SCHED"] is None, (xc.get("_raised"), xc["E28_SCHED_REC"]))
        n = sched_ns("off", dict(GC["PROFILE_TABLE"]["NB6COMMIT"]))
        check("T6 a profile without e28_sched: NOT_REQUESTED, arc2_duck.py untouched (pre sha), a stale E28_SCHED env cleared",
              n["E28_SCHED_REC"]["state"] == "NOT_REQUESTED" and n["_sha"] == DUCK_PRE and n["_env"]["E28_SCHED"] is None)
        f = sched_ns("foreign", dict(GC["P"]), foreign=True)
        check("T6 foreign arc2_duck.py: FAILED (PRE_SHA_MISMATCH), restored to the manifest sha from the payload, env unset, loud line, "
              "no raise", f["E28_SCHED_REC"]["state"] == "FAILED" and "PRE_SHA_MISMATCH" in f["E28_SCHED_REC"]["error"]
              and f["E28_SCHED_REC"]["restored"] is True and f["_sha"] == DUCK_PRE and f["_env"]["E28_SCHED"] is None
              and "E28 W4 SCHED FAILED" in f["_out"], f["E28_SCHED_REC"])
        e = sched_ns("noembed", dict(GC["P"]))
        e2 = dict(e)
        _blk_ns = {k: v for k, v in e.items() if not k.startswith("_") or k in ("_E12_ZIP_BYTES", "_E12_MANIFEST", "_io")}
        _blk_ns["RERUN"] = True
        _blk_ns["E28_SRC"] = {"e28w4_lib.py": GC["E28_SRC"]["e28w4_lib.py"]}
        shutil.rmtree(e["E12_DIR"])
        with zipfile.ZipFile(io.BytesIO(_zb)) as z:
            z.extractall(e["E12_DIR"])
        with contextlib.redirect_stdout(io.StringIO()):
            _o2 = dict(os.environ)
            try:
                exec(compile(_blk, "<cell3 sched>", "exec"), _blk_ns)
            finally:
                os.environ.clear()
                os.environ.update(_o2)
        check("T6 module not embedded (W12 load failure) -> FAILED (E28_SCHED_NOT_EMBEDDED), the original file stays",
              _blk_ns["E28_SCHED_REC"]["state"] == "FAILED" and "NOT_EMBEDDED" in _blk_ns["E28_SCHED_REC"]["error"]
              and sha((e["E12_DIR"] / "arc2_duck.py").read_bytes()) == DUCK_PRE)
        rv = M.revert_e28_sched(a["E12_DIR"])
        ra = M.apply_e28_sched(a["E12_DIR"])
        check("T6 round trip on the patched tree: revert -> ORIGINAL (pre sha), re-apply -> FULLY_PATCHED; verify ok",
              rv["ok"] and sha((a["E12_DIR"] / "arc2_duck.py").read_bytes()) == M.DUCK_POST and ra["applied"] == "applied"
              and M.verify_e28_sched(a["E12_DIR"])[0] is True)
        check("T6 the patched arc2_duck.py imports in a fresh python with E28_SCHED unset and set (solve_all_r6 / build_submission present)",
              all(subprocess.run([sys.executable, "-c", "import sys; sys.path[:0] = [sys.argv[1], sys.argv[1] + '/vendor']; import arc2_duck as A; "
                                  "assert callable(A.solve_all_r6) and callable(A.build_submission)", str(a["E12_DIR"])],
                                 capture_output=True, text=True, timeout=300, env=dict(os.environ, **envx)).returncode == 0
                  for envx in ({}, {"E28_SCHED": "1"})))
        for ns_ in (a, h, x, xc, n, f, e):
            shutil.rmtree(ns_["E12_DIR"], ignore_errors=True)

# ============================================================================ T7 push folder
if "T7" in SEC:
    if (PUSH / "kernel-metadata.json").is_file():
        m = json.loads((PUSH / "kernel-metadata.json").read_text(encoding="utf-8"))
        check("T7 push folder: id / title / code_file = E28 W4, the notebook in the folder = this build, GPU L4, private, no internet",
              m["id"] == "foysalemonshanto/q27-xhigh-e28-w4-eval120-6h" and m["code_file"] == "q27-xhigh-e28-w4-eval120-6h.ipynb"
              and (PUSH / m["code_file"]).read_bytes() == RAW and m["enable_gpu"] is True and m["machine_shape"] == "NvidiaL4"
              and m["is_private"] is True and m["enable_internet"] is False, m)
        check("T7 push folder sources = NB6Q's (W4 p1-p5, FP8 PLE overlay, Merlin env, the Duck payload dataset, the competition, image)",
              m["model_sources"] == FACTS["kernel_metadata"]["model_sources"] and m["dataset_sources"] == ["foysalemonshanto/q27-e15b-payload"]
              and m["competition_sources"] == ["arc-prize-2026-arc-agi-2"] and m["docker_image"].endswith("37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"))
    else:
        SKIPPED.append(f"T7 push folder: not here ({PUSH})")
    _ps = Path(ENV("E28T_PUSH_SAMI") or "D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e28-w4-eval120-6h__esfersami50")
    if (_ps / "kernel-metadata.json").is_file():            # local only (the kits carry the foysal folder)
        ms = json.loads((_ps / "kernel-metadata.json").read_text(encoding="utf-8"))
        check("T7 the esfersami50 GPU push folder: id esfersami50/q27-xhigh-e28-w4-eval120-6h, the same notebook bytes, every other field "
              "= the build facts' metadata", ms["id"] == "esfersami50/q27-xhigh-e28-w4-eval120-6h" and (_ps / ms["code_file"]).read_bytes() == RAW
              and {k: v for k, v in ms.items() if k != "id"} == {k: v for k, v in FACTS["kernel_metadata"].items() if k != "id"}, ms)

bad = [n for n, ok in RESULTS if not ok]
for s_ in SKIPPED:
    print("SKIP " + s_, flush=True)
print(f"{len(RESULTS) - len(bad)} / {len(RESULTS)} PASS ({len(SKIPPED)} skipped)", flush=True)
print("@@E28W4_UNITS_OK@@" if not bad else "@@E28W4_UNITS_FAIL@@", flush=True)
sys.exit(0 if not bad else 1)
