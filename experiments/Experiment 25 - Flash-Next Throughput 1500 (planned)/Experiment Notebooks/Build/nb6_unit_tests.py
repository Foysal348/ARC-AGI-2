"""E25 NB6 SUBMIT unit tests (quick, local; CPU kit: NB6T_NB / NB6T_NB5 / NB6T_PAYLOAD / NB6T_PUSH / NB6T_PUSH5).

Proven-only build (user order 2026-11-01): the NB6 notebook = the Kaggle-run NB5 QUALITY notebook (9c71ba89) + only the
marked edits E1-E5, E7, P1-P6; the rerun rules = E18 R2 (server ladder / readiness / stall / fast final kill / budget) and
E15B SUBMIT (CELL 0, writer, task set, order, sandbox fail closed, attempt policy); the Merlin engine config = NB5QUALITY_R2.
usage: python nb6_unit_tests.py   (exit 0 = all PASS)"""
import difflib
import time
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = Path(os.environ.get("NB6T_NB") or HERE.parent / "Q27-XHIGH-E25_NB6_SUBMIT.ipynb")
NB5 = Path(os.environ.get("NB6T_NB5") or HERE.parent / "Q27-XHIGH-E25_NB5_QUALITY.ipynb")
PAYLOAD = Path(os.environ.get("NB6T_PAYLOAD") or r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload/q27x_e15b_duck_9e4b89285136.bin")
NB5_SHA = "9c71ba89e4db75b5f9a63a4be652e860b629120685e8f534bed3bb8d06769049"
RESULTS = []


def check(name, cond, detail=""):
    try:
        ok = bool(cond() if callable(cond) else cond)
    except Exception as e:
        ok, detail = False, f"raised {type(e).__name__}: {e} | {detail}"
    RESULTS.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  [{str(detail)[:900]}]"), flush=True)


def cells_of(p):
    nb = json.loads(p.read_text(encoding="utf-8"))
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


import hashlib  # noqa: E402
raw = NB.read_bytes()
C6, C5 = cells_of(NB), cells_of(NB5)
code = [t for k, t in C6 if k == "code"]
print("NB6", NB.name, "sha256", hashlib.sha256(raw).hexdigest(), "|", len(raw), "bytes", flush=True)
check("NB6 notebook < 400 kB (Kaggle commit limit)", len(raw) < 400_000, len(raw))
check("NB5 QUALITY reference is the Kaggle-run sha 9c71ba89", hashlib.sha256(NB5.read_bytes()).hexdigest() == NB5_SHA)
check("same cell layout as NB5 QUALITY (9 cells, same kinds)", [k for k, _ in C6] == [k for k, _ in C5], [k for k, _ in C6])

# ---- 1. verbatim NB5: only the markdown, CELL 1, the engine cell, CELL 6/7 and CELL 8 differ, only by the marked edits
_diff_cells = [i for i, (a, b) in enumerate(zip(C5, C6)) if a != b]
check("only the markdown, CELL 1, the engine cell, CELL 6/7 and CELL 8 differ from NB5 QUALITY (CELL 0 / 3 / order / 9 verbatim)",
      _diff_cells == [0, 2, 5, 6, 7], _diff_cells)


def changed(a, b):
    rm, add = [], []
    for ln in difflib.ndiff(a.splitlines(), b.splitlines()):
        if ln.startswith("- "):
            rm.append(ln[2:])
        elif ln.startswith("+ "):
            add.append(ln[2:])
    return rm, add


_rm1, _add1 = changed(C5[2][1], C6[2][1])
check("CELL 1: the removed lines are exactly the default-profile line and the RERUN_ALL selector",
      sorted(_rm1) == sorted(['NB5_DEFAULT_PROFILE = "NB5QUALITY_R2"',
                              'PROFILE = "RERUN_ALL" if RERUN else os.environ.get("Q27X_PROFILE", NB5_DEFAULT_PROFILE).strip().upper()']), _rm1)
check("CELL 1: every added line belongs to the NB6 edits (NB6_* block, NB6-marked lines, the two profile rows)",
      all(("NB6" in ln) or ln.startswith(("NB6_", "                  ", "# ")) or ln.strip().startswith(('"RERUN_PROD"', '"NB6COMMIT"', "#"))
          or ln == 'NB5_DEFAULT_PROFILE = "NB6COMMIT"' for ln in _add1), _add1)
_rm6, _add6 = changed(C5[6][1], C6[6][1])
check("CELL 6/7: one line changed (E4: the b1_accepted_mode map gains 0 -> '')",
      len(_rm6) == 1 and len(_add6) == 1 and _rm6[0].replace('{1: "after3"', '{0: "", 1: "after3"') in _add6[0]
      and "NB6 E4" in _add6[0], (_rm6, _add6))
check("CELL 0 (dual mode + fallback submission) is NB5's, verbatim (= E18 R2 = E15B SUBMIT)", C5[1] == C6[1])
check("CELL 3 (Duck package + sandbox self-test, fail closed) and the task-order cell are NB5's, verbatim",
      C5[3] == C6[3] and C5[4] == C6[4] and C5[8] == C6[8])
check("the submission writer (e12_write_submission: RERUN builds from the records over the identity fallback) is NB5's",
      "            sub = identity_submission(test_challenges)\n            if RERUN:" in code[5]
      and "E12A.build_submission({t: CHALLENGES[t] for t in recs}, recs).items()" in code[5])
check("task order = ORDER2 (>=3 train + 1 test first, then fewest cells), no fixed task_order in the NB6 profiles",
      "E12_ORDER = sorted(TASK_IDS, key=lambda t: (-_e12_order2_score(t), _e12_total_cells(t), _E12_INDEX[t]))" in code[5])
_rm5, _add5 = changed(C5[5][1], C6[5][1])
check("engine cell: removed lines = E7 / P2 / P3 / P4 / P5 anchors only",
      sorted(_rm5) == sorted(['        if not fail["failed"] or not P.get("pin_ladder") or ABORT.get("why"):',
                              '            s.kill()', '        rel = L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus)',
                              '    o = with_servers(NB5_SERVERS, NB5_READY_TIMEOUT, body)',
                              '        ok, why = srv.wait_ready(max(60.0, min(NB5_READY_TIMEOUT, t)))',
                              '    def kill(self):', '            break']), _rm5)
_rm7, _add7 = changed(C5[7][1], C6[7][1])
check("CELL 8: only the sha-gate line changed (P6: no sha gate in the rerun)",
      _rm7 == ['    _ok, _why = sha_gate(min(1500, left() - 900))'] and len(_add7) == 2 and "NB6 P6" in _add7[1], (_rm7, _add7))
check("no NB6 heartbeat / sandbox retry / boot retry / sha continue code left",
      not any(x in "\n".join(code) for x in ("nb6_hb", "nb6_heartbeat", "nb6_sandbox", "nb6_boot", "TIMEOUT_CONTINUE", "boot_retry_once")))

# ---- 2. CELL 0 + CELL 1 executed for real
arc = Path(tempfile.mkdtemp(prefix="nb6_arc_"))
FAKE_TEST = {"aaaa0001": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[0, 1], [1, 0]]}]},
             "aaaa0002": {"train": [], "test": [{"input": [[3]]}, {"input": [[4, 4]]}]},       # two test inputs
             "aaaa0003": {"train": [], "test": [{"input": [[5], [6]]}]}}
json.dump(FAKE_TEST, open(arc / "arc-agi_test_challenges.json", "w"))
_CLR = ("E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN", "Q27X_PROFILE", "NB5_ATTEMPT_B0", "NB5_ATTEMPT_B1", "E12_MAX_GEN",
        "NB5_NB0_FP8_GATE", "E12_B_EXTRA_ATTEMPTS", "E12_B_SKIP_B1_IF_ACCEPTED", "E12_B_B1_ACC_MODE", "LOCAL_ANALYZER_SEED",
        "NB5_ADMIT_MODE", "NB5_THINK_FLOOR", "NB5_RESERVE", "E12_B_RESERVE", "E12_R6_MIN_ADMIT_SEC", "E12_B_MAX_ATTEMPTS",
        "Q27X_TIME_BUDGET", "NB5_KV_DTYPE", "NB5_COMMIT_MODE", "NB5_EXTRA_ARGS_JSON", "E12_B_WEAK_COMMIT")


def run_cells(env):
    work = Path(tempfile.mkdtemp(prefix="nb6_w_"))
    old = dict(os.environ)
    for k in _CLR:
        os.environ.pop(k, None)
    os.environ.update(Q27X_ARC_ROOT=str(arc), Q27X_WORKING=str(work), E23_TMP=str(work), **env)
    G = {"__name__": "__main__"}
    try:
        for c in code[:2]:
            exec(compile(c, "<cell>", "exec"), G)
    finally:
        os.environ.clear()
        os.environ.update(old)
    G["_WORK"] = work
    return G


GC = run_cells({})                                    # commit: the baked default
GR = run_cells({"KAGGLE_IS_COMPETITION_RERUN": "1"})  # the hidden rerun
T = GC["PROFILE_TABLE"]
R2 = T["NB5QUALITY_R2"]
check("profile table: NB5's profiles all kept + RERUN_PROD + NB6COMMIT",
      set(T) == {"NB5EVAL120", "NB5EVAL30", "NB5EVAL30MTP", "NB5QUALITY", "NB5SMOKEDIAG", "NB5BUDGET", "RERUN_ALL",
                 "NB5QUALITY_R2", "RERUN_PROD", "NB6COMMIT"}, sorted(T))
check("commit default PROFILE == NB6COMMIT (NB5_DEFAULT_PROFILE baked), RERUN False", GC["PROFILE"] == "NB6COMMIT"
      and GC["NB5_DEFAULT_PROFILE"] == "NB6COMMIT" and GC["RERUN"] is False)
check("fake rerun: PROFILE == RERUN_PROD, RERUN True", GR["PROFILE"] == "RERUN_PROD" and GR["RERUN"] is True)
check("fake rerun: Q27X_PROFILE cannot override the rerun profile",
      run_cells({"KAGGLE_IS_COMPETITION_RERUN": "1", "Q27X_PROFILE": "NB5EVAL30"})["PROFILE"] == "RERUN_PROD")
P = GR["P"]
# the Merlin engine config of the GPU-run NB5 QUALITY (sha 9c71ba89, profile NB5QUALITY_R2), key by key
_R2KEYS = {"admit_mode": "slots", "f_rule": "measured", "seqs_rule": "table", "caps16": True, "pin_ladder": True,
           "patch_60490": True, "lever_env": {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, "continue_after_failed_boot": True,
           "nb5_attempt": {"b0": 104000, "b1": 104000}, "nb5_gate": 1.0, "no_pure_decode": True}
for prof in ("RERUN_PROD", "NB6COMMIT"):
    for _k, _v in _R2KEYS.items():
        check(f"{prof} key {_k} = {_v!r} (= NB5QUALITY_R2)", T[prof].get(_k) == _v == R2.get(_k), (T[prof].get(_k), R2.get(_k)))
    check(f"{prof}: one phase EVAL, all tasks, until the solve end, MTP on, variant A, FP8 (= R2's Q_FP8_K3 apart from the name)",
          T[prof]["nb5_phases"] == [["EVAL", "all", None, "on", "A", "fp8"]] and R2["nb5_phases"][0][1:] == ["all", None, "on", "A", "fp8"])
    check(f"{prof}: no quality A/B / equal windows / fixed task order / boot_retry_once", not any(
        T[prof].get(k) for k in ("quality_ab", "equal_windows", "task_order", "hidden_equiv", "parity", "boot_retry_once")))
    check(f"{prof}: e18_ladder on (E18 R2 engine ladder)", T[prof].get("e18_ladder") is True)
    check(f"{prof}: nb5_env = E15B SUBMIT's attempt policy (no skip, no extra attempts, b1 mode '' = default); no fixed seed",
          T[prof]["nb5_env"] == {"E12_B_SKIP_B1_IF_ACCEPTED": "0", "E12_B_B1_ACC_MODE": "0", "E12_B_EXTRA_ATTEMPTS": "0"})
check("RERUN_PROD: source test (every hidden task), E18 R2 RERUN_ALL budget 12 h - 3 min, E18's margins",
      P["source"] == "test" and P.get("n_tasks") is None and not P.get("task_list") and P["time_budget_sec"] == 12.0 * 3600 - 3 * 60
      and all(P[k] == v for k, v in GR["_NB5_T"].items() if k != "time_budget_sec")
      and (P["admit_frac"], P["drain_frac"], P["hard_margin_sec"], P["verify_margin_sec"], P["drain_margin_sec"],
           P["admit_margin_sec"]) == (0.90, 0.94, 120, 60, 480, 120))
check("rerun: TASK_IDS source = every test task (the task-order cell takes test_challenges when source != eval)",
      "    CHALLENGES=dict(test_challenges)\n    TASK_IDS=sorted(CHALLENGES)" in code[3])
check("rerun E15B: skip 0, b1 mode 0, extra 0, attempts 104k / 104k",
      (GR["E15B"]["E12_B_SKIP_B1_IF_ACCEPTED"], GR["E15B"]["E12_B_B1_ACC_MODE"], GR["E15B"]["E12_B_EXTRA_ATTEMPTS"],
       GR["E15B"]["E12_B_ATTEMPT_TOKENS"], GR["E15B"]["E12_B_ATTEMPT_TOKENS_B1"]) == (0, 0, 0, 104000, 104000), GR["E15B"])
check("rerun engine params: slots / measured F / seqs table / lever env / gate 1.0 / NB5_CAP 99,904; readiness 30 min",
      GR["NB5_ADMIT_MODE"] == "slots" and GR["NB5_F_RULE"] == "measured" and GR["NB5_SEQS_RULE"] == "table"
      and GR["NB5_LEVER_ENV"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"} and GR["NB5_GATE"] == 1.0 and GR["NB5_CAP"] == 99904
      and GR["NB5_READY_TIMEOUT"] == 1800)
_cc = GR["NB5_EXTRA_ARGS"]
check("rerun CAPS16: --compilation-config FULL_AND_PIECEWISE with NB1's capture sizes",
      _cc[:1] == ["--compilation-config"] and json.loads(_cc[1]) == {"cudagraph_mode": "FULL_AND_PIECEWISE",
                                                                    "cudagraph_capture_sizes": GR["NB5_CAPS16"]}, _cc)
NB5L = GR["NB5L"]
check("pin ladder (FP8): RC2 pin + 1.0 GiB = 3,553,580,524 B first, then 0.75 / 0.5 / 0; launch seqs 9 / 8 / 8 / 6",
      [t["kv_bytes"] for t in NB5L.pin_tries(GR["NB5_KV_BYTES"]["fp8"], None)] == [3553580524, 3285145068, 3016709612, 2479838700]
      and [NB5L.seqs_for("fp8", t["gib"]) for t in NB5L.pin_tries(GR["NB5_KV_BYTES"]["fp8"], None)] == [9, 8, 8, 6])
check("CONFIG records the RERUN_PROD row (report)", GR["CONFIG"]["PROFILE"] == "RERUN_PROD" and GR["CONFIG"]["profile_row"] is P)


def _payload_src():
    b = PAYLOAD.read_bytes()
    try:
        z = zipfile.ZipFile(io.BytesIO(b))
        n = [x for x in z.namelist() if x.endswith("arc2_duck.py")][0]
        return z.read(n).decode("utf-8")
    except zipfile.BadZipFile:
        t = tarfile.open(fileobj=io.BytesIO(b))
        n = [x for x in t.getnames() if x.endswith("arc2_duck.py")][0]
        return t.extractfile(n).read().decode("utf-8")


_ds = _payload_src()
_mode = {0: "", 1: "after3", 2: "before3"}[GR["E15B"]["E12_B_B1_ACC_MODE"]]
check("Duck payload: skip False + mode '' -> solve_all_r6's defaults (= E15B SUBMIT's call, which passes neither)",
      "skip_b1_if_accepted: bool = False" in _ds and 'b1_accepted_mode: str = ""' in _ds and "extra_attempts: bool = False" in _ds
      and _mode == "" and not GR["E15B"]["E12_B_SKIP_B1_IF_ACCEPTED"])
check("the solver deadline = min(HARD_STOP - 180 s, phase end); min-admit 300 s (E15B SUBMIT's values)",
      "_end = min(HARD_STOP - E12_STOP_BEFORE_HARD_SEC, E21_HALF_END_PC)" in code[5] and GR["E12_R6"]["E12_R6_MIN_ADMIT_SEC"] == 300
      and GR["E12_STOP_BEFORE_HARD_SEC"] == 180)

# ---- 3. budget times
_t0 = GR["NOTEBOOK_T0"]
_m = lambda g, k: round((g[k] - g["NOTEBOOK_T0"]) / 60, 2)
check("rerun budget (E18 R2): 717 min; hard stop 715; solve end 712; drain 707 / admit 705 (recorded, not used by the Duck path)",
      GR["TOTAL_BUDGET_SEC"] == 43020 and _m(GR, "HARD_STOP") == 715 and _m(GR, "DRAIN_DEADLINE") == 707
      and _m(GR, "ADMIT_DEADLINE") == 705 and round((GR["HARD_STOP"] - 180 - _t0) / 60, 2) == 712,
      {k: _m(GR, k) for k in ("HARD_STOP", "DRAIN_DEADLINE", "ADMIT_DEADLINE", "VERIFY_DEADLINE")})
check("commit budget (NB6COMMIT): 50 min; hard stop 48; solve end 45",
      GC["TOTAL_BUDGET_SEC"] == 3000 and _m(GC, "HARD_STOP") == 48 and round((GC["HARD_STOP"] - 180 - GC["NOTEBOOK_T0"]) / 60, 2) == 45)
check("commit: a boot ready at 33 min still passes nb5_phase's check (left >= 15 min)", GC["TOTAL_BUDGET_SEC"] - 33 * 60 >= 15 * 60)
check("nb5_phase budget rule unchanged (need = window + 10 min; left >= need + 5 min)",
      "need = (window or 0) + 10 * 60\n        if left() < need + 5 * 60:" in code[4])

# ---- 4. commit profile is short
PC = GC["P"]
check("NB6COMMIT: eval source, 3 tasks (1818057f, 2ba387bc, 45a5af55, NB5 QUALITY solid group), scored",
      PC["source"] == "eval" and PC["task_list"] == ["1818057f", "2ba387bc", "45a5af55"]
      and all(t in GC["NB5_QAB_GROUPS"]["solid"] for t in PC["task_list"]))

# ---- 5. CELL 0 fallback under the fake rerun
_sub = json.loads((GR["_WORK"] / "submission.json").read_text(encoding="utf-8"))
check("fake rerun: CELL 0 wrote submission.json for every fake hidden task, one {attempt_1, attempt_2} per test input "
      "(identity fallback)",
      set(_sub) == set(FAKE_TEST) and all(len(_sub[t]) == len(FAKE_TEST[t]["test"]) for t in FAKE_TEST)
      and all(set(r) == {"attempt_1", "attempt_2"} and r["attempt_1"] == q["input"] == r["attempt_2"]
              for t in FAKE_TEST for r, q in zip(_sub[t], FAKE_TEST[t]["test"])), _sub)
for g in (GC, GR):
    shutil.rmtree(g["_WORK"], ignore_errors=True)
shutil.rmtree(arc, ignore_errors=True)

# ---- 6. E18 R2 server rules on the Merlin boot (P1-P6), executed with fakes
import ast  # noqa: E402
import textwrap  # noqa: E402
import types  # noqa: E402


def _src_of(text, name):
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    lines = text.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1:node.end_lineno])


_boot_src = _src_of(code[4], "nb5_boot")


def ladder_run(statuses, rerun=True, e18=True, left_min=600.0, abort_after=None):
    calls = []

    def fake_try(trec, name, kv_bytes, phases, mtp_on):
        st = statuses[len(calls)]
        calls.append({"name": name, "gib": trec["pin_gib"], "ready_sec": NS["NB6_READY_SEC"]})
        trec["status"] = st.get("status")
        if st.get("phases_started"):
            trec["phases_started"] = True
        trec["log_text_for_ladder"] = st.get("log", "")
        if abort_after is not None and len(calls) >= abort_after:
            NS["ABORT"]["why"] = "GPUs 0,1,2,3 still held after x"
    NS = {"P": {"pin_ladder": True, "patch_60490": True, "e18_ladder": e18}, "NB5_PIN": {"start_gib": None, "levers": None, "patch_60490": {"ok": True}},
          "NB5_KV_BYTES": {"fp8": 2479838700}, "NB5_KV_DTYPE": "fp8", "NB5_BOOTS": [], "NB5_EXTRA_ARGS": [], "NB5_MTP_OFF_ARGS": [],
          "NB5_LEVER_ENV": {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, "NB5_SEQS_RULE": "table", "NB5L": NB5L, "FAKE": None,
          "left": lambda: left_min * 60, "nb5_boot_try": fake_try, "log": lambda *a: None, "nb5_write_report": lambda: None,
          "json": json, "ABORT": {"why": None}, "NB5_READY_TIMEOUT": 1800, "RERUN": rerun}
    exec(compile(_boot_src, "<nb5_boot>", "exec"), NS)
    b = NS["nb5_boot"]("merlin_fp8_mtp", [], True, "fp8")
    return b, calls


_b, _c = ladder_run([{"status": "LAUNCH_FAILED"}, {"status": "NO_GO_CONTENT_PROBES_FAILED"}, {"status": "DONE"}])
check("P3 E18 ladder: any failure before solving -> the next pin step (+1.0 -> +0.75 -> +0.5); 30 / 15 / 15 min ready waits",
      [x["gib"] for x in _c] == [1.0, 0.75, 0.5] and [x["ready_sec"] for x in _c] == [1800, 900, 900] and _b["status"] == "DONE", _c)
_b, _c = ladder_run([{"status": "BODY_ERROR"}, {"status": "GPU_NOT_EMPTY"}, {"status": "HOST_RAM_GATE_FAILED"}, {"status": "DONE"}])
check("P3 E18 ladder: at most 3 engine attempts (the 4th never starts; CELL 0's fallback stays)", len(_c) == 3 and _b["status"] != "DONE", _c)
_b, _c = ladder_run([{"status": "PHASE_FAILED", "phases_started": True}, {"status": "DONE"}])
check("P3 E18 ladder: never after solving started", len(_c) == 1)
_b, _c = ladder_run([{"status": "LAUNCH_FAILED"}, {"status": "DONE"}], abort_after=1)
check("P3 E18 ladder: a GPU-release failure stops the ladder (E18 'gpu_memory_not_released' fatal)", len(_c) == 1)
_b, _c = ladder_run([{"status": "NO_GO_CONTENT_PROBES_FAILED"}, {"status": "DONE"}], e18=False)
check("without e18_ladder (NB5 profiles) the NB5 rule stays: a probe failure is not a ladder step", len(_c) == 1)

# P1: E18 R2's wait_ready / kill(fast) on MServer, executed against a fake process + log
_ms_src = _src_of(code[4], "MServer")
_silence_line = 'E18_SILENCE_SEC = float(os.environ.get("E18_SILENCE_SEC", 600))'
check("P1: E18_SILENCE_SEC = 600 s (E18 R2 cell 6)", _silence_line in code[4])


class _FakeBase:
    def __init__(self, logp):
        self.log_path, self.t0, self.port, self.proc = logp, time.perf_counter(), 9, types.SimpleNamespace(
            poll=lambda: None, returncode=None, pid=1, wait=lambda timeout=None: None)

    @property
    def url(self):
        return "http://127.0.0.1:9"

    def alive(self):
        return True

    def log(self):
        return self.log_path.read_text()


_lp = Path(tempfile.mkdtemp(prefix="nb6_p1_")) / "server.log"
_lp.write_text("loading\n")
_ns1 = {"L": types.SimpleNamespace(Server=_FakeBase), "os": os, "time": time, "FAKE": "1", "E18_SILENCE_SEC": 600.0,
        "_urlreq": __import__("urllib.request").request}
exec(compile(_ms_src, "<MServer>", "exec"), _ns1)
_srv = _ns1["MServer"](_lp)
_t = time.perf_counter()
_r = _srv.wait_ready(30, silence=2.0)
check("P1 wait_ready: a launch whose log stops growing for `silence` s is a stall (E18 B-M3)",
      _r[0] is False and _r[1].startswith("stall") and time.perf_counter() - _t < 10, _r)
_srv = _ns1["MServer"](_lp)
_r = _srv.wait_ready(3, silence=600.0)
check("P1 wait_ready: no /health within the timeout -> 'timeout'", _r == (False, "timeout"), _r)
check("P1: ready needs /health AND this server's own 'GPU KV cache size' line (E18 B-L8)",
      'if "GPU KV cache size" in self.log():' in _ms_src)
check("P1 kill(fast): SIGTERM, 0.1 s, SIGKILL of the group, 30 s wait (E18 kill(fast=True))",
      "self.proc.wait(timeout=0.1)" in _ms_src and "os.killpg(self.proc.pid, 9)" in _ms_src)
check("P2: rerun after the solve -> fast kill and no release wait; commit / earlier boots keep NB5's kill + release gate",
      "s.kill(fast=True) if _nb6_final else s.kill()" in code[4]
      and "rel = True if _nb6_final else L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus)" in code[4])
check("P5: watchdog relaunch waits 20 min (E18 R2 cell 6 L266)", "srv.wait_ready(max(60.0, min(20 * 60, t)))" in code[4])


def sha_line(rerun, res):
    ns = {"RERUN": rerun, "sha_gate": lambda t: res, "left": lambda: 3600, "min": min}
    blk = code[6][code[6].index("    _ok, _why = (sha_gate("):code[6].index('    R["setup"]["sha_gate"] = {"ok": _ok, "why": _why}')]
    exec(compile(textwrap.dedent(blk), "<sha>", "exec"), ns)
    return ns["_ok"]


check("P6: rerun has no sha gate (passes even when the hash is unfinished / mismatched)", sha_line(True, (False, "sha256 mismatch")) is True)
check("P6: commit keeps NB5's sha gate", sha_line(False, (False, "sha256 check not finished")) is False)
check("sandbox: CELL 8 keeps NB5's fail-closed raise (= E15B SUBMIT: no solver without a passing self-test)",
      'if not E10_SANDBOX_SELFTEST.get("ok"):\n        raise RuntimeError("sandbox self-test failed: "' in code[6])


# F1: the RC2 input block's ARC resolution (rerun, no evaluation file -> the test file; never raises)
_ab = code[4][code[4].index('    ARC = COMP / "arc-agi_evaluation_challenges.json"\n'):code[4].index('    if not ARC.is_file():\n        raise FileNotFoundError')]


def arc_pick(rerun, files):
    d = Path(tempfile.mkdtemp(prefix="nb6_f1_"))
    comp, root = d / "comp", d / "root"
    comp.mkdir()
    root.mkdir()
    for f in files:
        (root / f).write_text("{}")
    ns = {"COMP": comp, "ARC_ROOT": root, "RERUN": rerun}
    exec(compile(textwrap.dedent(_ab), "<f1>", "exec"), ns)
    return ns["ARC"].name if ns["ARC"].is_file() else None


check("F1: rerun, no eval file anywhere -> the probe prompts come from arc-agi_test_challenges.json (ARC_ROOT)",
      arc_pick(True, ["arc-agi_test_challenges.json"]) == "arc-agi_test_challenges.json")
check("F1: rerun with the eval file under ARC_ROOT -> the eval file",
      arc_pick(True, ["arc-agi_evaluation_challenges.json", "arc-agi_test_challenges.json"]) == "arc-agi_evaluation_challenges.json")
check("F1: commit never takes the test file (eval only)", arc_pick(False, ["arc-agi_test_challenges.json"]) is None)
check("F1: the probe prompt builder (_task_text) reads only train in/out + test inputs (fits a test file)",
      'for i, ex in enumerate(t["train"]):' in code[4] and 'for ex in t["test"]:\n            parts.append(f"Test input:' in code[4])
# ---- 7. push folder kernel-metadata
PUSH = Path(os.environ.get("NB6T_PUSH") or r"D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb6-submit__foysalemonshanto")
PUSH5 = Path(os.environ.get("NB6T_PUSH5") or r"D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb5-quality__foysalemonshanto")
if PUSH.exists() and PUSH5.exists():
    _m6 = json.load(open(PUSH / "kernel-metadata.json", encoding="utf-8"))
    _m5 = json.load(open(PUSH5 / "kernel-metadata.json", encoding="utf-8"))
    _srcs = [x for k in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources") for x in _m6.get(k) or []]
    check("kernel-metadata: no empty / blank source string", all(isinstance(x, str) and x.strip() for x in _srcs), _srcs)
    check("kernel-metadata: every model / dataset / competition source is NB5 QUALITY's (same lists), image + GPU + "
          "internet off + private unchanged",
          all(_m6.get(k) == _m5.get(k) for k in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources",
                                                 "docker_image", "machine_shape", "enable_gpu", "enable_internet", "is_private")))
    check("kernel-metadata: id / code_file / the notebook in the folder = this NB6 build",
          _m6["id"] == "foysalemonshanto/q27-xhigh-e25-nb6-submit" and (PUSH / _m6["code_file"]).read_bytes() == raw)
    _paths = set(m.lower() for m in __import__("re").findall(r'models/foysalemonshanto/([a-z0-9\-]+/[a-z]+/[a-z0-9\-]+/\d+)', "\n".join(code)))
    _decl = set(m.split("/", 1)[1].lower() for m in _m6["model_sources"])
    check("every Model path the notebook reads is declared in model_sources", _paths and _paths <= _decl, (_paths - _decl, _decl))
else:
    check("push folders present for the metadata check", False, (PUSH, PUSH5))

bad = [n for n, ok in RESULTS if not ok]
print(f"\n{len(RESULTS) - len(bad)} / {len(RESULTS)} PASS" + (f" | FAIL: {bad}" if bad else ""), flush=True)
print("@@NB6_UNITS_OK@@" if not bad else "@@NB6_UNITS_FAIL@@", flush=True)
sys.exit(1 if bad else 0)
