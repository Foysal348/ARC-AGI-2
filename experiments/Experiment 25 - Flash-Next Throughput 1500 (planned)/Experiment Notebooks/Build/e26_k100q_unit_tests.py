"""E26 K100 QUALITY quick local unit tests (no whole-notebook dry run): the plan, the worker rule, the gates, the report call,
the fallback / dual mode, and the build's provenance. usage: python e26_k100q_unit_tests.py   (exit 0 = all PASS)

Env overrides (CPU kit): K100QT_NB (notebook), K100QT_NB6 (NB6 SUBMIT), K100QT_STAGE (the staged tf-ref version folder),
K100QT_NB5_OUT (NB5 QUALITY output q27x folder), K100QT_E18 (E18 R2 notebook)."""
import ast
import difflib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FACTS = json.loads((HERE / "e26_k100q_build_facts.json").read_text(encoding="utf-8"))
NB = Path(os.environ.get("K100QT_NB") or FACTS["notebook"])
NB6 = Path(os.environ.get("K100QT_NB6") or HERE.parent / "Q27-XHIGH-E25_NB6_SUBMIT.ipynb")
STAGE = Path(os.environ.get("K100QT_STAGE") or FACTS["payload"]["staged"])
NB5_OUT = Path(os.environ.get("K100QT_NB5_OUT") or r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x")
E18 = Path(os.environ.get("K100QT_E18") or r"D:/ARC AGI 2/experiments/Experiment 18 - S1-60K on vLLM 0.32 (planned)/Experiment Notebooks/Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb")
RESULTS = []


def check(name, cond, detail=""):
    try:
        ok = bool(cond() if callable(cond) else cond)
    except Exception as e:
        ok, detail = False, f"raised {type(e).__name__}: {e} | {detail}"
    RESULTS.append((name, ok))
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  [{str(detail)[:900]}]"), flush=True)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def cells_of(p):
    return [(c["cell_type"], "".join(c["source"])) for c in json.loads(p.read_text(encoding="utf-8"))["cells"]]


raw = NB.read_bytes()
CK, C6 = cells_of(NB), cells_of(NB6)
code = [t for k, t in CK if k == "code"]
print("K100 QUALITY", NB.name, sha(raw), len(raw), "bytes", flush=True)

# ---- 1. build provenance / size / verbatim cells
check("notebook < 400 kB (Kaggle commit limit)", len(raw) < 400_000, len(raw))
check("notebook sha == build facts", sha(raw) == FACTS["notebook_sha256"])
check("no CR in the notebook (LF only)", b"\r" not in raw.replace(b"\\r", b""))
check("9 cells, NB6's layout (markdown + 8 code)", [k for k, _ in CK] == [k for k, _ in C6] == ["markdown"] + ["code"] * 8)
check("NB6 SUBMIT reference is sha 87afcc84", sha(NB6.read_bytes()) == FACTS["nb6_sha256"])
check("CELL 0 == E18 R2 cell 1 (the proven dual-mode CELL 0 + identity fallback)",
      CK[1][1].replace("\r\n", "\n") == "".join(json.loads(E18.read_text(encoding="utf-8"))["cells"][1]["source"]).replace("\r\n", "\n"))
check("CELL 0, CELL 3 (Duck), task order, CELL 6/7 are NB6's verbatim; markdown / CELL 1 / engine / CELL 8 / CELL 9 changed",
      [i for i, (a, b) in enumerate(zip(C6, CK)) if a != b] == [0, 2, 5, 7, 8])


def removed(a, b):
    return [ln[2:] for ln in difflib.ndiff(a.splitlines(), b.splitlines()) if ln.startswith("- ")]


_rm = {i: removed(C6[i][1], CK[i][1]) for i in (2, 5, 7, 8)}
_rm_ok = {2: ["NB5_LIB_SRC = (lambda _t:", 'NB5_DEFAULT_PROFILE = "NB6COMMIT"',
              'SGL_MODEL_PATH = INP / "models/foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1"   # tokenizer'],
          5: ["LAB_SRC = (lambda _t:", "    MERLIN_OK = all(PARTS.values())", '    R["inputs"] = {"parts":', "    _src = CKPT if MERLIN_PROMPTS else INTEL",
              '    TOK_SHA = "0997f410', '"captures_cover_workers": NB5L.captures_cover(NB5_CAPS16 if P.get("caps16") else [32], _w)}',
              '"captures": NB5_CAPS16 if P.get("caps16") else None,', "tries = NB5L.pin_tries(NB5_KV_BYTES[NB5_BOOT_KV], start)",
              'seqs = NB5L.seqs_for(NB5_BOOT_KV, tr["gib"]) if NB5_SEQS_RULE == "table" else None',
              "srv = merlin_fn_server(old.name, NB5_BOOT_KV, old.port, NB5_KV_BYTES[NB5_BOOT_KV])"],   # audit fix D (relaunch)
          7: [], 8: ['for fp in [BASE_OUT / "nb5_report.json", SUBMISSION_PATH,',
                     '        print("deployment_comparison:", _qa["deployment_comparison"]["status"]']}
check("only the anchored NB6 lines are replaced (every other NB6 line kept)",
      all(len(_rm[i]) == len(_rm_ok[i]) and all(any(ok_ in ln for ok_ in _rm_ok[i]) for ln in _rm[i]) for i in _rm),
      {i: [ln[:120] for ln in v] for i, v in _rm.items()})
_md = json.loads((Path(FACTS["push"]) / "kernel-metadata.json").read_text(encoding="utf-8"))
check("push folder: notebook bytes == the built notebook", (Path(FACTS["push"]) / "q27-xhigh-e26-k100-quality.ipynb").read_bytes() == raw)
check("kernel-metadata: GPU, NvidiaL4, private, no internet, NB5 / klee100 LOAD docker image",
      _md["enable_gpu"] is True and _md["machine_shape"] == "NvidiaL4" and _md["is_private"] is True and _md["enable_internet"] is False
      and _md["docker_image"].endswith("37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"), _md)
check("kernel-metadata sources: the klee100 Model + FP8 PLE overlay + Merlin env, tf-ref + the Duck payload dataset",
      _md["model_sources"] == ["foysalemonshanto/qwen38-flash-next-klee100-3bpw-mtp/PyTorch/autoround-3bpw-mtp-nople/1",
                               "foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
                               "foysalemonshanto/e23-merlin-env/Other/v1/1"]
      and _md["dataset_sources"] == ["foysalemonshanto/q27-e25-nb0-tf-ref", "foysalemonshanto/q27-e15b-payload"]
      and _md["competition_sources"] == ["arc-prize-2026-arc-agi-2"], _md)
# the staged tf-ref version
_src_ds = Path(r"D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb0_tf_ref")
check("staged tf-ref version: every current file byte-identical (SHA256SUMS.json unchanged)",
      all((STAGE / f.name).read_bytes() == f.read_bytes() for f in _src_ds.iterdir() if f.is_file()))
_pay = (STAGE / FACTS["payload"]["file"]).read_bytes()
check("staged payload file sha == the sha pinned in the notebook", sha(_pay) == FACTS["payload"]["sha256"] and FACTS["payload"]["sha256"] in code[1])
import base64  # noqa: E402
import lzma  # noqa: E402
_P = json.loads(_pay)
check("payload: every blob decodes to its sha (the klee100 LOAD rev 3 K100_EMB set + worker extension + nb5_lib)",
      sorted(_P) == ["k100kit_core.py", "k100load_lib.py", "klee100_shard_digests.json", "nb1_worker_ext.py", "nb5_lib.py",
                     "tables/klee100.json", "tables/merlin.json"]
      and all(sha(lzma.decompress(base64.b85decode(b))) == s for s, b in _P.values()))
check("payload nb5_lib.py == NB5 QUALITY's lib (sha 18efcbc1) == the Build copy",
      _P["nb5_lib.py"][0] == "18efcbc1dde9c590a14ac1937b45dec563ff76ed4d92b16632d273dec2c88500"
      == sha((HERE / "nb5_lib.py").read_bytes().replace(b"\r\n", b"\n")))
check("the K3_ONLY notebook's K100_EMB blobs are carried verbatim (same strings)",
      all(_P[n][1] in (HERE.parent / "Q27-XHIGH-E25_KLEE100_LOAD_K3ONLY.ipynb").read_text(encoding="utf-8").replace("\\\"", "\"")
          for n in ("k100load_lib.py", "k100kit_core.py", "tables/klee100.json")))
check("k100q_report.py (a pinned tf-ref file) / nb5_fp8_reference.json (embedded) have the shas of the Build files",
      FACTS["embedded"]["k100q_report.py"] == sha((HERE / "k100q_report.py").read_bytes())
      and FACTS["embedded"]["nb5_fp8_reference.json"] == sha((HERE / "nb5_fp8_reference.json").read_bytes()))

# ---- 2. CELL 0 + CELL 1 executed for real (commit, rerun, guard, missing payload)
arc = Path(tempfile.mkdtemp(prefix="k100q_arc_"))
FAKE_TEST = {"aaaa0001": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[0, 1], [1, 0]]}]},
             "aaaa0002": {"train": [], "test": [{"input": [[3]]}, {"input": [[4, 4]]}]}}
json.dump(FAKE_TEST, open(arc / "arc-agi_test_challenges.json", "w"))
_CLR = ("E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN", "Q27X_PROFILE", "NB5_ATTEMPT_B0", "NB5_ATTEMPT_B1", "E12_MAX_GEN",
        "NB5_NB0_FP8_GATE", "E12_B_EXTRA_ATTEMPTS", "E12_B_SKIP_B1_IF_ACCEPTED", "E12_B_B1_ACC_MODE", "LOCAL_ANALYZER_SEED",
        "NB5_ADMIT_MODE", "Q27X_TIME_BUDGET", "NB5_KV_DTYPE", "NB5_COMMIT_MODE", "NB5_EXTRA_ARGS_JSON", "NB5_BASE_ENV_JSON",
        "K100_PAYLOAD_DIR", "E12_TOKENIZER_JSON")


def run_cells(env, payload_dir=STAGE):
    work = Path(tempfile.mkdtemp(prefix="k100q_w_"))
    old = dict(os.environ)
    for k in _CLR:
        os.environ.pop(k, None)
    os.environ.update(Q27X_ARC_ROOT=str(arc), Q27X_WORKING=str(work), E23_TMP=str(work), E23_INPUT=str(work / "input"),
                      K100_PAYLOAD_DIR=str(payload_dir), **env)
    G = {"__name__": "__main__"}
    try:
        for c in code[:2]:
            exec(compile(c, "<cell>", "exec"), G)
        G["_ENV"] = dict(os.environ)
    finally:
        os.environ.clear()
        os.environ.update(old)
        for m in ("nb5_lib", "k100q_report"):
            sys.modules.pop(m, None)
    G["_WORK"] = work
    return G


GC = run_cells({})
GR = run_cells({"KAGGLE_IS_COMPETITION_RERUN": "1"})
T, P = GC["PROFILE_TABLE"], GC["P"]
R2 = T["NB5QUALITY_R2"]
check("commit: PROFILE == K100QUALITY (baked default), RERUN False", GC["PROFILE"] == "K100QUALITY" and GC["RERUN"] is False)
check("rerun: PROFILE == RERUN_PROD (NB6 E1), RERUN True", GR["PROFILE"] == "RERUN_PROD" and GR["RERUN"] is True)
for _k in ("nb5_gate", "nb5_attempt", "nb5_env", "quality_ab", "equal_windows", "no_pure_decode", "admit_mode",
           "f_rule", "pin_ladder", "patch_60490", "lever_env", "continue_after_failed_boot", "source"):
    check(f"K100QUALITY.{_k} == NB5QUALITY_R2's", P.get(_k) == R2.get(_k), (P.get(_k), R2.get(_k)))
check("K100QUALITY: one phase Q_K100_FP8_K3 = R2's Q_FP8_K3 apart from the name (all tasks, MTP on, variant A, FP8)",
      P["nb5_phases"] == [["Q_K100_FP8_K3", "all", None, "on", "A", "fp8"]] and R2["nb5_phases"][0][1:] == P["nb5_phases"][0][1:])
check("K100QUALITY: fixed 6,600-s window, 150-min budget (user cap), klee100 boot levers (seqs k100, captures k100, no CAPS16)",
      P["fixed_window_sec"] == 6600 and P["time_budget_sec"] == 150 * 60 and P["seqs_rule"] == "k100" and P["k100_caps"] is True
      and P["caps16"] is False and P["k100"] is True)
_nb6rp = dict(T["RERUN_PROD"])
check("RERUN_PROD = NB6's RERUN_PROD + only the klee100 boot levers (717-min budget, test source, e18 ladder, one EVAL phase)",
      {k: v for k, v in _nb6rp.items() if k not in ("seqs_rule", "caps16", "k100_caps", "k100")} ==
      {k: v for k, v in GC["NB6_LEVERS"].items() if k not in ("seqs_rule", "caps16")} | {k: GC["_NB5_T"][k] for k in GC["_NB5_T"]
                                                                                            if k != "time_budget_sec"}
      | {"time_budget_sec": 12.0 * 3600 - 3 * 60, "source": "test", "n_tasks": None}, sorted(_nb6rp))
NEW15 = ["3dc255db", "7491f3cf", "d59b0160", "35ab12c3", "e8686506", "7666fa5d", "898e7135", "edb79dae", "2d0172a1", "8e5c0c38",
         "136b0064", "db0c5428", "800d221b", "a395ee82", "20270e3b"]
check("30 tasks: NB5 QUALITY's 15 first in NB5's exact order (= R2's task_list / task_order), then the 15 new in the user's order",
      P["task_list"] == P["task_order"] == R2["task_list"] + NEW15 and R2["task_list"] == R2["task_order"] == GC["NB5_QAB_TASKS"][:15]
      and len(set(P["task_list"])) == 30 and P["task_list"][:3] == ["1818057f", "45a5af55", "bf45cf4b"])
check("groups: NB5's solid / sometimes / never for the first 15 + hard_long (10) / sometimes_new (5) for the new 15",
      [GC["NB5_QAB_GROUP_OF"][x] for x in NEW15] == ["hard_long"] * 10 + ["sometimes_new"] * 5
      and all(GC["NB5_QAB_GROUP_OF"][x] in ("solid", "sometimes", "never") for x in R2["task_list"])
      and list(GC["NB5_QAB_GROUPS"]) == ["solid", "sometimes", "never", "hard_long", "sometimes_new"])
check("the 27B R3K2 + ARC Prize reference is embedded and equals the Build file (30 tasks)",
      GC["K100Q_REF27"] == json.loads((HERE / "k100q_ref27b.json").read_text(encoding="utf-8")) and len(GC["K100Q_REF27"]["tasks"]) == 30)
_sc = FACTS["solutions_check"]
check("solutions: all 30 tasks are in the eval solutions file (38 test queries), the commit run's SOLUTIONS covers them",
      _sc["present"] == _sc["tasks"] == 30 and _sc["queries"] == 38, _sc)
check("hard stop plan <= budget = 150 min; window 110 min", FACTS["hard_stop_plan"]["sum_min"] <= FACTS["budget_min"] == 150
      and FACTS["hard_stop_plan"]["window_min"] == 110.0, FACTS["hard_stop_plan"])
# ---- the 150-min guarantee (user 2026-11-01): budget 150, HARD_STOP 145, the commit's absolute deadman at 150
_c1 = code[1]
_c1_lines = [ln for ln in _c1.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
check("deadman: the FIRST statements of CELL 1 (right after CELL 0's fallback submission) define and arm it, before PROFILE",
      _c1_lines[0] == "K100_DEADMAN_SEC = 9000" and _c1.index('K100_DEADMAN["thread"].start()') < _c1.index("PROFILE = ")
      and "daemon=True" in _c1[:_c1.index('K100_DEADMAN["thread"].start()')])
_dmt = GC["K100_DEADMAN"].get("thread")
check("deadman (commit): armed, a daemon thread alive, fire time = NOTEBOOK_T0 + 150 min, marker in /kaggle/working",
      GC["K100_DEADMAN"]["armed"] is True and _dmt is not None and _dmt.daemon and _dmt.is_alive()
      and GC["K100_DEADMAN"]["fire_pc"] == GC["NOTEBOOK_T0"] + 150 * 60
      and GC["K100_DEADMAN"]["marker"] == str(GC["WORKING"] / "k100q_deadman.txt"), {k: v for k, v in GC["K100_DEADMAN"].items() if k != "thread"})
check("deadman (rerun): NOT armed (RERUN_PROD keeps its own 717-min budget and behaviour, no 150-min deadman)",
      GR["K100_DEADMAN"]["armed"] is False and "thread" not in GR["K100_DEADMAN"] and GR["P"]["time_budget_sec"] == 12.0 * 3600 - 3 * 60)
check("budget: K100QUALITY 150 min, hard margin 5 -> HARD_STOP = T0 + 145 min, solve end cap = HARD_STOP - 180 s = 142 min",
      GC["P"]["time_budget_sec"] == 150 * 60 and GC["P"]["hard_margin_sec"] == 5 * 60
      and abs((GC["HARD_STOP"] - GC["NOTEBOOK_T0"]) - 145 * 60) < 1e-6
      and abs((GC["E21_HALF_END_PC"] - GC["NOTEBOOK_T0"]) - 142 * 60) < 1e-6
      and GC["NOTEBOOK_T0"] < GC["ADMIT_DEADLINE"] < GC["DRAIN_DEADLINE"] < GC["HARD_STOP"] < GC["VERIFY_DEADLINE"] < GC["K100_DEADMAN"]["fire_pc"] + 1e-6)
_hp = FACTS["hard_stop_plan"]
_wb = _hp["window_by_failed_rungs"]
check("budget plan: setup 15 + boot 10 + probes 2 + window 110 + stop 3 + hard margin 5 = 145 <= 150; window by failed rungs "
      "(budgeted 6,600 / 6,120 / 5,400 s; v2-measured 6,600 / 6,600 / 6,000 s)",
      _hp["sum_min"] == 145.0 and FACTS["budget_min"] == 150 and _hp["hard_stop_min"] == 145 and _hp["deadman_min"] == 150
      and [_wb[f"budgeted_{n}_failed"]["window_sec"] for n in (0, 1, 2)] == [6600, 6120, 5400]
      and [_wb[f"measured_v2_{n}_failed"]["window_sec"] for n in (0, 1, 2)] == [6600, 6600, 6000], _wb)

_DM_SRC = ast.get_source_segment(_c1, next(x for x in ast.parse(_c1).body if isinstance(x, ast.FunctionDef) and x.name == "k100_deadman"))


def _dm_fire(use_psutil):
    """A real fire in a child python: a dummy child process (its own session / group) + the deadman due in 1.5 s while the
    main thread sleeps 60 s. Expect: exit 0 within seconds, the marker on stdout + in the file, the dummy child dead."""
    d = Path(tempfile.mkdtemp(prefix="k100q_dm_"))
    scr = d / "fire.py"
    scr.write_text("\n".join([
        "import os, sys, time, threading, subprocess",
        "from pathlib import Path",
        "NOTEBOOK_T0 = time.perf_counter()",
        "K100_DEADMAN = {'pgids': []}",
        _DM_SRC,
        "kw = {'start_new_session': True} if os.name == 'posix' else {}",
        "ch = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], **kw)",
        "K100_DEADMAN['pgids'].append(ch.pid)",
        f"Path(r'{d / 'child.pid'}').write_text(str(ch.pid))",
        f"threading.Thread(target=k100_deadman, args=(time.perf_counter() + 1.5, r'{d / 'k100q_deadman.txt'}', 0.2, {use_psutil}), daemon=True).start()",
        "time.sleep(60)",
        "print('NOT_FIRED', flush=True)",
        "sys.exit(3)"]) + "\n", encoding="utf-8")
    t0 = __import__("time").time()
    cp = __import__("subprocess").run([sys.executable, str(scr)], capture_output=True, text=True, timeout=50)
    sec = __import__("time").time() - t0
    pid = int((d / "child.pid").read_text())
    __import__("time").sleep(1.0)
    try:
        import psutil as _psu
        alive = _psu.pid_exists(pid) and _psu.Process(pid).status() != _psu.STATUS_ZOMBIE
    except Exception:
        alive = None
    out = {"rc": cp.returncode, "sec": round(sec, 1), "stdout": cp.stdout[-200:], "marker_file": (d / "k100q_deadman.txt").read_text()
           if (d / "k100q_deadman.txt").exists() else None, "child_alive": alive}
    shutil.rmtree(d, ignore_errors=True)
    return out


_f1 = _dm_fire(True)
check("deadman fire (psutil): exit 0 within seconds, @@K100Q_DEADMAN_FIRED@@ on stdout + in the marker file, the child killed",
      _f1["rc"] == 0 and _f1["sec"] < 20 and "@@K100Q_DEADMAN_FIRED@@" in _f1["stdout"] and "NOT_FIRED" not in _f1["stdout"]
      and "@@K100Q_DEADMAN_FIRED@@" in (_f1["marker_file"] or "") and _f1["child_alive"] is False, _f1)
if hasattr(os, "killpg"):
    _f2 = _dm_fire(False)
    check("deadman fire (no psutil): os.killpg on the recorded group (+ pkill -9 -f vllm), exit 0, marker, the child killed",
          _f2["rc"] == 0 and "@@K100Q_DEADMAN_FIRED@@" in (_f2["marker_file"] or "") and _f2["child_alive"] is False, _f2)
else:
    print("SKIP deadman killpg path: no os.killpg on this OS (the Linux CPU kit runs it)", flush=True)

check("seed: LOCAL_ANALYZER_SEED 20261005 reaches the Duck env (commit)", GC["_ENV"].get("LOCAL_ANALYZER_SEED") == "20261005")
check("attempts 2 x 104k, no stage 3, the 60k floor holds", GC["E15B"]["E12_B_ATTEMPT_TOKENS"] == GC["E15B"]["E12_B_ATTEMPT_TOKENS_B1"] == 104000
      and GC["_ENV"].get("E12_B_EXTRA_ATTEMPTS") == "0" and GC["NB5_THINK_FLOOR"] == 60000)
_ea = GC["NB5_EXTRA_ARGS"]
_cc = json.loads(_ea[_ea.index("--compilation-config") + 1])
check("launch flags: captures = Merlin capture_sizes_for(21, 3) to 84, --language-model-only, the worker extension",
      _cc == {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 84]}
      and _ea.count("--compilation-config") == 1 and _ea[-3:] == ["--language-model-only", "--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt"], _ea)
check("server env: VLLM_SERVER_DEV_MODE=1 (/collective_rpc), applied after build() via NB5_BASE_ENV", GC["NB5_BASE_ENV"] == {"VLLM_SERVER_DEV_MODE": "1"})
check("Duck B3 tokenizer = the klee100 mount's tokenizer.json (the W4 p1 Model is not attached)",
      GC["_ENV"]["E12_TOKENIZER_JSON"].replace("\\", "/").endswith("qwen38-flash-next-klee100-3bpw-mtp/pytorch/autoround-3bpw-mtp-nople/1/tokenizer.json"))
check("nb5_lib from the payload == NB5's (sha checked at load)", GC["CONFIG"]["nb5"]["lib_sha256"] == "18efcbc1dde9c590a14ac1937b45dec563ff76ed4d92b16632d273dec2c88500")
check("k100q_report imported, the reference loaded", hasattr(GC["K100Q"], "build_quality_report") and isinstance(GC["K100Q_REF"], dict))
check("payload record: the file used + its sha", GC["K100Q_PAYLOAD_USED"].get("sha256") == FACTS["payload"]["sha256"])


def _raises(fn, msg):
    try:
        fn()
    except BaseException as e:
        return msg in str(e)
    return False


check("guard: Q27X_PROFILE=NB6COMMIT is refused (this notebook boots klee100 only)",
      _raises(lambda: run_cells({"Q27X_PROFILE": "NB6COMMIT"}), "boots klee100 only"))
_empty = Path(tempfile.mkdtemp(prefix="k100q_nopay_"))
check("a missing payload file raises K100Q_PAYLOAD (no silent run)", _raises(lambda: run_cells({}, payload_dir=_empty), "K100Q_PAYLOAD missing"))
_sub = json.loads((GR["_WORK"] / "submission.json").read_text(encoding="utf-8"))
check("rerun fallback: CELL 0 wrote the identity submission for every hidden task / test input",
      set(_sub) == set(FAKE_TEST) and all(len(_sub[t]) == len(FAKE_TEST[t]["test"]) for t in FAKE_TEST)
      and all(r["attempt_1"] == q["input"] == r["attempt_2"] for t in FAKE_TEST for r, q in zip(_sub[t], FAKE_TEST[t]["test"])), _sub)

# ---- 3. pin ladder + worker rule
NB5L = GC["NB5L"]
_tr = NB5L.pin_tries(GC["NB5_KV_BYTES"]["fp8"], None, steps=GC["K100_PIN_STEPS_GIB"])
check("K3 ladder: 7,933,526,016 -> 6,859,784,192 -> 5,270,142,976 -> 3,553,580,524 -> 2,479,838,700 B (nb5_lib.pin_tries)",
      [t["kv_bytes"] for t in _tr] == [7933526016, 6859784192, 5270142976, 3553580524, 2479838700], _tr)
check("ladder boot names pin508 / pin408 / pin260 / pin100 / pin0", [f"pin{int(round(t['gib'] * 100))}" for t in _tr]
      == ["pin508", "pin408", "pin260", "pin100", "pin0"])
check("#60490 not verified -> only Merlin's FP8 pin (NB5's fail-safe start 0.0)",
      [t["kv_bytes"] for t in NB5L.pin_tries(GC["NB5_KV_BYTES"]["fp8"], 0.0, steps=GC["K100_PIN_STEPS_GIB"])] == [2479838700])
check("workers rule reproduces NB5 QUALITY's FP8 phase (pool 214, F 6 -> 9)", NB5L.workers_rule(214, 4096, 6) == 9)
check("workers rule on the klee100 v1 boot (318 usable blocks, F 6 from 8.37x) -> 14", NB5L.workers_rule(318, 4096, 6) == 14
      and round(318 / 8.37 - 32, 1) == 6.0)
check("expected workers per rung 21 / 18 / 14 / 9 / 6; captures to 84 cover each (NB5's CAPS16 would not cover 21)",
      [e["workers_est"] for e in GC["K100_EXPECTED"]] == [21, 18, 14, 9, 6] and all(e["captures_cover"] for e in GC["K100_EXPECTED"])
      and NB5L.captures_cover(GC["K100_CAPS"], 21) and not NB5L.captures_cover(GC["NB5_CAPS16"], 21))
check("launch --max-num-seqs per rung == the expected workers (NB5 R2 SEQS_TABLE rule: max_num_seqs = workers)",
      [GC["K100_SEQS_TABLE"][round(t["gib"], 2)] for t in _tr] == [21, 18, 14, 9, 6]
      and all(GC["K100_SEQS_TABLE"][e["gib_step"]] == e["workers_est"] for e in GC["K100_EXPECTED"]))
_v2 = FACTS.get("k3_v2_gpu")
check("klee100 K3_ONLY GPU v2 (first rung 7,933,526,016 B): pool 479, F 6 -> rule 21 == the rung-1 launch seqs (build cross-check)",
      _v2 is None or (_v2["pool_blocks"] == 479 and _v2["F"] == 6 and NB5L.workers_rule(479, 4096, 6) == 21
                      and GC["K100_SEQS_TABLE"][round(_tr[0]["gib"], 2)] == 21 and _v2["method_map"] == "GPU_METHOD_MAP_PASS"), _v2)
check("not hard-coded: workers come from this boot's measured pool / F (nb5_capacity); K100_EXPECTED is a record only (never in the engine)",
      "K100_EXPECTED" not in code[4] and "_wr = NB5L.workers_rule(pool, block, plan[\"f_eff\"])" in code[4]
      and "1652197" not in "".join(code) and "1,652,197" not in "".join(code))
for g in (GC, GR):
    shutil.rmtree(g["_WORK"], ignore_errors=True)

# ---- 4. gates (k100_gate_verdict / k100_finish executed from the notebook source)
ENG = code[4]
_tree = ast.parse(ENG)


def fn_src(src, name):
    n = [x for x in ast.parse(src).body if isinstance(x, ast.FunctionDef) and x.name == name]
    assert len(n) == 1, name
    return ast.get_source_segment(src, n[0])


NS = {}
exec(fn_src(ENG, "k100_gate_verdict"), NS)
gv = NS["k100_gate_verdict"]
check("gate: method map gate_ok + scales ok -> pass", gv({"gate_ok": True, "verdict": "GPU_METHOD_MAP_PASS"}, {"ok": True}, "fp8")["fatal"] is None)
check("gate: METHOD_CLASS_ONLY (gate_ok True) passes (klee100 LOAD rev 3)", gv({"gate_ok": True, "verdict": "METHOD_CLASS_ONLY"}, {"ok": True}, "fp8")["ok"])
check("gate: a method-map mismatch -> fatal GPU_METHOD_MAP_PASS", gv({"gate_ok": False, "why": "x"}, {"ok": True}, "fp8")["fatal"] == "GPU_METHOD_MAP_PASS")
check("gate: scales not applied on every rank -> fatal STATIC_SCALES_APPLIED",
      gv({"gate_ok": True}, {"ok": False, "why": "ranks"}, "fp8")["fatal"] == "STATIC_SCALES_APPLIED")
check("gate: a fake server is recorded, never gated (unless K100_FAKE_GATES)",
      gv({"gate_ok": False}, None, "fp8", fake=True)["fatal"] is None and gv({"gate_ok": False}, None, "fp8", fake=True, fake_gates=True)["fatal"])
_bt = fn_src(ENG, "nb5_boot_try")
check("boot try: the klee100 gates run after the C-11 read-back and before the probes; fatal -> NO_GO_K100_* outside the rerun",
      _bt.index("config_readback(") < _bt.index("k100_boot_gates(s, mtp_on)") < _bt.index("nb5_probes(s)")
      and 'trec["status"] = "NO_GO_K100_"' in _bt and 'get("fatal") and not RERUN' in _bt)
_run = code[6]
check("runner: k100_setup() after #60490 and before the first boot; the fixed window preset",
      _run.index("nb5_apply_60490()") < _run.index("k100_setup()") < _run.index("nb5_boot(") and 'NB5_EQ.update(window_sec=float(P["fixed_window_sec"])' in _run)
check("runner: NB6 P6 kept (no sha gate in the rerun)", "rerun: no sha gate (E18 R2 / E15B SUBMIT have none)" in _run)
_ks = fn_src(ENG, "k100_setup")
check("k100_setup: OVERLAY_MERGE_K6 gate -> worker extension -> scales sidecar, each fatal", all(
    s in _ks for s in ("K100_OVERLAY_MERGE_K6_FAILED", "k100_worker_ext()", "k100_scales(", "K100_NO_SCALES")))
_mf = [x for x in _tree.body if isinstance(x, ast.FunctionDef) and x.name == "merlin_fn_server"]
check("merlin_fn_server rebound once (the klee100 wrapper after the NB5 one): K6 before launch, FP8 scales env, --help gate",
      len(_mf) == 2 and all(s in ast.get_source_segment(ENG, _mf[1]) for s in ("KL.k6_check(CKPT)", "VLLM_QSA_KV_SCALES", "FAIL_CLI_FLAGS")))
check("CKPT / merge_ckpt rebound by klee100 LOAD CELL 2b (fn_klee100, fixkeys overlay, K6, digests)",
      'CKPT = TMP / "fn_klee100"' in ENG and "def k100_overlay():" in ENG and "KL.verify_digests(plan" in ENG
      and ENG.index('CKPT = TMP / "fn_merlin"') < ENG.index('CKPT = TMP / "fn_klee100"'))
import k100q_report as _K100Q  # noqa: E402
NS2 = {"P": {"task_list": ["t1", "t2"]}, "Path": Path, "json": json, "K100Q": _K100Q}
exec(fn_src(ENG, "k100_finish"), NS2)
_pd = Path(tempfile.mkdtemp(prefix="k100q_fin_"))
for _d, _rows in (("t1_b0", [{"t0": 1000.0, "t1": 1100.0}, {"t0": 1100.0, "t1": 1150.5}]), ("t1_b1", [{"t0": 1000.0, "t1": 1300.0}])):
    (_pd / "duck" / _d).mkdir(parents=True)
    (_pd / "duck" / _d / "nb5_requests.jsonl").write_text("\n".join(json.dumps(r) for r in _rows) + "\n", encoding="utf-8")
_fin = NS2["k100_finish"]({"t1": {"finished": True, "solved": True, "attempts": [{"attempt": 1, "end_offset_sec": 4049.7, "outcome": "ACCEPTED"},
                                                                                {"attempt": 2, "end_offset_sec": 4049.8, "outcome": "BUDGET_END"}]},
                           "t2": {"finished": False, "attempts": [{"attempt": 1, "end_offset_sec": None, "outcome": "INCOMPLETE_WINDOW"}]}},
                          {"actual_sec": 4050}, _pd, 1000.0, {"t1": {"first_accept_sec": 120.0}})
check("finish: per-attempt end = the last request's t1 - window start (not the record's window-end t1); task = latest attempt",
      _fin["per_task"]["t1"]["attempts"][0]["end_offset_sec"] == 150.5 and _fin["per_task"]["t1"]["finish_offset_sec"] == 300.0
      and _fin["per_task"]["t1"]["attempts"][0]["requests"] == 2 and _fin["per_task"]["t1"]["first_accept_sec"] == 120.0
      and _fin["per_task"]["t2"]["finish_offset_sec"] is None and _fin["all_finished"] is False and _fin["tasks_finished"] == 1
      and _fin["last_finish_offset_sec"] == 300.0, _fin)
shutil.rmtree(_pd, ignore_errors=True)

# ---- audit fix D (2026-11-01): a mid-run relaunch boots exactly the try that came up (kv bytes, seqs + captures, levers)
_bt = fn_src(ENG, "nb5_boot_try")
_END = 'if k_ == "NB4_PPDEFER"})\n'
_set = _bt[_bt.index("        NB5_ACTIVE_BOOT.clear()"):_bt.index(_END) + len(_END)]
_set = "\n".join(l[8:] for l in _set.splitlines()) + "\n"
_STUBS = '''
class _Old:
    name, port = "K100_pin508", 8000
    def __init__(self, lp):
        self.log_path = lp
    def kill(self):
        pass
class _New:
    def __init__(self, a):
        self.args, self.url, self.env = a, "http://x", {}
    def start(self):
        pass
    def wait_ready(self, t):
        return True, "ready"
def merlin_fn_server(name, kv, port, kv_bytes):
    CALLS.append({"name": name, "kv": kv, "port": port, "kv_bytes": kv_bytes, "extra": list(NB5_BOOT_EXTRA),
                  "levers": dict(NB5_BOOT_LEVERS)})
    return _New(CALLS[-1])
def e21_snapshot_counters():
    pass
def nb5_host_gate(n):
    return {"ok": True}
def sampled_check(s):
    return {"ok": True}
class L:
    @staticmethod
    def wait_gpu_release(*a, **k):
        return True
'''


def _relaunch(relaunch_src, booted, qd=True):
    """booted = (kv_bytes, seqs) of the try that came up; afterwards the globals drift (a stale NB5_BOOT_EXTRA without the
    seqs, no levers), as when a later code path touched them. Returns merlin_fn_server's call."""
    lp = Path(tempfile.mkdtemp(prefix="k100q_rl_")) / "srv.log"
    lp.write_text("log\n", encoding="utf-8")
    ns = {"time": __import__("time"), "shutil": shutil, "json": json, "Path": Path, "CALLS": [], "FAKE": False,
          "E15B_SOLVE_END_PC": float("inf"), "NB5_RELAUNCH_MIN_SEC": 0, "NB5_REPORT": {}, "E12_RUN": {},
          "NB5_KV_BYTES": {"fp8": 2479838700}, "NB5_BOOT_KV": "fp8", "NB5_ACTIVE_BOOT": {}}
    exec(_STUBS, ns)
    caps = ["--compilation-config", '{"cudagraph_capture_sizes":[1,2,4,84]}']
    ns.update(name="K100_pin%d" % int(round(booted[2] * 100)), kv_bytes=booted[0],
              NB5_BOOT_EXTRA=caps + ["--max-num-seqs", str(booted[1])], NB5_BOOT_LEVERS={"MERLIN_SHARE_DRAFT_WORKSPACE": "1"},
              trec={"max_num_seqs_launch": booted[1], "pin_gib": booted[2]},
              s=type("S", (), {"env": {"NB4_PPDEFER": "1", "X": "y"} if qd else {"X": "y"}})())   # the server that came up
    exec(_set, ns)                                           # the block nb5_boot_try runs when this try's server is ready
    ns.update(NB5_BOOT_EXTRA=list(caps), NB5_BOOT_LEVERS={})  # drift
    ns["E11_SERVER"] = ns["_Old"](lp)
    ns["NB5_SERVERS"] = [ns["E11_SERVER"]]
    exec(relaunch_src, ns)
    ok, why = ns["e21_relaunch"]()
    shutil.rmtree(lp.parent, ignore_errors=True)
    return ns["CALLS"][-1] if ns["CALLS"] else None, ok, ns


_old_rl = fn_src(C6[5][1], "e21_relaunch")
_c0, _, _ = _relaunch(_old_rl, (7933526016, 21, 5.08))
check("fix D, failure reproduced: NB6's e21_relaunch after a 7.39-GiB / 21-seq boot relaunches at the base pin 2,479,838,700 B",
      _c0 is not None and _c0["kv_bytes"] == 2479838700, _c0)
_c1, _ok1, _ns1 = _relaunch(fn_src(ENG, "e21_relaunch"), (7933526016, 21, 5.08))
check("fix D: relaunch after the 7.39-GiB rung = exactly 7,933,526,016 B, --max-num-seqs 21, the captures, the lever env",
      _ok1 and _c1["kv_bytes"] == 7933526016 and _c1["extra"][-2:] == ["--max-num-seqs", "21"] and "--compilation-config" in _c1["extra"]
      and _c1["levers"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"} and _c1["kv"] == "fp8"
      and _ns1["NB5_REPORT"]["relaunch_boot"][-1]["kv_bytes"] == 7933526016, _c1)
_c2, _ok2, _ = _relaunch(fn_src(ENG, "e21_relaunch"), (5270142976, 14, 2.6))
check("fix D: relaunch after a lower rung (4.91 GiB / 14 seqs) = that rung's 5,270,142,976 B and seqs 14",
      _ok2 and _c2["kv_bytes"] == 5270142976 and _c2["extra"][-2:] == ["--max-num-seqs", "14"], _c2)
check("Q-defer: a relaunch after a Q-defer boot keeps NB4_PPDEFER=1 (NB5_ACTIVE_BOOT env_keep); without Q-defer none is added",
      _ns1["NB5_SERVERS"][0].env == {"NB4_PPDEFER": "1"} and _relaunch(fn_src(ENG, "e21_relaunch"), (7933526016, 21, 5.08), qd=False)[2]
      ["NB5_SERVERS"][0].env == {}, _ns1["NB5_SERVERS"][0].env)
check("fix D: NB5_ACTIVE_BOOT is set inside nb5_boot_try after the server is ready (before the gates / phases)",
      _bt.index('trec["ready_sec"]') < _bt.index("NB5_ACTIVE_BOOT.update(") < _bt.index("config_readback"))

# ---- Q-defer (user 2026-11-01): NB4S's nb4_ppack.py rev 2 from the tf-ref version, fail-safe apply, env only when applied
import importlib.util as _ilu  # noqa: E402
_PPK_FILE = STAGE / "nb4_ppack.py"
check("Q-defer: the staged tf-ref version carries nb4_ppack.py = the Build file = NB4S's tested sha 8dadf076, pinned in CELL 1",
      _PPK_FILE.is_file() and sha(_PPK_FILE.read_bytes()) == sha((HERE / "nb4_ppack.py").read_bytes())
      == "8dadf0763d43d22966603745198ad327ed6ed36001cc60a79925ea18818f2dea" and sha(_PPK_FILE.read_bytes()) in code[1])
check("the staged tf-ref version carries k100q_report.py = the Build file, its sha pinned in CELL 1 (not embedded any more)",
      sha((STAGE / "k100q_report.py").read_bytes()) == sha((HERE / "k100q_report.py").read_bytes()) == FACTS["embedded"]["k100q_report.py"]
      and FACTS["embedded"]["k100q_report.py"] in code[1] and "K100Q_REPORT_SRC = k100q_ds_text(" in code[1])
_spec = _ilu.spec_from_file_location("nb4_ppack_staged", _PPK_FILE)
PPKS = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(PPKS)
_QLOG = []
GC_ENG_CHAIN = ("vllm/v1/worker/gpu/pp_utils.py", "vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu_worker.py")
check("audit G: K100_QD_CHAIN in the engine cell = [pp_utils, model_runner, gpu_worker] (restored in reverse)",
      ast.literal_eval(next(l for l in ENG.splitlines() if l.startswith("K100_QD_CHAIN = ")).split(" = ", 1)[1]) == GC_ENG_CHAIN and "for rel in reversed(K100_QD_CHAIN):" in fn_src(ENG, "k100_qd_restore"))
check("audit G: the old revert-refusal path is gone (no revert_ppack_patch in k100_qdefer_apply; snapshot + k100_qd_restore)",
      "revert_ppack_patch" not in fn_src(ENG, "k100_qdefer_apply") and "k100_qd_restore(tree, snap, py)" in fn_src(ENG, "k100_qdefer_apply"))


def _qd_ns(ppk=PPKS, fake=False, os_mod=os):
    n = {"K100_QD": {"applied": False, "env": {}, "status": None}, "PPK": ppk, "PPK_ERROR": None if ppk else "missing", "FAKE": fake,
         "os": os_mod, "hashlib": hashlib, "json": json,
         "Path": Path, "log": lambda *a: _QLOG.append(" ".join(str(x) for x in a)),
         "merlin_install": lambda: (_ for _ in ()).throw(AssertionError("merlin_install must not run here"))}
    exec("K100_QD_CHAIN = " + repr(GC_ENG_CHAIN), n)
    for _fn in ("k100_qd_consistent", "k100_qd_restore", "k100_qdefer_apply"):
        exec(fn_src(ENG, _fn), n)
    exec(fn_src(ENG, "k100_qd_readback"), n)
    return n


_ks2 = fn_src(ENG, "k100_setup")
check("Q-defer: k100_setup applies it after the install / worker extension, before the scales and any boot, inside a try",
      _ks2.index("k100_worker_ext()") < _ks2.index("k100_qdefer_apply()") < _ks2.index("k100_scales(")
      and "    try:                                                     # Q-defer: the apply never raises" in _ks2)
_q0 = _qd_ns(None)
_r0 = _q0["k100_qdefer_apply"](Path("/nonexistent"), sys.executable)
check("Q-defer: no lib -> NO_LIB, applied False, no env, no exception", _r0["status"] == "NO_LIB" and _q0["K100_QD"]["env"] == {}
      and _q0["K100_QD"]["applied"] is False, _r0)
_qf = _qd_ns(PPKS, fake=True)
check("Q-defer: the local fake server (no tree) -> FAKE_SKIPPED, no env", _qf["k100_qdefer_apply"]()["status"] == "FAKE_SKIPPED"
      and _qf["K100_QD"]["env"] == {})
MERLIN_SRC = Path(os.environ.get("K100QT_MERLIN_SRC") or r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")


def _mk_tree(dst):
    for rel in PPKS.PATCHES:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(MERLIN_SRC / rel, dst / rel)
    return dst


if all((MERLIN_SRC / rel).is_file() for rel in PPKS.PATCHES):
    _qw = Path(tempfile.mkdtemp(prefix="k100q_qd_"))
    _t1 = _mk_tree(_qw / "ok")
    _q1 = _qd_ns()
    _r1 = _q1["k100_qdefer_apply"](_t1, sys.executable)
    check("Q-defer: the real Merlin source copy (pre-shas) -> APPLIED, post-shas + markers verified, env NB4_PPDEFER=1",
          _r1["status"] == "APPLIED" and _q1["K100_QD"]["applied"] is True and _q1["K100_QD"]["env"] == {"NB4_PPDEFER": "1"}
          and PPKS.verify_ppack_patch(_t1)["state"] == "applied", {k: _r1.get(k) for k in ("status", "error")})
    _rv = PPKS.revert_ppack_patch(_t1, sys.executable)
    check("Q-defer: the patched copy reverts to Merlin's exact pre-shas", _rv["ok"] and PPKS.verify_ppack_patch(_t1)["state"] == "not_applied")
    _t2 = _mk_tree(_qw / "bad")
    _f2 = _t2 / PPKS.PP_REL
    _f2.write_bytes(_f2.read_bytes().replace(b"self.queue.append(None)", b"self.queue.append(None) ", 1))
    _b2 = {rel: sha((_t2 / rel).read_bytes()) for rel in PPKS.PATCHES}
    _q2 = _qd_ns()
    _r2 = _q2["k100_qdefer_apply"](_t2, sys.executable)
    check("Q-defer failure path: a modified tree -> APPLY_FAILED (PRE_SHA_MISMATCH), nothing written, no env, the run continues",
          _r2["status"] == "APPLY_FAILED_RESTORED" and "PRE_SHA_MISMATCH" in str(_r2.get("error")) and _q2["K100_QD"]["env"] == {}
          and _b2 == {rel: sha((_t2 / rel).read_bytes()) for rel in PPKS.PATCHES} and any("NOT APPLIED" in x for x in _QLOG), _r2)
    _pre = {rel: PPKS.PATCHES[rel][0] for rel in GC_ENG_CHAIN}
    _post = {rel: PPKS.PATCHES[rel][1] for rel in GC_ENG_CHAIN}

    def _shas(tr):
        return {rel: sha((tr / rel).read_bytes()) for rel in GC_ENG_CHAIN}
    # (1) a failure on the 2nd apply write (pp_utils written, model_runner not): all three back to the original shas, no env
    _wa = PPKS._write_atomic
    _calls = []

    def _wa2(f, data):
        _calls.append(f.name)
        if len(_calls) == 2:
            raise OSError("injected: 2nd apply write")
        return _wa(f, data)
    _t4 = _mk_tree(_qw / "w2")
    PPKS._write_atomic = _wa2
    try:
        _q4 = _qd_ns()
        _r4 = _q4["k100_qdefer_apply"](_t4, sys.executable)
    finally:
        PPKS._write_atomic = _wa
    check("audit G: a failure on the 2nd apply write -> pp_utils restored, all three at Merlin's original shas, consistent, "
          "py_compile ok, no env, APPLY_FAILED_RESTORED",
          _calls[:2] == ["pp_utils.py", "model_runner.py"] and _r4["status"] == "APPLY_FAILED_RESTORED" and _shas(_t4) == _pre
          and _r4["restore"]["consistent"] is True and _r4["restore"]["files"]["vllm/v1/worker/gpu/pp_utils.py"] == "restored"
          and _r4["restore"]["py_compile"]["rc"] == 0 and _q4["K100_QD"]["env"] == {},
          {k: _r4.get(k) for k in ("status", "restore", "error")})
    # (2) a failure on each restore write: all three patched, the post-verify forced to fail, os.replace fails on one file
    for _bad, _left in (("gpu_worker.py", GC_ENG_CHAIN), ("model_runner.py", GC_ENG_CHAIN[:2]), ("pp_utils.py", GC_ENG_CHAIN[:1])):
        _t5 = _mk_tree(_qw / ("r_" + _bad))

        def _rep(src, dst, _b=_bad):
            if Path(dst).name == _b:
                raise OSError("injected: restore write " + _b)
            return os.replace(src, dst)
        import types as _types  # noqa: E402
        class _PPKV:                                   # the post-verify seen by k100_qdefer_apply fails (the module itself is intact)
            def __getattr__(self, n):
                return getattr(PPKS, n)

            @staticmethod
            def verify_ppack_patch(tr):
                return {"state": "foreign", "files": {}}
        _q5 = _qd_ns(ppk=_PPKV(), os_mod=_types.SimpleNamespace(replace=_rep))
        _r5 = _q5["k100_qdefer_apply"](_t5, sys.executable)
        _want = {rel: (_post[rel] if rel in _left else _pre[rel]) for rel in GC_ENG_CHAIN}
        check(f"audit G: restore write of {_bad} fails -> the restore stops there, patched set = {[r.split('/')[-1] for r in _left]} "
              "(a prefix: consistent), no env, no exception",
              _r5["status"] == "APPLY_FAILED_RESTORE_INCOMPLETE_CONSISTENT" and _r5["restore"]["consistent"] is True
              and _shas(_t5) == _want and _q5["K100_QD"]["env"] == {} and _q5["K100_QD"]["applied"] is False,
              {k: _r5.get(k) for k in ("status", "restore")})
    # (3) the old mixed-tree input (pp_utils already patched): the apply refuses, the snapshot is the input, nothing changes
    _t3 = _mk_tree(_qw / "mixed")
    _t3p = _mk_tree(_qw / "mixed_src")
    PPKS.apply_ppack_patch(_t3p, sys.executable)
    (_t3 / PPKS.PP_REL).write_bytes((_t3p / PPKS.PP_REL).read_bytes())
    _b3 = _shas(_t3)
    _q3 = _qd_ns()
    _r3 = _q3["k100_qdefer_apply"](_t3, sys.executable)
    check("audit G: a mixed input tree (pp_utils patched) -> refused, left as found (a consistent prefix), no env, no exception",
          _r3["status"] == "APPLY_FAILED_RESTORED" and _shas(_t3) == _b3 and _r3["restore"]["consistent"] is True
          and _q3["K100_QD"]["env"] == {}, {k: _r3.get(k) for k in ("status", "restore")})
    check("audit G: k100_qd_consistent flags a non-prefix set (model_runner patched over an original pp_utils) as inconsistent",
          _q3["k100_qd_consistent"](_mk_tree(_qw / "np_check"))["consistent"] is True and (lambda tr: (
              (tr / PPKS.MR_REL).write_bytes((_t3p / PPKS.MR_REL).read_bytes()), _q3["k100_qd_consistent"](tr))[1]["consistent"])(
              _mk_tree(_qw / "nonprefix")) is False)
    # ---- audit I (2026-11-01): the tree check before any boot (commit: NO_GO on an unverified tree; rerun: recorded, no env)
    _QD_SHAS = {rel: [PPKS.PATCHES[rel][0], PPKS.PATCHES[rel][1]] for rel in GC_ENG_CHAIN}
    check("audit I: K100_QD_SHAS in the engine cell = nb4_ppack.PATCHES (pre / post per file)",
          json.loads(next(l for l in ENG.splitlines() if l.startswith("K100_QD_SHAS = ")).split(" = ", 1)[1].split("   #")[0]) == _QD_SHAS)
    _ks3 = fn_src(ENG, "k100_setup")
    _gt = _ks3[_ks3.index("out[\"qdefer_tree\"] = k100_qd_gate()") - 200:_ks3.index("out[\"qdefer_tree\"] = k100_qd_gate()")]
    check("audit I: k100_setup calls k100_qd_gate after the apply, in a try / finally without except (a NO_GO is never swallowed)",
          _ks3.index("k100_qdefer_apply()") < _ks3.index("k100_qd_gate()") < _ks3.index("k100_scales(")
          and "    try:                                                     # audit I: the tree check, OUTSIDE the try above" in _gt)

    class _PPKV2:                                      # the post-verify seen by k100_qdefer_apply fails (the module itself is intact)
        def __getattr__(self, n):
            return getattr(PPKS, n)

        @staticmethod
        def verify_ppack_patch(tr):
            return {"state": "foreign", "files": {}}

    def _rep_fail(name):
        def _r(src, dst):
            if Path(dst).name == name:
                raise OSError("injected: restore write " + name)
            return os.replace(src, dst)
        return __import__("types").SimpleNamespace(replace=_r)

    def _icase(kind, rerun):
        """One fixture: apply in its own tree, then the gate in commit (rerun False) or rerun mode."""
        tr = _mk_tree(_qw / f"i_{kind}_{int(rerun)}")
        py, ppk, osm = sys.executable, PPKS, os
        if kind == "nonprefix":                         # model_runner patched over an original pp_utils (input as found)
            src = _mk_tree(_qw / f"i_src_{kind}_{int(rerun)}")
            PPKS.apply_ppack_patch(src, sys.executable)
            (tr / PPKS.MR_REL).write_bytes((src / PPKS.MR_REL).read_bytes())
        if kind in ("restored", "incomplete"):
            ppk = _PPKV2()
        if kind == "incomplete":
            osm = _rep_fail("gpu_worker.py")
        if kind == "pycompile":
            py = str(_qw / "no_such_python")
        n = _qd_ns(ppk=ppk, os_mod=osm)
        n["RERUN"] = rerun
        exec("K100_QD_SHAS = " + json.dumps(_QD_SHAS), n)
        for _fn in ("k100_qd_tree_check", "k100_qd_gate"):
            exec(fn_src(ENG, _fn), n)
        wa = PPKS._write_atomic
        calls = []
        if kind == "write2":
            def _w(f, data):
                calls.append(f.name)
                if len(calls) == 2:
                    raise OSError("injected: 2nd apply write")
                return wa(f, data)
            PPKS._write_atomic = _w
        try:
            n["k100_qdefer_apply"](tr, py)
        finally:
            PPKS._write_atomic = wa
        try:
            chk, err = n["k100_qd_gate"](tr), None
        except RuntimeError as e:
            chk, err = n["K100_QD"].get("tree_check"), str(e)
        return {"status": n["K100_QD"]["status"], "env": dict(n["K100_QD"]["env"]), "applied": n["K100_QD"]["applied"],
                "verified": (chk or {}).get("verified"), "state": (chk or {}).get("state"), "raised": err,
                "recorded": n["K100_QD"].get("tree_check") is not None}

    _ICASES = (("full", "continue WITH Q-defer", "continue WITH Q-defer"),
               ("restored", "continue without", "continue without"),
               ("write2", "continue without", "continue without"),
               ("incomplete", "NO_GO", "continue without, recorded"),
               ("nonprefix", "NO_GO", "continue without, recorded"),
               ("pycompile", "NO_GO", "continue without, recorded"))
    for _kind, _want_c, _want_r in _ICASES:
        for _rerun, _want in ((False, _want_c), (True, _want_r)):
            _o = _icase(_kind, _rerun)
            if _want == "continue WITH Q-defer":
                _ok = _o["raised"] is None and _o["verified"] and _o["env"] == {"NB4_PPDEFER": "1"} and _o["state"] == "applied"
            elif _want == "continue without":
                _ok = _o["raised"] is None and _o["verified"] and _o["env"] == {} and _o["state"] == "not_applied"
            elif _want == "NO_GO":
                _ok = bool(_o["raised"]) and _o["raised"].startswith("NO_GO_K100_QDEFER_TREE_UNVERIFIED") and _o["env"] == {} \
                    and _o["verified"] is False and _o["recorded"]
            else:
                _ok = _o["raised"] is None and _o["verified"] is False and _o["env"] == {} and _o["recorded"] and not _o["applied"]
            check(f"audit I [{_kind}, {'rerun' if _rerun else 'commit'}]: {_want}", _ok, _o)
    shutil.rmtree(_qw, ignore_errors=True)
else:
    print("SKIP Q-defer apply / revert on the real Merlin source: not at", MERLIN_SRC, flush=True)


class _QSrv:
    def __init__(self, env=None, text=""):
        self.env, self._t = dict(env or {}), text

    def log(self):
        return self._t


_wns = {"KL": type("KL", (), {"k6_check": staticmethod(lambda c: {"ok": True}), "cli_flags_check": staticmethod(lambda h: {"ok": True})}),
        "K1": {}, "HELP": {}, "save": lambda: None, "CKPT": Path("/ck"), "str": str,
        "_merlin_fn_server_nb5": lambda name, kv, port, b: _QSrv({"NB4_PPDEFER": "1", "NB4_PPACK": "1", "X": "y"})}
exec(ast.get_source_segment(ENG, _mf[1]), _wns)
_wns["K100_QD"] = {"applied": False, "env": {}}
_s0 = _wns["merlin_fn_server"]("K100_pin508", "fp8", 8000, 7933526016)
_wns["K100_QD"] = {"applied": True, "env": {"NB4_PPDEFER": "1"}}
_s1 = _wns["merlin_fn_server"]("K100_pin508", "fp8", 8000, 7933526016)
check("Q-defer env: not applied -> NB4_PPDEFER / NB4_PPACK removed even if inherited; applied -> NB4_PPDEFER=1 only (every launch)",
      "NB4_PPDEFER" not in _s0.env and "NB4_PPACK" not in _s0.env and _s0.env.get("X") == "y"
      and _s1.env.get("NB4_PPDEFER") == "1" and "NB4_PPACK" not in _s1.env and _wns["K100_QD"]["launch_env"] == {"K100_pin508": "1"},
      (_s0.env, _s1.env))


def _ppq_log(exec_n=965, park=1000):
    ln = []
    for pid, last in ((11, False), (12, False), (13, True), (14, True)):
        ln.append("NB4_PPQ " + json.dumps({"event": "init", "pid": pid, "is_last_rank": last, "ppack": False, "ppdefer": not last}))
    for pid in (11, 12):
        st = {"defer_park": park, "flush_exec": exec_n, "flush_empty": park - exec_n, "flush_recv": 0, "flush_consume": 0}
        ln.append("NB4_PPQ " + json.dumps({"event": "stats", "pid": pid, "key": "defer_park", "stats": st}))
        ln.append("NB4_PPQ " + json.dumps({"event": "stats", "pid": pid, "key": "flush_exec", "stats": st}))
    return "\n".join(ln) + "\n"


_qr = _qd_ns()
_qr["K100_QD"].update(applied=True, env={"NB4_PPDEFER": "1"})
_rb1 = _qr["k100_qd_readback"](_QSrv(text=_ppq_log()))
_rb2 = _qr["k100_qd_readback"](_QSrv(text="no NB4_PPQ lines"))
_rb3 = _qr["k100_qd_readback"](_QSrv(text=_ppq_log(exec_n=500)))
_qr["K100_QD"].update(applied=False)
_rb4 = _qr["k100_qd_readback"](_QSrv(text=_ppq_log()))
check("Q-defer read-back recorded (never fatal): ENGAGED + exec share 0.965 / NOT_INTEGRATED / ENGAGED_FALLBACK / NOT_APPLIED",
      (_rb1["label"], _rb1["exec_share_min"], _rb2["label"], _rb3["label"], _rb4["label"])
      == ("ENGAGED", 0.965, "NOT_INTEGRATED", "ENGAGED_FALLBACK", "NOT_APPLIED"), (_rb1, _rb2, _rb3, _rb4))
check("Q-defer read-back is taken at the boot (init lines) and after the phases, recorded in the try",
      'trec["qdefer_readback_boot"] = k100_qd_readback(s)' in _bt and 'trec["qdefer_readback"] = k100_qd_readback(E11_SERVER or s)' in _bt)

# ---- the admission / window end (kit agent's question, 2026-11-01): ADMIT_DEADLINE / DRAIN_DEADLINE are only defined
_all = "\n".join(code)
_uses = [ln.strip()[:80] for ln in _all.splitlines() if ("ADMIT_DEADLINE" in ln or "DRAIN_DEADLINE" in ln)
         and not ln.lstrip().startswith(("ADMIT_DEADLINE =", "DRAIN_DEADLINE =", "assert NOTEBOOK_T0 < ADMIT_DEADLINE", "#"))]
check("admission: ADMIT_DEADLINE / DRAIN_DEADLINE are never read after their definition (no 0.90 x budget admit cut)", _uses == [], _uses)
check("admission: the only admit cut is the branch deadline (= the solve window end) - E12_R6_MIN_ADMIT_SEC (300 s)",
      'self.deadline - float(E12_R6["E12_R6_MIN_ADMIT_SEC"])' in _all and '"E12_R6_MIN_ADMIT_SEC": 300' in _all)

# ---- 5. the report block on NB5 QUALITY's real FP8 phase (renamed to the klee100 phase): build_quality_report as called
REP = code[7]
_blk = REP[REP.index("# ---- E26 K100: the klee100 quality report"):REP.index('try:\n    _pack = WORKING / "Q27X_NB5_BUNDLE.zip"')]
check("report call passes order = the 30-task list, group_of = the extended groups, ref27 (review blocker fix)",
      'order=list(P["task_list"]), group_of=dict(NB5_QAB_GROUP_OF), ref27=K100Q_REF27' in _blk)
check("CELL 9: NB5's two-phase deployment_comparison line is not printed for the single phase (relabelled not applicable)",
      'if not P.get("fixed_window_sec"):' in REP and "deployment_comparison: not applicable (single phase" in REP)
if (NB5_OUT / "nb5_report.json").exists():
    import k100q_report as K100Q
    import k100q_report_tests as KRT
    nb5 = json.loads((NB5_OUT / "nb5_report.json").read_text(encoding="utf-8"))
    base = Path(tempfile.mkdtemp(prefix="k100q_rep_"))
    shutil.copytree(NB5_OUT / "phase_Q_FP8_K3", base / "phase_Q_K100_FP8_K3")
    rep = json.loads(json.dumps(nb5))
    rep["phases"] = {"Q_K100_FP8_K3": nb5["phases"]["Q_FP8_K3"]}
    rep["quality_ab"]["per_config"] = {"Q_K100_FP8_K3": nb5["quality_ab"]["per_config"]["Q_FP8_K3"]}

    _b0 = json.loads((NB5_OUT / "phase_Q_FP8_K3" / "duck" / "1818057f_b0" / "branch_result.json").read_text(encoding="utf-8"))
    _ep0 = float(_b0["t0"]) - nb5["phases"]["Q_FP8_K3"]["window"]["start_sec"] if "window" in nb5["phases"]["Q_FP8_K3"] else         float(_b0["t0"]) - nb5["quality_ab"]["per_config"]["Q_FP8_K3"]["window"]["start_sec"]      # the notebook start epoch (approx.)

    class _Tr:
        def summary(self):
            return nb5["phases"]["Q_FP8_K3"].get("tracer") or {}
    ns = {"NB5_REPORT": rep, "BASE_OUT": base, "NB5_BOOTS": nb5["boots"][:1], "RERUN": False, "K100Q": K100Q,
          "K100Q_REF": json.loads((HERE / "nb5_fp8_reference.json").read_text(encoding="utf-8")),
          "SOLUTIONS": KRT.SOL, "P": T["K100QUALITY"], "E12_RECORDS": None, "NB5_TRACE": _Tr(), "NB5_CAPACITY": {},
          "K100_EXPECTED": [], "json": json, "traceback": traceback, "time": __import__("time"), "Path": Path,
          "NB5_PIN": {"patch_60490": {"ok": True}}, "NB5_QAB_GROUP_OF": GC["NB5_QAB_GROUP_OF"], "K100Q_REF27": GC["K100Q_REF27"],
          "K100_QD": {"applied": False, "env": {}, "status": "FAKE_SKIPPED"},
          "NOTEBOOK_T0": __import__("time").perf_counter() - (__import__("time").time() - _ep0),
          "_load_json": lambda p: json.loads(Path(p).read_text(encoding="utf-8")),
          "write_json_atomic": lambda p, o: Path(p).write_text(json.dumps(o, default=str), encoding="utf-8")}
    exec(fn_src(ENG, "k100_finish"), ns)
    exec(compile(_blk, "<k100 report block>", "exec"), ns)
    out = ns["K100Q_OUT"]
    q = out.get("quality") or {}
    check("report block (audit F): the 30-task profile with gold for NB5's 15 only -> INVALID_MISSING_GOLD naming the 15 new, "
          "written to q27x/k100_quality_report.json", out.get("status") == "INVALID_MISSING_GOLD" and out.get("missing_gold") == sorted(NEW15)
          and (base / "k100_quality_report.json").exists(), out.get("error") or out.get("status"))
    check("report block: the same data as the reference -> paired both 13, k100-only 0, w4-only 0, errors = MISSING_GOLD for the new 15 only",
          (q.get("paired") or {}).get("counts", {}).get("both") == 13 and q["paired"]["counts"].get("k100_only") == 0
          and q["paired"]["counts"].get("w4_only") == 0 and q.get("errors") == [f"MISSING_GOLD:{x}" for x in NEW15], {"counts": (q.get("paired") or {}).get("counts"), "errors": q.get("errors")})
    check("report block: the engine facts reach the report (pool 214, workers 9)",
          json.dumps(q.get("k100"), default=str).count('"pool_blocks": 214') >= 1 and '"workers": 9' in json.dumps(q.get("k100"), default=str))
    check("report block: per-task finish table for the 15 tasks the run had + the markdown tables", out["finish"]["tasks"] == 15
          and bool(out.get("table_md")) and "Attempt 1 only" in out["table_md"] and "ARC Prize xhigh" in out["table_md"])
    check("report block (review blocker): order = the 30 tasks -> the k100 record covers 30 (15 never started here), no task outside order",
          len(q["k100"]["tasks"]) == 30 and q["inputs"]["tasks_outside_order"] == [] and q["run_summary"]["tasks"] == 30
          and q["k100"]["summary"]["hard_long"]["tasks"] == 10 and q["k100"]["summary"]["sometimes_new"]["tasks"] == 5)
    check("report block (review blocker): paired / delta / attempt 1 vs W4 stay on NB5's 15 (no fake k100_only / w4 unsolved)",
          q["scope"]["paired_delta_attempt1_tasks"] == 15 and q["paired"]["counts"]["tasks"] == 15 and q["delta"]["tasks"]["k100"] == 15
          and q["attempt1_vs_w4"]["counts"]["tasks"] == 15 and q["attempt1_vs_w4"]["counts"]["k100_only"] == 0
          and q["scope"]["tasks_not_in_reference"] == NEW15, q.get("scope"))
    check("report block: the 27B R3K2 / ARC Prize columns for all 30 (hard_long: R3K2 0 / 10, ARC Prize 10 / 10)",
          len(q["vs_27b"]["tasks"]) == 30 and q["vs_27b"]["by_group"]["hard_long"]["solved_r3k2"] == 0
          and q["vs_27b"]["by_group"]["hard_long"]["solved_arcprize_xhigh"] == 10, q["vs_27b"].get("by_group"))
    check("report block: solutions check names the tasks without solutions (here the test's 15-task solutions file)",
          out["solutions_check"]["with_solutions"] == 15 and out["solutions_check"]["missing"] == NEW15, out.get("solutions_check"))
    check("report block: NB5's 4,049.5-s window under the 6,600-s plan is flagged WINDOW SHORTENED (the flag works on real data)",
          [f[:16] for f in out.get("flags") or []] == ["WINDOW SHORTENED"], out.get("flags"))
    ns4 = dict(ns, P=dict(T["K100QUALITY"], fixed_window_sec=4050))
    exec(compile(_blk, "<k100 report block>", "exec"), ns4)
    check("report block: no flag when #60490 is verified and the window ran full (planned 4,050 s here)", ns4["K100Q_OUT"].get("flags") == [],
          ns4["K100Q_OUT"].get("flags"))
    ns2 = dict(ns, NB5_PIN={"patch_60490": {"ok": False}})
    exec(compile(_blk, "<k100 report block>", "exec"), ns2)
    check("report block: #60490 not verified -> a loud flag in the report notes (the ~6-worker rung only)",
          any("#60490 NOT VERIFIED" in f for f in ns2["K100Q_OUT"].get("flags") or [])
          and any("#60490 NOT VERIFIED" in ln for ln in ns2["K100Q_OUT"]["quality"]["notes"]["lines"]))
    _rep3 = json.loads(json.dumps(rep))
    _rep3["phases"]["Q_K100_FP8_K3"]["window_plan"] = {"planned_sec": 6600, "unequal": True, "short_by_sec": 900.0}
    ns3 = dict(ns, NB5_REPORT=_rep3)
    exec(compile(_blk, "<k100 report block>", "exec"), ns3)
    check("report block: a shortened window -> a WINDOW SHORTENED flag", any("WINDOW SHORTENED" in f for f in ns3["K100Q_OUT"].get("flags") or []),
          ns3["K100Q_OUT"].get("flags"))
    _f = out["finish"]["per_task"]
    check("report block: real finish times from the request traces (1818057f early, all inside the 4,050-s window, 12 / 15 finished)",
          0 < _f["1818057f"]["finish_offset_sec"] < 1000 and all(v["finish_offset_sec"] is None or v["finish_offset_sec"] <= 4060 for v in _f.values())
          and out["finish"]["tasks_finished"] == 12 and out["finish"]["all_finished"] is False,
          {t: v["finish_offset_sec"] for t, v in _f.items()})
    # ---- audit E (2026-11-01): the matched 4,050-s cutoff on the real data (k100 = NB5's own FP8 phase: same window)
    _m = q.get("paired_at_matched_4050s") or {}
    check("audit E: matched 4,050-s cutoff from k100_finish's request timings -> OK, the same 13 / 15 as W4, no discordant",
          _m.get("status") == "OK" and _m.get("offsets_source") == "k100_finish" and _m["counts_at_cutoff"]["both"] == 13
          and _m["counts_at_cutoff"]["discordant"] == 0, {k: _m.get(k) for k in ("status", "offsets_source", "counts_at_cutoff", "undecided_tasks")})
    check("audit J: on NB5's real records the matched cutoff uses Duck's accept time (t + verify_sec) for every exact attempt",
          _m.get("time_basis") == "ACCEPT_TIME" and all(b == "ACCEPT_TIME" for r in _m["tasks"].values()
                                                        for b in r["k100_exact_attempt_time_basis"].values()),
          {k: _m.get(k) for k in ("status", "time_basis")})
    check("audit E: the full-window pairing carries its window basis (here the same 4,049.5-s window)",
          q["paired"].get("window_basis") == "PAIRED_TASK_OUTCOME_SAME_WINDOW", q["paired"].get("window_basis"))
    check("audit E: the attempt-1 table shows each klee100 attempt 1's finish offset (1818057f early)",
          0 < q["attempt1_vs_w4"]["tasks"]["1818057f"]["k100"]["finish_offset_sec"] < 1000 and "K100 a1 end (s)" in out["table_md"])
    _es = out.get("engine_summary") or {}
    check("audit E: the summary has preemptions, fresh-prefill tokens, tasks finished per hour (12 in 4,049.5 s = 10.67 / h)",
          _es.get("preemptions") == 0 and _es.get("fresh_prefill_tokens") and _es.get("tasks_finished_per_hour") == round(12 * 3600 / 4049.5, 2)
          and _es.get("matched_4050s_status") == "OK" and q["notes"].get("engine_summary") == _es, _es)
    check("Q-defer report: not applied -> qdefer applied False, comparison 'klee100 vs W4' in the summary and the notes",
          out["qdefer"]["applied"] is False and out["engine_summary"]["comparison"].startswith("COMPARISON: klee100 vs W4")
          and any(ln.startswith("COMPARISON: klee100 vs W4") for ln in q["notes"]["lines"]), out.get("qdefer"))
    # ---- audit fixes A / B (2026-11-01): the status is BUILT only on a DONE phase with a clean report
    _P15 = dict(T["K100QUALITY"], task_list=list(R2["task_list"]), task_order=list(R2["task_list"]))   # the run this data is
    ns15 = dict(ns, P=_P15)
    exec(compile(_blk, "<k100 report block>", "exec"), ns15)
    _q15 = ns15["K100Q_OUT"].get("quality") or {}
    check("report block: the real DONE phase, consistent sources, gold for every task (15-task profile) -> VALID, no errors, units MATCH",
          ns15["K100Q_OUT"].get("status") == "VALID" and _q15.get("errors") == []
          and _q15["inputs"]["score_check"]["unit_check"]["status"] == "MATCH"
          and _q15["inputs"]["score_check"]["units_source"] == "generated_attempts",
          {"status": ns15["K100Q_OUT"].get("status"), "errors": _q15.get("errors")})
    check("report block: the CELL 9 summary carries preemptions + fresh-prefill tokens (prompt - prefix hits)",
          set(out.get("engine_summary") or {}) >= {"preemptions", "fresh_prefill_tokens", "prompt_tokens", "prefix_hit_tokens"}
          and out["engine_summary"]["fresh_prefill_tokens"] == out["engine_summary"]["prompt_tokens"] - out["engine_summary"]["prefix_hit_tokens"],
          out.get("engine_summary"))
    _repB = json.loads(json.dumps(rep))
    _repB["phases"]["Q_K100_FP8_K3"].update(status="PHASE_FAILED", solver_error="RuntimeError: server died")
    nsB = dict(ns, NB5_REPORT=_repB)
    exec(compile(_blk, "<k100 report block>", "exec"), nsB)
    check("fix B: a phase that is not DONE / has a solver error -> INCOMPLETE_PHASE + phase_error (the report is still built)",
          nsB["K100Q_OUT"].get("status") == "INCOMPLETE_PHASE" and "server died" in str(nsB["K100Q_OUT"].get("phase_error"))
          and bool(nsB["K100Q_OUT"].get("quality")), {k: nsB["K100Q_OUT"].get(k) for k in ("status", "phase_error")})
    _bq = json.loads(json.dumps(nb5["boots"][:1]))
    _bq[-1].setdefault("tries", [{}])[-1]["qdefer_readback"] = {"label": "ENGAGED", "exec_share_min": 0.965}
    nsQ = dict(ns15, NB5_BOOTS=_bq, K100_QD={"applied": True, "env": {"NB4_PPDEFER": "1"}, "status": "APPLIED",
                                             "launch_env": {"K100_pin508": "1"}})
    exec(compile(_blk, "<k100 report block>", "exec"), nsQ)
    _oq = nsQ["K100Q_OUT"]
    check("Q-defer report: applied -> the read-back (ENGAGED, exec share 0.965) in the K100 SUMMARY + notes, labelled "
          "'klee100 + Q-defer vs W4' (NB5 ran without it); the status stays VALID",
          _oq["engine_summary"]["qdefer"] == {"applied": True, "status": "APPLIED", "readback_label": "ENGAGED", "exec_share_min": 0.965}
          and _oq["engine_summary"]["comparison"].startswith("COMPARISON: klee100 + Q-defer vs W4")
          and any("read-back ENGAGED" in ln for ln in _oq["quality"]["notes"]["lines"]) and _oq["status"] == "VALID",
          {k: _oq.get(k) for k in ("status", "qdefer")})
    _baseA = Path(tempfile.mkdtemp(prefix="k100q_repA_"))
    shutil.copytree(base / "phase_Q_K100_FP8_K3", _baseA / "phase_Q_K100_FP8_K3")
    _rrA = json.loads((_baseA / "phase_Q_K100_FP8_K3" / "e12_run_report.json").read_text(encoding="utf-8"))
    _rrA["eval_query_hits"] = 999
    (_baseA / "phase_Q_K100_FP8_K3" / "e12_run_report.json").write_text(json.dumps(_rrA), encoding="utf-8")
    nsA = dict(ns15, BASE_OUT=_baseA)
    exec(compile(_blk, "<k100 report block>", "exec"), nsA)
    _qA = nsA["K100Q_OUT"].get("quality") or {}
    check("fix A: run report 999 hits vs the generated attempts -> INCONCLUSIVE_SCORE_MISMATCH, SCORE_SOURCE_MISMATCH, no score",
          nsA["K100Q_OUT"].get("status") == "INCONCLUSIVE_SCORE_MISMATCH" and any("SCORE_SOURCE_MISMATCH" in e for e in _qA.get("errors") or [])
          and _qA["k100"]["score"] is None and _qA["inputs"]["score_check"]["agree"] is False,
          {"status": nsA["K100Q_OUT"].get("status"), "errors": _qA.get("errors"), "score": (_qA.get("k100") or {}).get("score")})

    class _K100QErr:                                       # any other report error -> INCONCLUSIVE_REPORT_ERROR
        def __getattr__(self, n):
            return getattr(K100Q, n)

        @staticmethod
        def build_quality_report(*a, **k):
            r = K100Q.build_quality_report(*a, **k)
            r["errors"] = list(r.get("errors") or []) + ["delta: KeyError: x"]
            return r
    nsE = dict(ns15, K100Q=_K100QErr())
    exec(compile(_blk, "<k100 report block>", "exec"), nsE)
    check("fix A / F: a non-empty quality['errors'] -> INVALID_QUALITY_REPORT", nsE["K100Q_OUT"].get("status") == "INVALID_QUALITY_REPORT",
          nsE["K100Q_OUT"].get("status"))
    shutil.rmtree(_baseA, ignore_errors=True)
    shutil.rmtree(base, ignore_errors=True)
else:
    print("SKIP report block: NB5 QUALITY output not on this machine", flush=True)

n_bad = sum(1 for _, ok in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_bad}/{len(RESULTS)} PASS", flush=True)
shutil.rmtree(arc, ignore_errors=True)
sys.exit(1 if n_bad else 0)
