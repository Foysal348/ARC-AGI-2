"""CPU harness for the E15P notebook (Poetiq S60 on the E15B engine, build_e15p.py). Runs ALL 10 code cells of the real
notebook in order against e15p_fake_server.py (the E14 fake + the E11-05 Poetiq streaming behaviours; no GPU, no install),
with the setup of harness_e15b_nb.py (staging, payload dataset shape, fake SGLang package for the E14 patches). Only the
sandbox verifier is faked (the E11-05 harness's MARK verifier).
usage: python harness_e15p_nb.py <notebook> [scale]
env E15P_MODE: commit (default profile E15PEVAL120: all 120 eval tasks, scored) | kvgate (5 tasks, the server reports
KV 0.97: the gate must hold 10 running) | drain (8 tasks; 2 never close: drained, answered, verified; then the hard stop)
| rerun (KAGGLE_IS_COMPETITION_RERUN=1: every test task)
rev 2 (Notes/19): mambagate (20 tasks, the server reports mamba 0.97 with KV low: the gate must hold 10) | tokfail (the
tokenizer check fails: S60's vLLM FP8 fallback cell runs Poetiq on the E11-05 fake vllm) | smokefail (the 16-stream server
ignores the smoke's abort: one 12-stream relaunch, Poetiq at 12) | death (the 16-stream server crashes after 30 solve
requests: the running units are requeued, SGLang relaunches at 12 streams, the run completes). Unit tests of the shim's
take_pending(), the admission metric and the gate run in every SGLang mode."""
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

NB = sys.argv[1]
MODE = os.environ.get("E15P_MODE", "commit")
SCALE = sys.argv[2] if len(sys.argv) > 2 else "0.03"
HERE = Path(__file__).resolve().parent
E14H = Path(os.environ.get("E14_HARNESS_DIR", r"D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/src/harness"))
E14SRC = E14H.parent
FIX = Path(os.environ.get("E14_FIXTURES_DIR", r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures"))
SGL_SRC = Path(os.environ.get("E14_SGL_SRC", r"D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang"))
sys.path.insert(0, str(FIX))
import e12_fixtures as B  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
if os.name == "nt":   # the E12 harness's Windows-only shims (65 KB bootstrap as -c; no killpg)
    _real = subprocess.Popen
    _bdir = Path(tempfile.gettempdir()) / "e14_harness_bootstrap"
    _bdir.mkdir(parents=True, exist_ok=True)

    class _Shim(_real):
        def __init__(self, args, *a, **kw):
            if isinstance(args, list) and len(args) >= 5 and args[1:4] == ["-I", "-S", "-c"]:
                f = _bdir / "sandbox_bootstrap.py"
                if not f.exists() or f.read_text(encoding="utf-8") != args[4]:
                    f.write_text(args[4], encoding="utf-8")
                args = [args[0], "-I", "-S", str(f)] + args[5:]
            super().__init__(args, *a, **kw)
    subprocess.Popen = _Shim
    if not hasattr(os, "killpg"):
        import signal as _sig
        os.killpg = lambda pid, sig: os.kill(pid, 9)
        if not hasattr(_sig, "SIGKILL"):
            _sig.SIGKILL = 9

ROOT = Path(os.environ.get("E14_HARNESS_ROOT", r"D:/ARC AGI 2/_scratch/e15p/harness_runs"))
ROOT.mkdir(parents=True, exist_ok=True)
WORK = ROOT / f"{MODE}_{time.strftime('%H%M%S')}"
WORK.mkdir()
ch, sol, multi, single = B.load()
multi = sorted(t for t in ch if len(ch[t]["test"]) >= 2)[:3]
single = sorted(t for t in ch if len(ch[t]["test"]) == 1)[:5]
PICK = multi + single
# the E11-05 behaviours on the picked tasks (the rest of the 120: by a hash of the id, pass / fail_distinct / forced)
BEH = {multi[0]: "partial_then_fail", multi[1]: "forced_then_pass", multi[2]: "fail_distinct",
       single[0]: "pass", single[1]: "pass", single[2]: "fail_same", single[3]: "pass", single[4]: "pass"}
if MODE == "drain":
    BEH[single[0]] = BEH[single[3]] = "slow_pass"
FP8 = B.TOKDIR_FP8
MODELS = WORK / "inputs" / "models" / "foysalemonshanto"
INT4 = MODELS / "qwen3-8-27b-int4-redhat" / "pytorch" / "w4a16-g128" / "1"
shutil.copytree(B.TOKDIR_INT4, INT4)
hdr = json.dumps({"w": {"dtype": "U8", "shape": [3 << 20], "data_offsets": [0, 3 << 20]}}).encode()
hdr += b" " * (-len(hdr) % 8)
(INT4 / "model.safetensors").write_bytes(struct.pack("<Q", len(hdr)) + hdr + os.urandom(3 << 20))
DFL = MODELS / "qwen3-8-27b-dflash2" / "pytorch" / "bf16" / "1"
DFL.mkdir(parents=True)
json.dump({"architectures": ["DFlash2DraftModel"]}, open(DFL / "config.json", "w"))
(DFL / "model.safetensors").write_bytes((INT4 / "model.safetensors").read_bytes()[:8 + len(hdr)] + b"\0" * (3 << 20))
WH = WORK / "inputs" / "wheelhouse"
WH.mkdir(parents=True)
json.dump({"sglang": "0.5.21"}, open(WH / "BUILD_INFO.json", "w"))
(WH / "requirements.lock").write_text("", encoding="utf-8")
(WH / "SHA256SUMS").write_text("", encoding="utf-8")
ARC = WORK / "arc"
ARC.mkdir()
shutil.copy(B.DATA / "arc-agi_evaluation_challenges.json", ARC / "arc-agi_evaluation_challenges.json")
shutil.copy(B.DATA / "arc-agi_evaluation_solutions.json", ARC / "arc-agi_evaluation_solutions.json")
TEST = {t: {"train": ch[t]["train"], "test": [{"input": q["input"]} for q in ch[t]["test"]]} for t in PICK}
json.dump(TEST, open(ARC / "arc-agi_test_challenges.json", "w"))
sys.path.insert(0, str(E14SRC))
import e14_sglang_patches as E14P  # noqa: E402
PKG = WORK / "fake_sglang" / "sglang"
for name, hunks in E14P.PATCHES.items():
    for h in hunks:
        dst = PKG / h["file"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(SGL_SRC / h["file"], dst)
DEST = WORK / "shm"
DEST.mkdir()
PAYLOAD = WORK / "inputs"
_pd = PAYLOAD / "datasets" / "esfersami50" / "q27-e15b-payload"
_pd.mkdir(parents=True)
for _f in (Path(NB).resolve().parent / "Build" / "payload").glob("q27x_e14_modules_*.bin"):   # E15P needs only this one
    shutil.copy(_f, _pd / _f.name)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    PORT = s.getsockname()[1]
os.environ.update(Q27X_ARC_ROOT=str(ARC), Q27X_WORKING=str(WORK), Q27X_MODEL=str(FP8), Q27X_SKIP_INSTALL="1",
                  PYTHONIOENCODING="utf-8", E11_HARNESS="1", E11_PORT=str(PORT),
                  E11_FAKE_SERVER=str(HERE / "e15p_fake_server.py"), E11_INT4=str(INT4), E11_DFLASH=str(DFL),
                  E11_SGL_WHEELHOUSE=str(WH), E14_DIR=str(WORK / "e14_mod"), E14_DEST=str(DEST),
                  E14_FAKE_SGLANG_PKG=str(PKG), E14_TIME_SCALE=SCALE, E14_FIXTURES=str(FIX), FAKE_SPEED="600",
                  Q27X_PAYLOAD_DIR=str(PAYLOAD), E10_POETIQ_MAX_ITERATIONS="3", FAKE_POETIQ_BEH=json.dumps(BEH))
for _k in ("FAKE_CORRUPT_IF_FLAG", "Q27X_PROFILE", "Q27X_TIME_BUDGET", "FAKE_KV_USAGE", "KAGGLE_IS_COMPETITION_RERUN",
           "FAKE_MAMBA_USAGE", "FAKE_SMOKE_FAIL_MRR", "FAKE_DIE_MRR", "FAKE_DIE_AFTER", "FAKE_SCEN"):
    os.environ.pop(_k, None)
if MODE == "mambagate":
    os.environ["FAKE_MAMBA_USAGE"] = "0.97"
if MODE == "smokefail":
    os.environ["FAKE_SMOKE_FAIL_MRR"] = "17"
if MODE == "death":
    os.environ.update(FAKE_DIE_MRR="17", FAKE_DIE_AFTER="30")
E1105H = Path(os.environ.get("E1105_HARNESS_DIR", r"D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Review/E11_05_AUDIT_FOR_GPT/harness"))   # the Kaggle CPU kit sets it
if MODE == "tokfail":       # the E11-05 harness's fake vllm package (+ its behaviour module) for S60's fallback cell
    sys.path.insert(0, str(E1105H))
    sys.path.insert(0, str(E1105H / "fake_vllm"))
if MODE == "rerun":
    os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"
if MODE == "kvgate":
    os.environ["FAKE_KV_USAGE"] = "0.97"

nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
assert len(code) == 11 and "CELL 1b — E14" in code[2] and "Poetiq donor adaptation" in code[5] \
    and "E15P: the E15B engine cell" in code[6] and "CELL 5b — vLLM FP8 fallback" in code[7] \
    and "Poetiq iterative" in code[8] and "CELL 7b — E14" in code[10], \
    [c.splitlines()[0][:70] for c in code]
G = {"__name__": "__main__"}
ADMIT, ABORTS, EVENTS, FAKE_LOG = [], [], [], {}


def fake_verify(code_text, task, timeout_s=5.0):          # = the E11-05 harness verifier (MARK markers in the code)
    tid = next(t for t, v in G["CHALLENGES"].items() if v is task)
    truth = sol.get(tid) or [[[0]]] * len(task["test"])
    n_tr = len(task["train"])

    def res(passed, score, grids):
        tr = [{"success": passed, "soft_score": score, "feedback": "f", "error": None} for _ in range(n_tr)]
        tests = [{"grid": g, "error": None if g is not None else "boom"} for g in grids]
        usable = all(G["poetiq_valid_grid"](g) for g in grids)
        return {"train_results": tr, "test_results": tests, "passed": passed, "test_usable": usable,
                "verified_usable": passed and usable, "score": score, "feedback": "fb"}
    if "MARK:PASS_ALL" in code_text:
        return res(True, 1.0, [truth[i] for i in range(len(task["test"]))])
    if "MARK:PASS_PARTIAL" in code_text:
        return res(True, 1.0, [truth[i] if i < len(task["test"]) - 1 else None for i in range(len(task["test"]))])
    if "MARK:FAIL_ITER" in code_text:
        k = int(code_text.split("MARK:FAIL_ITER")[1].split()[0])
        return res(False, 0.5, [[[k % 10]] for _ in task["test"]])
    return res(False, 0.0, [[[9]] for _ in task["test"]])


cell_sec = []
for i, c in enumerate(code):
    if i == 8 and MODE == "drain" and G.get("ENGINE_READY"):
        G["DRAIN_DEADLINE"] = time.perf_counter() + 4.0
        G["ADMIT_DEADLINE"] = G["DRAIN_DEADLINE"] - 0.5
        G["HARD_STOP"] = G["DRAIN_DEADLINE"] + 10.0
        G["TOTAL_BUDGET_SEC"] = (G["HARD_STOP"] - G["NOTEBOOK_T0"]) + 360
    t = time.time()
    exec(compile(c, f"<cell{i}>", "exec"), G)
    cell_sec.append(round(time.time() - t, 1))
    if i == 1 and MODE not in ("commit", "rerun"):   # 20 tasks where > 10 must run at once, else the 8 picked
        _more = MODE in ("kvgate", "mambagate", "smokefail", "death")
        G["P"] = dict(G["P"], task_list=PICK + ([t for t in sorted(ch) if t not in PICK][:12] if _more else []))
    if i == 3 and MODE == "tokfail":    # the parity check failed (simulated): the install cell's own switch
        G["E11_TOKENIZER_CHECK"] = dict(G["E11_TOKENIZER_CHECK"], ok=False, harness="simulated mismatch")
        G["ENGINE_BACKEND"] = "vllm"
    if i == 5:
        G["E10_SANDBOX_SELFTEST_REAL"] = G.get("E10_SANDBOX_SELFTEST")
        G["E10_SANDBOX_SELFTEST"] = {"ok": True, "mode": "harness"}
        G["poetiq_verify_code"] = fake_verify
    if i == 6 and G.get("ENGINE_READY"):
        _add, _abort = G["engine"].add_request, G["engine"].abort_request
        G["engine"].add_request = lambda rid, p, params: (ADMIT.append(rid), EVENTS.append(("add", rid)), _add(rid, p, params))[2]
        G["engine"].abort_request = lambda ids: (ABORTS.extend(ids), EVENTS.extend(("abort", x) for x in ids), _abort(ids))[2]
    if i == 8:
        try:
            FAKE_LOG = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/fake_log", timeout=30).read())
        except Exception as e:
            print("fake log read failed", e)

checks = []


def check(label, cond, info=""):
    checks.append((label, bool(cond), "" if cond else str(info)[:900]))


S, F, E, SG = G["E10_SCHED"], G["E10_FINAL"], G["ENGINE_INFO"], G["E11_SGL"]
sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
R = json.load(open(WORK / "q27x" / "e14" / "e14_results.json", encoding="utf-8"))
ST = json.load(open(WORK / "e14_state.json", encoding="utf-8"))
PLOG = FAKE_LOG.get("poetiq") or {}
_want = "RERUN_ALL" if MODE == "rerun" else "E15PEVAL120"
check(f"mode: profile {_want}, RERUN {MODE == 'rerun'}, no E14 measurement", G["PROFILE"] == _want
      and bool(G["RERUN"]) == (MODE == "rerun") and G["E14_DO_MEASURE"] is False, (G["PROFILE"], G["RERUN"]))
check("submission schema = every test task, every test input, attempt_1 + attempt_2",
      set(sub) == set(TEST) and all(len(sub[t]) == len(TEST[t]["test"]) for t in TEST)
      and all(isinstance(a, dict) and {"attempt_1", "attempt_2"} <= set(a) for t in sub for a in sub[t]), list(sub))
fl = [str(x) for x in SG.get("launch_flags") or []]
_i = fl.index("--cuda-graph-bs-decode") if "--cuda-graph-bs-decode" in fl else -1
_v = lambda k, n=1: fl[fl.index(k) + 1:fl.index(k) + 1 + n] if k in fl else None
SGL = MODE != "tokfail"
S12 = MODE in ("smokefail", "death")
if SGL and not S12:
  check("E15B engine flags: DFLASH width 6, graphs 1..17, no max-bs, delayer 1, mrr 17, mamba 68, streaming interval 1",
      _v("--speculative-num-draft-tokens") == ["6"] and _i >= 0 and fl[_i + 1:_i + 18] == [str(b) for b in range(1, 18)]
      and "--cuda-graph-max-bs-decode" not in fl and "DFLASH" in fl and _v("--min-free-slots-delay") == ["1"]
      and _v("--max-running-requests") == ["17"] and _v("--max-mamba-cache-size") == ["68"]
      and "--incremental-streaming-output" in fl and _v("--stream-interval") == ["1"], fl)
if SGL and not S12:
  check("engine: SGLang ready on the first launch, capacity gate PASS, concurrency 15 (KV 881,336 // 56k), KV gate (10, 0.85)",
      str(E.get("backend", "")).startswith("sglang") and SG.get("ok") and len(SG.get("attempts") or []) == 1
      and SG["caps"]["gate"]["pass"] and G["CONCURRENCY"] == 15 and E.get("concurrency") == 15
      and E["kv_gate"] == {"min_active": 10, "admit_max": 0.85}, (E, SG.get("attempts"), SG.get("caps")))
check("S60 Poetiq values: 60,000 thinking, 5 feedback solutions, shuffle on, 1 expert, drain 12 min before the hard stop",
      G["THINK_BUDGET"] == 60000 and G["POETIQ_MAX_SOLUTIONS"] == 5 and G["POETIQ_SHUFFLE_EXAMPLES"] is True
      and G["POETIQ_NUM_EXPERTS"] == 1 and (MODE in ("drain", "tokfail") or E.get("drain_before_hard_stop_sec") == 720)
      and (not SGL or G["MAX_MODEL_LEN"] == G["SGL_CONTEXT_LEN"]), (G["THINK_BUDGET"], G["POETIQ_MAX_SOLUTIONS"],
                                                     G["POETIQ_SHUFFLE_EXAMPLES"], E.get("drain_before_hard_stop_sec")))
_p1 = [r for r, v in PLOG.items() if r.endswith("|1")]
check("every phase-1 request: max_new = 60,000 (the S1-60K thinking budget), temperature 1.0, top_k 20",
      not SGL or _p1 and all(PLOG[r]["max_new"] == 60000 and PLOG[r]["temperature"] == 1.0 and PLOG[r]["top_k"] == 20 for r in _p1),
      [(r, PLOG[r]["max_new"], PLOG[r]["temperature"], PLOG[r]["top_k"]) for r in _p1[:5]])
check("the solver ended without an error or a unit error", G["E10_SCHED_ERROR"] is None and not S["unit_errors"]
      and not [s for s in G["E10_STATES"].values() if s["status"] == "unit_error"],
      (str(G["E10_SCHED_ERROR"])[-400:], S["unit_errors"][:3]))
_first = [r.split("~e")[0] for r in ADMIT if r.endswith("|1") and "~i0|" in r]
if SGL and MODE != "death":   # death: the requeued units are admitted again at the front (expected)
  check("iteration-1 admissions follow ORDER2 + cells", _first == G["E10_ORDER"][:len(_first)], _first[:8])
_it2 = [k for k, r in enumerate(ADMIT) if r.endswith("|1") and "~i0|" not in r]
_it1 = [k for k, r in enumerate(ADMIT) if r.endswith("|1") and "~i0|" in r]
check("every task got iteration 1 before any iteration 2", MODE == "death" or not _it2 or not _it1 or min(_it2) > max(_it1),
      (min(_it2) if _it2 else None, max(_it1) if _it1 else None))
_think = list((G["OUT"] / "poetiq_thinking").glob("*.txt"))
_think = [f for f in _think if ".partial" not in f.name]
check("full thinking saved per iteration", len(_think) == S["iterations_finished"] > 0, (len(_think), S["iterations_finished"]))
_rep = json.load(open(G["OUT"] / "poetiq_run_report.json", encoding="utf-8"))
check("run report with the E11 engine section and the rev 2 e15p section; bundle written",
      "e11" in _rep and (not SGL or "error" not in _rep["e11"]) and _rep.get("e15p", {}).get("rev") == 2
      and _rep["e15p"].get("iteration1_completed_tasks") == len(G["TASK_IDS"]) and (WORK / "Q27X_E10_POETIQ_BUNDLE.zip").exists(), str(_rep.get("e11"))[:300])
check("server stopped at the end", G.get("E11_SERVER") is None or not G["E11_SERVER"].alive())
_pl = (R.get("build") or {}).get("payload") or {}
check("payload: notebook < 400 kB, only the E14 modules zip needed (found, sha-verified); no Duck package; variant E15P",
      Path(NB).stat().st_size < 400_000 and [k for k in _pl if k.endswith(".bin")] == ["q27x_e14_modules_55c1eb7911c6.bin"]
      and (R.get("build") or {}).get("variant") == "E15P" and not (WORK / "e12_duck").exists()
      and not (G["OUT"] / "e12_run_report.json").exists(), (Path(NB).stat().st_size, R.get("build")))
st, lp = R.get("staging") or {}, R.get("launch_paths") or {}
if SGL:
  check("staging: both model dirs copied to E14_DEST and the launch used the copies",
      len(st.get("copies") or {}) == 2 and str(DEST) in lp.get("model", "") and str(DEST) in lp.get("draft", "")
      and st.get("error") is None, (st, lp))
_ph = [p["phase"] for p in ST.get("phases") or []]
if SGL:
  check("phase record: PREPARE, STAGING_JOIN, LAUNCH, READY, REPORT; terminal done",
      ST.get("terminal_reason") == "done" and all(any(x.startswith(k) for x in _ph)
                                                  for k in ("PREPARE", "STAGING_JOIN", "LAUNCH", "READY", "REPORT")), _ph)
if MODE == "rerun":    # one Poetiq unit per task: the 8 test tasks all run at once
    check("rerun: all 8 test-task units ran at once", S["max_active"] == 8 == len(TEST), S["max_active"])
if MODE == "commit":
    # the server cap is client + 1 (rev 3c): a finished stream is still counted on the server while the next one starts
    check("15 Poetiq streams really ran at once (scheduler max_active 15; server in-flight peak 15-17 = within mrr 17: finishing streams overlap new ones)",
          S["max_active"] == 15 and FAKE_LOG.get("max_inflight") in (15, 16, 17), (S["max_active"], FAKE_LOG.get("max_inflight")))
    check("the KV gate never held (KV low)", S["kv_gate_blocks"] == 0, S["kv_gate_blocks"])
if MODE == "commit":
    check("commit: all 120 eval tasks, scored; every task started", len(G["TASK_IDS"]) == 120 and _rep.get("eval_query_total") == 172
          and S.get("not_started") == 0, (len(G["TASK_IDS"]), _rep.get("eval_query_total"), S.get("not_started")))
    _pass = [t for t in G["TASK_IDS"] if (BEH.get(t) or "") in ("pass",) or (t not in BEH and __import__("hashlib").sha256(t.encode()).digest()[0] % 3 != 1)]
    check("pass / forced-then-pass tasks solved on query 0 (attempt_1 = the solution)",
          all(F[t][0]["attempt_1"] == sol[t][0] for t in _pass), [t for t in _pass if F[t][0]["attempt_1"] != sol[t][0]][:5])
    check("F3: train-passing program with a crashing test input keeps query 0", F[multi[0]][0]["attempt_1"] == sol[multi[0]][0])
    check("a failing task gets all 3 iterations", G["E10_STATES"][(multi[2], 0)]["iteration"] == 3)
    check("A2: attempt_2 never repeats attempt_1", all(p["attempt_1"] != p["attempt_2"] for t in F for p in F[t]))
    check("commit: submission.json = the test-task fallback (eval answers only in the report)",
          all(sub[t] == G["identity_submission"](G["test_challenges"])[t] for t in sub))
    STOP = G["STOP_IDS"]
    a, b = PLOG.get(f"{multi[1]}~e0~i0|1"), PLOG.get(f"{multi[1]}~e0~i0|2")
    check("forced close: phase-2 prompt = phase-1 prompt + every streamed token + the stop text (exact)",
          a is not None and b is not None and b["input"] == a["input"] + a["sent"] + STOP and len(a["sent"]) == 60000)
if MODE == "kvgate":
    check("KV gate at 16 streams: held new admissions at 10 running when the server reports KV 0.97",
          S["max_active"] == 10 and S["kv_gate_blocks"] > 0 and S["kv_gate_last"] == 0.97, (S["max_active"], S["kv_gate_blocks"], S["kv_gate_last"]))
if MODE == "drain":
    STOP = G["STOP_IDS"]
    for t in (single[0], single[3]):
        a, b = PLOG.get(f"{t}~e0~i0|1"), PLOG.get(f"{t}~e0~i0|2")
        ok = a is not None and b is not None and a["finish"] == "abort" and b["input"][:len(a["input"])] == a["input"] \
            and b["input"][-len(STOP):] == STOP
        k = len(b["input"]) - len(a["input"]) - len(STOP) if ok else -1
        check(f"drain {t}: phase 1 aborted at {k} of 60,000; phase 2 = prompt + the received tokens + stop text; verified",
              ok and 0 < k < 60000 and G["E10_STATES"][(t, 0)]["status"] == "train_verified", (a and a["finish"], k))
    _d = [r for r in ABORTS if r.endswith("|1")]
    _i0 = EVENTS.index(("abort", _d[0])) if _d else len(EVENTS)
    check("drain: no new phase-1 admission after the drain; every verification harvested",
          bool(_d) and not [r for k_, r in EVENTS[_i0:] if k_ == "add" and r.endswith("|1")]
          and S.get("remaining_verifications") == 0, (_d, S.get("remaining_verifications")))
if MODE == "rerun":
    check("rerun: the solver took every test task", set(G["TASK_IDS"]) == set(TEST) and G["P"].get("source") == "test")
    check("rerun: solved tasks written into submission.json", sub[single[0]][0]["attempt_1"] == sol[single[0]][0]
          and sub[single[3]][0]["attempt_1"] == sol[single[3]][0])
if MODE == "mambagate":
    check("rev 2 F4: the gate reads max(KV, mamba): mamba 0.97 with KV low holds new admissions at 10 running",
          S["max_active"] == 10 and S["kv_gate_blocks"] > 0 and S["kv_gate_last"] == 0.97,
          (S["max_active"], S["kv_gate_blocks"], S["kv_gate_last"]))
if MODE == "tokfail":
    check("rev 2 F1: tokenizer check failed -> SGLang skipped, S60's vLLM FP8 fallback ran Poetiq",
          "fallback" in str(E.get("backend")) and str(SG.get("stage", "")).startswith("skipped") and G["ENGINE_READY"]
          and S["iterations_finished"] > 0, (E.get("backend"), SG.get("stage")))
    check("rev 2 F1: a pass task solved on the fallback engine", F[single[0]][0]["attempt_1"] == sol[single[0]][0])
if MODE == "smokefail":
    _rl = G.get("E15P_RELAUNCHES") or []
    check("rev 2 F15: 16-stream smoke FAIL -> one 12-stream relaunch (mrr 13), smoke PASS, Poetiq ran at 12",
          SG.get("smoke16", {}).get("ok") is False and len(_rl) == 1 and _rl[0]["reason"] == "smoke" and _rl[0].get("smoke_ok")
          and G["CONCURRENCY"] == 12 and S["max_active"] == 12 and _v("--max-running-requests") == ["13"]
          and str(E.get("backend", "")).startswith("sglang"), (SG.get("smoke16"), _rl, G["CONCURRENCY"], S["max_active"]))
if MODE == "death":
    _rs = G.get("E15P_RESTARTS") or []
    check("rev 2 F2: the server died mid-run -> the running units requeued, one 12-stream relaunch, the run completed",
          len(_rs) == 1 and _rs[0].get("relaunched") and _rs[0].get("requeued", 0) > 0 and G["CONCURRENCY"] == 12
          and S.get("not_started") == 0 and all(s["iteration"] >= 1 for s in G["E10_STATES"].values()),
          (_rs, G["CONCURRENCY"], S.get("not_started")))
    check("rev 2 F2: the thinking of the interrupted units saved (.restart1.partial.txt)",
          bool(_rs) and len(list((G["OUT"] / "poetiq_thinking").glob("*.restart1.partial.txt"))) == _rs[0].get("requeued", -1))
    check("rev 2 F2: pass tasks still solved after the restart",
          all(F[t][0]["attempt_1"] == sol[t][0] for t in (single[0], single[1], single[3], single[4])))
    check("rev 2 F12: the decode summary covers both server logs", len(G.get("E15P_OLD_LOGS") or []) == 1
          and (_rep.get("e15p") or {}).get("decode_log", {}).get("running_max", 0) > 12,   # > 12 exists only in the dead 15-stream server's log
          _rep.get("e15p", {}).get("decode_log"))
# unit tests of the rev 2 code (SGLang modes)
if SGL:
    import queue as _q

    class _Alive:
        def alive(self):
            return True
    _e = G["E11SglangEngine"].__new__(G["E11SglangEngine"])
    _e.server, _e.q, _e.held = _Alive(), _q.Queue(), []
    for it in (("a|1", [1, 2], None), ("b|1", [7], None), ("a|1", [3], "abort"), ("b|1", [8], None)):
        _e.q.put(it)
    _pd = _e.take_pending(["a|1"])
    _st = _e.step()
    check("rev 2 F3 unit: take_pending returns the queued deltas + final reason of the aborted rid; others kept in order",
          _pd == {"a|1": ([1, 2, 3], "abort")} and len(_st) == 1 and _st[0].request_id == "b|1"
          and list(_st[0].outputs[0].token_ids) == [7, 8], (_pd, [(o.request_id, o.outputs[0].token_ids) for o in _st]))
    _NS = __import__("types").SimpleNamespace
    _orig = G["e11_prom"]
    G["e11_prom"] = lambda timeout=10: [_NS(name="sglang:full_token_usage", labels=None, value=0.40),
                                        _NS(name="sglang:mamba_usage", labels=None, value=0.95),
                                        _NS(name="sglang:token_usage", labels=None, value=0.95)]
    _mv = [r.value for r in G["E11SglangLLM"]().get_metrics() if "kv_cache_usage" in r.name]
    G["e11_prom"] = lambda timeout=10: [_NS(name="sglang:full_token_usage", labels=None, value=0.40),
                                        _NS(name="sglang:mamba_usage", labels=None, value=0.95)]
    _mv2 = [r.value for r in G["E11SglangLLM"]().get_metrics() if "kv_cache_usage" in r.name]
    G["e11_prom"] = lambda timeout=10: []
    _mv3 = [r.value for r in G["E11SglangLLM"]().get_metrics() if "kv_cache_usage" in r.name]
    G["e11_prom"] = _orig
    check("rev 2 F4 unit: admission usage = token_usage (0.95 with KV 0.40), else max(full, mamba), none when absent",
          _mv == [0.95] and _mv2 == [0.95] and _mv3 == [], (_mv, _mv2, _mv3))
    _act, _kv, _sc = G["E10_ACTIVE"], G["_E10_KV"], G["E10_SCHED"]
    _save = (dict(_act), dict(_kv))
    _now = time.perf_counter() + 1e6         # cached value stays (read every 5 s)

    def _gate(n, v):
        _act.clear()
        _act.update({("t", i): {} for i in range(n)})
        _kv.update(t=_now, v=v)
        return G["e10_kv_gate_open"]()
    _res = [_gate(16, None), _gate(16, 0.97), _gate(16, 0.50), _gate(9, None), _gate(9, 0.97), _gate(10, 0.97)]
    _act.clear(); _act.update(_save[0]); _kv.clear(); _kv.update(_save[1])
    check("rev 2 F4 unit: gate closed on a missing metric and above 0.85 from 10 running; open below the floor of 10",
          _res == [False, False, True, True, True, False], _res)
print("cell seconds:", cell_sec)
print("scheduler:", json.dumps({k: v for k, v in S.items() if k not in ("error", "unit_errors")})[:1500])
print("eval:", _rep.get("eval_query_hits"), "/", _rep.get("eval_query_total"), "| work dir", WORK)
for label, ok, info in checks:
    print(("PASS " if ok else "FAIL ") + label + ("" if ok else "  <-- " + info))
nf = sum(not ok for _, ok, _ in checks)
print(f"\n{nf} failed of {len(checks)}" if nf else f"\nALL {len(checks)} CHECKS PASS ({MODE})")
sys.stdout.flush()
os._exit(1 if nf else 0)
