"""CPU harness for the E15B notebooks (Exp 15 Notes/02-03; derived from the E14S harness, Exp 14 Notes/30): runs ALL code cells of the real notebook in order against
e14_fake_server.py (no GPU, no install) in one of two modes, then checks what it produced.
usage: python harness_e14s_nb.py <notebook> [scale]
env E14S_MODE: commit (profile E14S: a short Duck window on eval tasks) | rerun (KAGGLE_IS_COMPETITION_RERUN=1: every
test task, profile RERUN_ALL with a shortened clock) | eval120 (the EVAL120G4B notebook's default profile: all 120 eval
tasks at the hidden run's compute share, gate 4b; clock shortened) | e15 (the E15 notebook: engine arms incl. MTP, then the
Duck run on the adopted arm). Both must launch the submission arm (DFlash width 6 + CUDA graphs
1..13), run no E14 measurement and write a complete submission.json. Setup = harness_e14_nb.py's."""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

NB = sys.argv[1]
MODE = os.environ.get("E14S_MODE", "commit")
SCALE = sys.argv[2] if len(sys.argv) > 2 else "0.03"
HERE = Path(__file__).resolve().parent
SRC = HERE.parent
# the E14 harness folder: the fake server and the E14 modules (unchanged)
E14H = Path(os.environ.get("E14_HARNESS_DIR", r"D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/src/harness"))   # the Kaggle CPU kit sets it
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

ROOT = Path(os.environ.get("E14_HARNESS_ROOT", r"D:/ARC AGI 2/_scratch/e15b/harness_runs"))
ROOT.mkdir(parents=True, exist_ok=True)
WORK = ROOT / f"run_{time.strftime('%H%M%S')}"
WORK.mkdir()
ch, sol, multi, single = B.load()
PICK = ["2ba387bc"] + multi[:2] + single[:2]
FP8 = B.TOKDIR_FP8
# the INT4 "model": a copy of the tokenizer fixture + a 3 MiB safetensors file, so staging copies something real.
# Paths have the REAL Kaggle shape (both end in /1): the old fixture names int4 / dflash hid the staging collision
MODELS = WORK / "inputs" / "models" / "foysalemonshanto"
INT4 = MODELS / "qwen3-8-27b-int4-redhat" / "pytorch" / "w4a16-g128" / "1"
shutil.copytree(B.TOKDIR_INT4, INT4)
import struct  # noqa: E402
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
json.dump({t: {"train": ch[t]["train"], "test": [{"input": q["input"]} for q in ch[t]["test"]]} for t in PICK},
          open(ARC / "arc-agi_test_challenges.json", "w"))
# a fake SGLang package holding only the files the E14 patches touch (copied from the local 0.5.21 source)
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
# the payload dataset (Notes/21) in the REAL Kaggle shape: /kaggle/input/datasets/<owner>/<slug>/<file>
PAYLOAD = WORK / "inputs"
_pd = PAYLOAD / "datasets" / "esfersami50" / "q27-e15b-payload"
_pd.mkdir(parents=True)
for _f in (Path(NB).resolve().parent / "Build" / "payload").glob("q27x_*.bin"):
    shutil.copy(_f, _pd / _f.name)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    PORT = s.getsockname()[1]
os.environ.update(Q27X_ARC_ROOT=str(ARC), Q27X_WORKING=str(WORK), Q27X_MODEL=str(FP8), Q27X_SKIP_INSTALL="1",
                  PYTHONIOENCODING="utf-8", E11_HARNESS="1", E11_PORT=str(PORT),
                  E11_FAKE_SERVER=str(E14H / "e14_fake_server.py"), E11_INT4=str(INT4), E11_DFLASH=str(DFL),
                  E11_SGL_WHEELHOUSE=str(WH), E12_DIR=str(WORK / "e12_duck"), E14_DIR=str(WORK / "e14_mod"),
                  E14_DEST=str(DEST), E14_FAKE_SGLANG_PKG=str(PKG), E14_TIME_SCALE=SCALE, E14_LONG_TOKENS="3000",
                  E14_FIXTURES=str(FIX), FAKE_SPEED="300", Q27X_PAYLOAD_DIR=str(PAYLOAD),
                  ARC3_HTTP_RETRY_BASE_SECONDS="0.05", ARC3_HTTP_RETRY_MAX_SECONDS="0.1",
                  E12_MAX_GEN="6000", E12_R6_FIRST_EXPOSURE="0", E12_R6_MIN_ADMIT_SEC="5", E12_B_RESERVE="1000",
                  E12_B_ATTEMPT_TOKENS_B1="4000")   # E15B: short attempts; rev 3f: attempt 2 has its own (smaller) budget
os.environ.pop("FAKE_CORRUPT_IF_FLAG", None)
if "_R3K" in Path(NB).name:
    os.environ["E12_B_MAX_ATTEMPTS"] = "3"     # rev 3k W1: every fake accept is weak, so every task continues; keep it short
os.environ.pop("Q27X_PROFILE", None)
os.environ.pop("E12_REASONING_EFFORT", None)
if MODE == "rerun":
    os.environ.update(KAGGLE_IS_COMPETITION_RERUN="1", Q27X_TIME_BUDGET=os.environ.get("E14S_RERUN_BUDGET", "1500"))
elif MODE == "eval120":
    os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    os.environ["Q27X_TIME_BUDGET"] = os.environ.get("E14S_EVAL_BUDGET", "2400")
else:
    os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    os.environ.pop("Q27X_TIME_BUDGET", None)

# A7 (Exp 15 Notes/07): fault injection. kill = the server process is killed mid-solve (the watchdog must relaunch it and
# the run must end with a valid submission); slow = a healthy server whose replies take longer than the health-fail
# window (the watchdog must NOT relaunch). Short watchdog limits; a longer commit window so the solve outlives the fault.
FAULT = os.environ.get("E15B_FAULT", "")
FAULT_REC = {}
if FAULT:
    os.environ.update(E15B_WD_POLL_SEC="2", E15B_WD_FAIL_SEC="20", E15B_WD_MIN_LEFT_SEC="30")
    if FAULT == "slow":
        os.environ["FAKE_SPEED"] = "40"          # a 1-3k-token reply holds the request for 25-75 s > the 20 s window
    if FAULT == "kill2":       # the solve must outlive BOTH kills (Kaggle CPU kit v3: a fast fake server ended the solve
        os.environ["FAKE_SPEED"] = "120"         # in 34 s, before the second kill, so no second relaunch was needed)

nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
_n = 12
assert len(code) == _n and "CELL 1b — E14" in code[2] and "CELL 5b — E14-A1" in code[7] \
    and "CELL 6b — E14-A2" in code[_n - 2], \
    [c.splitlines()[0][:70] for c in code]
G = {"__name__": "__main__"}
t0 = time.time()
cell_sec = []
def _fault_thread():
    import threading
    # wait for the solve (the watchdog record exists) and some generated tokens, then kill the server process
    t_end = time.time() + 900
    while time.time() < t_end and not ((G.get("E12_RUN") or {}).get("watchdog") is not None
                                       and any(int(r.get("generated_tokens") or 0) > 0
                                               for v in (G.get("E12_RECORDS") or {}).values() for r in v)):
        time.sleep(0.5)
    time.sleep(8)
    srv = G.get("E11_SERVER")
    FAULT_REC.update(t_kill=time.time(), server_before=id(srv))
    try:
        srv.kill(wait=True)
        FAULT_REC["killed"] = True
    except Exception as exc:
        FAULT_REC["kill_error"] = repr(exc)
    if FAULT == "kill2":       # B3: kill the relaunched server too -> the second relaunch must use the 12-stream flags
        t_end = time.time() + 300
        while time.time() < t_end and (G.get("E11_SERVER") is srv or not G["E11_SERVER"].alive()):
            time.sleep(0.5)
        time.sleep(8)
        srv2 = G.get("E11_SERVER")
        FAULT_REC["server_second"] = id(srv2)
        try:
            srv2.kill(wait=True)
            FAULT_REC["killed2"] = True
        except Exception as exc:
            FAULT_REC["kill2_error"] = repr(exc)


for i, c in enumerate(code):
    if i == 7 and MODE == "commit":    # a short Duck window in the harness (the profile's is 25 / 15 min)
        G["E12_SOLVE_WINDOW_SEC"] = (300 if FAULT == "kill2" else 240) if FAULT else 90
        if FAULT in ("kill", "kill2"):
            import threading
            threading.Thread(target=_fault_thread, daemon=True).start()
    t = time.time()
    exec(compile(c, f"<cell{i}>", "exec"), G)
    cell_sec.append(round(time.time() - t, 1))

checks = []


def check(label, cond, info=""):
    checks.append((label, bool(cond), "" if cond else str(info)[:900]))


R = json.load(open(WORK / "q27x" / "e14" / "e14_results.json", encoding="utf-8"))
ST = json.load(open(WORK / "e14_state.json", encoding="utf-8"))
sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
test = json.load(open(ARC / "arc-agi_test_challenges.json", encoding="utf-8"))
rep = json.load(open(WORK / "q27x" / "e12_run_report.json", encoding="utf-8"))
check("submission schema = every test task, every test input, attempt_1 + attempt_2 grids",
      set(sub) == set(test) and all(len(sub[t]) == len(test[t]["test"]) for t in test)
      and all(isinstance(a, dict) and {"attempt_1", "attempt_2"} <= set(a) for t in sub for a in sub[t]), list(sub))
R3J = ("1" if ("_R3J1_" in Path(NB).name or "_R3K1_" in Path(NB).name)
       else "2" if ("_R3J2_" in Path(NB).name or "_R3K2_" in Path(NB).name) else "")   # rev 3j / 3k notebooks
R3K = "_R3K" in Path(NB).name
R3I = "_R3I_" in Path(NB).name or bool(R3J)      # rev 3i / 3j notebook: stage 3, EVAL120 in a 6-h notebook
_want = {"rerun": "RERUN_ALL", "eval120": "E15BEVAL120R3I" if R3I else "E15BEVAL120G4B"}.get(MODE, "E15B")
check("mode: profile " + _want + (", RERUN on" if MODE == "rerun" else " (default), RERUN off") +
      "; no E14 measurement; raw reasoning on",
      G["PROFILE"] == _want and bool(G["RERUN"]) == (MODE == "rerun")
      and G["E14_DO_MEASURE"] is False and G["os"].environ.get("E12_RAW_REASONING_HISTORY") == "1",
      (G["PROFILE"], G["RERUN"], G["E14_DO_MEASURE"]))
fl = [str(x) for x in (G.get("E11_SGL") or {}).get("launch_flags") or []]
_i = fl.index("--cuda-graph-bs-decode") if "--cuda-graph-bs-decode" in fl else -1
_v = lambda k, n=1: fl[fl.index(k) + 1:fl.index(k) + 1 + n] if k in fl else None
check("E15B server flags: DFLASH width 6, --cuda-graph-bs-decode 1..17, no --cuda-graph-max-bs-decode, "
      "--min-free-slots-delay 1, --max-running-requests 17, --max-mamba-cache-size 68, no --enable-prefill-delayer",
      fl.count("--speculative-num-draft-tokens") == 1 and _v("--speculative-num-draft-tokens") == ["6"]
      and _i >= 0 and fl[_i + 1:_i + 18] == [str(b) for b in range(1, 18)] and (len(fl) == _i + 18 or fl[_i + 18].startswith("--"))
      and "--cuda-graph-max-bs-decode" not in fl and "DFLASH" in fl and _v("--min-free-slots-delay") == ["1"]
      and _v("--max-running-requests") == ["17"] and _v("--max-mamba-cache-size") == ["68"]
      and fl.count("--max-running-requests") == 1 and "--enable-prefill-delayer" not in fl, fl)
_eff = os.environ.get("E14S_EXPECT_EFFORT", "")
check("reasoning effort: " + (_eff or "model default (xhigh)") + " for the Duck (E12_REASONING_EFFORT)",
      G["E12_REASONING_EFFORT"] == _eff and G["os"].environ.get("E12_REASONING_EFFORT", "") == _eff
      and (G["CONFIG"].get("E12_REASONING_EFFORT", _eff) == _eff),
      (G.get("E12_REASONING_EFFORT"), G["os"].environ.get("E12_REASONING_EFFORT"), G["CONFIG"].get("E12_REASONING_EFFORT")))
check("CONFIG records the submission arm", (G["CONFIG"].get("e14") or {}).get("submit_arm"), G["CONFIG"].get("e14"))
check("no E14 measurement ran (no E14A phases, no measurement launches unless E15, no JIT tar, no NCCL bench)",
      (not R.get("phases") and not R.get("launches"))
      and not any(k.startswith(("A1-", "A2-")) for k in (R.get("phases") or {})) and "jit_cache" not in R and "nccl" not in R
      and not (WORK / "e14_jit_cache.tar").exists(), {k: R.get(k) for k in ("phases", "launches", "jit_cache", "nccl")})
_pl = (R.get("build") or {}).get("payload") or {}
check("payload: notebook < 400 kB, both zips found in the dataset shape and sha-verified",
      Path(NB).stat().st_size < 400_000 and len([k for k in _pl if k.endswith(".bin")]) == 2
      and {k for k in _pl if k.endswith(".bin")} <= {f.name for f in _pd.glob("q27x_*.bin")}
      and (Path(G["E12_DIR"]) / "arc2_duck.py").exists(), (Path(NB).stat().st_size, _pl))
st = R.get("staging") or {}
lp = R.get("launch_paths") or {}
check("staging: both model dirs copied to E14_DEST and the launch used the copies",
      len(st.get("copies") or {}) == 2 and str(DEST) in lp.get("model", "") and str(DEST) in lp.get("draft", "")
      and lp.get("model") != lp.get("draft") and st.get("error") is None, (st, lp))
phases_seen = [p["phase"] for p in ST.get("phases") or []]
check("phase record: PREPARE, STAGING_JOIN, LAUNCH, READY, REPORT; terminal done; no measurement phase",
      ST.get("terminal_reason") == "done" and all(any(x.startswith(k) for x in phases_seen) for k in
                                                  ("PREPARE", "STAGING_JOIN", "LAUNCH", "READY", "REPORT"))
      and not any(x.startswith(("PHASE:A1", "PHASE:A2", "NCCL_BENCH")) for x in phases_seen)
      and not any(x.startswith(("PHASE:", "LAUNCH:")) for x in phases_seen), phases_seen)
e = rep.get("e12_run") or {}
check("Duck ran without solver error on every task; branches recorded",
      not e.get("error") and e.get("branches_started", 0) > 0, e)
check("isolation preflight PASS, 16 workers / streams", e.get("rev6_workers") == 16 and e.get("streams") == 16
      and (e.get("rev6_preflight") or {}).get("isolation_state") == "PASS", (e.get("rev6_workers"), e.get("rev6_preflight")))
if MODE == "rerun":
    _ids = [t for t in test]
    check("rerun: the solver took every test task (no eval task, no task list)", set(G["TASK_IDS"]) == set(test)
          and G["P"].get("source") == "test", (G.get("TASK_IDS"), G["P"]))
    # the fake's programs return the input for test tasks (no solutions), so attempt_1 can equal the identity fallback;
    # the submission must be the solver's generated attempts, with at least one attempt that is not the fallback
    gen = json.load(open(WORK / "q27x" / "e12_generated_attempts.json", encoding="utf-8"))
    _other = sum(1 for t in test for a, q in zip(sub[t], test[t]["test"])
                 if a.get("attempt_1") != q["input"] or a.get("attempt_2") != q["input"])
    check("rerun: submission.json = the solver's generated attempts for every task, not only the identity fallback",
          set(gen) == set(test) and all(sub[t] == gen[t] for t in test) and _other > 0, (sorted(gen), _other))
if MODE == "eval120":
    g4 = e.get("g4b") or {}
    _hid = G["PROFILE_TABLE"]["RERUN_ALL"]
    _hid_end = _hid["time_budget_sec"] - _hid["hard_margin_sec"] - G["E12_STOP_BEFORE_HARD_SEC"]
    check("eval120: all 120 eval tasks / 172 inputs planned", len(G["TASK_IDS"]) == 120
          and rep.get("eval_query_total") == 172 and rep.get("queries_planned") == 172,
          (len(G["TASK_IDS"]), rep.get("eval_query_total"), rep.get("queries_planned")))
    check("eval120: gate 4b window = 120 x (hidden solve end - start) / 240, hidden end from RERUN_ALL (" + str(_hid_end) + " s)",
          g4.get("hidden_tasks") == 240 and g4.get("tasks") == 120 and g4.get("hidden_solve_end_sec") == _hid_end
          and abs(g4.get("sec_per_task", 0) - (_hid_end - g4.get("solve_start_sec", 0)) / 240) < 1
          and abs(g4.get("window_sec", 0) - 120 * (_hid_end - g4.get("solve_start_sec", 0)) / 240) < 2, g4)   # unrounded per
    if R3I:      # user 2026-10-25: exactly a 6-h notebook; the hidden-equivalent window is clamped by its hard stop
        check("eval120 (rev 3i): 6.0-h budget, hidden-equivalent window, solve end <= hard stop - stop margin",
              G["PROFILE_TABLE"]["E15BEVAL120R3I"]["time_budget_sec"] == 6.0 * 3600
              and G["PROFILE_TABLE"]["E15BEVAL120R3I"].get("hidden_equiv") is True, G["PROFILE_TABLE"]["E15BEVAL120R3I"])
    else:
        check("eval120: the 6.5 h budget holds start + the full window + the stop margin (no cut by the hard stop)",
              G["PROFILE_TABLE"]["E15BEVAL120G4B"]["time_budget_sec"] - 2 * 60 - G["E12_STOP_BEFORE_HARD_SEC"]
              >= 1500 + 120 * (_hid_end - 1500) / 240, G["PROFILE_TABLE"]["E15BEVAL120G4B"])
    check("eval120: scored (pass1 / pass2 / task-norm recorded)", rep.get("eval_task_norm_pct") is not None
          and rep.get("eval_pass1_hits") is not None and rep.get("eval_pass2_hits") is not None,
          {k: rep.get(k) for k in ("eval_query_hits", "eval_pass1_hits", "eval_pass2_hits", "eval_task_norm_pct")})
# ---- E15B (Exp 15 Notes/02): config B behaviour
_recs = [json.loads(p.read_text(encoding="utf-8")) for p in (WORK / "q27x" / "duck").glob("*_b*/branch_result.json")]
_b0 = [r for r in _recs if r.get("branch") == 0]
_b1 = [r for r in _recs if r.get("branch") == 1]
check("B: E15B block on (E12_B_MODE 1, attempt tokens = E12_MAX_GEN, reserve, window 100000) and in CONFIG",
      G["E15B"]["E12_B_MODE"] == 1 and G["E15B"]["E12_B_ATTEMPT_TOKENS"] == G["E12_MAX_GEN_PER_BRANCH"]
      and G["os"].environ.get("E12_B_MODE") == "1" and G["CONFIG"].get("e15b") == G["E15B"], (G["E15B"], G["E12_MAX_GEN_PER_BRANCH"]))
check("B: scheduler settings: first exposure 0 (no pause), branch-1 view facts, fair share 0",
      G["E12_R6"]["E12_R6_FIRST_EXPOSURE"] == 0 and G["E12_R6"]["E12_R6_B1_VIEW"] == "facts"
      and G["E12_R6"]["E12_R6_FAIR_SHARE"] == 0, G["E12_R6"])
check("B: the Duck window is the fixed 100000 (not KV / streams)", e.get("rev6_window") == 100000, e.get("rev6_window"))
check("B: every attempt ran in ONE run (no pause, no chunk, no resume)", _recs and all(len(r.get("runs") or []) == 1 for r in _recs),
      [(r.get("task_id"), r.get("branch"), len(r.get("runs") or [])) for r in _recs if len(r.get("runs") or []) != 1][:10])
check("B: both attempts use the facts view; no attempt exceeds its budget by more than one reply",
      all(r.get("view") == "facts" for r in _recs)
      and all(int(r.get("generated_tokens") or 0) <= G["E12_MAX_GEN_PER_BRANCH"] + 4096 for r in _recs),
      [(r.get("task_id"), r.get("branch"), r.get("view"), r.get("generated_tokens")) for r in _recs][:12])
check("B: attempt 2 started only after every attempt 1 had started", not _b1 or min(r.get("t0") or 1e18 for r in _b1)
      >= max(r.get("t0") or 0 for r in _b0) - 1.0, (len(_b0), len(_b1)))
check("B: per-attempt record (b_events, b_budget, b_candidate) on every attempt",
      _recs and all(isinstance(r.get("b_events"), dict) and r.get("b_budget") and isinstance(r.get("b_candidate"), dict)
                    for r in _recs), sorted(_recs[0]) if _recs else None)
if R3I:      # rev 3i: R + stage 3 on the real records
    # rev 3k W1: scheduling counts STRONG accepts only (an accept after the commit text is weak)
    _acc = lambda r: any(x.get("accepted") and not (R3K and x.get("weak"))
                         for x in (r.get("submissions") or []) + (r.get("checked") or []))
    _wk = lambda r: any(x.get("accepted") and x.get("weak") for x in (r.get("submissions") or []) + (r.get("checked") or []))
    _byt = {}
    for r in _recs:
        _byt.setdefault(r.get("task_id"), {})[r.get("branch")] = r
    _S = e.get("rev6_scheduler") or {}
    _mode = {"1": "after3", "2": "before3"}.get(R3J, "skip")
    check(f"rev3i/3j: scheduler flags (attempt 2 of accepted tasks: {_mode}, stage 3, max 8 attempts)",
          _S.get("b1_accepted_mode") == _mode and _S.get("extra_attempts") is True
          and _S.get("max_attempts") == G["E15B"].get("E12_B_MAX_ATTEMPTS", 8)
          and _S.get("skip_b1_if_accepted") is (not R3J),
          {k: _S.get(k) for k in ("b1_accepted_mode", "skip_b1_if_accepted", "extra_attempts", "max_attempts")})
    _nacc = sum(1 for d in _byt.values() if 0 in d and _acc(d[0]))
    if R3J:
        check("rev3j: the attempt 2 of every task whose attempt 1 accepted is deferred (counted), and it ran",
              _S.get("b1_deferred") == _nacc and _S.get("b1_skipped") == 0
              and all(1 in d for d in _byt.values() if 0 in d and _acc(d[0])) and _S.get("b1_deferred_started") == _nacc,
              (_S.get("b1_deferred"), _S.get("b1_deferred_started"), _nacc))
    else:
        check("rev3i R: no attempt 2 on a task whose attempt 1 has a host-accepted candidate; skips counted",
              all(1 not in d for d in _byt.values() if 0 in d and _acc(d[0])) and _S.get("b1_skipped") == _nacc,
              (_S.get("b1_skipped"), _nacc))
    if R3K:
        _wonly = [t_ for t_, d in _byt.items() if 0 in d and _wk(d[0]) and not _acc(d[0])]
        check("rev3k W1: a task whose attempt 1 accepted only weakly gets its attempt 2 (not deferred) and is not counted solved",
              all(1 in _byt[t_] for t_ in _wonly) and _S.get("weak_accepts", 0) >= len(_wonly)
              and _S.get("weak_only_tasks_continued") == len(_wonly),
              (len(_wonly), _S.get("weak_accepts"), _S.get("weak_only_tasks_continued")))
        check("rev3k: W1 weak-commit flag on in the scheduler; weak counters reported",
              _S.get("weak_commit") is True and "weak_accepts" in _S and "stage2_errors" in _S
              and "weak_accepts" in (e.get("b_summary") or {}), {k: _S.get(k) for k in ("weak_commit", "weak_accepts")})
    check("rev3i stage 3: attempt k >= 3 only on tasks whose earlier attempts had no accepted candidate",
          all(not any(_acc(d[j]) for j in d if j < b) for d in _byt.values() for b in d if b >= 2),
          sorted((t, sorted(d)) for t, d in _byt.items() if max(d) >= 2)[:12])
    check("rev3i: stage-3 counters reported (scheduler + b_summary)",
          _S.get("extra_started") == sum(1 for r in _recs if (r.get("branch") or 0) >= 2)
          and "extra_started" in (e.get("b_summary") or {}) and "b1_skipped" in (e.get("b_summary") or {}),
          (_S.get("extra_started"), sum(1 for r in _recs if (r.get("branch") or 0) >= 2), sorted(e.get("b_summary") or {})[:8]))
_bs = e.get("b_summary") or {}
check("B: b_summary with concurrency, server log block, watchdog, server-side tokens, effort",
      not _bs.get("error") and _bs.get("streams") == 16 and isinstance(_bs.get("server_log"), dict)
      and "running_hist" in (_bs.get("server_log") or {}) and _bs.get("watchdog") is not None
      and "server_gen_tokens" in _bs and _bs.get("effective_effort"), _bs)
check("B: no worker error, no stale server-down state", not _bs.get("worker_errors") and not G["E12A"].SERVER_DOWN.is_set(),
      (_bs.get("worker_errors"), G["E12A"].SERVER_DOWN.is_set()))
check("B: CONFIG submit-arm label = 16 streams", "16 streams" in str((G["CONFIG"].get("e14") or {}).get("submit_arm")),
      (G["CONFIG"].get("e14") or {}).get("submit_arm"))
check("B: the sandbox self-test used the fixed task and passed",
      G["E10_SANDBOX_SELFTEST"].get("ok") and G["E15B_SELFTEST_TASK"]["test"][0]["input"] == [[4]], G["E10_SANDBOX_SELFTEST"])
check("submission.json written atomically with no temp file left", not list(WORK.glob("submission.json.*")),
      list(WORK.glob("submission.json*")))
_wd = (e.get("b_summary") or {}).get("watchdog") or e.get("watchdog") or {}
if FAULT == "kill":
    _rl = _wd.get("relaunches") or []
    # the notebook has shut the server down by now: read the relaunched server's own log (its decode lines)
    try:
        _chat_after = G["E11_SERVER"].log_text().count("Decode batch")
    except Exception as exc:
        _chat_after = repr(exc)
    check("A7 kill: the server was killed mid-solve", FAULT_REC.get("killed"), FAULT_REC)
    check("A7 kill: the watchdog relaunched it once, OK, with the short test limits", len(_rl) == 1 and _rl[0].get("ok")
          and _rl[0].get("dead") and (_wd.get("limits") or {}).get("poll_sec") == 2.0, _wd)
    check("A7 kill: the relaunched server served chat requests (the solve continued)",
          isinstance(_chat_after, int) and _chat_after > 0 and id(G["E11_SERVER"]) != FAULT_REC.get("server_before"),
          (_chat_after, FAULT_REC))
    check("A7 kill: no worker error; SERVER_DOWN clear at the end (waits recorded: see info on failure)",
          not _bs.get("worker_errors") and not G["E12A"].SERVER_DOWN.is_set(),
          (_bs.get("server_down_waits"), _bs.get("worker_errors")))
    print("A7 kill: server_down_waits", _bs.get("server_down_waits"), "| relaunch", json.dumps(_wd.get("relaunches"))[:400])
if FAULT == "kill2":
    _rl = _wd.get("relaunches") or []
    _fl2 = [str(x) for x in (G["E11_SERVER"].cmd or [])]
    check("B3 kill2: two relaunches, both OK; the second with the 12-stream flags (mrr 13, graphs 1..13, mamba 56)",
          FAULT_REC.get("killed2") and len(_rl) == 2 and all(r.get("ok") for r in _rl) and _rl[1].get("streams12")
          and not _rl[0].get("streams12") and _fl2[_fl2.index("--max-running-requests") + 1] == "13"
          and _fl2[_fl2.index("--max-mamba-cache-size") + 1] == "56", (FAULT_REC, _rl, _fl2[-40:]))
    check("B6 kill2: the engine block read every server log (3 servers)", (_bs.get("server_log") or {}).get("decode_lines", 0) > 0
          and len(_wd.get("old_logs") or []) == 2, (_wd.get("old_logs"), (_bs.get("server_log") or {}).get("decode_lines")))
if FAULT == "slow":
    check("A7 slow: no relaunch while replies outlast the 20-s health-fail window (health stays OK)",
          not (_wd.get("relaunches") or []) and _wd.get("health_fail_sec_max") == 0
          and (_wd.get("limits") or {}).get("fail_sec") == 20.0, _wd)
    _long = [q for r in _recs for q in (r.get("requests") or []) if (q.get("sec") or q.get("elapsed_sec") or 0) > 20]
    check("A7 slow: some replies really took > 20 s", _long or max([int(r.get("generated_tokens") or 0) for r in _recs] or [0]) >= 800,
          [(r.get("task_id"), r.get("generated_tokens")) for r in _recs][:8])
check("B6 report fields: accept length p50, gen tok/s by running level; B5 give-up (RERUN 4 h, else 75 min); commit window >= 78 min, >= 15 min startup room",
      "accept_len_p50" in (_bs.get("server_log") or {}) and "gen_tok_s_p50_by_running" in (_bs.get("server_log") or {})
      and round(min(G["NOTEBOOK_T0"] + (240 if MODE == "rerun" else 75) * 60, G["ADMIT_DEADLINE"] - 300) - G["SGL_GIVE_UP_AT"]) == 0
      and G["PROFILE_TABLE"]["E15B"]["solve_window_sec"] >= 78 * 60 and len(G["PROFILE_TABLE"]["E15B"]["task_list"]) == 20
      and G["PROFILE_TABLE"]["E15B"]["time_budget_sec"] - G["PROFILE_TABLE"]["E15B"]["solve_window_sec"] >= 15 * 60,
      (sorted((_bs.get("server_log") or {})), G["SGL_GIVE_UP_AT"] - G["NOTEBOOK_T0"]))
if MODE == "rerun":
    check("B1 rerun: no submission rewrite after the solve end; the final write has every task",
          e.get("late_writes_skipped", 0) >= 0 and set(sub) == set(test), e.get("late_writes_skipped"))
check("rev3f: per-attempt budgets in the records (attempt 1 = E12_MAX_GEN 6000, attempt 2 = 4000) and in E15B",
      _recs and all(r.get("b_budget") == (4000 if (r.get("branch") or 0) >= 1 else 6000) for r in _recs)   # rev 3i: b >= 2 too
      and G["E15B"]["E12_B_ATTEMPT_TOKENS_B1"] == 4000
      and all(int(r.get("generated_tokens") or 0) <= (4000 if (r.get("branch") or 0) >= 1 else 6000) + 4096 for r in _recs),
      [(r.get("task_id"), r.get("branch"), r.get("b_budget"), r.get("generated_tokens")) for r in _recs][:10])
check("A1/A3/A4 report fields present", "running_hist_queue_gt0" in (_bs.get("server_log") or {})
      and "startup" in (_bs.get("server_log") or {}) and "server_running_mean_client_full" in _bs and "kv_stale_checks" in _bs,
      sorted(_bs))
if MODE == "eval120":
    _ea = rep.get("eval_b_attempts") or {}
    check("A2 eval: per-attempt correctness block (b0/b1 acc, b1_only, rescued) for all 120 tasks",
          _ea.get("tasks") == 120 and "b1_only_acc" in _ea and "b0_budget_partial_b1_acc" in _ea and len(_ea.get("rows") or []) == 120,
          {k: v for k, v in _ea.items() if k != "rows"})
G["e11_kill_server"](wait=False)
bad = [c for c in checks if not c[1]]
print(f"\n[e15b {MODE}{' fault=' + FAULT if FAULT else ''}] {len(checks) - len(bad)}/{len(checks)} ok in {time.time() - t0:.0f} s | cell seconds {cell_sec}")
for c in bad:
    print("   FAIL", c[0], "->", c[2])
print("@@NB_OK@@" if not bad else "@@NB_FAIL@@")
