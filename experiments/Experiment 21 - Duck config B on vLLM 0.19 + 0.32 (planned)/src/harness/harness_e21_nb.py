"""CPU harness for the E21 notebooks (Duck config B on vLLM 0.19 + 0.32, two halves). Runs ALL code cells of the real
notebook in order against fake_vllm_duck_server.py (no GPU, no install), then checks what it produced and prints a
PASS/FAIL table. Setup (Windows shims, fixtures, payload dataset shape, model-dir fakes) = Exp 15 harness_e15b_nb.py.
usage: python harness_e21_nb.py <notebook> [scale]
env E21_MODE:
  commit       both halves on a few eval tasks (profile E21H_PROFILE, default EVAL6) with short per-half windows
  rerun        KAGGLE_IS_COMPETITION_RERUN=1: half A only, every fixture test task, a complete submission.json
  fallbackA    half A's primary launch is rejected by the fake (--mamba-ssm-cache-dtype in a 0.19 launch: both
               variants' half A primary, rev 2j) -> its fallback launches; half B: NB2 repeats half A's
               fallback (E21_B_FOLLOWS_A), NB1 (rev 2i) runs its own INT4 primary
  halfAfail    the 0.19 server never becomes ready (FAKE_NEVER_READY=0.19) -> half A has no engine, half B still runs and
               the notebook finishes with a valid submission
  death        the half-A server dies mid-solve (FAKE_DIE_AFTER_SEC, 0.19 form, once) -> the watchdog relaunches it once
               (rev 2: e21_relaunch needs >= 12 min before the solve end, so this mode uses a 1,500-s window; the solve
               still ends as soon as every attempt is done); rev 2b: it dies twice, the second relaunch must switch to
               the half's other config
  reasoningkey = commit; the reasoning-key checks (history sent under `reasoning`, never only `reasoning_content`)
               are run in every mode, this mode only names them
  gateAB       rev 2: the 0.19 server logs fewer KV blocks (FAKE_NUM_BLOCKS) -> e21_capacity gives half A 10 streams,
               half B its full count; 20 tasks (profile E15B); half B must really run its streams concurrently and
               its fresh Duck gate must never block
  wdhold       rev 2c: half A's watchdog is held inside a /health probe (harness urlopen wrapper; the 0.19 fake prints
               no stats lines, so the watchdog probes /health) across the A->B switch, then released: it must exit and
               never relaunch anything in half B
  overrun      rev 2c: the fake holds one half-A response 600 s, past the 90-s window: the solver must return within
               120 s of its deadline (stragglers counted), the half must end on time, the submission stays valid
  kvoutage     rev 2c/2d: the 0.19 /metrics answers 500 for 150 s mid-solve (20 tasks): for its first 120 s the KV
               probe reports 1.0 (probe_failures > 0) and no new branch reaches the server while requests run; after
               120 s Duck's own fail-open applies (probe_fail_open > 0); no deadlock
  gpuheldB     rev 2d: E21_FAKE_GPU_HELD=1 from half B's begin: half B makes no launch, fatal "GPU memory still held"
SMOKE notebooks (*_SMOKE.ipynb, rev 2d): commit mode with the notebook's own default profile E21SMOKE (Q27X_PROFILE
and the time budget are not overridden; only the per-half window is shortened)
rev 2c checks in every two-half mode: a late half-A branch callback invoked after half B's solve (harness wrapper of
E12A.solve_all_r6) changes nothing of half B (records, branches_accepted, submission.json bytes) and counts
late_cross_half_callbacks; the deadline contract (solve_overrun_sec <= 120, end_sec <= planned_end_sec + 300);
config_status / comparison_valid / lmo_in_log. A heartbeat-log gap > 300 s (host suspend) prints HOST_SUSPENDED
(@@E21_HARNESS_HOST_SUSPENDED@@, exit 3) instead of PASS / FAIL.
rev 2 checks in every mode where Duck runs: the cut tool call (fake: finish "tool_calls" at max_tokens) is restored to
"length" (length_restored > 0) and Duck dropped / saved it; the contract probe passed; e21_report per-half fields.
Other env: E21_HARNESS_ROOT (run folders), E21_FAKE_SERVER (default: next to this file), E14_FIXTURES_DIR,
Q27X_PAYLOAD_DIR, E14_HARNESS_DIR (accepted, unused), E21H_PROFILE, E21H_HALF_SOLVE_SEC, E21H_MAX_SEC (hard limit, 1800).
Output: one line per check `[e21 PASS] ...` / `[e21 FAIL] ... -> info`, then @@E21_HARNESS_OK@@ and exit 0 only when
every check passed (else @@E21_HARNESS_FAIL@@, exit 1). Runs on Windows (shims behind os.name) and Linux / Py 3.12."""
import faulthandler
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

NB = str(Path(sys.argv[1]).resolve())
MODE = os.environ.get("E21_MODE", "commit")
assert MODE in ("commit", "rerun", "fallbackA", "halfAfail", "death", "reasoningkey", "gateAB", "wdhold", "overrun",
                "kvoutage", "gpuheldB"), MODE
SCALE = sys.argv[2] if len(sys.argv) > 2 else "0.03"
HERE = Path(__file__).resolve().parent
NBN = Path(NB).name
VAR = "NB1" if "_NB1_" in NBN else "NB2" if "_NB2_" in NBN else None
SMOKE = NBN.endswith("_SMOKE.ipynb")      # rev 2d: the GPU-smoke notebooks (default profile E21SMOKE)
assert VAR, NBN
FAKE = Path(os.environ.get("E21_FAKE_SERVER") or (HERE / "fake_vllm_duck_server.py"))
FIX = Path(os.environ.get("E14_FIXTURES_DIR", r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures"))
# Q27X_PAYLOAD_DIR (the Kaggle CPU kit: a flat folder of q27x_*.bin) is passed to the notebook as it is; without it the
# E15B payload is copied into the real dataset shape below
PAYLOAD_ENV = os.environ.get("Q27X_PAYLOAD_DIR")
PAYSRC = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload")
sys.path.insert(0, str(FIX))
import e12_fixtures as B  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
faulthandler.dump_traceback_later(float(os.environ.get("E21H_MAX_SEC", 1800)), exit=True)   # never hang forever
if os.name == "nt":   # the E12 harness's Windows-only shims (65 KB sandbox bootstrap as -c; no killpg)
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

ROOT = Path(os.environ.get("E21_HARNESS_ROOT") or os.environ.get("E21H_ROOT") or r"D:/ARC AGI 2/_scratch/e21/harness_runs")
ROOT.mkdir(parents=True, exist_ok=True)
WORK = ROOT / f"run_{VAR}{'_SMOKE' if SMOKE else ''}_{MODE}_{time.strftime('%H%M%S')}"
WORK.mkdir()
STATE = WORK / "fake_state"
ch, sol, multi, single = B.load()
PICK = ["2ba387bc"] + multi[:2] + single[:2]
FP8 = B.TOKDIR_FP8
# the INT4 "model": the tokenizer fixture + a 3 MiB safetensors file, in the REAL Kaggle path shape (staging copies it)
MODELS = WORK / "inputs" / "models" / "foysalemonshanto"
INT4 = MODELS / "qwen3-8-27b-int4-redhat" / "pytorch" / "w4a16-g128" / "1"
shutil.copytree(B.TOKDIR_INT4, INT4)
hdr = json.dumps({"w": {"dtype": "U8", "shape": [3 << 20], "data_offsets": [0, 3 << 20]}}).encode()
hdr += b" " * (-len(hdr) % 8)
(INT4 / "model.safetensors").write_bytes(struct.pack("<Q", len(hdr)) + hdr + os.urandom(3 << 20))
DFL = MODELS / "qwen3-8-27b-dflash2" / "pytorch" / "bf16" / "1"      # CELL 1 still resolves the (unused) drafter path
DFL.mkdir(parents=True)
json.dump({"architectures": ["DFlash2DraftModel"]}, open(DFL / "config.json", "w"))
ARC = WORK / "arc"
ARC.mkdir()
shutil.copy(B.DATA / "arc-agi_evaluation_challenges.json", ARC / "arc-agi_evaluation_challenges.json")
shutil.copy(B.DATA / "arc-agi_evaluation_solutions.json", ARC / "arc-agi_evaluation_solutions.json")
json.dump({t: {"train": ch[t]["train"], "test": [{"input": q["input"]} for q in ch[t]["test"]]} for t in PICK},
          open(ARC / "arc-agi_test_challenges.json", "w"))
DEST = WORK / "shm"
DEST.mkdir()
# the payload dataset in the REAL Kaggle shape: /kaggle/input/datasets/<owner>/<slug>/<file>
if PAYLOAD_ENV:
    PAYLOAD = Path(PAYLOAD_ENV)
else:
    PAYLOAD = WORK / "inputs"
    _pd = PAYLOAD / "datasets" / "esfersami50" / "q27-e15b-payload"
    _pd.mkdir(parents=True)
    for _f in PAYSRC.glob("q27x_*.bin"):
        shutil.copy(_f, _pd / _f.name)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    PORT = s.getsockname()[1]
WIN = os.environ.get("E21H_HALF_SOLVE_SEC", "1500" if MODE == "death" else "150" if MODE == "gateAB" else
                     "420" if MODE == "kvoutage" else "90")
os.environ.update(Q27X_ARC_ROOT=str(ARC), Q27X_WORKING=str(WORK), Q27X_MODEL=str(FP8), Q27X_SKIP_INSTALL="1",
                  PYTHONIOENCODING="utf-8", E11_HARNESS="1", E11_PORT=str(PORT), E21_FAKE_SERVER=str(FAKE),
                  E11_FAKE_SERVER=str(FAKE), E11_INT4=str(INT4), E11_DFLASH=str(DFL),
                  E12_DIR=str(WORK / "e12_duck"), E14_DIR=str(WORK / "e14_mod"), E14_DEST=str(DEST), E14_TIME_SCALE=SCALE,
                  E14_FIXTURES=str(FIX), FAKE_SPEED="300", FAKE_STATE_DIR=str(STATE), Q27X_PAYLOAD_DIR=str(PAYLOAD),
                  ARC3_HTTP_RETRY_BASE_SECONDS="0.05", ARC3_HTTP_RETRY_MAX_SECONDS="0.1",
                  E12_MAX_GEN="6000", E12_R6_FIRST_EXPOSURE="0", E12_R6_MIN_ADMIT_SEC="5", E12_B_RESERVE="1000",
                  E12_B_ATTEMPT_TOKENS_B1="4000", E12_B_MAX_ATTEMPTS="3",
                  Q27X_GPU_WAIT_MAX="1",                  # no nvidia-smi here: _wait_gpu_release would wait 180 s per call
                  E21_HALF_SOLVE_SEC=WIN, E21_B_SETUP_RESERVE_SEC="300", E21_ANY_PROFILE="1",   # rev 2: profile assert
                  E21_FAKE_GPU_HELD="0",
                  E18_SILENCE_SEC=os.environ.get("E18_SILENCE_SEC", "15" if MODE == "halfAfail" else "120"))
for k in ("FAKE_REJECT", "FAKE_REJECT_FLAG", "FAKE_NEVER_READY", "FAKE_DIE_AFTER_SEC", "FAKE_DIE_FORM", "Q27X_PROFILE",
          "FAKE_QUIET", "FAKE_HOLD_ONE", "FAKE_METRICS_FAIL",
          "FAKE_NUM_BLOCKS", "FAKE_DROP_BLOCK_LINE", "FAKE_DIE_TIMES", "FAKE_ACCEPT_LEN",
          "E12_REASONING_EFFORT", "KAGGLE_IS_COMPETITION_RERUN", "Q27X_TIME_BUDGET", "E15B_WD_POLL_SEC", "E15B_WD_FAIL_SEC",
          "E15B_WD_MIN_LEFT_SEC"):
    os.environ.pop(k, None)
if MODE == "rerun":
    # RERUN_ALL has no solve window: the solve ends when every attempt is done (the fake's programs end quickly)
    os.environ.update(KAGGLE_IS_COMPETITION_RERUN="1", Q27X_TIME_BUDGET=os.environ.get("E21H_RERUN_BUDGET", "2400"))
else:
    # a long notebook budget: e21_begin_half skips the fallback when < 30 min are left, and CELL 1 asserts that two
    # windows + the half-B reserve fit; the short per-half windows (E21_HALF_SOLVE_SEC) keep the run short
    if SMOKE:            # the notebook's own default profile and budget (E21SMOKE: 12 tasks, 2 h); windows shortened
        os.environ.pop("E21_ANY_PROFILE", None)
    else:
        os.environ.update(Q27X_PROFILE=os.environ.get("E21H_PROFILE", "E15B" if MODE in ("gateAB", "kvoutage") else "EVAL6"),
                          Q27X_TIME_BUDGET="10800")
if MODE == "fallbackA":
    os.environ["FAKE_REJECT"] = "0.19:--mamba-ssm-cache-dtype"     # rev 2j: both variants' half A primary has SSM fp16
if MODE == "halfAfail":
    os.environ["FAKE_NEVER_READY"] = "0.19"
if MODE == "death":
    # rev 2b: two deaths -> the second relaunch switches to the half's other config (relaunch_cfg_switched)
    os.environ.update(FAKE_DIE_AFTER_SEC="8", FAKE_DIE_FORM="0.19", FAKE_DIE_TIMES="2", FAKE_SPEED="60",
                      E15B_WD_POLL_SEC="2", E15B_WD_FAIL_SEC="20", E15B_WD_MIN_LEFT_SEC="30")
if MODE == "wdhold":
    os.environ.update(FAKE_QUIET="0.19", E15B_WD_POLL_SEC="2")
if MODE == "overrun":
    os.environ.update(FAKE_HOLD_ONE="0.19:6:600")
if MODE == "kvoutage":
    os.environ.update(FAKE_METRICS_FAIL="0.19:3:150", FAKE_SPEED="40")
GATE_A_STREAMS = 10
if MODE == "gateAB":
    # 0.19 logs fewer blocks: FP8 / INT4 (B 1,600, 51 blocks per 57k request) 510 -> 10 streams; NB2's SSM-16 primary
    # (B 832, 84 blocks per 57k request) 850 -> 10 streams (510 would give 6 < 8: refused -> fallback)
    os.environ.update(FAKE_NUM_BLOCKS="0.19:850", FAKE_SPEED="100",       # rev 2j: both half A primaries SSM fp16
                      FAKE_ACCEPT_LEN="0.19:1.2")      # rev 2b: half A's MTP acceptance 1.2 < 1.5 -> mtp_acceptance_warning



import atexit
LOCK_PATH = Path(tempfile.gettempdir()) / "e21_harness_ports.lock"


def _pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except Exception:
        pass
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259           # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _lock_acquire(wait_sec):
    """One E21 harness run per host (two runs started at the same second both passed the port check, 2026-10-29).
    O_EXCL lock file holding the owner's pid; a dead owner's lock is taken over."""
    t_end = time.time() + wait_sec
    while True:
        try:
            fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {NBN} {MODE}".encode())
            os.close(fd)
            atexit.register(lambda: LOCK_PATH.unlink(missing_ok=True) if LOCK_PATH.exists()
                            and LOCK_PATH.read_text(errors="replace").split(" ")[0] == str(os.getpid()) else None)
            return True
        except FileExistsError:
            try:
                owner = LOCK_PATH.read_text(errors="replace").split(" ")[0]
                if owner.isdigit() and not _pid_alive(owner):
                    LOCK_PATH.unlink(missing_ok=True)
                    continue
            except Exception:
                pass
        if time.time() > t_end:
            return False
        time.sleep(3)


def _port_busy(port):
    with socket.socket() as s_:
        s_.settimeout(0.5)
        return s_.connect_ex(("127.0.0.1", port)) == 0


# the notebook's ports are fixed (half A 1234, half B 8011): another harness run on this host would answer our requests
# (Windows SO_REUSEADDR lets two servers bind one port). Wait up to E21H_PORT_WAIT_SEC (default 900) for them, else stop.
if not _lock_acquire(float(os.environ.get("E21H_PORT_WAIT_SEC", 900))):
    print(f"[e21 PORT_BUSY] another E21 harness run holds {LOCK_PATH}: not run", flush=True)
    print("@@E21_HARNESS_PORT_BUSY@@", flush=True)
    sys.exit(4)
_t_wait = time.time() + float(os.environ.get("E21H_PORT_WAIT_SEC", 900))
while any(_port_busy(p_) for p_ in (1234, 8011)) and time.time() < _t_wait:
    time.sleep(5)
if any(_port_busy(p_) for p_ in (1234, 8011)):
    print(f"[e21 PORT_BUSY] ports 1234 / 8011 are in use by another process (another harness run?): not run", flush=True)
    print("@@E21_HARNESS_PORT_BUSY@@", flush=True)
    sys.exit(4)
nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
heads = [c.splitlines()[0][:90] for c in code]
print(f"[e21 harness] {NBN} | variant {VAR} | mode {MODE} | {len(code)} code cells | work {WORK}", flush=True)
G = {"__name__": "__main__"}
t0 = time.time()
cell_sec, cell_err = [], None
LATE, HOLD = {"cb": {}}, {}
TWO_RUN_MODES = ("commit", "reasoningkey", "fallbackA", "death", "gateAB", "wdhold", "overrun", "kvoutage")


def _install_wrappers():
    """rev 2c regression hooks. (1) E12A.solve_all_r6: keep each half's on_branch_done; right after half B's solve
    returns, invoke half A's callback with an accepted record and compare half B's state before / after. (2) wdhold:
    urllib.request.urlopen holds half A's watchdog inside its first /health probe until half B is solving."""
    import threading
    import urllib.error
    import urllib.request
    E = G["E12A"]
    orig = E.solve_all_r6

    def wrapped(*a, **kw):
        H = G.get("E21_HALF")
        cb = kw.get("on_branch_done")
        LATE["cb"][H] = cb
        out = orig(*a, **kw)
        if H == "B" and LATE["cb"].get("A") is not None and MODE in TWO_RUN_MODES:
            try:
                cbA = LATE["cb"]["A"]
                co = cbA.__code__
                names = co.co_varnames[co.co_argcount - len(cbA.__defaults__):co.co_argcount]
                dA = dict(zip(names, cbA.__defaults__))
                src = next((r for v in dA["_recs"].values() for r in v), None)
                snap = lambda: (json.dumps(G["E12_RECORDS"], sort_keys=True, default=str),
                                G["E12_RUN"].get("branches_accepted"), (WORK / "submission.json").read_bytes())
                b0, n0 = snap(), dA["_run"].get("late_cross_half_callbacks", 0)
                cbA(dict(src, publish_seq=10 ** 9, accepted=True, status="done"))
                b1 = snap()
                LATE.update(ran=True, same=b0 == b1, same_parts=[x == y for x, y in zip(b0, b1)], half_bound=dA.get("_half"),
                            late_delta=dA["_run"].get("late_cross_half_callbacks", 0) - n0,
                            b_recs_is_a=dA["_recs"] is G["E12_RECORDS"])
            except Exception as e:
                LATE.update(ran=False, error=repr(e)[:400])
        return out
    E.solve_all_r6 = wrapped
    if MODE == "gpuheldB":
        _orig_begin = G["e21_begin_half"]

        def begin(H):
            if H == "B":
                os.environ["E21_FAKE_GPU_HELD"] = "1"
            return _orig_begin(H)
        G["e21_begin_half"] = begin
    if MODE == "wdhold":
        real = urllib.request.urlopen

        def urlopen(url, *a, **kw):
            u = url if isinstance(url, str) else getattr(url, "full_url", "")
            th = threading.current_thread()
            if (u.endswith("/health") and th.name == "e15b_watchdog" and G.get("E21_HALF") == "A" and not HOLD):
                HOLD.update(thread=th, t0=time.time(), half_at_start="A")
                t_end = time.time() + 600
                while time.time() < t_end and not (G.get("E21_HALF") == "B" and (G.get("E12_RUN") or {}).get("started_sec")):
                    time.sleep(0.5)
                HOLD.update(t1=time.time(), half_at_release=G.get("E21_HALF"))
                raise urllib.error.URLError("harness: held /health released")
            return real(url, *a, **kw)
        urllib.request.urlopen = urlopen


for i, c in enumerate(code):
    t = time.time()
    if i == 7 and "E12A" in G and "e21_begin_half" in G:      # after CELL 5 (the engine library)
        _install_wrappers()
    try:
        exec(compile(c, f"<cell{i}>", "exec"), G)
    except BaseException:
        import traceback
        cell_err = (i, heads[i], traceback.format_exc()[-4000:])
        print(f"[e21 harness] UNCAUGHT in code cell {i} ({heads[i]}):\n{cell_err[2]}", flush=True)
        break
    finally:
        cell_sec.append(round(time.time() - t, 1))
faulthandler.cancel_dump_traceback_later()

checks = []


def check(label, cond, info=""):
    try:
        ok = bool(cond() if callable(cond) else cond)
    except Exception as e:
        ok, info = False, f"check raised {type(e).__name__}: {e} | {info}"
    checks.append((label, ok, "" if ok else str(info)[:1500]))


def jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception as e:
        return {"__error__": repr(e)[:200]}


Q = WORK / "q27x"
st = G.get("E21_STATE") or {}
halves_run = list(G.get("E21_HALVES") or [])
want_halves = ["A"] if MODE == "rerun" else ["A", "B"]
launches = [json.loads(l) for l in open(STATE / "launches.jsonl", encoding="utf-8")] if (STATE / "launches.jsonl").exists() else []
stats = {}
for p in STATE.glob("stats_*.json"):
    d = jload(p)
    if "pid" in d:
        stats[d["pid"]] = d
port_of = lambda L: int(L["argv"][L["argv"].index("--port") + 1]) if "--port" in L["argv"] else None
la = [L for L in launches if port_of(L) == 1234]
lb = [L for L in launches if port_of(L) == 8011]
exp_seqs = 20 if VAR == "NB1" else 16       # half B's primary (rev 2i: NB1 half A = FP8, 16)
EXP_A_SEQS = 18 if VAR == "NB1" else 16      # rev 2j
sub = jload(WORK / "submission.json")
test = jload(ARC / "arc-agi_test_challenges.json")

check("notebook: every code cell ran without an uncaught exception", cell_err is None, cell_err)
check("notebook: 10 code cells (CELL 0, 1, 1b, 2, 3, order, 5 engine, 6/7 functions, 8 runner, 9 report)",
      len(code) == 10 and "CELL 5" in heads[6] and "CELL 8" in heads[8] and "CELL 9" in heads[9], heads)
check(f"halves run = {want_halves} (rerun: half A only)", halves_run == want_halves, halves_run)
check("submission.json: every test task, every test input, attempt_1 + attempt_2 grids (lists of lists)",
      lambda: set(sub) == set(test) and all(len(sub[t]) == len(test[t]["test"]) for t in test)
      and all(isinstance(a, dict) and all(isinstance(a.get(k), list) and a[k] and all(isinstance(r, list) for r in a[k])
                                          for k in ("attempt_1", "attempt_2")) for t in sub for a in sub[t]),
      list(sub)[:8])
check("submission.json written atomically, no temp file left", not list(WORK.glob("submission.json.*")),
      list(WORK.glob("submission.json*")))
# ---- launches: argv form per half, Duck flags, the variant's config
check("every half-A launch (port 1234) used the vLLM 0.19 api_server form (--model <path>)",
      la and all(L["form"] == "0.19" and L["argv"][0] == "--model" for L in la), [(L["form"], L["argv"][:2]) for L in la])
if MODE == "gpuheldB":
    check("gpuheldB: half B made 0 launches (port 8011)", not lb, [(L["form"], L["outcome"]) for L in lb])
elif "B" in want_halves:
    check("every half-B launch (port 8011) used the vLLM 0.32 `vllm serve <path>` form", lb and all(
        L["form"] == "0.32" and not L["argv"][0].startswith("--") for L in lb), [(L["form"], L["argv"][:2]) for L in lb])
else:
    check("rerun: half B never launched (no 0.32-form launch, no port 8011)",
          not lb and not any(L["form"] == "0.32" for L in launches), [(L["form"], port_of(L)) for L in launches])
check("no launch was refused for missing Duck serving flags (fake exit 2)", not any(L["outcome"] == "refused" for L in launches),
      [L.get("problems") for L in launches if L["outcome"] == "refused"])


def first_ready(ls):
    return next((L for L in ls if L["outcome"] == "ready"), None)


def flag(L, k):
    return L["argv"][L["argv"].index(k) + 1] if L and k in L["argv"] else None


fa, fb = first_ready(la), first_ready(lb)
model_of = lambda L: (L["argv"][1] if L["form"] == "0.19" else L["argv"][0]) if L else ""
if MODE not in ("fallbackA", "halfAfail"):
    if VAR == "NB1":
        check("NB1 half A primary (rev 2j): FP8 weights, --max-num-seqs 18, --mamba-ssm-cache-dtype float16; no 0.19 INT4 launch",
              fa and "w4a16" not in model_of(fa) and flag(fa, "--max-num-seqs") == "18"
              and flag(fa, "--mamba-ssm-cache-dtype") == "float16"
              and not any("w4a16" in model_of(L) for L in la), fa and fa["argv"])
    else:
        check("NB2 half A primary: FP8 weights, --max-num-seqs 16, --mamba-ssm-cache-dtype float16",
              fa and "w4a16" not in model_of(fa) and flag(fa, "--max-num-seqs") == "16"
              and flag(fa, "--mamba-ssm-cache-dtype") == "float16", fa and fa["argv"])
if "B" in want_halves and MODE not in ("fallbackA", "gpuheldB"):
    check(f"{VAR} half B primary: {'INT4' if VAR == 'NB1' else 'FP8'} weights / seqs {exp_seqs} / SSM {'default' if VAR == 'NB1' else 'fp16'}",
          fb and flag(fb, "--max-num-seqs") == str(exp_seqs) and (("w4a16" in model_of(fb)) == (VAR == "NB1"))
          and ((flag(fb, "--mamba-ssm-cache-dtype") == "float16") == (VAR == "NB2")), fb and fb["argv"])
if MODE == "fallbackA" and VAR == "NB1":     # rev 2i: NB1 never follows A; half B runs its own INT4 primary
    check("fallbackA NB1: half B launched its own primary (INT4, seqs 20), first try; no spec_note",
          fb and lb and lb[0] is fb and "w4a16" in model_of(fb) and flag(fb, "--max-num-seqs") == "20"
          and not (((st.get("halves") or {}).get("B") or {}).get("engine") or {}).get("spec_note"), fb and fb["argv"])
elif MODE == "fallbackA":      # rev 2 E21_B_FOLLOWS_A: a half-A fallback is repeated in half B (only the version differs)
    check(f"fallbackA {VAR}: half B launched half A's FALLBACK config (FP8, seqs 16, no SSM flag), first try",
          fb and lb and lb[0] is fb and "w4a16" not in model_of(fb) and flag(fb, "--max-num-seqs") == "16"
          and "--mamba-ssm-cache-dtype" not in fb["argv"], fb and fb["argv"])
    _eb = (((st.get("halves") or {}).get("B") or {}).get("engine") or {})
    check("fallbackA: half B engine record has spec_note (E21_B_FOLLOWS_A) and the e21_report row carries it",
          _eb.get("spec_note") and ((jload(WORK / "q27x" / "e21_report.json").get("halves") or {}).get("B") or {}).get("spec_note"),
          {k: _eb.get(k) for k in ("spec_note", "spec", "used")})
check("served model names are never a Duck alias and are the E21 names",
      all(flag(L, "--served-model-name") in ("Qwen/Qwen3.8-27B-FP8", "Qwen/Qwen3.8-27B-INT4") for L in launches),
      [flag(L, "--served-model-name") for L in launches])
# ---- capacity: real running <= max-num-seqs, stats per server
check("real running never exceeded --max-num-seqs on any server (the fake's own counter)",
      stats and all(d["max_running"] <= d["max_num_seqs"] for d in stats.values()),
      {p: (d["form"], d["max_running"], d["max_inflight"], d["max_num_seqs"]) for p, d in stats.items()})
# ---- the reasoning key (vLLM reads only message["reasoning"])
hr = sum(d.get("hist_msgs_reasoning", 0) + d.get("hist_msgs_both", 0) for d in stats.values())
hrc = sum(d.get("hist_msgs_reasoning_content_only", 0) for d in stats.values())
rrc = sum(d.get("requests_history_reasoning_content_only", 0) for d in stats.values())
check("reasoning key: assistant history carried `reasoning` (> 0 messages) and never only `reasoning_content`",
      hr > 0 and hrc == 0 and rrc == 0, {"reasoning": hr, "reasoning_content_only_msgs": hrc, "requests_rc_only": rrc})
check("reasoning key: ARC3_REASONING_HISTORY_KEY == 'reasoning' at the end of the run",
      G.get("os") and G["os"].environ.get("ARC3_REASONING_HISTORY_KEY") == "reasoning",
      G.get("os") and G["os"].environ.get("ARC3_REASONING_HISTORY_KEY"))
check("no request exceeded --max-model-len (fake 400s)", sum(d.get("context_rejections", 0) for d in stats.values()) == 0,
      {p: d.get("context_rejections") for p, d in stats.items()})
# ---- per half: engine, Duck run, artefacts
expect_engine = {"A": MODE != "halfAfail", "B": MODE != "gpuheldB"}
for H in want_halves:
    hs = (st.get("halves") or {}).get(H) or {}
    eng = hs.get("engine") or {}
    d = Q / f"half{H}"
    rep = jload(d / "e12_run_report.json")
    e = rep.get("e12_run") or {}
    if expect_engine[H]:
        want_used = "fallback" if (MODE == "fallbackA" and H == "A") else "primary"
        _hseqs = EXP_A_SEQS if H == "A" else exp_seqs
        check(f"half {H}: engine ready on the {want_used} config ({eng.get('backend')})", eng.get("used") == want_used
              and str(eng.get("backend") or "").startswith("vllm-" + ("0.19.0" if H == "A" else "0.32.0")),
              {k: eng.get(k) for k in ("used", "backend", "fatal")} | {"attempts": [(a.get("tag"), a.get("why"), str(a.get("error"))[:200]) for a in eng.get("attempts") or []]})
        if MODE == "fallbackA" and H == "A":
            a0 = (eng.get("attempts") or [{}])[0]
            check("fallbackA: half A primary rejected at start (exited), then the fallback config was ready",
                  a0.get("tag") == "primary" and "exited" in str(a0.get("why") or a0.get("error"))
                  and any(L["outcome"] == "rejected" for L in la), (a0.get("why"), a0.get("error")))
            fl = first_ready(la)
            if VAR == "NB1":
                check("fallbackA NB1: half A fallback = FP8 weights, --max-num-seqs 16, no SSM flag (rev 2j: the S1 engine)",
                      fl and "w4a16" not in model_of(fl) and flag(fl, "--max-num-seqs") == "16"
                      and "--mamba-ssm-cache-dtype" not in fl["argv"], fl and fl["argv"])
            else:
                check("fallbackA NB2: half A fallback = FP8 without --mamba-ssm-cache-dtype, seqs 16",
                      fl and "--mamba-ssm-cache-dtype" not in fl["argv"] and flag(fl, "--max-num-seqs") == "16",
                      fl and fl["argv"])
        cap = eng.get("capacity") or {}
        want_streams = (GATE_A_STREAMS if (MODE == "gateAB" and H == "A")
                        else 16 if (MODE == "fallbackA" and (VAR == "NB2" or H == "A")) else _hseqs)
        check(f"half {H}: capacity check from the vLLM lines -> {want_streams} streams; Duck streams / workers = that",
              cap.get("streams") == want_streams and e.get("streams") == want_streams and e.get("rev6_workers") == want_streams,
              (cap, e.get("streams"), e.get("rev6_workers")))
        check(f"half {H}: rev 2i content probes (7 * 6, copy, sequence) passed on the ready launch",
              any(len(a.get("content_probes") or []) == 3 and all(x.get("ok") and x.get("clean") for x in a["content_probes"])
                  for a in eng.get("attempts") or [] if (a.get("smoke") or {}).get("ok")),
              [a.get("content_probes") for a in eng.get("attempts") or []])
        check(f"half {H}: smoke OK, /v1/models = the served name", any((a.get("smoke") or {}).get("ok") for a in eng.get("attempts") or [])
              and any(a.get("models") for a in eng.get("attempts") or []), [(a.get("smoke"), a.get("models")) for a in eng.get("attempts") or []])
        check(f"half {H}: Duck ran without solver error; branches started; isolation preflight PASS",
              not e.get("error") and (e.get("branches_started") or 0) > 0
              and (e.get("rev6_preflight") or {}).get("isolation_state") == "PASS",
              {k: e.get(k) for k in ("error", "branches_started", "branches_done")} | {"iso": (e.get("rev6_preflight") or {}).get("isolation_state")})
        bs = e.get("b_summary") or {}
        sl = bs.get("server_log") or {}
        check(f"half {H}: b_summary from the vLLM probes (gauges, server-side tokens, stats-line summary with lines)",
              not bs.get("error") and bs.get("gauge_samples", 0) >= 0 and bs.get("server_gen_tokens") is not None
              and ((sl.get("decode_lines") or 0) > 0 or (MODE == "wdhold" and H == "A"))
              and sl.get("startup", {}).get("kv_tokens_logged"),
              {k: bs.get(k) for k in ("error", "server_gen_tokens", "server_running_mean", "gauge_samples")} | {"sl": sl})
        check(f"half {H}: no worker error, no stale SERVER_DOWN", not bs.get("worker_errors"), bs.get("worker_errors"))
        # rev 2i: the capacity record is the half's FIRST ready server (a switching relaunch keeps it, VAL2_engine F11)
        _fr = first_ready(la if H == "A" else lb)
        _ssm16 = bool(_fr) and flag(_fr, "--mamba-ssm-cache-dtype") == "float16"
        check(f"half {H}: capacity from the rev 2 vLLM lines (block size {832 if _ssm16 else 1600}, num_blocks, F = 15 with MTP-3)",
              cap.get("block_size") == (832 if _ssm16 else 1600) and cap.get("num_blocks")
              and cap.get("fixed_blocks_per_request") == 15, cap)
        _ct = next((a.get("contract") for a in reversed(eng.get("attempts") or []) if a.get("contract")), None) or {}
        check(f"half {H}: contract probe passed with a real tool call and a turn-2 history check",
              _ct.get("ok") is True and _ct.get("tool_calls") == 1 and isinstance(_ct.get("turn2_shortfall"), int)
              and _ct["turn2_shortfall"] <= 0 and (_ct.get("reasoning_chars") or 0) > 0, _ct)
        _br = rep.get("branches") or []
        _drop = sum(int((b.get("rev5_events") or {}).get("cut_tool_calls_dropped") or 0) for b in _br)
        _lr = (((jload(WORK / "q27x" / "e21_report.json").get("halves") or {}).get(H) or {}).get("length_restored"))
        # the fake counts a cut reply once it was written to the client (a reply the client abandoned at its deadline or
        # a server death never reaches Duck)
        _cuts = sum(d_.get("cut_delivered", d_.get("cut_tool_calls", 0)) for d_ in stats.values()
                    if d_.get("port") == (1234 if H == "A" else 8011))
        check(f"half {H}: cut tool calls (fake: finish 'tool_calls' at max_tokens) restored to 'length' (exactly those) and handled by Duck",
              _cuts > 0 and (_lr == _cuts if MODE != "death" else 0 < _lr <= _cuts)    # death: a written reply can still be lost
              and (_drop > 0 or (bs.get("cut_code_saved") or 0) > 0),
              {"fake_cut_calls": _cuts, "length_restored": _lr, "cut_tool_calls_dropped": _drop,
               "cut_code_saved": bs.get("cut_code_saved")})
        check(f"half {H}: prefix cache reported (cached tokens > 0 in the Duck usage)",
              ((rep.get("usage_all") or {}).get("cached_tokens") or 0) > 0, rep.get("usage_all"))
        if MODE != "rerun":
            check(f"half {H}: scored (eval_task_norm_pct, pass1, pass2)", rep.get("eval_task_norm_pct") is not None
                  and rep.get("eval_pass1_hits") is not None, {k: rep.get(k) for k in ("eval_query_hits", "eval_task_norm_pct")})
        logs = sorted(p.name for p in d.glob("vllm_server_*.log"))
        check(f"half {H}: artefacts: e12_run_report.json, vllm_server_half{H}_<tag>.log, sglang_decode_summary.json, duck/, "
              f"Q27X_E21_{VAR}_HALF{H}_DUCK_BUNDLE.zip",
              "__error__" not in rep and logs and all(n.startswith(f"vllm_server_half{H}_") for n in logs)
              and (d / "sglang_decode_summary.json").exists() and (d / "duck").is_dir()
              and (WORK / f"Q27X_E21_{VAR}_HALF{H}_DUCK_BUNDLE.zip").exists(),
              (logs, sorted(p.name for p in d.glob("*")) if d.exists() else None))
    else:
        check(f"half {H}: no engine (expected in {MODE}); " + ("no launch at all (GPUs held)" if MODE == "gpuheldB" else
              "both configs tried") + "; Duck not run; recorded, not raised",
              not eng.get("used") and eng.get("fatal")
              and (len(eng.get("attempts") or []) == 0 and eng.get("fatal") == "GPU memory still held before the launch"
                   and any(c.get("fake_held") for c in eng.get("gpu_checks") or []) if MODE == "gpuheldB"
                   else len(eng.get("attempts") or []) == 2)
              and "not run" in str(e.get("error") or (hs.get("e12_run") or {}).get("error") or ""),
              {k: eng.get(k) for k in ("used", "fatal")} | {"attempts": [(a.get("tag"), a.get("why"), str(a.get("error"))[:200]) for a in eng.get("attempts") or []], "e12_error": e.get("error")})
    check(f"half {H}: no step error recorded by the runner (begin / cell6 / cell7 / end)", not hs.get("errors"),
          [(x.get("step"), x.get("traceback", "")[-1500:]) for x in hs.get("errors") or []])
    check(f"half {H}: the server was stopped at the half's end (end_sec set)", hs.get("end_sec") is not None, hs.get("end_sec"))
    _last = H == want_halves[-1]
    check(f"half {H}: e21_end_half {'did NOT wait for the GPU release (last half)' if _last else 'waited for the GPU release'}",
          ("gpu_release" in hs) == (not _last), {k: hs.get(k) for k in ("gpu_release", "end_sec")})
    if expect_engine[H]:
        _r = (jload(Q / "e21_report.json").get("halves") or {}).get(H) or {}
        check(f"half {H}: e21_report row has recomputed_prompt_tokens, counters (server), contract ok, gate",
              isinstance(_r.get("recomputed_prompt_tokens"), int) and _r.get("prompt_tokens", 0) >= _r["recomputed_prompt_tokens"] > 0
              and isinstance(_r.get("counters"), dict) and "error" not in _r["counters"]
              and _r["counters"].get("vllm:prefix_cache_queries_total", 0) > 0 and "vllm:num_preemptions_total" in _r["counters"]
              and (_r.get("contract") or {}).get("ok") is True and isinstance(_r.get("gate"), dict)
              and isinstance((_r.get("gate") or {}).get("stats"), dict),
              {k: _r.get(k) for k in ("recomputed_prompt_tokens", "prompt_tokens", "counters", "contract", "gate")})
        _s1, _p1 = _r.get("stage1") or {}, _r.get("score_attempt1_strict_pass1")
        _nq = sum(len(G["CHALLENGES"][t]["test"]) for t in G.get("TASK_IDS") or [])
        check(f"half {H}: rev 2b report fields: stage1 (every attempt 1 present), strict pass-1 score, window_sec",
              _s1.get("tasks") == len(G.get("TASK_IDS") or []) and _s1.get("attempt1_missing") == 0
              and (_s1.get("attempt1_records") or 0) >= _s1["tasks"]
              and (_p1 is None if MODE == "rerun" else (isinstance(_p1, dict) and _p1.get("queries") == _nq
                                                         and 0 <= _p1.get("pass1_query_hits", -1) <= _nq))
              and (_r.get("window_sec") is None if MODE == "rerun" else float(_r.get("window_sec") or 0) > 0),
              {"stage1": _s1, "pass1": _p1, "window_sec": _r.get("window_sec"), "queries": _nq})
        check(f"half {H}: e21_report prefix_cache_hit_pct_last (stats lines) and prefix_hit_pct_whole_run (counters) are filled",
              (isinstance(_r.get("prefix_cache_hit_pct_last"), (int, float)) or (MODE == "wdhold" and H == "A"))
              and isinstance(_r.get("prefix_hit_pct_whole_run"), (int, float)) and 0 < _r["prefix_hit_pct_whole_run"] <= 100,
              {"report": _r.get("prefix_cache_hit_pct_last"), "whole_run": _r.get("prefix_hit_pct_whole_run"),
               "server_log_keys": sorted(((((hs.get("e12_run") or {}).get("b_summary") or {}).get("server_log")) or {}).keys())})
        _mw = hs.get("mtp_acceptance_warning")
        if MODE == "gateAB":
            check(f"half {H}: mtp_acceptance_warning {'= 1.2 (fake 0.19 acceptance 1.2 < 1.5)' if H == 'A' else 'absent (3.08)'}",
                  (_mw == 1.2) if H == "A" else (_mw is None),
                  {"warning": _mw, "decode_summary_keys": sorted((hs.get("decode_summary") or {}).keys())})
        else:
            check(f"half {H}: no mtp_acceptance_warning at the fake's 3.08 acceptance", _mw is None, _mw)
if "A" in want_halves and "B" in want_halves:
    ea, eb = (st.get("halves") or {}).get("A", {}).get("end_sec"), ((st.get("halves") or {}).get("B", {}).get("engine") or {}).get("begin_sec")
    check("half B began after half A ended (one server at a time)", ea is not None and eb is not None and eb >= ea - 1, (ea, eb))
    tA = [L["t"] for L in la]
    tB = [L["t"] for L in lb]
    check("every half-B launch came after every half-A launch", not tA or not tB or min(tB) > max(tA), (tA, tB))
R21 = jload(Q / "e21_report.json")
check("e21_report.json: one row per half with score_full, score_attempt1_only, engine_used, config_used, attempts, errors",
      "__error__" not in R21 and set((R21.get("halves") or {})) == set(want_halves)
      and all({"score_full", "score_attempt1_only", "engine_used", "config_used", "attempts", "errors"} <= set(h)
              for h in (R21.get("halves") or {}).values()), {k: sorted(v) for k, v in (R21.get("halves") or {}).items()})
check(f"final bundle Q27X_E21_{VAR}_BUNDLE.zip holds e21_report.json + submission.json",
      lambda: {"q27x/e21_report.json", "submission.json"} <= set(zipfile.ZipFile(WORK / f"Q27X_E21_{VAR}_BUNDLE.zip").namelist()),
      list(WORK.glob("*.zip")))
if all(expect_engine[H] for H in want_halves) and len(want_halves) == 2:
    _dl = [((R21.get("halves") or {}).get(H) or {}).get("deadline_in_sec") for H in want_halves]
    check("both halves got the same solve window (deadline_in_sec within 300 s; windows_equal true)",
          all(isinstance(x, (int, float)) for x in _dl) and abs(_dl[0] - _dl[1]) <= 300 and R21.get("windows_equal") is True,
          (_dl, R21.get("windows_sec"), R21.get("windows_equal")))
# ---- rev 2d: the scheduler handoff of a non-last half; the SMOKE profile
for H in want_halves[:-1]:
    hs_ = (st.get("halves") or {}).get(H) or {}
    check(f"half {H}: handoff: no Duck scheduler thread left (handoff_scheduler_threads_left), wait recorded",
          "handoff_scheduler_threads_left" in hs_ and not hs_["handoff_scheduler_threads_left"]
          and isinstance(hs_.get("handoff_wait_sec"), (int, float)) and hs_["handoff_wait_sec"] <= 120,
          {k: hs_.get(k) for k in ("handoff_scheduler_threads_left", "handoff_wait_sec")})
if SMOKE:
    check("SMOKE: the notebook's default profile E21SMOKE ran (12 eval tasks, 20-min profile window, 2.5-h budget (rev 2f); "
          "window shortened by E21_HALF_SOLVE_SEC only)",
          G.get("PROFILE") == "E21SMOKE" and len(G.get("TASK_IDS") or []) == 12
          and (G.get("PROFILE_TABLE") or {}).get("E21SMOKE", {}).get("solve_window_sec") == 20 * 60
          and G.get("TOTAL_BUDGET_SEC") == 2.5 * 3600 and abs(float(G.get("E12_SOLVE_WINDOW_SEC") or 0) - float(WIN)) <= 5,
          (G.get("PROFILE"), len(G.get("TASK_IDS") or []), G.get("TOTAL_BUDGET_SEC"), G.get("E12_SOLVE_WINDOW_SEC")))
# ---- rev 2c: late-callback isolation, deadline contract, report status
if MODE in TWO_RUN_MODES:
    check("rev 2c late-callback isolation: half A's callback invoked after half B's solve left half B's records, "
          "branches_accepted and submission.json byte-identical; late_cross_half_callbacks +1; bound to half A",
          LATE.get("ran") and LATE.get("same") and LATE.get("late_delta") == 1 and LATE.get("half_bound") == "A"
          and LATE.get("b_recs_is_a") is False, {k: v for k, v in LATE.items() if k != "cb"})
for H in want_halves:
    _r = (R21.get("halves") or {}).get(H) or {}
    if expect_engine[H]:
        check(f"half {H}: deadline contract: solve_overrun_sec <= 120, no deadline_violation, end_sec <= planned_end_sec + 300",
              isinstance(_r.get("solve_overrun_sec"), (int, float)) and _r["solve_overrun_sec"] <= 120
              and not _r.get("deadline_violation") and isinstance(_r.get("planned_solve_end_sec"), (int, float))
              and isinstance(_r.get("end_sec"), (int, float)) and isinstance(_r.get("planned_end_sec"), (int, float))
              and _r["end_sec"] <= _r["planned_end_sec"] + 300,
              {k: _r.get(k) for k in ("solve_overrun_sec", "deadline_violation", "planned_solve_end_sec", "end_sec",
                                      "planned_end_sec", "stragglers")})
        check(f"half {H}: lmo_in_log True (the fake prints 'language_model_only': True in its non-default args line)",
              _r.get("lmo_in_log") is True, _r.get("lmo_in_log"))
_he_a = (((st.get("halves") or {}).get("A") or {}).get("engine") or {})
_want_cs = ("NOT_RUN" if MODE in ("halfAfail", "gpuheldB") else
            ("CONFOUNDED_FALLBACK" if VAR == "NB1" else "FALLBACK_BOTH") if MODE == "fallbackA" else
            "SWITCHED_MIDHALF" if (MODE == "death" and _he_a.get("relaunch_cfg_switched")) else "PRIMARY_BOTH")
check(f"rev 2c/2e: config_status = {_want_cs}" + (" (half A switched to its fallback in a relaunch)" if MODE == "death" else "")
      + "; configs_ran recorded per half",
      R21.get("config_status") == _want_cs and (MODE in ("halfAfail", "gpuheldB") or isinstance(R21.get("configs_ran"), (dict, list))), {"config_status": R21.get("config_status"),
                                               "used": {H: (h.get("config_used"), h.get("spec_note")) for H, h in (R21.get("halves") or {}).items()},
                                               "relaunch_cfg_switched": _he_a.get("relaunch_cfg_switched"),
                                               "configs_ran": R21.get("configs_ran")})
# gateAB: the halves ran 10 vs full streams; since rev 2e (kit run 2) unequal streams make the comparison invalid
_want_cv = (MODE in TWO_RUN_MODES and _want_cs in ("PRIMARY_BOTH", "FALLBACK_BOTH") and MODE != "gateAB"
            and VAR != "NB1")       # rev 2i: NB1's halves differ by design (FP8 16 vs INT4 20): never a valid A/B
check(f"rev 2c/2e: comparison_valid = {_want_cv}" + (" (gateAB: 10 vs full streams -> invalid, rev 2e rule)" if MODE == "gateAB" else ""),
      R21.get("comparison_valid") is _want_cv,
      {"comparison_valid": R21.get("comparison_valid"), "windows_sec": R21.get("windows_sec"),
       "streams": {H: h.get("streams") for H, h in (R21.get("halves") or {}).items()}})
if MODE == "wdhold":
    _hA, _hB = ((st.get("halves") or {}).get("A") or {}), ((st.get("halves") or {}).get("B") or {})
    th = HOLD.get("thread")
    check("wdhold: half A's watchdog was held in /health from half A until half B was solving, then released",
          HOLD.get("half_at_start") == "A" and HOLD.get("half_at_release") == "B" and _hA.get("e15b_watchdog_alive_after_join") is True,
          {k: v for k, v in HOLD.items() if k != "thread"} | {"alive_after_join_A": _hA.get("e15b_watchdog_alive_after_join")})
    check("wdhold: half A's watchdog thread exited after the release; no relaunch in half A or half B",
          th is not None and not th.is_alive() and not ((_hA.get("e12_run") or {}).get("watchdog") or {}).get("relaunches")
          and not ((_hB.get("e12_run") or {}).get("watchdog") or {}).get("relaunches")
          and not (_hB.get("engine") or {}).get("relaunches") and len([L for L in lb if L["outcome"] == "ready"]) == 1,
          {"alive": th is not None and th.is_alive(), "A": ((_hA.get("e12_run") or {}).get("watchdog") or {}).get("relaunches"),
           "B": ((_hB.get("e12_run") or {}).get("watchdog") or {}).get("relaunches"), "B_launches": [L["outcome"] for L in lb]})
if MODE == "overrun":
    _rA = (R21.get("halves") or {}).get("A") or {}
    _held = [x for d_ in stats.values() if d_.get("port") == 1234 for x in d_.get("held") or []]
    check("overrun: the fake held exactly one half-A response for 600 s, in flight past the solve deadline",
          len(_held) == 1 and _held[0]["sec"] == 600 and isinstance(_rA.get("planned_solve_end_sec"), (int, float))
          and _held[0]["t0"] + 600 > _rA["planned_solve_end_sec"] + (time.time() - (time.perf_counter() - G["NOTEBOOK_T0"]))
          and _held[0]["t0"] < _rA["planned_solve_end_sec"] + (time.time() - (time.perf_counter() - G["NOTEBOOK_T0"])),
          {"held": _held, "planned_solve_end_sec": _rA.get("planned_solve_end_sec")})
    check("overrun: half A's solver returned within 120 s of its deadline (stragglers = information) and half A ended "
          "<= its solve start + window + 300 s; the submission stayed complete (checked above)",
          (_rA.get("solve_overrun_sec") or 0) <= 120
          and _rA.get("end_sec") is not None and _rA.get("solve_start_sec") is not None
          and _rA["end_sec"] - _rA["solve_start_sec"] <= (_rA.get("deadline_in_sec") or 0) + 300,
          {k: _rA.get(k) for k in ("solve_overrun_sec", "stragglers", "end_sec", "solve_start_sec", "deadline_in_sec",
                                   "late_cross_half_callbacks")})
if MODE == "kvoutage":
    _sa = [d_ for d_ in stats.values() if d_.get("port") == 1234]
    _w = next((d_.get("metrics_fail_window") for d_ in _sa if d_.get("metrics_fail_window")), None)
    _bad_starts = [x for d_ in _sa for x in d_.get("conv_starts") or []
                   if _w and _w[0] + 7 <= x[0] <= min(_w[1], _w[0] + 115) and x[1] > 0]
    _after = [x for d_ in _sa for x in d_.get("conv_starts") or [] if _w and _w[0] + 125 <= x[0] <= _w[1]]
    _in = [x for d_ in _sa for x in d_.get("conv_starts") or [] if _w and _w[0] <= x[0] <= _w[1]]
    _schA = (((st.get("halves") or {}).get("A") or {}).get("e12_run") or {}).get("rev6_scheduler") or {}
    check("kvoutage: /metrics failed (fake 500s) and the notebook counted probe failures (KV reported 1.0)",
          sum(d_.get("metrics_500", 0) for d_ in _sa) > 0 and (st.get("probe_failures") or 0) > 0,
          {"metrics_500": [d_.get("metrics_500") for d_ in _sa], "probe_failures": st.get("probe_failures")})
    check("kvoutage (rev 2d): after 120 s of outage Duck's fail-open applied (probe_fail_open > 0)",
          (st.get("probe_fail_open") or 0) > 0, {"probe_fail_open": st.get("probe_fail_open"),
                                                  "starts_after_120s": _after[:10]})
    check("kvoutage: no new branch reached the server while requests ran in the outage's first 115 s (7-s grace); gate held",
          _w and not _bad_starts and ((_schA.get("kv_holds") or 0) > 0 or (_schA.get("kv_idle_overrides") or 0) > 0),
          {"window": _w, "violations": _bad_starts[:10], "starts_in_window": _in[:10],
           "kv_holds": _schA.get("kv_holds"), "kv_idle_overrides": _schA.get("kv_idle_overrides")})
check("rev 2b: tool_calls_with_content recorded in e21_report (a count; the fake never adds text to a call -> 0)",
      isinstance(R21.get("tool_calls_with_content"), dict) and R21["tool_calls_with_content"].get("n") == 0,
      R21.get("tool_calls_with_content"))
check("rev 2b: vLLM 0.32 background install thread " + ("never started (rerun)" if MODE == "rerun" else "started once half A was up"),
      (st.get("v032_thread") is None) if MODE == "rerun" else
      (st.get("v032_thread") is not None or MODE == "halfAfail") and isinstance(st.get("v032_install"), dict)
      and not st["v032_install"].get("error"), {k: (str(v)[:200]) for k, v in st.items() if k.startswith("v032")})
_tj = (G.get("os") and G["os"].environ.get("E12_TOKENIZER_JSON")) or ""
check("rev 2b: E12_TOKENIZER_JSON points to an existing tokenizer.json", _tj.endswith("tokenizer.json") and Path(_tj).is_file(), _tj)
if MODE == "gateAB":
    _hA, _hB = ((R21.get("halves") or {}).get("A") or {}), ((R21.get("halves") or {}).get("B") or {})
    check(f"gateAB: half A got {GATE_A_STREAMS} streams (fewer KV blocks), half B {exp_seqs}",
          _hA.get("streams") == GATE_A_STREAMS and _hB.get("streams") == exp_seqs, (_hA.get("streams"), _hB.get("streams")))
    _sb = [d_ for d_ in stats.values() if d_.get("port") == 8011]
    _sa = [d_ for d_ in stats.values() if d_.get("port") == 1234]
    check(f"gateAB: half B's server really ran {exp_seqs} requests at once; half A's never more than {GATE_A_STREAMS}",
          _sb and max(d_["max_running"] for d_ in _sb) == exp_seqs and _sa and max(d_["max_running"] for d_ in _sa) <= GATE_A_STREAMS,
          {"B": [(d_["max_running"], d_["max_inflight"]) for d_ in _sb], "A": [(d_["max_running"], d_["max_inflight"]) for d_ in _sa]})
    _gB = ((_hB.get("gate") or {}).get("stats") or {})
    check("gateAB: half B's fresh Duck gate never blocked (blocked == 0) and was used (enqueued > 0); slots env = B streams",
          _gB.get("blocked") == 0 and (_gB.get("enqueued") or 0) > 0 and str((_hB.get("gate") or {}).get("slots_env")) == str(exp_seqs),
          _hB.get("gate"))
if MODE == "rerun":
    gen = jload(Q / "halfA" / "e12_generated_attempts.json")
    _other = sum(1 for t in test for a, q in zip(sub.get(t, []), test[t]["test"])
                 if a.get("attempt_1") != q["input"] or a.get("attempt_2") != q["input"])
    check("rerun: the solver took every test task; submission.json = half A's generated attempts (not only identity)",
          set(G.get("TASK_IDS") or []) == set(test) and set(gen) == set(test) and all(sub[t] == gen[t] for t in test) and _other > 0,
          (sorted(gen) if isinstance(gen, dict) else gen, _other))
else:
    P21 = (G.get("PROFILE_TABLE") or {}).get("E21X2EVAL120") or {}
    check("profile E21X2EVAL120 (the Kaggle default): 120 eval tasks, 5.45-h window; 2 windows + the 30-min reserve fit",
          P21.get("n_tasks") == 120 and P21.get("solve_window_sec") == 5.45 * 3600
          and 2 * P21["solve_window_sec"] + 30 * 60 < P21["time_budget_sec"] - P21["hard_margin_sec"], P21)
if MODE == "death":
    ha = ((st.get("halves") or {}).get("A") or {})
    wd = (ha.get("e12_run") or {}).get("watchdog") or {}
    rl = wd.get("relaunches") or []
    he = ha.get("engine") or {}
    check("death: the 0.19 server died mid-solve twice (fake 'died' records), the third survived",
          sum(1 for L in la if L["outcome"] == "died") == 2, [L["outcome"] for L in la])
    check("death: the watchdog relaunched twice, both OK (e21_relaunch), short test limits",
          len(rl) == 2 and all(r.get("ok") and r.get("dead") for r in rl) and (wd.get("limits") or {}).get("poll_sec") == 2.0, wd)
    _cf = G["e21_cfgs"](he.get("spec")) if he.get("spec") else []
    _pri = [c for t_, c in _cf if t_ == "primary"][0] if _cf else None
    _alt = [c for t_, c in _cf if t_ == "fallback"][0] if _cf else None
    # rev 2g (VAL2_engine F1): switch only to a config with the same weights (the served name = Duck's model_id)
    _exp = _alt if (_alt and _pri and _alt["weights"] == _pri["weights"]) else _pri
    _last = [L for L in la if L["outcome"] == "ready"][-1:] or [None]
    check("death: the second relaunch switched to the other config only if it has the same weights (rev 2g), "
          "and that is what the last server ran; its served name = Duck's model_id",
          _exp and _last[0]
          and (he.get("relaunch_cfg_switched") == _alt if _exp is _alt else he.get("relaunch_cfg_switched") is None)
          and flag(_last[0], "--max-num-seqs") == str(_exp["seqs"])
          and ("--mamba-ssm-cache-dtype" in _last[0]["argv"]) == bool(_exp.get("ssm"))
          and ("w4a16" in model_of(_last[0])) == (_exp["weights"] == "int4")
          and flag(_last[0], "--served-model-name") == (ha.get("e12_run") or {}).get("model_id"),
          {"switched": he.get("relaunch_cfg_switched"), "expected": _exp, "last_argv": _last[0] and _last[0]["argv"],
           "model_id": (ha.get("e12_run") or {}).get("model_id")})
    check("death: both relaunch logs exist and the last server served Duck tool requests",
          (Q / "halfA" / "vllm_server_halfA_relaunch0.log").exists() and (Q / "halfA" / "vllm_server_halfA_relaunch1.log").exists()
          and any(d["form"] == "0.19" and d["tool_requests"] > 0 for p, d in stats.items()
                  if p == (la[-1]["pid"] if la else None)), (sorted(p.name for p in (Q / "halfA").glob("*.log")), la[-1:] if la else None))
    check("death: half A engine record lists both relaunches with their cfg", len(he.get("relaunches") or []) == 2
          and all(r.get("ready") for r in he.get("relaunches") or []), he.get("relaunches"))

# ---- clean up every fake server still alive
try:
    G["e11_kill_server"](wait=False)
except Exception:
    pass
for L in launches:
    try:
        os.kill(L["pid"], 9)
    except Exception:
        pass
bad = [c for c in checks if not c[1]]
SUSP = []
try:     # e14_heartbeat.log: "HH:MM:SS +x.y min phase ..." every 60 s; a gap > 300 s = the host was suspended
    _hb = [float(l.split("+", 1)[1].split(" min", 1)[0]) * 60 for l in open(WORK / "e14_heartbeat.log", encoding="utf-8")
           if "+" in l and " min" in l]
    SUSP = [(round(a_), round(b_ - a_)) for a_, b_ in zip(_hb, _hb[1:]) if b_ - a_ > 300]
except Exception:
    pass
print(f"\n[e21 SUMMARY] {VAR} {MODE}: {len(checks) - len(bad)}/{len(checks)} PASS in {time.time() - t0:.0f} s | cell seconds {cell_sec}")
for c in checks:
    print(f"[e21 {'PASS' if c[1] else 'FAIL'}] {c[0]}" + ("" if c[1] else f" -> {c[2]}"))
print("info: fake servers", json.dumps({p: {k: d.get(k) for k in ("form", "port", "max_num_seqs", "max_running", "max_inflight",
                                                                    "tool_requests", "preflight", "smoke", "hist_msgs_reasoning",
                                                                    "cached_reported", "kv_tokens", "num_blocks", "block",
                                                                    "cut_tool_calls", "contract_turn1", "contract_turn2",
                                                                    "preempt")} for p, d in stats.items()}))
print("info: launches", [(L["form"], port_of(L), L["outcome"]) for L in launches])
# another harness run on this host talks to OUR fake servers through the fixed ports (seen 2026-10-28): conversations
# of tasks this run does not have = foreign traffic -> no verdict
_tids = set(G.get("TASK_IDS") or []) | {"None"}
FOREIGN = sorted({x[2].rsplit("r", 1)[0] for d_ in stats.values() for x in d_.get("conv_starts") or []} - _tids)
if FOREIGN:
    print(f"[e21 FOREIGN_TRAFFIC] the fake servers saw conversations of tasks outside this run: {FOREIGN[:10]}; another "
          "harness run used ports 1234 / 8011 at the same time; the results above are not a verdict", flush=True)
    print("@@E21_HARNESS_FOREIGN_TRAFFIC@@", flush=True)
    sys.exit(5)
if SUSP:
    print(f"[e21 HOST_SUSPENDED] heartbeat gaps (start sec, gap sec): {SUSP}; the results above are not a verdict", flush=True)
    print("@@E21_HARNESS_HOST_SUSPENDED@@", flush=True)
    sys.exit(3)
print("@@E21_HARNESS_OK@@" if not bad else "@@E21_HARNESS_FAIL@@", flush=True)
sys.exit(0 if not bad else 1)
