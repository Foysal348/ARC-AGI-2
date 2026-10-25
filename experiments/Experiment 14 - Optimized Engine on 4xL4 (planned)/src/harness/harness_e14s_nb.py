"""CPU harness for the E14S submission notebook (Notes/30): runs ALL code cells of the real notebook in order against
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

ROOT = Path(os.environ.get("E14_HARNESS_ROOT", r"D:/DevCache/Temp/claude/d--ARC-AGI-2/e14_harness"))
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
sys.path.insert(0, str(SRC))
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
_pd = PAYLOAD / "datasets" / "foysalemonshanto" / "q27-e14a-payload"
_pd.mkdir(parents=True)
for _f in (Path(NB).resolve().parent / "Build" / "payload").glob("q27x_*.bin"):
    shutil.copy(_f, _pd / _f.name)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    PORT = s.getsockname()[1]
os.environ.update(Q27X_ARC_ROOT=str(ARC), Q27X_WORKING=str(WORK), Q27X_MODEL=str(FP8), Q27X_SKIP_INSTALL="1",
                  PYTHONIOENCODING="utf-8", E11_HARNESS="1", E11_PORT=str(PORT),
                  E11_FAKE_SERVER=str(HERE / "e14_fake_server.py"), E11_INT4=str(INT4), E11_DFLASH=str(DFL),
                  E11_SGL_WHEELHOUSE=str(WH), E12_DIR=str(WORK / "e12_duck"), E14_DIR=str(WORK / "e14_mod"),
                  E14_DEST=str(DEST), E14_FAKE_SGLANG_PKG=str(PKG), E14_TIME_SCALE=SCALE, E14_LONG_TOKENS="3000",
                  E14_FIXTURES=str(FIX), FAKE_SPEED="300", Q27X_PAYLOAD_DIR=str(PAYLOAD),
                  ARC3_HTTP_RETRY_BASE_SECONDS="0.05", ARC3_HTTP_RETRY_MAX_SECONDS="0.1",
                  E12_MAX_GEN="10000", E12_R6_FIRST_EXPOSURE="4000", E12_R6_MIN_ADMIT_SEC="5")
os.environ.pop("FAKE_CORRUPT_IF_FLAG", None)
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

nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
_n = 13 if any("CELL 5c — E15" in c for c in code) else 12      # the E15 notebook has one more cell
assert len(code) == _n and "CELL 1b — E14" in code[2] and "CELL 5b — E14-A1" in code[7] \
    and "CELL 6b — E14-A2" in code[_n - 2] and (MODE != "e15" or "CELL 5c — E15" in code[8]), \
    [c.splitlines()[0][:70] for c in code]
G = {"__name__": "__main__"}
t0 = time.time()
cell_sec = []
for i, c in enumerate(code):
    if i == 7 and MODE in ("commit", "e15"):    # a short Duck window in the harness (the profile's is 25 / 15 min)
        G["E12_SOLVE_WINDOW_SEC"] = 90
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
_want = {"rerun": "RERUN_ALL", "eval120": "E14SEVAL120G4B", "e15": "E15"}.get(MODE, "E14S")
check("mode: profile " + _want + (", RERUN on" if MODE == "rerun" else " (default), RERUN off") +
      "; no E14 measurement; raw reasoning on",
      G["PROFILE"] == _want and bool(G["RERUN"]) == (MODE == "rerun")
      and G["E14_DO_MEASURE"] is False and G["os"].environ.get("E12_RAW_REASONING_HISTORY") == "1",
      (G["PROFILE"], G["RERUN"], G["E14_DO_MEASURE"]))
fl = [str(x) for x in (G.get("E11_SGL") or {}).get("launch_flags") or []]
_i = fl.index("--cuda-graph-bs-decode") if "--cuda-graph-bs-decode" in fl else -1
check("the server ran the submission arm: --speculative-num-draft-tokens 6, --cuda-graph-bs-decode 1..13, no "
      "--cuda-graph-max-bs-decode, DFLASH on, no --min-free-slots-delay",
      fl.count("--speculative-num-draft-tokens") == 1 and fl[fl.index("--speculative-num-draft-tokens") + 1] == "6"
      and _i >= 0 and fl[_i + 1:_i + 14] == [str(b) for b in range(1, 14)] and (len(fl) == _i + 14 or fl[_i + 14].startswith("--"))
      and "--cuda-graph-max-bs-decode" not in fl and "DFLASH" in fl and "--min-free-slots-delay" not in fl, fl)
_eff = os.environ.get("E14S_EXPECT_EFFORT", "")
check("reasoning effort: " + (_eff or "model default (xhigh)") + " for the Duck (E12_REASONING_EFFORT)",
      G["E12_REASONING_EFFORT"] == _eff and G["os"].environ.get("E12_REASONING_EFFORT", "") == _eff
      and (G["CONFIG"].get("E12_REASONING_EFFORT", _eff) == _eff),
      (G.get("E12_REASONING_EFFORT"), G["os"].environ.get("E12_REASONING_EFFORT"), G["CONFIG"].get("E12_REASONING_EFFORT")))
check("CONFIG records the submission arm", (G["CONFIG"].get("e14") or {}).get("submit_arm"), G["CONFIG"].get("e14"))
check("no E14 measurement ran (no E14A phases, no measurement launches unless E15, no JIT tar, no NCCL bench)",
      (MODE == "e15" or (not R.get("phases") and not R.get("launches")))
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
      and (MODE == "e15" or not any(x.startswith(("PHASE:", "LAUNCH:")) for x in phases_seen)), phases_seen)
if MODE == "e15":
    E5 = R.get("e15") or {}
    A = E5.get("arms") or {}
    L = {x["arm"]: x for x in R.get("launches") or []}
    _want_arms = ("ref_w6g13", "w8", "w6_mrr14", "w6_nodelay_mrr12", "mtp_3_1_4", "mtp_3_1_4_mem86", "mtp_2_1_3",
                  "mtp_replay_fp32", "mtp_replay_fp16")
    check("e15: every arm recorded (" + ", ".join(_want_arms) + ") with a gate verdict and both windows",
          all(k in A for k in _want_arms) and all(A[k].get("gate") in ("PASS", "reference") for k in _want_arms)
          and all(A[k].get("short_tok_s") and A[k].get("long_tok_s") for k in _want_arms),
          {k: {f: (A.get(k) or {}).get(f) for f in ("gate", "short_tok_s", "long_tok_s")} for k in _want_arms})
    _mpath = str(G["SGL_MODEL_PATH"])

    def _val(fl, k, n=1):
        return fl[fl.index(k) + 1:fl.index(k) + 1 + n] if k in fl else None
    _m = [str(x) for x in (L.get("mtp_3_1_4") or {}).get("flags") or []]
    check("e15: the MTP launch = EAGLE 3/1/4 with the draft path = the (staged) INT4 model path, no DFLASH, no DFlash draft",
          _val(_m, "--speculative-algorithm") == ["EAGLE"] and _val(_m, "--speculative-draft-model-path") == [_mpath]
          and _val(_m, "--speculative-num-steps") == ["3"] and _val(_m, "--speculative-eagle-topk") == ["1"]
          and _val(_m, "--speculative-num-draft-tokens") == ["4"] and "DFLASH" not in _m
          and str(G["SGL_DRAFT_PATH"]) not in _m and _m.count("--speculative-num-draft-tokens") == 1, _m)
    _r32 = [str(x) for x in (L.get("mtp_replay_fp32") or {}).get("flags") or []]
    _r16 = [str(x) for x in (L.get("mtp_replay_fp16") or {}).get("flags") or []]
    check("e15: ReplaySSM arms carry --enable-linear-replayssm-spec; fp32 sets --mamba-ssm-dtype float32 once, fp16 keeps float16",
          "--enable-linear-replayssm-spec" in _r32 and _val(_r32, "--mamba-ssm-dtype") == ["float32"]
          and _r32.count("--mamba-ssm-dtype") == 1 and "--enable-linear-replayssm-spec" in _r16
          and _val(_r16, "--mamba-ssm-dtype") == ["float16"], (_r32, _r16))
    _m14 = [str(x) for x in (L.get("w6_mrr14") or {}).get("flags") or []]
    _nd = [str(x) for x in (L.get("w6_nodelay_mrr12") or {}).get("flags") or []]
    check("e15: 11-of-12 arms: w6_mrr14 = mrr 14 + graphs 1..14, delayer kept; w6_nodelay_mrr12 = delay 1 + mrr 12; "
          "both DFLASH width 6; latch measured on them and on the reference",
          _val(_m14, "--max-running-requests") == ["14"] and _val(_m14, "--cuda-graph-bs-decode", 14) == [str(b) for b in range(1, 15)]
          and "--min-free-slots-delay" not in _m14 and _val(_m14, "--speculative-num-draft-tokens") == ["6"]
          and _val(_nd, "--min-free-slots-delay") == ["1"] and _val(_nd, "--max-running-requests") == ["12"]
          and "DFLASH" in _nd and _m14.count("--max-running-requests") == 1 and _nd.count("--max-running-requests") == 1
          and all(A.get(k, {}).get("latch_ge12") is not None for k in ("ref_w6g13", "w6_mrr14", "w6_nodelay_mrr12")),
          (_m14, _nd, {k: (A.get(k) or {}).get("latch_ge12") for k in ("ref_w6g13", "w6_mrr14", "w6_nodelay_mrr12")}))
    _a14 = [str(x) for x in (L.get("w6_at14") or {}).get("flags") or []]
    _a16 = [str(x) for x in (L.get("w6_at16") or {}).get("flags") or []]
    _a20 = [str(x) for x in (L.get("w6_at20") or {}).get("flags") or []]
    check("e15: DFlash W6 at 14 / 16 / 20 streams: mrr 17 / 20 / 24, mamba 4 x mrr, graphs 1..mrr, all measured",
          all(_val(f, "--max-running-requests") == [str(m)] and _val(f, "--max-mamba-cache-size") == [str(4 * m)]
              and _val(f, "--cuda-graph-bs-decode", m) == [str(b) for b in range(1, m + 1)] and "DFLASH" in f
              for f, m in ((_a14, 17), (_a16, 20), (_a20, 24)))
          and all(A.get(k, {}).get("short_tok_s") for k in ("w6_at14", "w6_at16", "w6_at20")),
          (_a14, _a16, _a20))
    _to = [str(x) for x in (L.get("target_only") or {}).get("flags") or []]
    check("e15: long probe (c12 x 1024): reference A/A, target-only launch without --speculative-*, probes on w6_at14/16/20 "
          "and mtp_3_1_4, all compared",
          (E5.get("probe_ref_aa") or {}).get("compared") and (E5.get("probe_target_only") or {}).get("compared")
          and _to and not any(f.startswith("--speculative-") for f in _to)
          and all(A.get(k, {}).get("probe_flips") is not None for k in ("w6_at14", "w6_at16", "w6_at20", "mtp_3_1_4")),
          (E5.get("probe_ref_aa"), E5.get("probe_target_only"), {k: A.get(k, {}).get("probe_flips")
                                                                 for k in ("w6_at14", "w6_at16", "w6_at20", "mtp_3_1_4")}))
    _w8 = [str(x) for x in (L.get("w8") or {}).get("flags") or []]
    check("e15: the w8 arm = DFLASH width 8 on graphs 1..13", _val(_w8, "--speculative-num-draft-tokens") == ["8"]
          and "DFLASH" in _w8 and _val(_w8, "--cuda-graph-bs-decode", 13) == [str(b) for b in range(1, 14)], _w8)
    _b = E5.get("best_mtp_at_12")
    _h = {lvl: [str(x) for x in (L.get(f"{_b}_at{lvl}") or {}).get("flags") or []] for lvl in (14, 16)}
    check("e15: the best MTP arm at 12 was re-run at 14 / 16 (mrr 15 / 17, mamba 60 / 68, graphs 1..15 / 1..17)",
          _b in A and all(_val(_h[lv], "--max-running-requests") == [str(lv + 1)]
                          and _val(_h[lv], "--max-mamba-cache-size") == [{14: "60", 16: "68"}[lv]]
                          and _val(_h[lv], "--cuda-graph-bs-decode", lv + 1) == [str(x) for x in range(1, lv + 2)]
                          and _h[lv].count("--max-running-requests") == 1 for lv in (14, 16)), (_b, _h))
    _d = E5.get("duck_arm")
    check("e15: the Duck arm is a 12-stream arm (no _atN, not target-only)", _d and "_at" not in _d and _d != "target_only", _d)
    check("e15: the Duck ran on a fresh forced launch of the adopted arm, recorded in decisions",
          _d in A and any(x.get("arm") == "duck_" + _d and x.get("force") and x.get("ok") for x in R.get("launches") or [])
          and (R.get("decisions") or {}).get("e15", {}).get("duck_arm") == _d, (_d, [x.get("arm") for x in R.get("launches") or []]))
e = rep.get("e12_run") or {}
check("Duck ran without solver error on every task; branches recorded",
      not e.get("error") and e.get("branches_started", 0) > 0, e)
check("isolation preflight PASS, 12 workers", e.get("rev6_workers") == 12
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
    check("eval120: the 6.5 h budget holds start + the full window + the stop margin (no cut by the hard stop)",
          G["PROFILE_TABLE"]["E14SEVAL120G4B"]["time_budget_sec"] - 2 * 60 - G["E12_STOP_BEFORE_HARD_SEC"]
          >= 1500 + 120 * (_hid_end - 1500) / 240, G["PROFILE_TABLE"]["E14SEVAL120G4B"])
    check("eval120: scored (pass1 / pass2 / task-norm recorded)", rep.get("eval_task_norm_pct") is not None
          and rep.get("eval_pass1_hits") is not None and rep.get("eval_pass2_hits") is not None,
          {k: rep.get(k) for k in ("eval_query_hits", "eval_pass1_hits", "eval_pass2_hits", "eval_task_norm_pct")})
check("submission.json written atomically with no temp file left", not list(WORK.glob("submission.json.*")),
      list(WORK.glob("submission.json*")))
G["e11_kill_server"](wait=False)
bad = [c for c in checks if not c[1]]
print(f"\n[e14s {MODE}] {len(checks) - len(bad)}/{len(checks)} ok in {time.time() - t0:.0f} s | cell seconds {cell_sec}")
for c in bad:
    print("   FAIL", c[0], "->", c[2])
print("@@NB_OK@@" if not bad else "@@NB_FAIL@@")
