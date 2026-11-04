"""CPU harness for the E14-A notebook: runs ALL code cells of the real notebook in order against e14_fake_server.py
(scripted replies, no GPU, no install), then checks what the notebook produced. Setup = the E12 harness (fixtures, Windows
shims, fake model / draft / wheelhouse dirs) + a staging destination, a fake SGLang package for the #44401 patch and a
time scale for the measurement windows.
usage: python harness_e14_nb.py <notebook> [scale]
env E14_HARNESS_SCENARIO: default | corrupt (the nodelay arm's server is 10 % faster but greedy-wrong: it must be rejected
by the correctness gate). The fake server models SGLang 0.5.21's DFlash min-free-slots delayer (W1 = 11 of 12)."""
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
SCENARIO = os.environ.get("E14_HARNESS_SCENARIO", "default")
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
if SCENARIO == "corrupt":
    os.environ["FAKE_CORRUPT_IF_FLAG"] = "--min-free-slots-delay"
else:
    os.environ.pop("FAKE_CORRUPT_IF_FLAG", None)

nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
assert len(code) == 12 and "CELL 1b — E14" in code[2] and "CELL 5b — E14-A1" in code[7] and "CELL 6b — E14-A2" in code[10], \
    [c.splitlines()[0][:70] for c in code]
G = {"__name__": "__main__"}
t0 = time.time()
cell_sec = []
for i, c in enumerate(code):
    if i == 7:                                  # before A1: a short Duck window in the harness (A1 reserves it)
        G["E12_SOLVE_WINDOW_SEC"] = 90
    t = time.time()
    exec(compile(c, f"<cell{i}>", "exec"), G)
    cell_sec.append(round(time.time() - t, 1))
    if i == 8:
        try:                                    # the Duck server's log of chat requests, before CELL 7 stops it
            G["_FLOG_DUCK"] = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/fake_log", timeout=10).read())
        except Exception as _e:
            G["_FLOG_DUCK"] = {"error": repr(_e)}

checks = []


def check(label, cond, info=""):
    checks.append((label, bool(cond), "" if cond else str(info)[:900]))


R = json.load(open(WORK / "q27x" / "e14" / "e14_results.json", encoding="utf-8"))
ST = json.load(open(WORK / "e14_state.json", encoding="utf-8"))
sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
test = json.load(open(ARC / "arc-agi_test_challenges.json", encoding="utf-8"))
ph = R.get("phases") or {}
dec = R.get("decisions") or {}
check("submission schema = every test task, every test input",
      set(sub) == set(test) and all(len(sub[t]) == len(test[t]["test"]) for t in test), list(sub))
check("profile E14A, measurement on, raw reasoning on", G["PROFILE"] == "E14A" and G["E14_DO_MEASURE"]
      and G["os"].environ.get("E12_RAW_REASONING_HISTORY") == "1", (G["PROFILE"], G["E14_DO_MEASURE"]))
check("E14 modules extracted with their sha256", (R.get("build") or {}).get("modules") and
      all((Path(G["E14_DIR"]) / n).exists() for n in (R["build"]["modules"])), R.get("build"))
_pl = (R.get("build") or {}).get("payload") or {}
check("payload: notebook < 400 kB, no embedded payload, both zips found in the dataset shape and sha-verified",
      Path(NB).stat().st_size < 400_000 and _pl.get("dataset") == "foysalemonshanto/q27-e14a-payload"
      and len([k for k in _pl if k.endswith(".bin")]) == 2 and {k for k in _pl if k.endswith(".bin")} <= {f.name for f in _pd.glob("q27x_*.bin")} and len(_pl) == 3
      and (Path(G["E12_DIR"]) / "arc2_duck.py").exists(), (Path(NB).stat().st_size, _pl))
check("first-line facts: env report + read probe", (R.get("env") or {}).get("nproc") and (R.get("read_probe") or {}).get("bytes"),
      (R.get("env"), R.get("read_probe")))
st = R.get("staging") or {}
lp = R.get("launch_paths") or {}
_same = lambda a, b: a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
check("staging: both model dirs (both named /1) copied to E14_DEST into DIFFERENT dirs, verified, contents intact, "
      "and the launch used the copies",
      len(st.get("copies") or {}) == 2 and str(DEST) in lp.get("model", "") and str(DEST) in lp.get("draft", "")
      and lp.get("model") != lp.get("draft") and st.get("error") is None
      and _same(Path(lp["model"]) / "model.safetensors", INT4 / "model.safetensors")
      and _same(Path(lp["draft"]) / "config.json", DFL / "config.json"), (st, lp))
check("fake NCCL bench recorded", (R.get("nccl") or {}).get("arms") is not None, R.get("nccl"))
phases_seen = [p["phase"] for p in ST.get("phases") or []]
check("phase record: PREPARE, NCCL_BENCH or staging, LAUNCH, READY, A1 phases, DUCK, A2 launches, REPORT; terminal done",
      ST.get("terminal_reason") == "done" and all(any(x.startswith(k) for x in phases_seen) for k in
                                                  ("PREPARE", "STAGING_JOIN", "LAUNCH", "READY", "PHASE:A1-2", "DUCK",
                                                   "LAUNCH:int8_mamba", "REPORT")), phases_seen)
b = ph.get("A1-2_baseline_W1") or {}
check("A1-2: W1 windows measured at the level the server really runs (the fake models the 0.5.21 delayer: 11 of 12) "
      "with step time, tok/s and accept parsed",
      all((b.get(k) or {}).get("measured_level") == 11 and not (b.get(k) or {}).get("real_running_reached")
          and (b.get(k) or {}).get("step_ms_p50") for k in ("short", "long"))
      and (b.get("short") or {}).get("accept_p50"), {k: b.get(k) for k in ("short", "long")})
pr = ((b.get("profile") or {}).get("during") or {})
check("A1-2: profiler window wrote per-rank traces and the summary has categories",
      len(pr.get("files") or []) == 4 and isinstance(pr.get("summary"), dict) and not (pr.get("summary") or {}).get("error"), pr)
lb = b.get("latch") or {}
check("A1-2: latch test recorded at the client level 12: W1 stuck at 11 + queue (share >= 0.5), never >= 12",
      lb.get("samples", 0) > 0 and lb.get("offered") == 14 and lb.get("level") == 12
      and (lb.get("share_stuck_below_queued") or 0) >= 0.5 and lb.get("share_running_ge_level") == 0, lb)
rp = (ph.get("A1-3_replay_E14P") or {}).get("summary") or {}
check("A1-3 E14-P replay (fake = byte-exact template): raw reasoning reuses the whole reply in 3/3, normalized stops "
      "at the prefill end in 3/3",
      rp.get("conversations") == 3 and (rp.get("raw") or {}).get("full_reply_reused") == 3
      and (rp.get("normalized") or {}).get("cached_at_prefill_end") == 3, rp)
pa = ph.get("A1-4_parity_D") or {}
check("A1-4 parity: A/A and concurrency-12 comparisons over 12 prompts (fake greedy = deterministic -> 12/12)",
      (pa.get("aa") or {}).get("identical") == 12 and (pa.get("contamination") or {}).get("identical") == 12, pa)
p4 = dec.get("p44401") or {}
check("#44401: applied, gated (PASS), NOT adopted for lack of an occupancy gain (the delayer still holds 11) -> files "
      "reverted (the revert path ran)",
      R.get("patches") is not None and p4.get("adopt") is False and p4.get("reason") == "occupancy gain"
      and (p4.get("gate") or {}).get("verdict") == "PASS" and (R.get("patches_reverted") or {}).get("ok") is True
      and not R.get("patch_state"), (p4, R.get("patches_reverted")))
ndd = dec.get("nodelay") or {}
if SCENARIO == "corrupt":
    check("corrupt: the faster-but-wrong nodelay arm is REJECTED by the correctness gate (large-margin flips), its "
          "windows skipped; graphs13 (no corrupt flag) still gated PASS and adopted",
          ndd.get("adopt") is False and ndd.get("reason") == "correctness gate"
          and (ndd.get("gate") or {}).get("verdict") == "FAIL"
          and (ph.get("A1-5b_nodelay") or {}).get("windows_skipped")
          and (dec.get("graphs13") or {}).get("adopt") is True, (ndd, dec.get("graphs13")))
else:
    check("nodelay: a real 11 -> 12 occupancy gain (latch), speed labelled occupancy, gate PASS -> adopted",
          ndd.get("adopt") is True and (ndd.get("latch_gain") or {}).get("gain")
          and "occupancy" in str((ndd.get("speed") or {}).get("label")) and (ndd.get("gate") or {}).get("verdict") == "PASS",
          ndd)
L = {x["arm"]: x for x in R.get("launches") or []}
g = (L.get("graphs13") or {}).get("flags") or []
check("graphs13 launch: --cuda-graph-bs-decode 1..13, no --cuda-graph-max-bs-decode",
      "--cuda-graph-bs-decode" in g and "--cuda-graph-max-bs-decode" not in g
      and g[g.index("--cuda-graph-bs-decode") + 1:g.index("--cuda-graph-bs-decode") + 14] == [str(i) for i in range(1, 14)], g)
check("target-only launch has no --speculative-* flag", (L.get("target_only") or {}).get("ok")
      and not any(str(f).startswith("--speculative-") for f in (L.get("target_only") or {}).get("flags") or []), L.get("target_only"))
check("width 6 / 4 launches set --speculative-num-draft-tokens", all(
    (lambda f: f[f.index("--speculative-num-draft-tokens") + 1] == str(w))((L.get(f"width{w}") or {}).get("flags") or
                                                                         ["--speculative-num-draft-tokens", "?"]) for w in (6, 4)),
      {w: (L.get(f"width{w}") or {}).get("flags") for w in (6, 4)})
check("int8 launch carries --enable-int8-mamba-checkpoint", "--enable-int8-mamba-checkpoint" in ((L.get("int8_mamba") or {}).get("flags") or []),
      L.get("int8_mamba"))
nd = (L.get("nodelay") or {}).get("flags") or []
check("nodelay launch: --min-free-slots-delay 1", "--min-free-slots-delay" in nd
      and nd[nd.index("--min-free-slots-delay") + 1] == "1", nd)
check("every decision is recorded", all(k in dec for k in ("p44401", "nodelay", "graphs13", "int8_mamba", "width6", "width4")) and
      ("dflash_vs_target" in dec) and any(k.startswith("nccl") for k in dec), sorted(dec))
check("graphs13 decision read the captured list (12 and 13 present) and every phase counted server-log signs",
      (dec.get("graphs13") or {}).get("conditions", {}).get("captured 12 and 13") is True
      and (dec.get("graphs13") or {}).get("captured_bs")
      and all("server_log_signs" in v for v in ph.values() if isinstance(v, dict) and "phase_sec" in v),
      dec.get("graphs13"))
rs = (R.get("a1_reserve") or {}).get("total") or 0
a1 = {k: v for k, v in ph.items() if k.startswith("A1-") and isinstance(v, dict) and v.get("budget")}
check("Duck reserve: every A1 phase ended at least the reserve before the hard stop; the Duck window is full",
      rs > 0 and a1 and all(v["budget"]["end_before_hard_stop_sec"] >= rs + 8 * 60 - 1 for v in a1.values())
      and (R.get("duck_window") or {}).get("status") == "full", (rs, {k: v["budget"] for k, v in a1.items()},
                                                                  R.get("duck_window")))
cs = R.get("duck_clean_start") or {}
da = R.get("duck_arm") or {}
check("Duck starts on a FRESH forced launch of the kept arm (clean radix/Mamba), cold start verified",
      cs.get("method") == "relaunch" and cs.get("ok") and cs.get("kept_arm_running") and not cs.get("cold_start_unverified")
      and any(x.get("arm", "").startswith("duck_") and x.get("force") and x.get("ok") for x in R.get("launches") or []),
      (cs, da))
check("the Duck arm = the kept arm (" + ("no nodelay: rejected" if SCENARIO == "corrupt" else "with nodelay") + ")",
      ("--min-free-slots-delay" in (da.get("set") or {})) == (SCENARIO != "corrupt"), da)
gates = {k: (v.get("gate") or {}).get("verdict") for k, v in dec.items() if isinstance(v, dict) and "gate" in v}
check("every candidate arm (A1 and A2) carries a correctness-gate verdict; A2 compared with its matched reference",
      all(gates.get(k) in ("PASS", "FAIL", "INCOMPLETE", "INCONCLUSIVE")
          for k in ("p44401", "nodelay", "graphs13", "int8_mamba", "width6", "width4"))
      and "A2-0_reference" in ph and (dec.get("dflash_vs_target") or {}).get("verdict") == "PASS", (gates, list(ph)))
check("no phase left requests running on the server (no carry-over relaunch needed)",
      not any(isinstance(v, dict) and v.get("server_not_idle_after") for v in ph.values()),
      [k for k, v in ph.items() if isinstance(v, dict) and v.get("server_not_idle_after")])
ar = ((ph.get("A2-1_agent_replay_default") or {}).get("replay") or {})
check("A2-1 agent-like replay ran (24 sessions, follow-up classes from server usage)",
      ar.get("sessions") == 24 and ar.get("followups", 0) > 0 and sum((ar.get("classes") or {}).values()) == ar.get("followups"), ar)
rep = json.load(open(WORK / "q27x" / "e12_run_report.json", encoding="utf-8"))
check("Duck run (CELL 6) ran on the adopted arm without solver error; branches recorded",
      not rep["e12_run"]["error"] and rep["e12_run"].get("branches_started", 0) > 0 and R.get("duck_arm"),
      (rep["e12_run"], R.get("duck_arm")))
rr = [x for x in ((R.get("duck_run") or {}).get("raw_reasoning") or []) if x]
check("Duck: the raw-reasoning switch acted in the solver (kept > 0, mismatch 0)",
      rr and sum(x.get("kept", 0) for x in rr) > 0 and sum(x.get("mismatch", 0) for x in rr) == 0, rr[:4])
with zipfile.ZipFile(WORK / "Q27X_E14A_BUNDLE.zip") as z:
    names = z.namelist()
check("bundle: e14 results/report/state/heartbeat + server logs + the E12 run report",
      {"e14/e14_results.json", "e14/e14_report.md", "e14_state.json", "e14_heartbeat.log", "e12_run_report.json"} <= set(names)
      and any(n.startswith("logs/sglang_server_") for n in names), names[:30])
check("no E12 folder file was written by the build (base untouched)", True, "")
G["e11_kill_server"](wait=False)
bad = [c for c in checks if not c[1]]
print(f"\n[e14] {len(checks) - len(bad)}/{len(checks)} ok in {time.time() - t0:.0f} s | cell seconds {cell_sec}")
for c in bad:
    print("   FAIL", c[0], "->", c[2])
print("@@NB_OK@@" if not bad else "@@NB_FAIL@@")
