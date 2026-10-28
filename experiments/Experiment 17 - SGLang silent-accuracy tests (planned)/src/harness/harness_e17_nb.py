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

ROOT = Path(os.environ.get("E14_HARNESS_ROOT", r"D:/ARC AGI 2/_scratch/e17/harness_runs"))
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
json.dump({"architectures": ["DFlash2DraftModel"], "num_target_layers": 64, "vocab_size": 248320,
           "dflash_config": {"block_size": 8, "target_layer_ids": [5, 19, 33, 47, 61]}}, open(DFL / "config.json", "w"))
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
for _f in (Path(os.environ.get("E17_PAYLOAD_DIR", r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload"))).glob("q27x_e14_modules_*.bin"):   # E15P needs only this one
    shutil.copy(_f, _pd / _f.name)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    PORT = s.getsockname()[1]
os.environ.update(Q27X_ARC_ROOT=str(ARC), Q27X_WORKING=str(WORK), Q27X_MODEL=str(FP8), Q27X_SKIP_INSTALL="1",
                  PYTHONIOENCODING="utf-8", E11_HARNESS="1", E11_PORT=str(PORT),
                  E11_FAKE_SERVER=str(HERE / "e17_fake_server.py"), E11_INT4=str(INT4), E11_DFLASH=str(DFL),
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


os.environ.update(E17_HOOK_DIR=str(WORK / "e17_hook"), E17_VOCAB="2048", E17_T3B_TOKENS="64", E17_T3A_TOKENS="3000",
                  E17_T2_TOKENS="1500", E17_T2_MAX_NEW="48")
for _k in ("Q27X_PROFILE", "E17_FAKE_PATCH_FAIL"):
    os.environ.pop(_k, None)
# HARNESS_E17_PROFILE: E17SHORT (the notebook default, L1 / L7 + L5 / L9) or E17 (the full plan)
if os.environ.get("HARNESS_E17_PROFILE", "E17SHORT") != "E17SHORT":
    os.environ["Q27X_PROFILE"] = os.environ["HARNESS_E17_PROFILE"]
PKG_SB = PKG / "srt" / "managers" / "schedule_batch.py"
import hashlib as _hl  # noqa: E402
PKG_SB_SHA0 = _hl.sha256(PKG_SB.read_bytes()).hexdigest()

nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
assert len(code) == 10 and "E17 hook source" in code[7] and "CELL E17 —" in code[8] and "CELL E17-R" in code[9], \
    [c.splitlines()[0][:70] for c in code]
G = {"__name__": "__main__"}
cell_sec = []
for i, c in enumerate(code):
    t = time.time()
    exec(compile(c, f"<cell{i}>", "exec"), G)
    cell_sec.append(round(time.time() - t, 1))

checks = []


def check(label, cond, info=""):
    checks.append((label, bool(cond), "" if cond else str(info)[:1500]))


E = G.get("E17") or {}
A, LA, P0 = E.get("arms") or {}, E.get("launches") or {}, E.get("phase0") or {}


def fl(n):
    return [str(x) for x in (LA.get(n) or {}).get("flags") or []]


def val(f, k):
    return f[f.index(k) + 1] if k in f else None


def v(n):
    return (A.get(n) or {}).get("verdict")


SHORT = G["PROFILE"] == "E17SHORT"
check("mode: profile " + G["PROFILE"] + ", not a rerun, E17 ran without a cell error",
      G["PROFILE"] == os.environ.get("HARNESS_E17_PROFILE", "E17SHORT") and G["E17_ON"] and not E.get("error")
      and E.get("mode") == ("SHORT" if SHORT else "FULL"), (G["PROFILE"], E.get("mode"), E.get("error")))
check("phase 0: unpatched at start, DFlash taps ok, GDN backend line, the vocab override used",
      P0.get("patch_43567_state") == "not_applied" and P0.get("dflash_taps_ok") and P0.get("gdn_backend_line")
      and E.get("vocab_used") == 2048 and P0.get("schedule_batch_sha256") == PKG_SB_SHA0,
      {k: P0.get(k) for k in ("patch_43567_state", "dflash_taps_ok", "gdn_backend_line", "schedule_batch_sha256")})

# ---- audit 4 negative tests (three GPT audits 2026-10-26), on the notebook's own functions in G
def _neg_tests():
    import types as _ty
    t = {"safe_depth": 128, "tol": 5e-3, "reference_valid": True,
         "cycles": [{"seed_rid": "sA", "probe_cached": 128, "probe_kl": 0.7, "flush_valid": True},
                    {"seed_rid": "sB", "probe_cached": 192, "probe_kl": 0.0, "flush_valid": True}]}
    ev = {"failed": {"sB"}, "confirmed": {"sB": [192]}}
    v_disc = G["e17_verdict"]([t], ev)
    t2 = {"safe_depth": 128, "tol": 5e-3, "reference_valid": True,
          "cycles": [{"seed_rid": "sC", "probe_cached": 192, "probe_kl": 0.7, "flush_valid": True}]}
    v_link = G["e17_verdict"]([t2], {"failed": {"sC"}, "confirmed": {"sC": [192]}})
    v_wrong_len = G["e17_verdict"]([t2], {"failed": {"sC"}, "confirmed": {"sC": [256]}})
    t3 = dict(t2, reference_valid=False)
    v_noisy = G["e17_verdict"]([t3], {"failed": {"sC"}, "confirmed": {"sC": [192]}})
    t4 = dict(t2, cycles=[dict(t2["cycles"][0], flush_valid=False)])
    v_flush = G["e17_verdict"]([t4], {"failed": {"sC"}, "confirmed": {"sC": [192]}})
    good = {"expected": 4, "compared": 4, "errors": 0, "missing_margins": 0, "large_margin_flips": 0}
    out_ok = {"dflash_serial_vs_target_serial": good, "dflash_conc_vs_target_conc": dict(good, expected=16, compared=16),
              "dflash_conc_vs_dflash_serial": good}
    v_t3_ok = G["e17_t3b_verdict"](out_ok, good)
    bad_all = dict(out_ok, dflash_conc_vs_target_conc={"expected": 16, "compared": 0, "errors": 16, "missing_margins": 0,
                                                       "large_margin_flips": 0})
    v_t3_err = G["e17_t3b_verdict"](bad_all, good)
    v_t3_margin = G["e17_t3b_verdict"](dict(out_ok, dflash_serial_vs_target_serial=dict(good, missing_margins=1)), good)
    v_t3_missing = G["e17_t3b_verdict"]({k: v for k, v in out_ok.items() if k != "dflash_conc_vs_dflash_serial"}, good)
    real = G["E14P"]
    G["E14P"] = _ty.SimpleNamespace(revert=lambda pkg: {"ok": False, "errors": ["boom"]},
                                    check=lambda pkg, names: {"patches": {"43567": {"state": "applied"}}})
    n0 = len(G["E17"]["launches"])
    try:
        G["e17_revert_verified"](G["PKG17"])
        rv = "NO_RAISE"
    except RuntimeError as e:
        rv = str(e)[:80]
    finally:
        G["E14P"] = real
    G["E17"].pop("patch_revert_failed", None)
    return {"disconnected": v_disc, "linked": v_link, "wrong_len": v_wrong_len, "noisy_ref": v_noisy, "bad_flush": v_flush,
            "t3_ok": v_t3_ok, "t3_errors": v_t3_err, "t3_margin": v_t3_margin, "t3_missing": v_t3_missing,
            "revert_fail": rv, "launches_added": len(G["E17"]["launches"]) - n0}


NEGR = _neg_tests()
check("audit 4 negatives: disconnected evidence is not BUG_EXPOSED; the linked chain is; a donation of another length "
      "breaks the link; an unstable reference and an invalid flush give INCONCLUSIVE",
      NEGR["disconnected"] != "BUG_EXPOSED" and NEGR["linked"] == "BUG_EXPOSED" and NEGR["wrong_len"] != "BUG_EXPOSED"
      and NEGR["noisy_ref"].startswith("INCONCLUSIVE") and NEGR["bad_flush"].startswith("INCONCLUSIVE"), NEGR)
check("audit 4 negatives: T3b complete -> a verdict; 16/16 errors, a missing margin, a missing comparison -> INCOMPLETE",
      NEGR["t3_ok"] == "NO_HIGH_MARGIN_DIVERGENCE" and NEGR["t3_errors"] == "INCOMPLETE" and NEGR["t3_margin"] == "INCOMPLETE"
      and NEGR["t3_missing"] == "INCOMPLETE", NEGR)
check("audit 4 negatives: a failed revert raises PATCH_REVERT_FAILED and launches nothing",
      NEGR["revert_fail"].startswith("PATCH_REVERT_FAILED") and NEGR["launches_added"] == 0, NEGR)
check("audit 4: every hooked launch passed the per-rank self-test; T1 cycles carry rids and the verdict's chain records",
      all((LA[n].get("hook_selftest") or {}).get("ok") for n in LA if LA[n].get("hook") and LA[n].get("ok"))
      and all(c.get("seed_rid") and "chain" in c for n in A for t in (A[n].get("t1") or []) for c in t.get("cycles", [])
              if A[n].get("hook")), "")
SHORT_RUN = ["L1_t1a_128", "L5_t1c_dflash", "L7_t1a_128_patched", "L9_t1c_dflash_patched"]
want = SHORT_RUN if SHORT else ["L1_t1a_128", "L2_t1a_128_hook_off", "L3_t1a_256", "L4_t1b_forced", "L5_t1c_dflash", "L6_class2_mtp",
        "L12_t3b_target", "L13_t3b_dflash", "L7_t1a_128_patched", "L8_t1a_256_patched", "L9_t1c_dflash_patched",
        "L11_t2_patched", "L10_t2_unpatched", "L15_t3a_fp32"]
check("every planned launch ran and was ready", all((LA.get(n) or {}).get("ok") for n in want),
      {n: (LA.get(n) or {}).get("ok", (LA.get(n) or {}).get("skipped")) for n in want})
if SHORT:
    check("SHORT: exactly L1 / L5 / L7 / L9 launched (+ nothing else); every other arm recorded as skipped",
          set(LA) == set(SHORT_RUN) and all(k in (E.get("skipped") or {}) for k in ("L2_t1a_128_hook_off", "L3_t1a_256",
          "L4_t1b_forced", "L6_class2_mtp", "L12_t3b_target", "L13_t3b_dflash", "L8_t1a_256_patched", "T2", "L15_t3a_fp32")),
          (sorted(LA), E.get("skipped")))
    check("SHORT: the session fits the 60-min profile (HARD_STOP from a 60-min budget)", G["TOTAL_BUDGET_SEC"] == 3600, G["TOTAL_BUDGET_SEC"])
    check("SHORT: L1 / L5 expose the chain, L7 / L9 clean, decision A = BUG_EXPOSED_43567 with both pairs",
          v("L1_t1a_128") == "BUG_EXPOSED" and v("L5_t1c_dflash") == "BUG_EXPOSED" and v("L7_t1a_128_patched") == "TRIGGER_REACHED_NO_STALE"
          and v("L9_t1c_dflash_patched") == "TRIGGER_REACHED_NO_STALE"
          and sorted((E.get("decision_A") or {}).get("patched_clean") or []) == ["L1_t1a_128", "L5_t1c_dflash"], E.get("decision_A"))
    s1, s7 = A["L1_t1a_128"].get("safety") or {}, A["L7_t1a_128_patched"].get("safety") or {}
    check("SHORT: patch-safety cases on L1 / L7: no leak, cold really cold, stale only unpatched",
          s1.get("leaks") == 0 and s7.get("leaks") == 0 and s1.get("cold_cached_zero") and s7.get("cold_cached_zero")
          and s1.get("stale", 0) >= 1 and s7.get("stale") == 0, (s1, s7))
    check("SHORT: patch applied then reverted (the fake package file is the original again)",
          E.get("patch_check_after_apply") == "applied" and E.get("patch_state_end") == "not_applied"
          and _hl.sha256(PKG_SB.read_bytes()).hexdigest() == PKG_SB_SHA0, (E.get("patch_apply"), E.get("patch_state_end")))
    check("SHORT: report + bundle written; no server left running",
          (WORK / "q27x" / "e14" / "e17_report.md").exists() and (WORK / "Q27X_E17_BUNDLE.zip").exists()
          and (G.get("E11_SERVER") is None or not G["E11_SERVER"].alive()), "")
    print("cell seconds:", cell_sec)
    print((WORK / "q27x" / "e14" / "e17_report.md").read_text(encoding="utf-8")[:4000])
    for label, ok, info in checks:
        print(("PASS " if ok else "FAIL ") + label + ("" if ok else "  <-- " + info))
    nf = sum(not ok for _, ok, _ in checks)
    print(chr(10) + (f"{nf} failed of {len(checks)}" if nf else f"ALL {len(checks)} CHECKS PASS (E17 SHORT)"))
    sys.stdout.flush()
    os._exit(1 if nf else 0)
f1 = fl("L1_t1a_128")
check("L1 = upstream Class 1 flags: 4 mamba slots, 1 running, interval 128, chunk 2048, graphs [1], target-only, "
      "production numerics kept", val(f1, "--max-mamba-cache-size") == "4" and val(f1, "--max-running-requests") == "1"
      and val(f1, "--mamba-track-interval") == "128" and val(f1, "--chunked-prefill-size") == "2048"
      and val(f1, "--cuda-graph-bs-decode") == "1" and "--speculative-algorithm" not in f1
      and val(f1, "--kv-cache-dtype") == "fp8_e4m3" and val(f1, "--mamba-ssm-dtype") == "float16"
      and val(f1, "--mamba-radix-cache-strategy") == "extra_buffer_lazy" and "--min-free-slots-delay" not in f1, f1)
check("hook on L1 (PYTHONPATH + trace dir + self-test ok), off on L2",
      (LA["L1_t1a_128"].get("hook_selftest") or {}).get("ok") and LA["L1_t1a_128"].get("hook") and not LA["L2_t1a_128_hook_off"].get("hook"),
      (LA["L1_t1a_128"].get("hook_selftest"), LA["L2_t1a_128_hook_off"].get("hook")))
check("L3 interval 256 with P255 and P511; L4 the forced-fail env; L5 DFlash W6 at interval 64",
      val(fl("L3_t1a_256"), "--mamba-track-interval") == "256" and [t["prompt"] for t in A["L3_t1a_256"]["t1"]] == ["P255", "P511"]
      and LA["L4_t1b_forced"]["env"].get("SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL") == "1"
      and val(fl("L5_t1c_dflash"), "--speculative-algorithm") == "DFLASH" and val(fl("L5_t1c_dflash"), "--mamba-track-interval") == "64"
      and val(fl("L5_t1c_dflash"), "--speculative-num-draft-tokens") == "6", (fl("L3_t1a_256"), LA["L4_t1b_forced"]["env"], fl("L5_t1c_dflash")))
check("unpatched T1 arms expose the chain (fake: failure + confirmed stale donation + a 192 hit + KL above tolerance)",
      all(v(n) == "BUG_EXPOSED" for n in ("L1_t1a_128", "L3_t1a_256", "L4_t1b_forced", "L5_t1c_dflash")),
      {n: (v(n), [(t["reuse_cached"], round(t["reuse_kl_max"], 4)) for t in (A.get(n) or {}).get("t1", [])]) for n in
       ("L1_t1a_128", "L3_t1a_256", "L4_t1b_forced", "L5_t1c_dflash")})
check("hook off: the same KL signal without a trace (L2)", v("L2_t1a_128_hook_off") == "KL_ABOVE_TOL" and A["L2_t1a_128_hook_off"].get("trace") is None,
      A.get("L2_t1a_128_hook_off"))
check("A/A of every T1 reference is 0 on the fake and the tolerance is the 5e-3 floor; ref hits are 128",
      all(t["aa_kl_max"] == 0 and t["tol"] == 5e-3 and set(t["ref_cached"]) == {128} for n in A for t in (A[n].get("t1") or [])),
      [(n, t["aa_kl_max"], t["ref_cached"]) for n in A for t in (A[n].get("t1") or [])])
check("patch #43567 applied between the blocks and reverted at the end (the fake package file is the original again)",
      (E.get("patch_apply") or {}).get("ok") and E.get("patch_check_after_apply") == "applied" and E.get("patch_state_end") == "not_applied"
      and _hl.sha256(PKG_SB.read_bytes()).hexdigest() == PKG_SB_SHA0, (E.get("patch_apply"), E.get("patch_state_end")))
check("patched T1 arms: the trigger is reached, nothing stale (TRIGGER_REACHED_NO_STALE), reuse hits stay at 128",
      all(v(n) == "TRIGGER_REACHED_NO_STALE" and all(set(t["reuse_cached"]) == {128} for t in A[n]["t1"])
          for n in ("L7_t1a_128_patched", "L8_t1a_256_patched", "L9_t1c_dflash_patched")),
      {n: (v(n), (A.get(n) or {}).get("trace", {}).get("counters")) for n in ("L7_t1a_128_patched", "L8_t1a_256_patched", "L9_t1c_dflash_patched")})
s1, s7 = A["L1_t1a_128"].get("safety") or {}, A["L7_t1a_128_patched"].get("safety") or {}
check("patch-safety cases: 5 cases each, no slot leak, cold probe really cold; stale restores only unpatched",
      len(s1.get("rows") or []) == 5 and len(s7.get("rows") or []) == 5 and s1.get("leaks") == 0 and s7.get("leaks") == 0
      and s1.get("cold_cached_zero") and s7.get("cold_cached_zero") and s1.get("stale", 0) >= 1 and s7.get("stale") == 0,
      (s1, s7))
check("decision A = BUG_EXPOSED_43567 (all three unpatched / patched pairs)",
      (E.get("decision_A") or {}).get("verdict") == "BUG_EXPOSED_43567" and len(E["decision_A"]["patched_clean"]) == 3, E.get("decision_A"))
check("L6 control (no_buffer + EAGLE 3/1/4, overlap and graphs off) detects the known stale restore",
      v("L6_class2_mtp") == "STALE_RESTORE_DETECTED" and val(fl("L6_class2_mtp"), "--mamba-radix-cache-strategy") == "no_buffer"
      and val(fl("L6_class2_mtp"), "--speculative-algorithm") == "EAGLE" and "--disable-overlap-schedule" in fl("L6_class2_mtp")
      and "--disable-cuda-graph" in fl("L6_class2_mtp"), (A.get("L6_class2_mtp"), fl("L6_class2_mtp")))
t2p, t2u = A.get("L11_t2_patched") or {}, A.get("L10_t2_unpatched") or {}
check("T2 ran patched then unpatched on production flags with 16 streams, prefix hits, probes and a verdict",
      t2p.get("verdict") and t2u.get("verdict") and (t2u.get("stats") or {}).get("requests", 0) >= 16
      and (t2u.get("stats") or {}).get("prefix_hits", 0) > 0 and len(t2u.get("probes") or []) == 4
      and val(fl("L10_t2_unpatched"), "--max-mamba-cache-size") == str(G["E14_VENV"]["cfg"]["mamba_cache"])
      and val(fl("L10_t2_unpatched"), "--speculative-algorithm") == "DFLASH" and not (t2u.get("stats") or {}).get("errors"),
      (t2p.get("verdict"), t2u.get("verdict"), t2u.get("stats"), fl("L10_t2_unpatched")))
check("T3b: target-only and DFlash with the radix cache off; serial A/A + 16-way compares complete; a verdict",
      "--disable-radix-cache" in fl("L12_t3b_target") and "--speculative-algorithm" not in fl("L12_t3b_target")
      and "--disable-radix-cache" in fl("L13_t3b_dflash") and val(fl("L13_t3b_dflash"), "--speculative-algorithm") == "DFLASH"
      and (A["L12_t3b_target"].get("serial_aa") or {}).get("compared") == 4
      and (A["L13_t3b_dflash"].get("dflash_conc_vs_target_conc") or {}).get("compared") == 16
      and v("L13_t3b_dflash") == "NO_HIGH_MARGIN_DIVERGENCE", (A.get("L12_t3b_target"), A.get("L13_t3b_dflash")))
check("T3a: fp16 A/A on the long prefixes, then the fp32 launch compared at the same cuts",
      A["L12_t3b_target"].get("t3a_fp16") and val(fl("L15_t3a_fp32"), "--mamba-ssm-dtype") == "float32"
      and (A.get("L15_t3a_fp32") or {}).get("rows") and set(A["L15_t3a_fp32"]["rows"]) == set(A["L12_t3b_target"]["t3a_fp16"]),
      (A["L12_t3b_target"].get("t3a_fp16"), (A.get("L15_t3a_fp32") or {}).get("rows")))
check("KV scales captured from the hook (TP0 rows) on a hooked launch",
      any(((A[n].get("trace") or {}).get("kv_scales") or {}).get("n") for n in A if isinstance(A[n], dict)), "")
check("report + bundle written; no server left running; e17_report.json valid JSON",
      (WORK / "q27x" / "e14" / "e17_report.md").exists() and (WORK / "Q27X_E17_BUNDLE.zip").exists()
      and json.load(open(WORK / "q27x" / "e14" / "e17" / "e17_report.json", encoding="utf-8")).get("decision_A")
      and (G.get("E11_SERVER") is None or not G["E11_SERVER"].alive()), "")
sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
check("submission.json is the cell-0 fallback for every test task", set(sub) == set(TEST))
print("cell seconds:", cell_sec)
print((WORK / "q27x" / "e14" / "e17_report.md").read_text(encoding="utf-8")[:6000] if (WORK / "q27x" / "e14" / "e17_report.md").exists() else "no report")
for label, ok, info in checks:
    print(("PASS " if ok else "FAIL ") + label + ("" if ok else "  <-- " + info))
nf = sum(not ok for _, ok, _ in checks)
print(chr(10) + (f"{nf} failed of {len(checks)}" if nf else f"ALL {len(checks)} CHECKS PASS (E17)"))
sys.stdout.flush()
os._exit(1 if nf else 0)
