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
E15H = Path(os.environ.get("E15P_HARNESS_DIR", r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/src/harness"))
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

ROOT = Path(os.environ.get("E14_HARNESS_ROOT", r"D:/ARC AGI 2/_scratch/ems/harness_runs"))
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
for _f in (Path(os.environ.get("EMS_PAYLOAD_DIR", r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload"))).glob("q27x_e14_modules_*.bin"):   # E15P needs only this one
    shutil.copy(_f, _pd / _f.name)
with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    PORT = s.getsockname()[1]
os.environ.update(Q27X_ARC_ROOT=str(ARC), Q27X_WORKING=str(WORK), Q27X_MODEL=str(FP8), Q27X_SKIP_INSTALL="1",
                  PYTHONIOENCODING="utf-8", E11_HARNESS="1", E11_PORT=str(PORT),
                  E11_FAKE_SERVER=str(E15H / "e15p_fake_server.py"), E11_INT4=str(INT4), E11_DFLASH=str(DFL),
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
assert len(code) == 9 and "CELL EMS —" in code[7] and "CELL EMS-R" in code[8], [c.splitlines()[0][:70] for c in code]
# rev 2f: HARNESS_EMS_PART=ALL (default) runs every arm (EMS_PART=ALL) with the full checks; SAFE / RISKY run the
# notebook with its baked-in part and check the part split
HPART = os.environ.get("HARNESS_EMS_PART", "ALL")
if HPART == "ALL":
    os.environ["EMS_PART"] = "ALL"
else:
    os.environ.pop("EMS_PART", None)
G = {"__name__": "__main__"}
cell_sec = []
for i, c in enumerate(code):
    t = time.time()
    exec(compile(c, f"<cell{i}>", "exec"), G)
    cell_sec.append(round(time.time() - t, 1))

checks = []


def check(label, cond, info=""):
    checks.append((label, bool(cond), "" if cond else str(info)[:1200]))


R = json.load(open(WORK / "q27x" / "e14" / "e14_results.json", encoding="utf-8"))
EM = R.get("ems") or {}
A = EM.get("arms") or {}
L = {x["arm"]: x for x in R.get("launches") or [] if "arm" in x}


def fl(arm):
    return [str(x) for x in (L.get(arm) or {}).get("flags") or []]


def val(f, k, n=1):
    return f[f.index(k) + 1:f.index(k) + 1 + n] if k in f else None


def finish():
    check("report table + bundle written; no server left running",
          (WORK / "q27x" / "e14" / "ems_report.md").exists() and any(WORK.glob("Q27X_EMS*_BUNDLE.zip"))
          and (G.get("E11_SERVER") is None or not G["E11_SERVER"].alive()))
    sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
    check("submission.json is the cell-0 fallback for every test task", set(sub) == set(TEST))
    print("cell seconds:", cell_sec)
    print((WORK / "q27x" / "e14" / "ems_report.md").read_text(encoding="utf-8") if (WORK / "q27x" / "e14" / "ems_report.md").exists() else "no report")
    for label, ok, info in checks:
        print(("PASS " if ok else "FAIL ") + label + ("" if ok else "  <-- " + info))
    nf = sum(not ok for _, ok, _ in checks)
    print(chr(10) + (f"{nf} failed of {len(checks)}" if nf else f"ALL {len(checks)} CHECKS PASS (EMS {HPART})"))
    sys.stdout.flush()
    os._exit(1 if nf else 0)


if HPART != "ALL":
    _pa = set(G["EMS_PART_ARMS"][HPART]) | {"ref_dflash_w6_s16", "target_only"}
    _ran = {k for k, v in A.items() if not v.get("skipped")}
    # rev 2g: both parts run their own best MTP at 20 after the plan; RISKY also runs 2 x TP2
    _dyn = {k for k in _ran if (k.startswith(("mtp", "ntn")) and k.endswith("_s20"))} |         ({k for k in _ran if k.startswith("tp2x2_")} if HPART == "RISKY" else set())
    _ndyn = {"SAFE": 1, "RISKY": 3, "MTPX": 0}[HPART]       # rev 2h: MTPX runs no best-MTP-at-20 and no TP2
    check(f"rev 2f part {HPART}: EMS_PART baked in", G["EMS_PART"] == HPART and EM.get("part") == HPART, (G.get("EMS_PART"), EM.get("part")))
    check(f"rev 2f part {HPART}: exactly its arms ran, plus the oracle and reference and "
          + ("the best MTP at 20" if HPART == "SAFE" else "the best MTP at 20 and 2 x TP2 (DFlash + the best MTP)"),
          _ran - _dyn == _pa and len(_dyn) == _ndyn, (sorted(_ran), sorted(_dyn)))
    check(f"rev 2f part {HPART}: every arm that ran is ok; every other arm recorded OTHER_PART",
          all(A[k].get("ok") for k in _ran) and all(str(A[k].get("skipped", "")).startswith("OTHER_PART") for k in A if k not in _ran)
          and any(k not in _ran for k in A), {k: (A[k].get("ok"), A[k].get("skipped")) for k in A})
    check(f"rev 2f part {HPART}: no EMS cell error; the #45151 patch only in RISKY",
          not EM.get("error") and (("part: not needed" in str(EM["pr45151_patch"].get("why")))      # rev 2g-d: the overlay runs in SAFE too (PCIe arms)
                                   if HPART in ("SAFE", "MTPX") else "part: not needed" not in str(EM.get("pr45151_patch"))),
          (EM.get("error"), EM.get("pr45151_patch"), EM.get("fi07_overlay")))
    check(f"rev 2f part {HPART}: the reference is adoptable (probe_complete_vs_target set), the oracle canary clean",
          A["ref_dflash_w6_s16"].get("adoptable") and (EM.get("canary_oracle") or {}).get("foreign_leaks") == 0,
          {x: A["ref_dflash_w6_s16"].get(x) for x in ("adoptable", "gate", "probe_complete_vs_target", "canary_leaks", "xl_gate")})
    finish()


check("mode: profile EMS, not a rerun, the matrix ran", G["PROFILE"] == "EMS" and G["EMS_ON"] and not G["RERUN"], G["PROFILE"])
want = ["ref_dflash_w6_s16", "target_only", "mtp_3_1_4", "ntn_fp8", "ntn_int4", "dflash_w6_s20", "dflash_w6_s24",
        "mtp_2_1_3", "mtp_3_1_4_mem86", "mtp_replay_fp16", "mtp_replay_fp32", "dflash_w4", "dflash_w8",
        "tp2x2_dflash_w6", "dflash_w6_pcie", "mtp_3_1_4_pr45151", "mtp_3_1_4_pcie"]
check("every planned arm was run and recorded ok", all(k in A and A[k].get("ok") for k in want),
      {k: (A.get(k) or {}).get("ok") for k in want})
check("measured tok/s for every single-server arm (8k and 40k bands)",
      all(A[k].get("short_tok_s") and A[k].get("long_tok_s") for k in want if k in A and not k.startswith("tp2")),
      {k: (A[k].get("short_tok_s"), A[k].get("long_tok_s")) for k in want if k in A})
check("rev 2: target-only is the gate reference; the DFlash reference judged against it (PASS on the fake)",
      EM.get("gate_reference", "").startswith("target_only") and (EM.get("dflash_vs_target_gate") or {}).get("verdict") == "PASS"
      and A.get("ref_dflash_w6_s16", {}).get("gate") == "PASS" and (EM.get("probe_tgt_aa") or {}).get("compared")
      and (EM.get("dflash_vs_target_probe") or {}).get("compared"), (EM.get("gate_reference"), EM.get("dflash_vs_target_gate"),
                                                                     A.get("ref_dflash_w6_s16", {}).get("gate")))
check("rev 2: every gated INT4 arm also compared with the target-only probe",
      all(A[k].get("probe_flips_vs_target") is not None for k in ("mtp_3_1_4", "ntn_int4", "dflash_w4", "dflash_w8") if k in A),
      {k: A.get(k, {}).get("probe_flips_vs_target") for k in ("mtp_3_1_4", "ntn_int4", "dflash_w4", "dflash_w8")})
check("rev 2: the linear-attention arms are dropped with their source reasons, never launched",
      set(EM.get("dropped") or {}) == {"linattn_flashinfer", "linattn_cutedsl"} and not any(k.startswith("linattn") for k in L),
      (EM.get("dropped"), [k for k in L if k.startswith("linattn")]))
check("rev 2: no arm flagged CAPPED on the fake (it admits what it is asked)", not any(A[k].get("capped") for k in A),
      {k: (A[k].get("level_wanted"), A[k].get("per_replica_streams"), A[k].get("level_short")) for k in A if A[k].get("capped")})
check("INT4 arms pass the correctness gate vs the target-only A/A",
      all(A[k].get("gate") == "PASS" for k in ("mtp_3_1_4", "ntn_int4", "dflash_w6_s20", "dflash_w4", "dflash_w8") if k in A),
      {k: A.get(k, {}).get("gate") for k in ("mtp_3_1_4", "ntn_int4", "dflash_w6_s20", "dflash_w4", "dflash_w8")})
check("FP8 is diagnostic, target-only is the reference",
      A.get("ntn_fp8", {}).get("gate") == "diagnostic" and A.get("target_only", {}).get("gate") == "reference",
      (A.get("ntn_fp8", {}).get("gate"), A.get("target_only", {}).get("gate")))
f = fl("mtp_3_1_4")
check("MTP arm: EAGLE 3/1/4, draft path = the INT4 dir, no DFLASH, no delayer flag",
      val(f, "--speculative-algorithm") == ["EAGLE"] and val(f, "--speculative-num-steps") == ["3"]
      and val(f, "--speculative-eagle-topk") == ["1"] and val(f, "--speculative-num-draft-tokens") == ["4"]
      and val(f, "--speculative-draft-model-path") == val(f, "--model-path") and "DFLASH" not in f
      and "--min-free-slots-delay" not in f and "--speculative-draft-model-quantization" not in f, f)
f = fl("ntn_fp8")
check("NTN-FP8: FP8 model + its draft, flashinfer attention, triton linear attention, chunk 2048, max prefill 4096, "
      "extra_buffer, mamba ratio 0.5 + rev 2b mamba 85 (5 x 17), SSM fp32, EAGLE 3/1/4",
      val(f, "--model-path") == [str(G["MODEL_PATH"])] and val(f, "--speculative-draft-model-path") == [str(G["MODEL_PATH"])]
      and val(f, "--attention-backend") == ["flashinfer"] and val(f, "--linear-attn-backend") == ["triton"]
      and val(f, "--chunked-prefill-size") == ["2048"] and val(f, "--max-prefill-tokens") == ["4096"]
      and val(f, "--mamba-radix-cache-strategy") == ["extra_buffer"] and val(f, "--mamba-full-memory-ratio") == ["0.5"]
      and val(f, "--max-mamba-cache-size") == ["85"] and val(f, "--mamba-ssm-dtype") == ["float32"]
      and val(f, "--speculative-algorithm") == ["EAGLE"] and f.count("--model-path") == 1, f)
f = fl("ntn_int4")
check("NTN-INT4: the same flags on the INT4 model", val(f, "--model-path") != [str(G["MODEL_PATH"])]
      and val(f, "--mamba-ssm-dtype") == ["float32"] and val(f, "--chunked-prefill-size") == ["2048"], f)
f = fl("dflash_w6_s20")
check("20 streams: mrr 21, graphs 1..21, mamba 84, DFLASH width 6",
      val(f, "--max-running-requests") == ["21"] and val(f, "--cuda-graph-bs-decode", 21) == [str(b) for b in range(1, 22)]
      and val(f, "--max-mamba-cache-size") == ["84"] and val(f, "--speculative-num-draft-tokens") == ["6"], f)
check("20 / 24 streams really measured at 20 / 24", A.get("dflash_w6_s20", {}).get("level_short") == 20
      and A.get("dflash_w6_s24", {}).get("level_short") == 24, (A.get("dflash_w6_s20", {}).get("level_short"),
                                                                  A.get("dflash_w6_s24", {}).get("level_short")))
check("each flag appears once in every launch (no duplicate after the edits)",
      all(max(fl(k).count(x) for x in fl(k) if x.startswith("--")) == 1 for k in L if fl(k)),
      [k for k in L if fl(k) and max(fl(k).count(x) for x in fl(k) if x.startswith("--")) > 1])
check("width / ReplaySSM arms set what they say",
      val(fl("dflash_w4"), "--speculative-num-draft-tokens") == ["4"] and val(fl("dflash_w8"), "--speculative-num-draft-tokens") == ["8"]
      and val(fl("mtp_replay_fp32"), "--mamba-ssm-dtype") == ["float32"] and "--enable-linear-replayssm-spec" in fl("mtp_replay_fp16"))
_best = [k for k in A if k.startswith("mtp_") and k.endswith("_s20")]
check("the best gated MTP was re-run at 20 streams", len(_best) == 1 and A[_best[0]].get("level_short") == 20, _best)
tp = (R.get("phases") or {}).get("EMS_tp2x2_dflash_w6") or {}
reps = (tp.get("launch") or {}).get("replicas") or []
check("2 x TP2: both replicas ready on their own ports, 8 + 8 streams measured, sum of tok/s recorded",
      A.get("tp2x2_dflash_w6", {}).get("replicas_ready") == [True, True] and len({r["port"] for r in reps}) == 2
      and (A["tp2x2_dflash_w6"].get("short_tok_s") or 0) > 0 and (A["tp2x2_dflash_w6"].get("long_tok_s") or 0) > 0,
      (A.get("tp2x2_dflash_w6"), reps))
_t2 = A.get("tp2x2_dflash_w6", {})
_cd = _t2.get("cache_dirs") or []
check("rev 2: 2 x TP2 pairs from the topology (default 0-1 / 2-3 without nvidia-smi), own cache dirs per replica, "
      "custom all-reduce state recorded", _t2.get("pairs") == [[0, 1], [2, 3]] and len(_cd) == 2
      and all(len({d[k] for d in _cd}) == 2 for k in _cd[0]) and all(v.endswith(("-r0", "-r1")) or "-r0" in v or "-r1" in v
                                                                      for d in _cd for v in d.values())
      and len(_t2.get("custom_allreduce_disabled") or []) == 2 and "pairs" in (EM.get("topology") or {}),
      (_t2.get("pairs"), _cd, _t2.get("custom_allreduce_disabled"), EM.get("topology")))
f = fl("mtp_3_1_4")
check("rev 2b: oracle flag, reference name from the real flags, family tags, MTP gate vs the DFlash reference recorded",
      EM.get("oracle") == "target_only" and EM.get("ref_name") == "ref_dflash_w6_s16" and A["mtp_3_1_4"].get("family") == "mtp"
      and A["ntn_int4"].get("family") == "mtp" and A["dflash_w4"].get("family") == "dflash" and A["target_only"].get("family") is None
      and A["mtp_3_1_4"].get("gate_vs_dflash_ref") == "PASS" and A["mtp_3_1_4"].get("probe_complete_vs_target")
      and A["mtp_3_1_4"].get("server_alive_after") is True and not EM.get("error"),
      (EM.get("oracle"), EM.get("ref_name"), {k: (A[k].get("family"), A[k].get("gate_vs_dflash_ref")) for k in A}, EM.get("error")))
_t2r = (R.get("phases") or {}).get("EMS_tp2x2_dflash_w6") or {}
check("rev 2b: 2 x TP2 = both replicas gated, all four windows ok, KV-for-production flag, custom all-reduce off",
      len(((_t2r.get("result") or {}).get("gate_replicas")) or []) == 2 and (_t2r.get("result") or {}).get("all_windows_ok")
      and "kv_ok_for_production" in A.get("tp2x2_dflash_w6", {})
      and len((_t2r.get("launch") or {}).get("replica_flags") or []) == 2
      and all("--disable-custom-all-reduce" in fl_ and "--cuda-graph-max-bs-decode" not in fl_ for fl_ in (_t2r.get("launch") or {}).get("replica_flags")),
      ((_t2r.get("result") or {}).get("all_windows_ok"), (_t2r.get("launch") or {}).get("replica_flags")))
check("rev 2c: the PR #45151 arm ran with EMS_PR45151=1 (the patch itself is skipped in the harness: no SGLang install)",
      A.get("mtp_3_1_4_pr45151", {}).get("ok") and A["mtp_3_1_4_pr45151"].get("env") == {"EMS_PR45151": "1"}
      and (EM.get("pr45151_patch") or {}).get("why") == "harness: no SGLang install" and ("p2p_read" in (EM.get("topology") or {}) or "error" in (EM.get("topology") or {})),   # no nvidia-smi locally
      (A.get("mtp_3_1_4_pr45151"), EM.get("pr45151_patch"), (EM.get("topology") or {}).keys()))
_rep = (WORK / "q27x" / "e14" / "ems_report.md").read_text(encoding="utf-8") if (WORK / "q27x" / "e14" / "ems_report.md").exists() else ""
check("rev 2d-b: the report carries the rev 2b / 2d lines (oracle flag, cell error none, overlay, XL / PCIe columns) and the cell is "
      "inside try / except", "oracle flag: target_only" in _rep and "EMS cell error: none" in _rep and "FlashInfer 0.7 overlay" in _rep
      and "XL tok/s" in _rep and "PCIe-IPC workspace built" in _rep and "EMS cell error:" in code[7] and code[7].count(chr(10) + "try:" + chr(10) + "    if EMS_ON and ENGINE_READY:") == 1,
      _rep[:1500])
check("rev 2d: XL band (~85k prompts) on every single-server arm at the streams the KV holds; none on 2 x TP2; budget 260 min",
      # rev 2e-e: an arm whose own phase cap ran out records xl {"skipped": "deadline"} (time-scaled harness jitter);
      # that is allowed, a silent miss is not; the core arms must be measured
      all((A[k].get("xl_tok_s") and A[k].get("level_xl"))
          or str(((G["E14_RESULTS"]["phases"].get("EMS_" + k) or {}).get("xl") or {}).get("skipped", "")).startswith("deadline")
          for k in want if k in A and not k.startswith("tp2x2"))
      and all(A[k].get("xl_tok_s") for k in ("ref_dflash_w6_s16", "target_only", "mtp_3_1_4", "mtp_3_1_4_pr45151", "dflash_w6_s20"))
      and not any(A[k].get("xl_tok_s") for k in A if k.startswith("tp2x2")) and (EM.get("prompts") or {}).get("xl_p50", 0) > 60000
      and G["P"]["time_budget_sec"] == 260 * 60,
      ({k: (A[k].get("xl_tok_s"), A[k].get("level_xl")) for k in A}, EM.get("prompts"), G["P"].get("time_budget_sec")))
check("rev 2d: PCIe-IPC arms carry the overlay env (PYTHONPATH + SGLANG_ENABLE_PCIE_IPC_ALLREDUCE); the overlay is skipped in "
      "the harness; no other arm has it", all((A.get(k, {}).get("env") or {}).get("SGLANG_ENABLE_PCIE_IPC_ALLREDUCE") == "1"
                                              and (A[k]["env"].get("PYTHONPATH") or "").endswith("ems-fi07") for k in ("dflash_w6_pcie", "mtp_3_1_4_pcie"))
      and (EM.get("fi07_overlay") or {}).get("why") == "harness: no SGLang install"
      and not any((A[k].get("env") or {}).get("SGLANG_ENABLE_PCIE_IPC_ALLREDUCE") for k in A if not k.endswith("_pcie")),
      ({k: A.get(k, {}).get("env") for k in ("dflash_w6_pcie", "mtp_3_1_4_pcie")}, EM.get("fi07_overlay")))
check("rev 2e: oracle canary + XL A/A; INT4 arms (DFlash ref, MTP, widths) PASS_EXACT, XL gate PASS, 0 canary leaks, adoptable; "
      "ntn_fp8 after the 20 / 24 arms", (EM.get("canary_oracle") or {}).get("foreign_leaks") == 0
      and all(A[k].get("gate_class") == "PASS_EXACT" and A[k].get("xl_gate") == "PASS" and A[k].get("canary_leaks") == 0 and A[k].get("adoptable")
              for k in ("ref_dflash_w6_s16", "mtp_3_1_4", "mtp_3_1_4_pr45151", "dflash_w4", "dflash_w6_s20"))
      and not A["ntn_fp8"].get("adoptable") and EM["order"].index("ntn_fp8") > EM["order"].index("dflash_w6_s24"),
      {k: (A[k].get("gate_class"), A[k].get("xl_gate"), A[k].get("canary_leaks"), A[k].get("adoptable")) for k in A})
check("rev 2e-d: certified_exact = adoptable and PASS_EXACT on every arm; the radix-cache hit rate is recorded on "
      "every measured band (the fake server reports #cached-token 0)",
      all(bool(A[k].get("certified_exact")) == bool(A[k].get("adoptable") and A[k].get("gate_class") == "PASS_EXACT") for k in A)
      and A["mtp_3_1_4"].get("certified_exact") and A["ref_dflash_w6_s16"].get("prefix_hit_short") is not None
      and A["ref_dflash_w6_s16"].get("prefix_hit_long") is not None,
      {k: (A[k].get("certified_exact"), A[k].get("prefix_hit_short"), A[k].get("prefix_hit_long")) for k in A})
# rev 2g: the GPT rev 2f audit negative tests on _gated / _text_bad (synthetic copies of a real arm)
import copy as _cp
_gd, _txb = G["_gated"], G["_text_bad"]
_good = _cp.deepcopy(A["mtp_3_1_4"])
def _gt(**mut):
    a = _cp.deepcopy(_good); a.update(mut); return bool(_gd(a))
_oc = _cp.deepcopy(G["EMSR"].get("canary_oracle"))
_neg = {"base adoptable": _gt(), "missing XL": not _gt(xl_gate=None), "XL INCOMPLETE": not _gt(xl_gate="INCOMPLETE"),
        "PCIe unverified": not _gt(env={"SGLANG_ENABLE_PCIE_IPC_ALLREDUCE": "1"}, pcie_ipc={"workspace_built": False})}
G["EMSR"]["canary_oracle"] = dict(_oc or {}, errors=1); _neg["oracle canary errors"] = not _gt()
G["EMSR"]["canary_oracle"] = None; _neg["no oracle canary"] = not _gt()
G["EMSR"]["canary_oracle"] = _oc
_g30 = chr(10).join(" ".join("0" for _ in range(30)) for _ in range(30))
_neg["text 0*250 bad"] = _txb("0" * 250); _neg["text 0+!+0 bad"] = _txb("0" * 40 + "!" * 60 + "0" * 60)
_neg["text 0 0 0 x200 bad"] = _txb(" ".join("0" * 200)); _neg["text grids ok"] = not any(_txb(x) for x in ("[[0,0],[0,0]]", "0 0 1" + chr(10) + "0 0 0", _g30))
_neg["canary classification recorded"] = (G["EMSR"].get("canary_oracle") or {}).get("classification") in ("PASS_NO_OBSERVED_LEAK", "INCONCLUSIVE")
_neg["every arm has production_approved False"] = all(A[k].get("production_approved") is False for k in A if "adoptable" in A[k])
check("rev 2g negative tests (GPT rev 2f audit R2F-02 / R6 / R2F-06 / R2F-07): fail-closed _gated, structural grid rule",
      all(_neg.values()), {k: v for k, v in _neg.items() if not v})
_tb = G["_tail_bad"]
_enc = lambda s: G["tokenizer"].encode(s, add_special_tokens=False)
check("rev 2e-c negative tests: the tail check flags repeated punctuation and passes an ARC grid and normal prose",
      _tb(_enc("!" * 2000)) and _tb(_enc(", " * 400)) and not _tb(_enc(chr(10).join(" ".join("0" for _ in range(30)) for _ in range(12))))
      and not _tb(_enc("Let me look at the first example carefully. The input grid has a red line. " * 6)), None)
check("2 x TP2 also ran on the best MTP", any(k.startswith("tp2x2_mtp_") and A[k].get("ok") for k in A), list(A))
check("the 2 x TP2 slot was not skipped (reserved slot)", not any((v or {}).get("skipped") for k, v in (R.get("phases") or {}).items()
                                                                  if k.startswith("EMS_tp2")), R.get("phases", {}).get("EMS_tp2x2_dflash_w6"))
finish()
