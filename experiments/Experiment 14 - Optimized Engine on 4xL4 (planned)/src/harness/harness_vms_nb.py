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

ROOT = Path(os.environ.get("E14_HARNESS_ROOT", r"D:/ARC AGI 2/_scratch/vms/harness_runs"))
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


with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    VPORT = s.getsockname()[1]
os.environ.update(VMS_FAKE_SERVER=str(HERE / "fake_vllm_server.py"), VMS_PORT=str(VPORT), FAKE_SPEED="400")
# rev 2: the RedHat INT4 safetensors index (real key names, kernel_issue/redhat_index.json) for the static MTP check;
# VMS_BAD_INT4=1 = the #58807 case (the ignore list without the mtp entry) -> every INT4 arm must be skipped
_ri = json.load(open(os.environ.get("VMS_REDHAT_INDEX", r"D:/ARC AGI 2/_scratch/kernel_issue/redhat_index.json"), encoding="utf-8"))
json.dump(_ri, open(INT4 / "model.safetensors.index.json", "w"))
BAD_INT4 = os.environ.get("VMS_BAD_INT4") == "1"
if BAD_INT4:
    _c = json.load(open(INT4 / "config.json", encoding="utf-8"))
    _q = _c.get("quantization_config") or _c["text_config"]["quantization_config"]
    _q["ignore"] = [x for x in _q["ignore"] if "mtp" not in x]
    json.dump(_c, open(INT4 / "config.json", "w"))
nb = json.load(open(NB, encoding="utf-8"))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
assert len(code) == 8 and "CELL V1" in code[5] and "CELL V2" in code[6] and "CELL V3" in code[7]
G = {"__name__": "__main__"}
cell_sec = []
for i, c in enumerate(code):
    t = time.time()
    exec(compile(c, f"<cell{i}>", "exec"), G)
    cell_sec.append(round(time.time() - t, 1))
checks = []


def check(label, cond, info=""):
    checks.append((label, bool(cond), "" if cond else str(info)[:1200]))


V = G["VMS"]
A = V["arms"]
L = {x["arm"]: x for x in V["launches"]}


def fl(arm):
    return [str(x) for x in (L.get(arm) or {}).get("flags") or []]


def val(f, k):
    return f[f.index(k) + 1] if k in f else None


want = ["ref_fp8_mtp3_s16", "fp8_target_only_noprefix", "fp8_mtp2", "fp8_mtp4", "int4_mtp3_s16", "fp8_mtp3_s20", "fp8_mtp3_s24",
        "int4_mtp3_s20", "fp8_mtp3_ssm_fp16", "fp8_mtp3_noprefix", "fp8_mtp3_lb_triton", "fp8_mtp3_pcie", "tp2x2_fp8_mtp3",
        "tp2x2_int4_mtp3"]
if BAD_INT4:
    want = [k for k in want if "int4" not in k]
check("mode: profile VMS, matrix ran, staging off, venv = fake", G["PROFILE"] == "VMS" and G["VMS_ON"]
      and G["E14_STAGING"] is None and V["install"].get("fake"), (G["PROFILE"], V["install"]))
check("rev 2b: no arm recorded an exception (Kaggle CPU kit run1: Thread._stop shadowed -> TypeError on Py3.12)",
      not any(A[k].get("error") for k in A), {k: A[k].get("error") for k in A if A[k].get("error")})
check("rev 2d: XL band (~85k prompts) measured on every TP4 arm at the streams the KV holds, none on DP2 x TP2; budget 260 min",
      all(A[k].get("xl_tok_s") and A[k].get("level_xl") for k in want if k in A and not k.startswith("tp2x2"))
      and not any(A[k].get("xl_tok_s") for k in A if k.startswith("tp2x2")) and V["prompts"].get("xl_p50", 0) > 60000
      and G["P"]["time_budget_sec"] == 260 * 60,
      ({k: (A[k].get("xl_tok_s"), A[k].get("level_xl")) for k in A}, V.get("prompts"), G["P"].get("time_budget_sec")))
check("rev 2d: the PCIe-IPC arm set VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1 and the log shows it initialized",
      (L.get("fp8_mtp3_pcie") or {}).get("env") == {"VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "1"}
      and (A.get("fp8_mtp3_pcie", {}).get("pcie_ipc") or {}).get("initialized") and not A["ref_fp8_mtp3_s16"].get("pcie_ipc"),
      (L.get("fp8_mtp3_pcie"), A.get("fp8_mtp3_pcie", {}).get("pcie_ipc")))
check("rev 2e: the oracle is ready; FP8 MTP arms PASS_EXACT, XL gate PASS, 0 canary leaks, adoptable; INT4 arms never adoptable",
      V.get("oracle_ready") and all(A[k].get("gate_class") == "PASS_EXACT" and A[k].get("xl_gate") == "PASS"
                                    and (A[k].get("canary") or {}).get("foreign_leaks") == 0 and A[k].get("adoptable")
                                    for k in ("fp8_mtp2", "fp8_mtp4", "fp8_mtp3_s20", "fp8_mtp3_noprefix", "fp8_mtp3_lb_triton"))
      and A["ref_fp8_mtp3_s16"].get("adoptable") and not any(A[k].get("adoptable") for k in A if k.startswith("int4") or k.startswith("tp2x2_int4")),
      {k: (A[k].get("gate_class"), A[k].get("xl_gate"), (A[k].get("canary") or {}).get("foreign_leaks"), A[k].get("adoptable")) for k in A})
check("rev 2e: every window clean: prefill settled, idle after, threads joined, 0 stream errors / empties; e2e tok/s and "
      "prefix-cache counters recorded (0 hits on the no-prefix arm)",
      all((A[k].get(b) or {}).get("prefill_settled") and (A[k].get(b) or {}).get("idle_after") and (A[k].get(b) or {}).get("threads_not_joined") == 0
          and (A[k].get(b) or {}).get("stream_errors") == 0 and (A[k].get(b) or {}).get("empty_outputs") == 0
          and (A[k].get(b) or {}).get("end_to_end_tok_s") for k in want if k in A for b in ("short", "long"))
      and (A["ref_fp8_mtp3_s16"].get("long") or {}).get("prefix_cache_hits", 0) > 0 and (A["fp8_mtp3_noprefix"].get("long") or {}).get("prefix_cache_hits") == 0,
      {k: {b: {x: (A[k].get(b) or {}).get(x) for x in ("prefill_settled", "idle_after", "stream_errors", "empty_outputs", "end_to_end_tok_s", "prefix_cache_hits")}
           for b in ("short", "long")} for k in want[:3] if k in A})
# rev 2e: the audits' negative tests on the notebook's own functions (synthetic rows, no server)
_dv, _jd, _dg, _tb = G["vms_diverge"], G["vms_judge"], G["vms_degenerate"], G["vms_tail_bad"]
_row = lambda ids, done=True, fin="length", m=None, text="x": {"tokens": ids, "margins": m or [2.0] * len(ids), "error": None,
                                                              "done_seen": done, "finish_reason": fin, "text": text}
_inc = _dv([_row([1, 2, 3])], [_row([1, 2, 3], done=False, fin=None)])
_ids_same_text = _dv([_row([10, 11])], [_row([10, 12])])
# rev 2g: the planned row counts (6 serial, every prompt all at once; GPT R5): fewer rows are INCOMPLETE by design
_NS, _NC = min(6, len(G["_S"])), len(G["_S"])
_base = {"serial_a": [_row(list(range(40)))] * _NS, "serial_b": [_row(list(range(40)))] * _NS,
         "conc_a": [_row(list(range(40)))] * _NC, "conc_b": [_row(list(range(40)))] * _NC}
_enc = lambda s: G["tokenizer"].encode(s, add_special_tokens=False)
_bad = [_row([0, 1] + _enc("!" * 2000), m=[2.0, 0.2] + [2.0] * len(_enc("!" * 2000)))]
_grid = _enc(chr(10).join(" ".join("0" for _ in range(30)) for _ in range(12)))
_v_bad, _d_bad = _jd(_bad + _base["serial_a"][1:], _bad + _base["conc_a"][1:], _base)
_v_ex, _d_ex = _jd(_base["serial_a"], _base["conc_a"], _base)
check("rev 2e negative tests: incomplete SSE row = error; different ids = divergence; a near-tie divergence followed by a "
      "degenerate tail FAILS; identical rows = PASS_EXACT; 1 empty + 1 '!'x100 = 2 degenerate",
      _inc["errors"] == 1 and _inc["identical"] == 0 and _ids_same_text["differ"] == 1 and _v_bad == "FAIL" and _tb(_enc("!" * 2000)) and not _tb(_grid)
      and _v_ex == "PASS" and _d_ex.get("gate_class") == "PASS_EXACT" and _dg(["", "!" * 100, "normal reasoning text " * 20, chr(10).join(" ".join("0" for _ in range(30)) for _ in range(12))]) == 2,
      (_inc, _ids_same_text, _v_bad, _d_bad.get("gate_class"), _v_ex, _d_ex.get("gate_class")))
check("rev 2e-d: certified_exact = adoptable and PASS_EXACT on every arm (FP8 MTP arms certified, INT4 never)",
      all(bool(A[k].get("certified_exact")) == bool(A[k].get("adoptable") and A[k].get("gate_class") == "PASS_EXACT") for k in A)
      and A["fp8_mtp2"].get("certified_exact") and not any(A[k].get("certified_exact") for k in A if k.startswith("int4")),
      {k: A[k].get("certified_exact") for k in A})
# rev 2g: the GPT rev 2f audit negative tests, on the notebook's own functions (synthetic copies of a real arm)
import copy as _cp, tempfile as _tf
_cl, _wh, _txb, _ist = G["vms_classify"], G["vms_window_healthy"], G["vms_text_bad"], G["vms_int4_mtp_static"]
_good = _cp.deepcopy(A["fp8_mtp2"])
def _ad(**mut):
    a = _cp.deepcopy(_good)
    for k, v in mut.items():
        if v is _DEL:
            a.pop(k, None)
        else:
            a[k] = v
    return _cl(a)["adoptable"]
_DEL = object()
_lvl = _good.get("level_wanted") or 16
_neg = {"base adoptable": _ad(), "missing XL": not _ad(xl_gate=None), "XL INCOMPLETE": not _ad(xl_gate="INCOMPLETE"),
        "no canary": not _ad(canary=_DEL), "canary errors": not _ad(canary={"level": _lvl, "errors": 1, "foreign_leaks": 0, "own_hits": 3}),
        "canary leak": not _ad(canary={"level": _lvl, "errors": 0, "foreign_leaks": 1, "own_hits": 3}),
        "canary other level": not _ad(canary={"level": 12 if _lvl != 12 else 16, "errors": 0, "foreign_leaks": 0, "own_hits": 3}),
        "DP diagnostic": not _ad(diagnostic_only=True),
        "PCIe unverified": not _ad(env={"VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "1"}, pcie_ipc={"initialized": False})}
_w = _cp.deepcopy(_good["short"]); _w1 = dict(_w, no_first_token=1)
_neg["window healthy"] = _wh(_w, _lvl); _neg["window no_first_token=1 unhealthy"] = not _wh(_w1, _lvl)
_g30 = chr(10).join(" ".join("0" for _ in range(30)) for _ in range(30))
_neg["text 0*250 bad"] = _txb("0" * 250); _neg["text 0+!+0 bad"] = _txb("0" * 40 + "!" * 60 + "0" * 60)
_neg["text 0 0 0 x200 bad"] = _txb(" ".join("0" * 200)); _neg["text grids ok"] = not any(_txb(x) for x in ("[[0,0],[0,0]]", "0 0 1" + chr(10) + "0 0 0", _g30))
_neg["text prose ok"] = not _txb("Let me look at the first example carefully. The input grid has a red line. " * 6)
_one = {k: v[:1] for k, v in G["VMS_REF"].items() if k in ("serial_a", "serial_b", "conc_a", "conc_b")}
_neg["judge 1 row INCOMPLETE"] = G["vms_judge"](_one["serial_a"], _one["conc_a"], _one)[0] == "INCOMPLETE"
def _st(keys, ign):
    d = Path(_tf.mkdtemp())
    (d / "config.json").write_text(json.dumps({"quantization_config": {"ignore": ign, "format": "pack-quantized"}}), encoding="utf-8")
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "x" for k in keys}}), encoding="utf-8")
    return bool(_ist(d))      # vms_int4_mtp_static returns the ok flag
_saved = G["VMS"].get("int4_mtp_static")
_neg["static dense+ignored ok"] = _st(["mtp.fc.weight", "mtp.layers.0.mlp.gate_proj.weight"], ["re:^mtp.*"])
_neg["static dense not ignored FAIL"] = not _st(["mtp.fc.weight", "mtp.layers.0.mlp.gate_proj.weight"], [])
_neg["static packed+ignored FAIL"] = not _st(["mtp.fc.weight_packed", "mtp.fc.weight_scale"], ["re:^mtp.*"])
_neg["static packed not ignored ok"] = _st(["mtp.fc.weight_packed", "mtp.fc.weight_scale"], [])
G["VMS"]["int4_mtp_static"] = _saved
check("rev 2g negative tests (GPT rev 2f audit R1-R5 / R2F-01..10): fail-closed classifier, first-token coverage, "
      "structural grid rule, planned row counts, INT4 MTP static check", all(_neg.values()), {k: v for k, v in _neg.items() if not v})
check("every planned arm ran and recorded ok with tok/s in both bands",
      all(k in A and A[k].get("ok") and A[k].get("short_tok_s") and A[k].get("long_tok_s") for k in want),
      {k: (A.get(k, {}).get("ok"), A.get(k, {}).get("short_tok_s"), A.get(k, {}).get("long_tok_s")) for k in want})
check("rev 2: the oracle (FP8 target-only, prefix caching off) is the gate reference; the production reference judged "
      "against it (PASS on the fake); 0 degenerate outputs", V.get("gate_reference", "").startswith("fp8_target_only_noprefix")
      and A["fp8_target_only_noprefix"]["gate"] == "oracle" and (V.get("ref_vs_oracle") or {}).get("verdict") == "PASS"
      and A["ref_fp8_mtp3_s16"]["gate"] == "PASS" and A["fp8_target_only_noprefix"]["aa"]["degenerate"] == 0
      and "--no-enable-prefix-caching" in fl("fp8_target_only_noprefix") and "--speculative-config" not in fl("fp8_target_only_noprefix"),
      (V.get("gate_reference"), V.get("ref_vs_oracle"), A.get("fp8_target_only_noprefix", {}).get("gate")))
check("rev 2: degenerate counts in every gate and window", all(A[k].get("short", {}).get("degenerate") == 0 for k in want if k in A)
      and all((A[k].get("gate_detail") or {}).get("degenerate", 0) == 0 for k in want if k in A),
      {k: (A[k].get("short", {}).get("degenerate"), (A[k].get("gate_detail") or {}).get("degenerate")) for k in want if k in A})
check("rev 2b: --linear-backend triton set; lb_marlin + the old env arm dropped (SM 8.9 auto = Marlin), kernels from the log",
      val(fl("fp8_mtp3_lb_triton"), "--linear-backend") == "triton" and "fp8_mtp3_lb_marlin" not in A and "fp8_mtp3_marlin" not in A
      and not any("VLLM_TEST_FORCE_FP8_MARLIN" in json.dumps(x.get("env") or {}) for x in V["launches"])
      and any("Triton" in s for s in A["fp8_mtp3_lb_triton"].get("kernels") or [])
      and any("Marlin" in s for s in A["ref_fp8_mtp3_s16"].get("kernels") or [])
      and any("lb_marlin" in k for k in V.get("dropped") or {}), (A.get("fp8_mtp3_lb_triton", {}).get("kernels"), V.get("dropped")))
_rows = [json.loads(x) for x in open(G["VMS_DIR"] / "vms_greedy_rows.jsonl", encoding="utf-8")] if (G["VMS_DIR"] / "vms_greedy_rows.jsonl").exists() else []
check("rev 2b: token + margin gate: rows saved with tokens and margins, every gate detail counts large-margin flips",
      _rows and all(r["tokens"] and len(r["margins"]) == len(r["tokens"]) for x in _rows for v in x["rows"].values() for r in (v or []))
      and all("large_margin_flips" in (A[k].get("gate_detail") or {}).get("serial", {}) for k in want if A.get(k, {}).get("gate_detail")),
      (len(_rows), [A[k].get("gate_detail", {}).get("serial") for k in want[:3] if k in A]))
check("rev 2b: one deadline per arm recorded, both bands measured, server alive after, every KV line kept",
      all(A[k].get("deadline_in_sec") and A[k].get("server_alive_after") for k in want if k in A)
      and all(x.get("kv_tokens_all") for x in V["launches"] if x.get("ok")),
      {k: (A[k].get("deadline_in_sec"), A[k].get("server_alive_after")) for k in want if k in A})
_st = V.get("int4_mtp_static") or {}
check("rev 2: INT4 MTP static check " + ("FAILS on the #58807 ignore list and every INT4 arm is skipped" if BAD_INT4 else
      "passes on the RedHat index (15 dense mtp keys, re:^mtp.* in the ignore list)"),
      (not _st.get("ok") and all("static check" in str(A.get(k, {}).get("skipped")) for k in ("int4_mtp3_s16", "int4_mtp3_s20", "tp2x2_int4_mtp3"))
       and not any(fl(k)[:1] == [str(G["SGL_MODEL_PATH"])] for k in L)) if BAD_INT4 else
      (_st.get("ok") and _st.get("mtp_keys") == 15 and _st.get("packed") == 0 and _st.get("mtp_ignore_entries") == ["re:^mtp.*"]
       and not _st.get("not_ignored")), _st)
check("levels really reached: 16 / 20 / 24, 2 x TP2 = 16 (8 per DP rank)",
      A["ref_fp8_mtp3_s16"]["level_short"] == 16 and A["fp8_mtp3_s20"]["level_short"] == 20
      and A["fp8_mtp3_s24"]["level_short"] == 24 and A["tp2x2_fp8_mtp3"]["level_short"] == 16,
      {k: A[k].get("level_short") for k in want if k in A})
f = fl("ref_fp8_mtp3_s16")
check("reference = S1-60K engine: FP8 model, TP4, fp8 KV, MTP 3, FLASHINFER, prefix caching, 16 seqs, 4096 batched, "
      "mem 0.90, custom all-reduce off", f[0] == str(G["MODEL_PATH"]) and val(f, "--tensor-parallel-size") == "4"
      and val(f, "--kv-cache-dtype") == "fp8_e4m3" and json.loads(val(f, "--speculative-config")) == {"method": "mtp", "num_speculative_tokens": 3}
      and val(f, "--attention-backend") == "FLASHINFER" and "--enable-prefix-caching" in f and val(f, "--max-num-seqs") == "16"
      and val(f, "--max-num-batched-tokens") == "4096" and val(f, "--gpu-memory-utilization") == "0.9"
      and "--disable-custom-all-reduce" in f, f)
check("arms set what they say", json.loads(val(fl("fp8_mtp4"), "--speculative-config"))["num_speculative_tokens"] == 4
      and (BAD_INT4 or fl("int4_mtp3_s16")[0] == str(G["SGL_MODEL_PATH"])) and val(fl("fp8_mtp3_s24"), "--max-num-seqs") == "24"
      and val(fl("fp8_mtp3_ssm_fp16"), "--mamba-ssm-cache-dtype") == "float16" and "--no-enable-prefix-caching" in fl("fp8_mtp3_noprefix")
      and val(fl("tp2x2_fp8_mtp3"), "--tensor-parallel-size") == "2" and val(fl("tp2x2_fp8_mtp3"), "--data-parallel-size") == "2"
      and val(fl("tp2x2_fp8_mtp3"), "--max-num-seqs") == "8" and val(fl("tp2x2_fp8_mtp3"), "--api-server-count") == "1"
      and "--disable-custom-all-reduce" in fl("tp2x2_fp8_mtp3") and "--disable-custom-all-reduce" in fl("ref_fp8_mtp3_s16"))
check("each flag once per launch", all(max(fl(k).count(x) for x in fl(k) if x.startswith("--")) == 1 for k in L if fl(k)))
check("gates: FP8 arms PASS vs the oracle; INT4 diagnostic",
      all(A[k]["gate"] == "PASS" for k in ("fp8_mtp2", "fp8_mtp4", "fp8_mtp3_s20", "fp8_mtp3_lb_triton", "tp2x2_fp8_mtp3"))
      and (BAD_INT4 or A["int4_mtp3_s16"]["gate"] == "diagnostic"),
      {k: A[k].get("gate") for k in want if k in A})
check("accept length measured for MTP arms, none for target-only", A["ref_fp8_mtp3_s16"]["accept_short"]
      and A["fp8_target_only_noprefix"]["accept_short"] is None,
      (A["ref_fp8_mtp3_s16"]["accept_short"], A["fp8_target_only_noprefix"]["accept_short"]))
check("KV tokens parsed from the server log", all(A[k].get("kv_tokens") for k in want if k in A))
check("2 x TP2 not skipped (reserved slot)", not A["tp2x2_fp8_mtp3"].get("skipped") and (BAD_INT4 or not A["tp2x2_int4_mtp3"].get("skipped")))
check("report + bundle; server stopped", (G["VMS_DIR"] / "vms_report.md").exists() and (WORK / "Q27X_VMS_BUNDLE.zip").exists()
      and (G["VMS_SERVER"] is None or not G["VMS_SERVER"].alive()))
sub = json.load(open(WORK / "submission.json", encoding="utf-8"))
check("submission.json = the cell-0 fallback for every test task", set(sub) == set(TEST))
print("cell seconds:", cell_sec)
print((G["VMS_DIR"] / "vms_report.md").read_text(encoding="utf-8") if (G["VMS_DIR"] / "vms_report.md").exists() else "no report")
for label, ok, info in checks:
    print(("PASS " if ok else "FAIL ") + label + ("" if ok else "  <-- " + info))
nf = sum(not ok for _, ok, _ in checks)
print(f"\n{nf} failed of {len(checks)}" if nf else f"\nALL {len(checks)} CHECKS PASS (VMS)")
sys.stdout.flush()
os._exit(1 if nf else 0)
