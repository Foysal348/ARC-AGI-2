"""E27 NBDF Kaggle CPU kits (pinned Py3.12 image, no GPU, no internet, private; account foysalemonshanto). Pattern: the NB4T
kits (build_e25_nb4t_cpu_kit.py). Each kernel is fail-soft, writes its JSON report and prints exactly one line
`@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`.

  units  q27-e27-nbdf-kit-units: nbdf_unit_tests.py (T1-T10, incl. SeamSeq + nb4_ppack on real copies of the Merlin files
         and the real NB4T Q worker record) and dflash_seam_tests.py (the seam agent's 520 checks) on Py3.12.
  real   q27-e27-nbdf-kit-real (the 7 RC2 Models + the drafter Model + ARC): NBDF's own code (RC2_UNIT + RC2_CPU_KIT) on the REAL
         mounts: CELL 0 identity submission; inputs, merge, full sha256, prompts; the drafter gate on the real mount (pinned
         file / embed tensor / mask-row shas, no model load); the real Merlin install + the flag-source shas; #60490 + the
         extension; nb4_ppack + dflash_seam applied by the notebook's own calls (three-state check, combined_tree_state);
         launch.build() + host_phase() + the NBDF launch check for ALL 5 arms on the patched tree, each env read back in a
         spawned child; the argv fields; the spec-config preflight in the Merlin venv (every key a SpeculativeConfig field,
         the Literal values, the backend enum, draft_tensor_parallel_size unset -> the target TP); the patched-model import
         probe; the drafter loaded-weights probe; Merlin's get_kv_cache_configs on per-stage specs (Q calibrated against the
         NB4T GPU record: 215 blocks; then the drafter group per K) = the KV gate the notebook embeds; then seam + ppack
         reverted (pre-shas), #60490 / extension restored.
Run: python build_e27_nbdf_cpu_kit.py -> D:/ARC AGI 2/_scratch/push_e25/q27-e27-nbdf-kit-{units,real}__foysalemonshanto/
The real kit's report (nbdf_kit_real_report.json) is the --kv-gate-report input of build_e27_nbdf.py.
"""
import base64
import hashlib
import json
import lzma
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(E25B))
import nbdf_lib as N  # noqa: E402
import nb4_ppack as PPK  # noqa: E402
import dflash_seam as DSM  # noqa: E402
K100 = os.environ.get("NBDF_VARIANT") == "k100"      # E27 NBDF-K100: build_e27_nbdf_k100_cpu_kit.py
NB_PATH = HERE.parent / ("Q27-XHIGH-E27_NBDF_K100.ipynb" if K100 else "Q27-XHIGH-E27_NBDF_DFLASH2.ipynb")
NB_SHA = hashlib.sha256(NB_PATH.read_bytes()).hexdigest()
FACTS = json.loads((HERE / ("nbdf_k100_build_facts.json" if K100 else "nbdf_build_facts.json")).read_text(encoding="utf-8"))
assert FACTS["sha256"] == NB_SHA, "build facts are not for this notebook: rebuild"
EXPORT = (HERE / ("nbdf_k100_code_export.py" if K100 else "nbdf_code_export.py")).read_text(encoding="utf-8")
assert hashlib.sha256(EXPORT.encode("utf-8")).hexdigest() == FACTS["export_sha256"], "export != build facts"
MERLIN_SRC = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
NB4T_RES = Path(r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb4t-levers/nb4t/nb4t_results.json")
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
OWNER = "foysalemonshanto"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
MODELS = FACTS["model_sources"]
SRC_FILES = ("tools/merlin/launch.py", "tools/merlin/registry.py", "vllm/config/speculative.py",
             "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py", "vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py",
             "vllm/platforms/cuda.py", "vllm/v1/worker/gpu/cudagraph_utils.py", "vllm/v1/core/kv_cache_utils.py",
             "vllm/v1/kv_cache_interface.py")
SRC_SHAS = {rel: hashlib.sha256((MERLIN_SRC / rel).read_bytes()).hexdigest() for rel in SRC_FILES}
SEAM_TEST_RELS = ("vllm/model_executor/layers/attention/attention.py", "vllm/model_executor/models/config.py",
                  "vllm/model_executor/models/interfaces.py", "vllm/model_executor/models/interfaces_base.py",
                  "vllm/model_executor/models/qwen3.py", "vllm/model_executor/models/qwen3_5.py",
                  "vllm/model_executor/models/qwen3_dflash.py", "vllm/model_executor/models/qwen3_vl.py",
                  "vllm/model_executor/models/utils.py", "vllm/models/qwen4_exp/common/qsa_cache.py",
                  "vllm/models/qwen4_exp/nvidia/model.py", "vllm/models/qwen4_exp/nvidia/ops/hc.py",
                  "vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu/pp_utils.py",
                  "vllm/v1/worker/gpu/spec_decode/dflash/utils.py", "vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py",
                  "vllm/v1/worker/gpu/spec_decode/eagle/utils.py", "vllm/v1/worker/gpu_worker.py",
                  "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py", "vllm/v1/worker/utils.py")


def blob(files):
    return base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


R_ = json.loads(NB4T_RES.read_text(encoding="utf-8"))
Q_REC = R_["sections"]["Q"]["nb4t_q"]
Q_SLIM = {"workers": Q_REC["workers"], "pool": Q_REC["pool"], "env_readback": Q_REC["env_readback"],
          "config_readback": Q_REC["config_readback"], "status": Q_REC["status"], "quality_gate": Q_REC["quality_gate"],
          "q_readback_final": Q_REC.get("q_readback_final"), "free_after_ready_min": Q_REC.get("free_after_ready_min"),
          "capacity": Q_REC.get("capacity"), "pin": Q_REC.get("pin"), "not_integrated": Q_REC.get("not_integrated")}
UNIT_FILES = {f"build/{n}": (HERE / n).read_text(encoding="utf-8") for n in ("nbdf_lib.py", "nbdf_unit_tests.py", "dflash_seam.py",
                                                                             "dflash_seam_tests.py", "nbdf_cell7_body.py")}
_BODY_KEY = "build/nbdf_k100_cell7_body.py" if K100 else "build/nbdf_cell7_body.py"
if K100:                                   # E27 K100: + the K100 body / tests, klee100 LOAD's lib and the QSA scales sidecar
    UNIT_FILES.pop("build/nbdf_cell7_body.py")   # size (< 400 kB): the W4 body is the W4 units kit's; its body tests print NOTE here
    UNIT_FILES[_BODY_KEY] = (HERE / "nbdf_k100_cell7_body.py").read_text(encoding="utf-8")
    UNIT_FILES["build/nbdf_k100_unit_tests.py"] = (HERE / "nbdf_k100_unit_tests.py").read_text(encoding="utf-8")
    UNIT_FILES["build/nbdf_k100_nan_tests.py"] = (HERE / "nbdf_k100_nan_tests.py").read_text(encoding="utf-8")   # DFlash2 conv NaN isolation
    UNIT_FILES["build/e25/k100load_lib.py"] = (E25B / "k100load_lib.py").read_text(encoding="utf-8")
    UNIT_FILES["build/e25/qsa_kv_scales_262k.json"] = Path(r"D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb0_tf_ref/qsa_kv_scales_262k.json").read_text(encoding="utf-8")
assert hashlib.sha256(UNIT_FILES[_BODY_KEY].encode("utf-8")).hexdigest() == FACTS["body_file_sha256"], "body != build facts"
for n in ("nb4t_lib.py", "nb4_ppack.py"):
    UNIT_FILES[f"build/e25/{n}"] = (E25B / n).read_text(encoding="utf-8")
UNIT_FILES["nb4t_q_record.json"] = json.dumps({"sections": {"Q": {"nb4t_q": Q_SLIM}}})
UNIT_FILES["drafter_config.json"] = Path(r"D:/ARC AGI 2/_scratch/dflash2_drafter/kernel_out/bf16-embed/config.json").read_text(encoding="utf-8")
for rel in sorted(set(SEAM_TEST_RELS) | set(DSM.PATCHES) | set(PPK.PATCHES)):
    UNIT_FILES["merlin/" + rel] = (MERLIN_SRC / rel).read_bytes().decode("utf-8")
for rel, (pre, *_r) in list(DSM.PATCHES.items()) + list(PPK.PATCHES.items()):
    assert hashlib.sha256(UNIT_FILES["merlin/" + rel].encode("utf-8")).hexdigest() == pre, rel
if K100:                                   # E27 K100 (size < 400 kB): only what the K100 + NaN tests read; the shared nbdf_lib's
    _keep = {"build/nbdf_lib.py", "build/dflash_seam.py", _BODY_KEY, "build/nbdf_k100_unit_tests.py", "build/nbdf_k100_nan_tests.py",
             "build/e25/k100load_lib.py", "build/e25/qsa_kv_scales_262k.json", "build/e25/nb4t_lib.py", "build/e25/nb4_ppack.py",
             "drafter_config.json", "merlin/vllm/model_executor/models/qwen3_dflash2.py",
             "merlin/vllm/v1/e2_guards.py"}   # E27 B1: K8 (the seam's e2_guards edit on Merlin's own file)   # 281 W4 / seam tests: the W4 units kit
    UNIT_FILES = {k: v for k, v in UNIT_FILES.items() if k in _keep}
    assert set(UNIT_FILES) == _keep, sorted(_keep - set(UNIT_FILES))
for k, v in UNIT_FILES.items():
    assert "\r" not in v, f"CR in {k}"
UNITS_BLOB = blob(UNIT_FILES)
EXPORT_BLOB = blob({"nbdf_code_export.py": EXPORT})
FIXTURE_BLOB = blob({"q_workers": Q_REC["workers"], "q_pool": Q_REC["pool"]})
KV_Q_PIN, KV_D_PIN, KV_Q_BLOCKS, KIT_ARMS = 3553580524, 3016709612, 215, ("Q", "D4s", "D3s", "D4", "D4t", "D7", "D4b")
if K100:                                   # E27 K100: R9 on the klee100 K3_ONLY v2 GPU record (7,933,526,016 B -> 480 blocks = 479 usable, F 6)
    _k3v2 = json.loads(Path(r"D:/ARC AGI 2/_scratch/e25_gpu_out_k3_v2/q27-xhigh-e25-klee100-load-k3only/k100/k100_results.json").read_text(encoding="utf-8"))
    _k3r = _k3v2["sections"]["K3"]["k100_K3_fp8_mtp3_pin739_r0"]
    assert _k3r["kv_bytes"] == 7933526016 and _k3r["capacity"]["pool_blocks"] == 479 and _k3r["capacity"]["F"] == 6
    FIXTURE_BLOB = blob({"q_workers": _k3r["workers"], "q_pool": _k3r.get("pool")})
    _cfg = FACTS["k100"]["cfg"]
    # the probe's num_blocks is the TOTAL pool (the GPU record's kv_cache_config num_blocks 480 = 479 usable + the null block)
    KV_Q_PIN, KV_D_PIN, KV_Q_BLOCKS, KIT_ARMS = _cfg["q_pin"], _cfg["q_pin"], 480, tuple(_cfg["arms"])

HEAD = r'''# E27 NBDF CPU kit __NAME__ (pinned image, no GPU): see the builder docstring for the checks and the verdict rule
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib, re, base64, lzma
from pathlib import Path
T0 = time.time()
KIT = "__NAME__"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")
K = Path(os.environ.get("KIT_TMP", "/tmp/nbdfkit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": f"E27 NBDF CPU kit {KIT}", "notebook_sha256": "__NBSHA__", "python": sys.version, "checks": {}, "variant": "__VARIANT__"}
def save():
    (W / f"nbdf_kit_{KIT}_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def unpack(b):
    return json.loads(lzma.decompress(base64.b85decode(b)).decode("utf-8"))
def sh(name, args, env=None, timeout=3600, cwd=None):
    t = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})}, cwd=str(cwd or K))
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out)
    return {"rc": rc, "sec": round(time.time() - t), "tail": out[-1500:], "_out": out}
def verdict(ok, recorded=None):
    REP["verdict"] = {"checks": ok, "ALL_OK": all(ok.values()), "recorded": recorded or {}, "minutes": round((time.time() - T0) / 60, 1)}
    save()
    bad = [k for k, v in ok.items() if not v]
    print("VERDICT", json.dumps(REP["verdict"], default=str)[:6000], flush=True)
    print(f"@@KIT {KIT} ALL OK@@" if not bad else f"@@KIT {KIT} NOT OK: {', '.join(bad)}@@", flush=True)
'''

UNITS = r'''
U1 = {}
try:
    F = unpack("__UNITSBLOB__")
    B = K / "kit"
    for rel, txt in F.items():
        (B / rel).parent.mkdir(parents=True, exist_ok=True)
        (B / rel).write_text(txt, encoding="utf-8", newline="\n")
    U1["files"] = {rel: hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12] for rel, txt in F.items() if not rel.startswith("merlin/")}
    for k_ in list(os.environ):
        if k_.startswith(("NB1_", "E23_", "RC2_", "NB4_", "Q27X_", "DFLASH_")):
            os.environ.pop(k_, None)
    r = sh("units", [sys.executable, str(B / "build/nbdf_unit_tests.py"), "--merlin-src", str(B / "merlin"),
                     "--nb4t-results", str(B / "nb4t_q_record.json"), "--drafter-config", str(B / "drafter_config.json")],
           timeout=1800, cwd=B / "build")
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
    U1["units"] = {"rc": r["rc"], "sec": r["sec"], "summary": m.group(0) if m else None, "fails": [l[:300] for l in lines if l.startswith("FAIL ")][:40],
                   "skips": [l for l in lines if l.startswith("SKIP ")], "tail": r["tail"][-800:]}
    U1["units"]["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0 and not U1["units"]["skips"]
    log("units", json.dumps(U1["units"])[:1500])
    r = sh("seam_tests", [sys.executable, str(B / "build/dflash_seam_tests.py"), "--merlin-src", str(B / "merlin"), "--nb4-dir", str(B / "build/e25")],
           timeout=1800, cwd=B / "build")
    U1["seam_tests"] = {"rc": r["rc"], "sec": r["sec"], "all_pass": r["rc"] == 0 and "ALL PASS" in r["_out"],
                        "fails": [l[:300] for l in r["_out"].splitlines() if l.startswith("FAIL")][:40], "tail": r["tail"][-1500:]}
    log("seam tests", json.dumps(U1["seam_tests"])[:1500])
    if (B / "build/nbdf_k100_unit_tests.py").is_file():          # E27 K100
        r = sh("k100_units", [sys.executable, str(B / "build/nbdf_k100_unit_tests.py")], timeout=1800, cwd=B / "build")
        lines = r["_out"].splitlines()
        m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
        U1["k100_units"] = {"rc": r["rc"], "summary": m.group(0) if m else None, "fails": [l[:300] for l in lines if l.startswith("FAIL ")][:40],
                            "skips": [l for l in lines if l.startswith("SKIP ")], "tail": r["tail"][-800:]}
        U1["k100_units"]["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0 and not U1["k100_units"]["skips"]
        log("k100 units", json.dumps(U1["k100_units"])[:1500])
except Exception as e:
    U1["fatal"] = f"{type(e).__name__}: {e}"[:800]; U1["tb"] = traceback.format_exc()[-1500:]
REP["checks"]["U1"] = U1
verdict(dict({"nbdf_units_all_pass": (U1.get("units") or {}).get("all_pass") is True,
              "seam_tests_all_pass": (U1.get("seam_tests") or {}).get("all_pass") is True},
             **({"k100_units_all_pass": (U1.get("k100_units") or {}).get("all_pass") is True} if "__VARIANT__" == "k100" else {})),
        recorded={"units": (U1.get("units") or {}).get("summary"), "k100_units": (U1.get("k100_units") or {}).get("summary"), "files": U1.get("files")})
'''

REAL = r'''
FILES = unpack("__EXPORTBLOB__")
FIX = unpack("__FIXBLOB__")
(K / "nbdf_code_export.py").write_text(FILES["nbdf_code_export.py"], encoding="utf-8", newline="\n")
REP["export_sha256"] = hashlib.sha256(FILES["nbdf_code_export.py"].encode("utf-8")).hexdigest()
ARMS5 = tuple(json.loads(r"""__KITARMS__"""))   # every arm of the plan (W4: Q D4s D3s D4 D4t D7 D4b; K100: its arm set)
SRC_SHAS = json.loads(r"""__SRCSHAS__""")
R0, R1, R2, R3, R3P, R3S, R4, R5, R6, R9, R10, R11, R12, R13 = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
G = None
try:
    os.environ.update(RC2_UNIT="1", RC2_CPU_KIT="1", E23_OUT=str(K / "out"), E23_TMP=str(K / "tmp"), E23_INPUT=INPUT_ROOT,
                      RC2_SHM=str(K / "shm"), Q27X_WORKING=str(K / "work0"))
    for k_ in [x for x in os.environ if x.startswith(("DFLASH_", "NB4_"))] + ["E23_FAKE_SERVER", "KAGGLE_IS_COMPETITION_RERUN", "DRAFT_HEAD",
                                                                             "VLLM_PP_LAYER_PARTITION"]:
        os.environ.pop(k_, None)
    for d in ("out", "tmp", "shm", "work0"):
        (K / d).mkdir(exist_ok=True)
    t = time.time()
    G = runpy.run_path(str(K / "nbdf_code_export.py"), run_name="nbdfkit")["nbdf_main"].__globals__
    R = G["R"]
    sub = json.loads((K / "work0" / "submission.json").read_text())
    tc = G["test_challenges"]
    R0.update(tasks=len(tc), sub_tasks=len(sub),
              identity=all(len(sub[t_]) == len(v["test"]) and all(a["attempt_1"] == q["input"] == a["attempt_2"]
                                                                 for a, q in zip(sub[t_], v["test"])) for t_, v in tc.items()) and set(sub) == set(tc))
    R0["ok"] = R0["identity"] and R0["tasks"] > 0
    R1.update(load_sec=round(time.time() - t), inputs=R["inputs"], merge=R["sections"]["A0"]["merge"], arms=list(G["ARMS"]),
              rc2_rows_loaded=sorted(k for k in G["ROWS"] if k.startswith("rc2_")), prompts={b: len(v) for b, v in G["PROMPTS"].items()},
              embed=G["NBDF"].get("seam"), drafter=G["DRAFTER"], seam_envs=G["SEAM_ENVS"], seam_notes=G["SEAM_NOTES"],
              kv_kit_gate=G["KV_KIT_GATE"])
    R1["drafter_ok"] = G["DRAFTER"].get("ok") is True and G["DRAFTER"].get("pins_ok") is True
    log("R0", json.dumps(R0)); log("R1 drafter", json.dumps({k: G["DRAFTER"].get(k) for k in ("ok", "status", "why", "dir", "sec")})[:900])
    save()
    ok, why = G["sha_gate"](4 * 3600)
    R1["sha"] = {"ok": ok, "why": why}
except Exception as e:
    R1["fatal"] = f"{type(e).__name__}: {e}"[:1000]; R1["tb"] = traceback.format_exc()[-2500:]
    log("R1 FATAL", R1["fatal"])
REP["checks"]["R0"] = R0; REP["checks"]["R1"] = R1; save()

eng = None
try:
    eng = G["merlin_install"]()
    tree0 = Path(eng["kv"]["MERLIN_TREE"])
    R2.update(ok=True, kv=dict(eng["kv"]), src_shas={rel: hashlib.sha256((tree0 / rel).read_bytes()).hexdigest() if (tree0 / rel).is_file() else None
                                                     for rel in SRC_SHAS})
    R2["src_equal"] = R2["src_shas"] == SRC_SHAS
except Exception as e:
    R2.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R2 install", json.dumps({k: v for k, v in R2.items() if k != "tb"})[:900])
REP["checks"]["R2"] = R2; save()


STUB = {}
try:
    import ctypes, glob
    try:
        ctypes.CDLL("libcuda.so.1")
        STUB["libcuda_so_1"] = "loadable"
    except OSError as e:
        STUB["libcuda_so_1"] = f"not loadable: {e}"[:200]
    cands = []
    for root in [eng["cuda_home"] if eng else "", "/usr/local/cuda", "/usr/local/cuda/targets/x86_64-linux"]:
        if root:
            cands += glob.glob(f"{root}/lib*/stubs/libcuda.so") + glob.glob(f"{root}/**/stubs/libcuda.so", recursive=True)
    if cands:
        d_ = K / "cudastub"
        d_.mkdir(exist_ok=True)
        if not (d_ / "libcuda.so.1").exists():
            os.symlink(cands[0], d_ / "libcuda.so.1")
        STUB.update(stub=cands[0], dir=str(d_))
except Exception as e:
    STUB["error"] = repr(e)[:300]
REP["checks"]["libcuda_stub"] = STUB

# E25 klee100 kit (k100kit_real.py:74-148, run on Kaggle CPU): vllm.platforms._current_platform = a NonNvmlCudaPlatform
# subclass reporting sm_89 "NVIDIA L4" BEFORE any vllm config / model import (else "Failed to infer device type"), and a
# single-process gloo group (TP1 / PP1) inside set_current_vllm_config(VllmConfig()) for layers built on meta.
PLATFORM = r"""
import json as _j
PLAT = {}
try:
    import torch
    import vllm.platforms as VP
    from vllm.platforms.interface import DeviceCapability
    from vllm.platforms import cuda as VPC

    class FakeL4(VPC.NonNvmlCudaPlatform):
        @classmethod
        def get_device_capability(cls, device_id=0):
            return DeviceCapability(8, 9)

        @classmethod
        def has_device_capability(cls, capability, device_id=0):
            want = tuple(capability) if isinstance(capability, tuple) else (capability // 10, capability % 10)
            return (8, 9) >= want

        @classmethod
        def get_device_name(cls, device_id=0):
            return "NVIDIA L4"

        @classmethod
        def get_device_total_memory(cls, device_id=0):
            return 23034 * 2 ** 20

        @classmethod
        def is_fully_connected(cls, physical_device_ids):
            return False

    VP._current_platform = FakeL4()
    from vllm.platforms import current_platform as CP
    PLAT.update(ok=bool(CP.is_cuda() and not CP.is_cpu()), cls=type(CP).__name__, capability=list(CP.get_device_capability()))
except BaseException as e:
    PLAT.update(ok=False, error=f"{type(e).__name__}: {e}"[:400])


def init_gloo():
    import socket
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.config import VllmConfig as _VC, set_current_vllm_config as _scvc
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with _scvc(_VC()):
        init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=0, backend="gloo")
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, backend="gloo")
"""


def venv_py(code, args=(), name="probe", timeout=1200, env_extra=None, base_env=None, platform=False):
    tree, py = eng["kv"]["MERLIN_TREE"], eng["kv"]["MERLIN_VENV"] + "/bin/python"
    if platform:
        code = PLATFORM + code
        env_extra = dict(env_extra or {}, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
        if STUB.get("dir"):
            env_extra["LD_LIBRARY_PATH"] = STUB["dir"] + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    env = dict(base_env) if base_env is not None else {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_PRELOAD")}
    if base_env is None:
        env.update(PYTHONPATH=tree, CUDA_HOME=eng["cuda_home"], VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PATH=f"{eng['kv']['MERLIN_VENV']}/bin:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}")
        if eng["kv"].get("MERLIN_SHIM"):
            env["LD_PRELOAD"] = eng["kv"]["MERLIN_SHIM"]
    env.update(env_extra or {})
    (K / f"{name}.py").write_text(code)               # never named after a stdlib module (no argparse.py etc.)
    try:
        r = subprocess.run([py, str(K / f"{name}.py"), *args], capture_output=True, text=True, timeout=timeout, env=env, cwd=str(K))
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out)
    js = next((json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("@JSON ")), None)
    return {"rc": rc, "json": js, "tail": out[-1500:]}


SPAWN_ENV = r"""
import json, multiprocessing as mp, os, sys
KEYS = sys.argv[1].split(",")
def child(q):
    q.put({k: os.environ.get(k) for k in KEYS})
if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(q,))
    p.start(); r = q.get(timeout=120); p.join(60)
    print("@JSON " + json.dumps({"parent": {k: os.environ.get(k) for k in KEYS}, "child": r}))
"""
SPECPROBE = r"""
import json, sys, typing
out = {}
specs = json.loads(sys.argv[1])
try:
    from vllm.config import speculative as S
    hints = typing.get_type_hints(S.SpeculativeConfig)
    out["fields_missing"] = {k: [f for f in sp if f not in hints] for k, sp in specs.items()}
    lit = lambda t: set(x for a in typing.get_args(t) for x in (typing.get_args(a) or (a,)))
    out["method_ok"] = {k: sp["method"] in lit(S.SpeculativeMethod) for k, sp in specs.items()}
    out["draft_sample_ok"] = {k: sp.get("draft_sample_method") in typing.get_args(S.DraftSampleMethod) for k, sp in specs.items()}
    from vllm.config.cache import CacheDType
    out["kv_dtype_ok"] = {k: sp.get("kv_cache_dtype") in typing.get_args(CacheDType) for k, sp in specs.items()}
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as E
    out["backend_ok"] = {k: S.SpeculativeConfig._parse_attention_backend(sp["attention_backend"]) == E[sp["attention_backend"]]
                         for k, sp in specs.items() if "attention_backend" in sp}
    out["draft_tp_default"] = str(hints.get("draft_tensor_parallel_size"))
    import inspect
    src = inspect.getsource(S.SpeculativeConfig._verify_and_get_draft_tp)
    out["draft_tp_unset_is_target_tp"] = "target_parallel_config.tensor_parallel_size" in src
    out["ok"] = (not any(out["fields_missing"].values()) and all(out["method_ok"].values()) and all(out["draft_sample_ok"].values())
                 and all(out["kv_dtype_ok"].values()) and all(out["backend_ok"].values()) and out["draft_tp_unset_is_target_tp"])
except BaseException as e:
    out.update(ok=False, error=f"{type(e).__name__}: {e}"[:600])
try:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    p = EngineArgs.add_cli_args(FlexibleArgumentParser())
    out["argparse"] = {k: str(vars(p.parse_args(["--model", "/tmp/x", "--speculative-config", json.dumps(sp)])).get("speculative_config"))[:300]
                       for k, sp in specs.items()}
except BaseException as e:
    out["argparse_error"] = f"{type(e).__name__}: {e}"[:400]
out["platform"] = PLAT
print("@JSON " + json.dumps(out, default=str))
"""
GRDPROBE = r"""
# E27 B1: the REAL e2_guards.assert_resolved_cudagraph_modes of the seam-patched installed tree, with a fake DFlash2Speculator
import json, traceback
out = {}
try:
    from vllm.v1 import e2_guards as G
    from vllm.config.compilation import CUDAGraphMode
    arch = sorted(G.QWEN4EXP_TARGET_ARCHITECTURES)[0]
    cfg = type("C", (), {"model_config": type("M", (), {"architectures": [arch]})()})()

    def mgr(m):
        return type("Mgr", (), {"cudagraph_mode": m})()

    class DFlash2Speculator:
        pass

    def run(mode):
        sp = DFlash2Speculator()
        sp.query_cudagraph_manager = mgr(mode)
        try:
            G.assert_resolved_cudagraph_modes(cfg, mgr(CUDAGraphMode.FULL_AND_PIECEWISE), sp)
            return None
        except RuntimeError as e:
            return str(e)[:400]
    out["file"] = G.__file__
    out["marker"] = "E27 DFLASH_SEAM" in open(G.__file__, encoding="utf-8").read()
    out["full_decode_only"] = run(CUDAGraphMode.FULL_DECODE_ONLY)
    out["none"] = run(CUDAGraphMode.NONE)
    out["ok"] = out["marker"] and out["full_decode_only"] is None and bool(out["none"]) and "query_cudagraph_manager" in out["none"]
except Exception as e:
    out.update(ok=False, error=repr(e)[:400], tb=traceback.format_exc()[-1500:])
print("@JSON " + json.dumps(out))
"""
IMPORTPROBE = r"""
import json
out = {}
try:
    import vllm.models.qwen4_exp.nvidia.model as M
    out["import"] = M.__file__
    cls = M.Qwen4ExpModel
    out["mro"] = [c.__name__ for c in cls.__mro__][:8]
    out["eagle_mixin"] = any(c.__name__ == "EagleModelMixin" for c in cls.__mro__)
    out["aux_over_pp"] = getattr(cls, "supports_aux_hidden_states_over_pp", None)
    out["compiled_wrapper"] = any("compile" in c.__name__.lower() for c in cls.__mro__) or hasattr(cls, "__vllm_compile__") or "support_torch_compile" in repr(getattr(cls, "__init__", ""))
    out["served_eagle3"] = any(c.__name__ == "SupportsEagle3" for c in M.Qwen4ExpForConditionalGeneration.__mro__)
    out["ok"] = out["eagle_mixin"] and out["aux_over_pp"] is True and out["served_eagle3"]
except BaseException as e:
    out.update(ok=None, error=f"{type(e).__name__}: {e}"[:600])
print("@JSON " + json.dumps(out, default=str))
"""
WEIGHTPROBE = r"""
import json, sys, io, os, traceback, contextlib, dataclasses
out = {"platform": PLAT}
d, ckpt = sys.argv[1], sys.argv[2]
os.environ["DFLASH_SEAM"] = "1"                 # the seam's allowlist + loaded-weights check (qwen3_dflash.py _e27_check_loaded) ON
try:
    from safetensors import safe_open
    with safe_open(d + "/model.safetensors", framework="pt", device="cpu") as f:
        names = sorted(f.keys())
        shapes = {n: (list(f.get_slice(n).get_shape()), f.get_slice(n).get_dtype()) for n in names}
    out["checkpoint_tensors"] = len(names)
    out["has_embed"] = any(n.endswith("embed_tokens.weight") for n in names)
    import torch
    from vllm.config import VllmConfig, ModelConfig, SpeculativeConfig, set_current_vllm_config
    init_gloo()
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
    from vllm.v1.worker.gpu.spec_decode.dflash import utils as DU
    out["class"] = DFlash2Qwen3ForCausalLM.__name__
    # the engine's path: the TARGET model config + SpeculativeConfig(method dflash, the drafter dir), then the draft vllm_config
    # exactly as load_dflash_model builds it (dflash/utils.py: attention_config non-causal + backend, cache dtype, load config)
    mc = ModelConfig(model=ckpt, tokenizer=ckpt, dtype="bfloat16", max_model_len=131072)
    vc0 = VllmConfig(model_config=mc)
    sc = SpeculativeConfig(method="dflash", model=d, num_speculative_tokens=4, draft_sample_method="probabilistic",
                           kv_cache_dtype="auto", attention_backend="FLASH_ATTN", target_model_config=mc,
                           target_parallel_config=vc0.parallel_config)
    vc = VllmConfig(model_config=mc, speculative_config=sc)
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    dvc = dataclasses.replace(vc, attention_config=dataclasses.replace(vc.attention_config, use_non_causal=dflash_has_any_non_causal(
        sc.draft_model_config.hf_config), backend=sc.attention_backend),
        cache_config=dataclasses.replace(vc.cache_config, cache_dtype=sc.kv_cache_dtype) if sc.kv_cache_dtype is not None else vc.cache_config)
    out["draft_arch"] = getattr(sc.draft_model_config.hf_config, "architectures", None)
    DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    buf = io.StringIO()
    from vllm.utils.torch_utils import set_default_torch_dtype
    out["draft_dtype"] = str(sc.draft_model_config.dtype)
    with set_current_vllm_config(dvc), set_default_torch_dtype(sc.draft_model_config.dtype), torch.device("meta"):
        m = DFlash2Qwen3ForCausalLM(vllm_config=dvc, prefix="")
        params = sorted(n for n, _ in m.named_parameters())
        out["named_parameters"] = len(params)
        out["params_sample"] = params[:6]
        with contextlib.redirect_stdout(buf):
            try:
                ret = m.load_weights((n, torch.empty(shapes[n][0], dtype=DT.get(shapes[n][1], torch.bfloat16), device="meta")) for n in names)
                out["load_weights"] = "returned None" if ret is None else f"returned {type(ret).__name__}"
            except BaseException as e:
                out["load_weights_error"] = f"{type(e).__name__}: {e}"[:900]
    out["seam_events"] = [l for l in buf.getvalue().splitlines() if "E27_DFLASH" in l][:6]
    chk = [l for l in out["seam_events"] if "draft_load_check" in l]
    out["draft_load_check_line"] = chk[:1]
    out["false_raise_suspect"] = "got no tensor" in str(out.get("load_weights_error", ""))
    out["ok"] = bool(chk) and "load_weights_error" not in out and '"skipped": true' not in (chk[0] if chk else "")
    if not out["ok"] and not out["false_raise_suspect"] and "load_weights_error" in out:
        out["ok"] = False
except BaseException as e:
    out.update(ok=None, error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1800:])
print("@JSON " + json.dumps(out, default=str))
"""
KVPROBE = r"""
import json, sys, dataclasses, re, copy, traceback
from types import SimpleNamespace as NS
out = {"runs": {}, "platform": PLAT}
fx = json.loads(open(sys.argv[1]).read())
Q_PIN, D_PIN = int(sys.argv[2]), int(sys.argv[3])
ckpt = sys.argv[4] if len(sys.argv) > 4 else None
try:
    import torch
    from vllm.v1 import kv_cache_interface as KI
    from vllm.v1.core import kv_cache_utils as KU
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends import utils as AU
    tdt = lambda v: getattr(torch, str(v).replace("torch.", "")) if v else None

    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as MBE
    PAGE_MISMATCH = []

    def mk(s, name=""):
        sp = _mk(s, name)
        f = {x.name for x in dataclasses.fields(type(sp))}
        want = s.get("page_size_bytes")
        have = getattr(sp, "page_size_bytes", None)
        if "tokens_per_state" in f and isinstance(want, int) and isinstance(have, int) and have != want and want and have % want == 0:
            sp = dataclasses.replace(sp, tokens_per_state=have // want)          # e.g. the QSA compressed indexer cache (ratio 4)
            have = sp.page_size_bytes
        if isinstance(want, int) and have != want and len(PAGE_MISMATCH) < 12:
            PAGE_MISMATCH.append([s["type"], name[:60], have, want])
        return sp

    def _mk(s, name=""):
        cls = getattr(KI, s["type"])
        f = {x.name for x in dataclasses.fields(cls)}
        kw = {}
        if "block_size" in f: kw["block_size"] = s["block_size"]
        if "num_kv_heads" in f: kw["num_kv_heads"] = s.get("num_kv_heads")
        if "head_size" in f: kw["head_size"] = s.get("head_size")
        if "dtype" in f and s.get("dtype"): kw["dtype"] = tdt(s["dtype"])
        if "shapes" in f: kw["shapes"] = tuple(tuple(x) for x in s.get("state_shapes") or ())
        if "dtypes" in f: kw["dtypes"] = tuple(tdt(x) for x in s.get("state_dtypes") or ())
        if "page_size_padded" in f: kw["page_size_padded"] = s.get("page_size_padded")
        if "mamba_type" in f and s.get("mamba_type"): kw["mamba_type"] = MBE[str(s["mamba_type"]).split(".")[-1]]
        if "mamba_cache_mode" in f and s["type"] == "MambaSpec": kw["mamba_cache_mode"] = "align"     # --mamba-cache-mode align
        if "tp_replicated" in f and ".ple" in name: kw["tp_replicated"] = True                    # "folded replicated Mamba group (ple)"
        if "sliding_window" in f and s.get("sliding_window"): kw["sliding_window"] = s["sliding_window"]
        return cls(**{k: v for k, v in kw.items() if v is not None})

    def k_shapes(s, k):
        # the K-dependent state rows (nvidia/model.py:980-1062 + mamba_utils.py:331-375): GDN conv (conv_kernel - 1 + K),
        # the RecoverSSM replay record (K + 1), the PLE short conv (conv_state_len + K); the recorded specs are at K = 3
        s = copy.deepcopy(s)
        sh = [list(x) for x in s.get("state_shapes") or []]
        ex = str(s.get("example") or "")
        if s["type"] == "MambaSpec" and sh:
            sh[0][0] += k - 3
            if "linear_attn" in ex and len(sh) >= 3:
                sh[2][1] += k - 3
        s["state_shapes"] = sh
        return s

    LT = None
    try:
        _c = json.load(open(f"{ckpt}/config.json"))
        LT = (_c.get("text_config") or _c).get("layer_types")
    except BaseException:
        LT = None
    out["layer_types_from_ckpt"] = bool(LT)

    def layer_names(stage, s, n):
        # the real per-layer names on this PP stage (25 / 23): GDN on linear_attention layers, the QSA caches on
        # full_attention layers (+ the MTP layer 48 on the last stage), the PLE at layers.1; else stage-unique names
        lo, hi = (0, 25) if stage == 0 else (25, 48)
        ex = str(s.get("example") or "layer")
        suf = ex.split(".", 4)[-1] if ex.startswith("language_model.model.layers.") else None
        if LT and suf is not None:
            if ".ple" in ex:
                idx = [1]
            elif "linear_attn" in ex:
                idx = [i for i in range(lo, hi) if LT[i] == "linear_attention"]
            else:
                idx = [i for i in range(lo, hi) if LT[i] == "full_attention"]
            names = [f"language_model.model.layers.{i}.{suf}" for i in idx]
            if stage == 1 and len(names) + 1 == n and "linear_attn" not in ex:
                names.append(f"mtp.layers.48.{suf}")
            if len(names) == n:
                return names
        return [re.sub(r"layers\.\d+", f"layers.{1000 + 100 * stage + i}", ex) for i in range(n)]

    def worker_specs(ranks, k=3, drafter=None, drop_mtp=False):
        ws = []
        for w in ranks:
            d = {}
            stage = 0 if w["rank"] < 2 else 1
            for s in w["kv_cache_specs"]:
                s = k_shapes(s, k)
                for nm in layer_names(stage, s, int(s.get("layers") or 1)):
                    if drop_mtp and nm.startswith("mtp."):
                        continue                          # the MTP layer (layer 48) is not built on a DFlash arm
                    d[nm] = mk(s, nm)
            if drafter and w["rank"] >= 2:
                for i in range(5):                        # qwen3_dflash.py:489: prefix model.layers.{i + target layers}
                    d[f"model.layers.{48 + i}.self_attn.attn"] = mk(drafter)
            ws.append(d)
        return ws

    MC = {}
    try:
        from vllm.config import ModelConfig
        MC["mc"] = ModelConfig(model=ckpt, tokenizer=ckpt, dtype="bfloat16", max_model_len=131072) if ckpt else None
        MC["how"] = "real ModelConfig of the merged W4 checkpoint"
    except BaseException as e:
        MC["error"] = f"{type(e).__name__}: {e}"[:400]
    out["model_config"] = {k: v for k, v in MC.items() if k != "mc"}
    BK = {}
    try:
        from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend
        from vllm.models.qwen4_exp.common.qsa_cache import QSAStateBackend
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
        from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
        BK = {"target": [Qwen4ExpQSAFlashAttentionBackend, QSAStateBackend], "FLASH_ATTN": FlashAttentionBackend,
              "TRITON_ATTN": TritonAttentionBackend}
    except BaseException as e:
        out["backend_import_error"] = f"{type(e).__name__}: {e}"[:400]

    def summary(c, drafter_names):
        groups, builders, wasted = [], 0, 0
        for g in c.kv_cache_groups:
            sp = g.kv_cache_spec
            inner = getattr(sp, "kv_cache_specs", None)
            types_ = sorted({type(x).__name__ for x in (inner.values() if isinstance(inner, dict) else [sp])})
            builders += len(types_)
            for x in (inner.values() if isinstance(inner, dict) else [sp]):
                pad, real = getattr(x, "page_size_padded", None), getattr(x, "real_page_size_bytes", None)
                if isinstance(pad, int) and isinstance(real, int) and pad > real:
                    wasted += pad - real
            groups.append({"spec": type(sp).__name__, "inner": types_, "layers": len(g.layer_names),
                           "block_size": getattr(sp, "block_size", None), "page_size_bytes": getattr(sp, "page_size_bytes", None),
                           "is_eagle_group": getattr(g, "is_eagle_group", None),
                           "drafter_layers": sum(1 for n in g.layer_names if n in drafter_names)})
        return {"kv_group_count": len(groups), "groups": groups, "attn_metadata_builders": builders,
                "drafter_group_count": sum(1 for g in groups if g["drafter_layers"]), "wasted_bytes_per_block": wasted or None}

    import logging
    class _Cap(logging.Handler):
        msgs = []
        def emit(self, r):
            try:
                _Cap.msgs.append(r.getMessage())
            except Exception:
                pass
    logging.getLogger("vllm").addHandler(_Cap())

    def run(tag, ws, pin, draft_backend=None):
        _Cap.msgs.clear()
        vc = VllmConfig()
        if MC.get("mc") is not None:
            vc.model_config = MC["mc"]
        else:
            vc.model_config = NS(max_model_len=131072, original_max_model_len=131072, hf_config=NS(), use_mla=False, is_hybrid=True,
                                 get_num_layers=lambda *a: 48)
        vc.cache_config.block_size = 4096
        vc.cache_config.enable_prefix_caching = True
        vc.cache_config.cache_dtype = "fp8_e4m3"
        rec = {"pin": pin}
        try:
            # the engine core's own call (v1/engine/core.py:388-392): resolve_kv_cache_layout(vllm_config, the workers'
            # supported layouts (worker_base.py:125-128 = get_supported_kv_cache_layouts(the backends)), every spec)
            bks = list(BK.get("target") or []) + ([BK[draft_backend]] if draft_backend and BK.get(draft_backend) else [])
            sup = [l.name for l in AU.get_supported_kv_cache_layouts(bks)] if bks else None
            rec["supported_layouts"], rec["backends"] = sup, [b.__name__ for b in bks]
            lay = AU.resolve_kv_cache_layout(vc, [sup] * len(ws), [s for w in ws for s in w.values()])
            rec["layout"] = lay.name
            cfgs = KU.get_kv_cache_configs(vc, ws, [pin] * len(ws))
            dn = {n for w in ws for n in w if n.startswith("model.layers.4") or n.startswith("model.layers.5")}
            rec.update(num_blocks=[x.num_blocks for x in cfgs], **summary(cfgs[2], dn))
            try:
                rec["max_concurrency_131072"] = KU.get_max_concurrency_for_kv_cache_config(vc, cfgs[2])
            except BaseException as e:
                rec["max_concurrency_error"] = f"{type(e).__name__}: {e}"[:200]
        except BaseException as e:
            rec.update(error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
        rec["draft_group_warning"] = any("no KV cache group could be identified as the draft model" in m for m in _Cap.msgs)
        rec["fine_grained_disabled"] = any("Disabling fine-grained prefix-cache hits" in m for m in _Cap.msgs)
        out["runs"][tag] = rec

    ranks = fx["q_workers"]["ranks"]
    run("Q", worker_specs(ranks, 3), Q_PIN)
    out["page_mismatch"] = PAGE_MISMATCH
    out["q_target"] = {"num_blocks": __QBLOCKS__, "groups_rank0": [[g.get("layers"), g.get("block_size"), g.get("page_size_bytes")]
                                                          for g in (ranks[0].get("kv_cache_config") or {}).get("groups") or []],
                       "groups_rank2": [[g.get("layers"), g.get("block_size"), g.get("page_size_bytes")]
                                        for g in (ranks[2].get("kv_cache_config") or {}).get("groups") or []]}
    q = out["runs"]["Q"]
    out["q_calibrated"] = bool(q.get("num_blocks")) and __QBLOCKS__ - 2 <= min(q["num_blocks"]) <= __QBLOCKS__ + 2
    bf16 = {"type": "SlidingWindowSpec", "block_size": 4096, "num_kv_heads": 4, "head_size": 128, "dtype": "torch.bfloat16", "sliding_window": 2048}
    fp8 = dict(bf16, dtype="torch.uint8")
    if out["q_calibrated"]:                   # no D number counts before Q reproduces NB4T's 215 blocks
        for tag, k, dr, bk in (("K3", 3, bf16, "FLASH_ATTN"), ("K4", 4, bf16, "FLASH_ATTN"), ("K7", 7, bf16, "FLASH_ATTN"),
                               ("K4_fp8", 4, fp8, "TRITON_ATTN"), ("K4_swblock1024", 4, dict(bf16, block_size=1024), "FLASH_ATTN"),
                               ("K3_swblock1024", 3, dict(bf16, block_size=1024), "FLASH_ATTN")):
            run(tag, worker_specs(ranks, k, dr, drop_mtp=True), D_PIN, bk)
    out["note"] = ("target specs = the NB4T GPU record (K = 3) with the K-dependent state rows scaled to K (GDN conv, the RecoverSSM "
                   "replay record, PLE short conv), the MTP layer removed on PP1, the drafter's 5 SWA layers added (4 KV heads x 128 "
                   "per TP rank); layout via the engine core's resolve_kv_cache_layout with the target's attention backends")
    out["ok"] = out["q_calibrated"] and all(bool((out["runs"].get(t) or {}).get("num_blocks")) and "error" not in out["runs"].get(t, {})
                                            for t in ("K4_swblock1024", "K3_swblock1024"))
except BaseException as e:
    out.update(ok=False, error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
print("@JSON " + json.dumps(out, default=str))
"""

orig, qsa_file, tree = None, None, None
try:
    tree = Path(eng["kv"]["MERLIN_TREE"])
    qsa_file = tree / G["P60490_REL"]
    orig = qsa_file.read_bytes()
    rec = G["apply_60490"]()
    R3["ok"] = rec.get("applied") == "applied" and rec.get("sha_after") == G["P60490_POST"] and (rec.get("py_compile") or {}).get("rc") == 0
except Exception as e:
    R3.update(ok=False, error=f"{type(e).__name__}: {e}"[:800])
REP["checks"]["R3"] = R3; save()

plans = {}


def plan_one(key):
    a = G["ARMS"][key]
    nm = a["name"]
    G["NB1_EXTRA"][nm], G["NB1_ENV"][nm], G["NB1_KV"][nm] = list(a["extra"]), dict(a["env"]), a["kv_bytes"]
    G["ARM_OF"][nm] = a
    b = {"arm": key, "tree_check": G["tree_check"](a)}
    try:
        s = G["merlin_fn_server"](nm, "fp8", a["port"], a["kv_bytes"])
        b.update(ok=True, host_phase=G["HOST_PHASE"].get(nm), argv=s.argv, launch_check=G["N"].launch_check(a, s.env, s.argv, a["kv_bytes"]),
                 server_started=s.proc is not None, spec=G["T"].spec_from_argv(s.argv),
                 env={k: s.env.get(k) for k in G["N"].WATCHED + ("KV_BYTES", "VLLM_SERVER_DEV_MODE", "PYTHONPATH")})
        sp = venv_py(SPAWN_ENV, [",".join(G["N"].WATCHED)], name=f"r4_spawn_{key.lower()}", base_env=dict(s.env))
        b["spawn_env"] = sp["json"] or {"error": sp["tail"]}
        b["spawn_ok"] = isinstance(sp["json"], dict) and sp["json"]["child"] == a["expect"] and sp["json"]["parent"] == a["expect"]
    except Exception as e:
        b.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], host_phase=G["HOST_PHASE"].get(nm))
    plans[key] = b
    log("R4", key, json.dumps({k: v for k, v in b.items() if k not in ("argv",)}, default=str)[:900]); save()


try:
    R3P["apply"] = G["_ppack_apply"]()
    R3P["after_apply"] = G["P"].verify_ppack_patch(tree)["state"]
    R3S["pre_state"] = G["DS"].tree_state(tree)
    R2["merlin_commit"] = G["DS"].MERLIN_COMMIT if R3S["pre_state"] == "ORIGINAL" else None
    R2["merlin_commit_evidence"] = "every dflash_seam file at its Merlin ad42e37 pre-sha on the installed tree (tree_state ORIGINAL before apply)"
    REP["checks"]["R2"] = R2
    ok_s, why_s = G["SSEQ"].apply()
    R3S.update(apply_ok=ok_s, why=why_s, ring=G["SSEQ"].ring, ring_status=list(G["N"].ring_status(G["SSEQ"])),
               combined=G["DS"].combined_tree_state(tree, G["P"], ring=G["SSEQ"].ring), check=list(G["SSEQ"].check("kit")))
    for k in ARMS5:
        plan_one(k)
except Exception as e:
    R4["fatal"] = f"{type(e).__name__}: {e}"[:800]; R4["tb"] = traceback.format_exc()[-1500:]
R4["plans"] = plans
R4["host_phase_ok"] = len(plans) == len(ARMS5) and all(p.get("host_phase") == "ok" for p in plans.values())
R4["launch_check_ok"] = len(plans) == len(ARMS5) and all(p.get("ok") and p.get("launch_check") == [] for p in plans.values())
R4["spawn_env_ok"] = len(plans) == len(ARMS5) and all(p.get("spawn_ok") is True for p in plans.values())
R4["tree_ok"] = len(plans) == len(ARMS5) and all((p.get("tree_check") or {}).get("ok") for p in plans.values())
REP["checks"]["R3P"] = R3P; REP["checks"]["R3S"] = R3S; REP["checks"]["R4"] = R4; save()

try:
    fields = {}
    for k in ARMS5:
        x, a, bad = plans.get(k, {}).get("argv") or [], G["ARMS"][k], []
        lv = G["T"].last_value
        if G["T"].spec_from_argv(x) != a["spec"]: bad.append(f"spec {G['T'].spec_from_argv(x)}")
        if G["N"].captures_from_argv(x) != a["captures"]: bad.append(f"captures {G['N'].captures_from_argv(x)}")
        if lv(x, "--max-num-seqs") != "8": bad.append("seqs")
        if lv(x, "--kv-cache-memory-bytes") != str(a["kv_bytes"]): bad.append("kv bytes")
        if lv(x, "--kv-cache-dtype") != "fp8_e4m3": bad.append("kv dtype")
        if "--language-model-only" not in x: bad.append("lmo")
        if any(f in x for f in ("--no-async-scheduling", "--no-enable-prefix-caching")): bad.append("never-do")
        fields[k] = bad
    R5["field_check"] = fields
    R5["ok"] = len(fields) == len(ARMS5) and all(not v for v in fields.values())
except Exception as e:
    R5.update(ok=False, error=f"{type(e).__name__}: {e}"[:600])
REP["checks"]["R5"] = R5; save()

R10 = venv_py(SPECPROBE, [json.dumps({k: G["ARMS"][k]["spec"] for k in ARMS5 if k != "Q"})], name="r10_spec_preflight", platform=True)
R11 = venv_py(IMPORTPROBE, name="r11_patched_model_import", platform=True)
R13 = venv_py(GRDPROBE, name="r13_e2_guard_dflash", platform=True)   # E27 B1: on the seam-patched tree, before the revert
R12 = (venv_py(WEIGHTPROBE, [str(G["DRAFTER"].get("dir")), str(G["CKPT"])], name="r12_drafter_weights", timeout=1800, platform=True)
       if G and G["DRAFTER"].get("dir") else {"json": {"ok": None, "error": "no drafter dir"}})
(K / "q_fixture.json").write_text(json.dumps(FIX))
R9 = venv_py(KVPROBE, [str(K / "q_fixture.json"), "__QPIN__", "__DPIN__", str(G["CKPT"])], name="r9_kv_cache_configs", timeout=1800, platform=True)
R9["summary"] = R9.get("json")
for nm, r in (("R9", R9), ("R10", R10), ("R11", R11), ("R12", R12), ("R13", R13)):
    REP["checks"][nm] = r
    log(nm, json.dumps(r.get("json") or r.get("tail"), default=str)[:1500])
save()

try:
    R3S["finish"] = G["SSEQ"].finish()
    R3S["after"] = G["DS"].combined_tree_state(tree, G["P"])
    R3P["revert"] = G["_ppack_revert"]()
    R3P["after_revert"] = G["P"].verify_ppack_patch(tree)["state"]
    R3S["pristine"] = G["DS"].tree_state(tree) == "ORIGINAL" and R3P["after_revert"] == "not_applied"
    if orig is not None:
        qsa_file.write_bytes(orig)
        ext = tree / "nb1_worker_ext.py"
        if ext.exists():
            ext.unlink()
except Exception as e:
    R3S["pristine"] = False; R3S["restore_error"] = repr(e)[:300]
REP["checks"]["R3S"] = R3S; REP["checks"]["R3P"] = R3P; save()
verdict({"cell0_identity_submission": R0.get("ok") is True, "inputs_found": bool((R1.get("inputs") or {}).get("merlin_inputs_ok")),
         "merge_ok": bool((R1.get("merge") or {}).get("ok")), "sha256_all_ok": bool((R1.get("sha") or {}).get("ok")),
         "drafter_pins_ok": R1.get("drafter_ok") is True, "arms_q_d4_d3_d7_d4b": R1.get("arms") == list(ARMS5),
         "merlin_install_ok": R2.get("ok") is True, "merlin_flag_sources_equal": R2.get("src_equal") is True,
         "patch_60490_ok": R3.get("ok") is True, "seam_apply_verify_revert_ok": R3S.get("apply_ok") is True and R3S.get("ring") is True
         and (R3S.get("combined") or {}).get("valid") is True and R3S.get("pristine") is True,
         "host_phase_all_5_ok": R4.get("host_phase_ok") is True, "launch_check_all_5_ok": R4.get("launch_check_ok") is True,
         "env_in_spawned_child_all_5": R4.get("spawn_env_ok") is True, "tree_check_all_5": R4.get("tree_ok") is True,
         "argv_fields_ok": R5.get("ok") is True, "spec_config_preflight_ok": (R10.get("json") or {}).get("ok") is True,
         "kv_cache_configs_ok": (R9.get("json") or {}).get("ok") is True,
         "e2_guard_dflash_ok": (R13.get("json") or {}).get("ok") is True,
         "drafter_loaded_weights_ok": (R12.get("json") or {}).get("ok") is True},
        recorded={"patched_model_import": R11.get("json"), "spec_argparse": (R10.get("json") or {}).get("argparse"),
                  "kv_runs": (R9.get("json") or {}).get("runs"), "drafter": {k: (R1.get("drafter") or {}).get(k) for k in ("status", "why", "dir")}})
'''


def _k100_sub(text, old, new, where):
    """E27 K100: one exact, unique anchor in a kit template (the W4 templates are never touched)."""
    n = text.count(old)
    assert n == 1, f"K100 anchor {where}: {n} matches"
    return text.replace(old, new)


if K100:                                   # E27 K100: the record-and-check items 1-9 + R9_IDX (real kit), the NaN test (units)
    UNITS = _k100_sub(UNITS, '''        log("k100 units", json.dumps(U1["k100_units"])[:1500])
''', '''        log("k100 units", json.dumps(U1["k100_units"])[:1500])
    if (B / "build/nbdf_k100_nan_tests.py").is_file():          # E27 K100: DFlash2 grouped-conv NaN isolation
        r = sh("k100_nan", [sys.executable, str(B / "build/nbdf_k100_nan_tests.py"), "--merlin-src", str(B / "merlin"),
                            "--drafter-config", str(B / "drafter_config.json")], timeout=900, cwd=B / "build")
        lines = r["_out"].splitlines()
        m = next((re.match(r"(\\d+)/(\\d+) PASS", l) for l in lines[::-1] if re.match(r"\\d+/\\d+ PASS", l)), None)
        U1["k100_nan"] = {"rc": r["rc"], "summary": m.group(0) if m else None, "fails": [l[:300] for l in lines if l.startswith("FAIL ")][:40],
                          "skips": [l for l in lines if l.startswith("SKIP ")], "tail": r["tail"][-600:]}
        U1["k100_nan"]["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0 and not U1["k100_nan"]["skips"]
        log("k100 nan", json.dumps(U1["k100_nan"])[:1200])
''', "units nan run")
    UNITS = _k100_sub(UNITS, '''**({"k100_units_all_pass": (U1.get("k100_units") or {}).get("all_pass") is True} if "__VARIANT__" == "k100" else {})),''',
                      '''**({"k100_units_all_pass": (U1.get("k100_units") or {}).get("all_pass") is True} if "__VARIANT__" == "k100" else {}),
             k100_nan_all_pass=(U1.get("k100_nan") or {}).get("all_pass") is True),''', "units nan verdict")
    UNITS = _k100_sub(UNITS, '''"k100_units": (U1.get("k100_units") or {}).get("summary"), ''',
                      '''"k100_units": (U1.get("k100_units") or {}).get("summary"), "k100_nan": (U1.get("k100_nan") or {}).get("summary"), ''',
                      "units nan recorded")

    _i = UNITS.index('    r = sh("units", [')
    _e = '    log("seam tests", json.dumps(U1["seam_tests"])[:1500])\n'
    _j = UNITS.index(_e) + len(_e)
    UNITS = UNITS[:_i] + ('    U1["w4_units"] = "the shared nbdf_lib.py (sha " + hashlib.sha256((B / "build/nbdf_lib.py").read_bytes()).hexdigest()[:12] + '
                          '") and the seam tests run in the W4 units kit"\n') + UNITS[_j:]
    UNITS = _k100_sub(UNITS, '''{"nbdf_units_all_pass": (U1.get("units") or {}).get("all_pass") is True,
              "seam_tests_all_pass": (U1.get("seam_tests") or {}).get("all_pass") is True}''', "{}", "units w4 verdict")

    # ---- REAL: the KV probe's attention layer names (klee100's config calls its QSA layers "qwen_sparse_attention", so
    # the W4 test == "full_attention" found none, the names fell back to synthetic ones and drop_mtp never matched mtp.layers.48:
    # the MTP layer's QSA / indexer KV stayed in the D-SW simulation; Merlin builds no MTP module on a dflash arm)
    REAL = _k100_sub(REAL, '''idx = [i for i in range(lo, hi) if LT[i] == "full_attention"]''',
                     '''idx = [i for i in range(lo, hi) if LT[i] != "linear_attention"]   # E27 K100: qwen_sparse_attention / full_attention''',
                     "R9 attention layer types")
    # ---- REAL: the notebook's scales-sidecar step (CELL 2b k100_setup, the same order as nbdf_main: after #60490, before any
    # launch) before the R4 host-phase / launch dry-run (Merlin's host_phase refuses FP8 KV without the sidecar)
    REAL = _k100_sub(REAL, '''plans = {}\n''', '''K100_SETUP = {}
try:
    K100_SETUP.update(G["k100_setup"]())
    K100_SETUP["sidecar_exists"] = (Path(G["CKPT"]) / "qsa_kv_scales_262k.json").is_file()
except Exception as e:
    K100_SETUP.update(ok=False, error=f"{type(e).__name__}: {e}"[:600])
REP["checks"]["K100_SETUP"] = K100_SETUP; save()
log("K100 setup (scales sidecar)", json.dumps(K100_SETUP, default=str)[:900])
plans = {}
''', "k100_setup before R4")

    # ---- REAL (round 4): the notebook's launch path, step by step, through its own functions (nbdf_main + arm_boot + the
    # K100 / NBDF / CELL 2 launcher chain). nb4_ppack via PSEQ.before_arm(ARMS["Q"]) (nbdf_main), every D arm SSEQ.check(k) +
    # PSEQ.before_arm(a) (nbdf_main), then arm_boot's first try: tree_check, Q's I8 draft head (Q_I8 -> arm draft_head /
    # expect_cfg, NB4T_BUILD_ENV DRAFT_HEAD=int8 into Merlin's build()), NB1_EXTRA / NB1_ENV / NB1_KV at K100_CFG q_pin, ARM_OF;
    # launch_check at that pin (the pin the notebook passes)
    REAL = _k100_sub(REAL, '''    R3P["apply"] = G["_ppack_apply"]()
''', '''    _okp, _whyp = G["PSEQ"].before_arm(G["ARMS"]["Q"])          # E27 K100: nbdf_main's own call (PatchSeq -> _ppack_apply)
    R3P["before_arm_q"], R3P["apply"] = [_okp, _whyp], G["PSEQ"].apply_rec
    if not _okp:
        raise RuntimeError("PSEQ.before_arm(Q): " + str(_whyp))
''', "R3P via PSEQ.before_arm")
    REAL = _k100_sub(REAL, '''    G["NB1_EXTRA"][nm], G["NB1_ENV"][nm], G["NB1_KV"][nm] = list(a["extra"]), dict(a["env"]), a["kv_bytes"]
    G["ARM_OF"][nm] = a
    b = {"arm": key, "tree_check": G["tree_check"](a)}
''', '''    b = {"arm": key}
    if key != "Q":                                            # E27 K100: nbdf_main's per-D-arm checks before step(...)
        b["sseq_check"] = list(G["SSEQ"].check(key))
        b["pseq_before_arm"] = list(G["PSEQ"].before_arm(a))
    b["tree_check"] = G["tree_check"](a)                       # arm_boot: the first try
    kvb = int(G["K100_CFG"]["q_pin"])
    if key == "Q":
        a["draft_head"] = "int8" if G["Q_I8"]["on"] else None
        a["expect_cfg"]["draft_head"] = a["draft_head"]
        G["NB4T_BUILD_ENV"][nm] = {"DRAFT_HEAD": "int8"} if G["Q_I8"]["on"] else {}
    G["NB1_EXTRA"][nm], G["NB1_ENV"][nm], G["NB1_KV"][nm] = list(a["extra"]), dict(a["env"]), kvb
    G["ARM_OF"][nm] = a
    b.update(pin=kvb, q_i8=dict(G["Q_I8"]) if key == "Q" else None, build_env=G["NB4T_BUILD_ENV"].get(nm))
    _gate_bad = [x for x in ("sseq_check", "pseq_before_arm") if x in b and b[x][0] is not True]
    if _gate_bad or not (b["tree_check"] or {}).get("ok"):
        b.update(ok=False, error=f"nbdf_main / arm_boot pre-launch check failed: {_gate_bad} tree {b['tree_check']}")
        plans[key] = b
        log("R4", key, json.dumps(b, default=str)[:900]); save()
        return
''', "plan_one = arm_boot")
    REAL = _k100_sub(REAL, '''        s = G["merlin_fn_server"](nm, "fp8", a["port"], a["kv_bytes"])
''', '''        s = G["merlin_fn_server"](nm, "fp8", a["port"], kvb)
''', "launch at the K100 pin")
    REAL = _k100_sub(REAL, '''launch_check=G["N"].launch_check(a, s.env, s.argv, a["kv_bytes"]),''',
                     '''launch_check=G["N"].launch_check(a, s.env, s.argv, kvb), cli_flags=(G["K1"].get("cli_flags") or {}).get(nm),
                 k6_before_launch=(G["K1"].get("k6_before_launch") or {}).get(nm), qsa_scales_env=s.env.get("VLLM_QSA_KV_SCALES"),''',
                     "launch_check at the K100 pin")
    # ---- REAL (round 4): the --help flag gate. The notebook's own L.help_check (e23_lab: `vllm serve --help=all` with the
    # launch env) died in vLLM's device inference on the GPU-less kit box (traceback >= 2000 chars, ran True, every flag
    # missing). VLLM_TARGET_DEVICE=cpu (kit only, as the klee100 LOAD / K100Q kits) reaches it through os.environ -> Merlin's
    # build() env; popped after R4. The help text is then checked to be the real argparser output.
    REAL = _k100_sub(REAL, '''    for k in ARMS5:
        plan_one(k)
''', '''    os.environ["VLLM_TARGET_DEVICE"] = "cpu"     # kit only (the GPU notebook never sets it)
    try:
        for k in ARMS5:
            plan_one(k)
    finally:
        os.environ.pop("VLLM_TARGET_DEVICE", None)
        _ht = G["L"]._HELP.get("merlin") or ""
        R4["help_text_head"], R4["help_text_len"] = _ht[:3000], len(_ht)
        R4["help_real"] = ("--max-num-seqs" in _ht and "--speculative-config" in _ht and "--kv-cache-memory-bytes" in _ht
                           and "Failed to infer device" not in _ht and "Traceback" not in _ht)
        R4["cli_flags"] = dict(G["K1"].get("cli_flags") or {})
''', "help gate on CPU")
    # ---- REAL (round 4): R5's expected spec = nbdf_lib.launch_check's want_sp (Q with the I8 head: draft_lm_head_quant int8)
    REAL = _k100_sub(REAL, '''        if G["T"].spec_from_argv(x) != a["spec"]: bad.append(f"spec {G['T'].spec_from_argv(x)}")''',
                     '''        _want_sp = dict(a["spec"], draft_lm_head_quant=a["draft_head"]) if a.get("draft_head") and a.get("method") != "dflash" else dict(a["spec"])
        if G["T"].spec_from_argv(x) != _want_sp: bad.append(f"spec {G['T'].spec_from_argv(x)} != {_want_sp}")''', "R5 spec = launch_check want_sp")
    # ---- REAL: R5 seqs 24
    REAL = _k100_sub(REAL, '''if lv(x, "--max-num-seqs") != "8": bad.append("seqs")''',
                     '''if lv(x, "--max-num-seqs") != "24": bad.append("seqs")''', "R5 seqs")

    # ---- KVPROBE: keep every run's configs + the per-worker pool block bytes (item 1)
    REAL = _k100_sub(REAL, '''    def run(tag, ws, pin, draft_backend=None):
''', '''    CF = {}

    def k100_pages(vc, cfgs):
        # E27 K100 R9: per worker the pool block bytes (the divisor get_kv_cache_config_from_groups uses) + the tensor view
        res = []
        for c in cfgs:
            r_ = {"num_blocks": c.num_blocks, "groups": len(c.kv_cache_groups)}
            try:
                r_["pool_bytes_per_block"] = int(KU._pool_bytes_per_block(vc, c.kv_cache_groups))
            except BaseException as e:
                r_["pool_bytes_error"] = f"{type(e).__name__}: {e}"[:200]
            try:
                r_["tensor_page_max"] = max(int(t.size) // max(1, c.num_blocks) for t in c.kv_cache_tensors)
            except BaseException as e:
                r_["tensor_page_error"] = f"{type(e).__name__}: {e}"[:200]
            res.append(r_)
        return res

    def run(tag, ws, pin, draft_backend=None):
''', "kvprobe CF")
    REAL = _k100_sub(REAL, '''            cfgs = KU.get_kv_cache_configs(vc, ws, [pin] * len(ws))
''', '''            cfgs = KU.get_kv_cache_configs(vc, ws, [pin] * len(ws))
            CF[tag] = (cfgs, ws)
            rec["pages"] = k100_pages(vc, cfgs)
''', "kvprobe pages")

    # ---- KVPROBE: R9_IDX (record only), item 2 (F with the real max_in_flight_tokens), item 3 (core.py's block drop)
    REAL = _k100_sub(REAL, '''    out["note"] = ("target specs = the NB4T GPU record''', '''    # ---- E27 K100 (the kit computes the verdicts): R9_IDX, D-SW F (item 2), the engine-core block drop (item 3)
    import math
    import os as _os
    KX = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    KO = out.setdefault("k100", {"kx": KX})
    try:                                   # R9_IDX: the QSA indexer compressed-key cache at fp8 (indexer_qsa.py:155-182)
        ranks_i = copy.deepcopy(ranks)
        n_i = 0
        for w in ranks_i:
            for s in w["kv_cache_specs"]:
                if s["type"] == "MLAAttentionSpec" and "compressed_key_cache" in str(s.get("example") or ""):
                    s["dtype"] = "torch.float8_e4m3fn"
                    if isinstance(s.get("page_size_bytes"), int):
                        s["page_size_bytes"] //= 2
                    n_i += 1
        run("Q_idxfp8", worker_specs(ranks_i, 3), Q_PIN)
        qi = out["runs"]["Q_idxfp8"]
        KO["r9_idx"] = {"specs_changed": n_i, "q_blocks": out["runs"]["Q"].get("num_blocks"), "q_idxfp8_blocks": qi.get("num_blocks"),
                        "q_pages": out["runs"]["Q"].get("pages"), "q_idxfp8_pages": qi.get("pages"), "error": qi.get("error")}
    except BaseException as e:
        KO["r9_idx"] = {"error": f"{type(e).__name__}: {e}"[:400]}

    def real_vc(seqs):
        # the served config's capacity inputs: PP2 x TP2, V2 runner (launch.py sets VLLM_USE_V2_MODEL_RUNNER=1), async
        # scheduling, the argv's --max-num-batched-tokens / --prefix-match-unit, align mamba cache, prefix caching
        _os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        vc = VllmConfig()
        vc.model_config = MC["mc"] if MC.get("mc") is not None else NS(
            max_model_len=131072, original_max_model_len=131072, hf_config=NS(), use_mla=False, is_hybrid=True, get_num_layers=lambda *a: 48)
        cc, pc, sc_ = vc.cache_config, vc.parallel_config, vc.scheduler_config
        cc.block_size, cc.enable_prefix_caching, cc.cache_dtype = 4096, True, "fp8_e4m3"
        cc.prefix_match_unit = int(KX.get("prefix_match_unit") or 64)
        if hasattr(cc, "mamba_cache_mode"):
            cc.mamba_cache_mode = "align"
        pc.pipeline_parallel_size, pc.tensor_parallel_size = int(KX.get("pp") or 2), int(KX.get("tp") or 2)
        sc_.async_scheduling = True
        sc_.max_num_batched_tokens = int(KX.get("max_num_batched_tokens") or 1024)
        sc_.max_num_seqs = int(seqs)
        return vc

    def f_parts(vc, sched):
        # per scheduler group: blocks one max_model_len request needs (get_max_concurrency_for_kv_cache_config's sum);
        # the FullAttention / MLA groups carry the long context, every other group is F (fixed per request)
        rows, fixed, full, shared = [], 0, 0, 0
        for i, g in enumerate(sched.kv_cache_groups):
            sp = g.kv_cache_spec
            r_ = {"group": i, "spec": type(sp).__name__, "block_size": getattr(sp, "block_size", None), "layers": len(g.layer_names),
                  "page_size_bytes": getattr(sp, "page_size_bytes", None), "slots_per_block": getattr(g, "slots_per_block", 1),
                  "prefix_cacheable": bool(getattr(sp, "prefix_cacheable", False))}
            if r_["slots_per_block"] > 1 and not getattr(g, "host_resident", False):
                r_["shared_blocks"] = math.ceil(vc.scheduler_config.max_num_seqs / r_["slots_per_block"])
                shared += r_["shared_blocks"]
            else:
                r_["required"] = math.ceil(sp.max_memory_usage_bytes(vc) / sp.page_size_bytes)
                if isinstance(sp, KI.SlidingWindowSpec):
                    r_["sw_admission_blocks"] = sp.max_admission_blocks_per_request(max_in_flight_tokens=vc.max_in_flight_tokens,
                                                                                    max_model_len=vc.model_config.max_model_len)
                    r_["extra_retained_tokens"] = getattr(sp, "extra_retained_tokens", None)
                    fixed += r_["required"]
                elif isinstance(sp, (KI.FullAttentionSpec, KI.MLAAttentionSpec)):
                    full += r_["required"]
                else:
                    fixed += r_["required"]
            rows.append(r_)
        return rows, fixed, full, shared

    def layout_ranks(cfgs, ws):
        # the NB1 worker-extension shape (kv_cache_specs + kv_cache_config per rank) for nbdf_lib.kv_layout_readback
        res = []
        for i, (c, w) in enumerate(zip(cfgs, ws)):
            agg = {}
            for nm, sp in w.items():
                key = (type(sp).__name__, getattr(sp, "block_size", None), getattr(sp, "page_size_bytes", None), str(getattr(sp, "dtype", None)))
                a = agg.setdefault(key, {"type": key[0], "block_size": key[1], "page_size_bytes": key[2], "dtype": key[3],
                                         "num_kv_heads": getattr(sp, "num_kv_heads", None), "head_size": getattr(sp, "head_size", None),
                                         "layers": 0, "example": nm})
                a["layers"] += 1
            res.append({"rank": ranks[i]["rank"] if i < len(ranks) else i, "kv_cache_specs": list(agg.values()),
                        "kv_cache_config": {"num_blocks": c.num_blocks, "groups": [
                            {"spec": type(g.kv_cache_spec).__name__, "layers": len(g.layer_names), "block_size": getattr(g.kv_cache_spec, "block_size", None),
                             "page_size_bytes": getattr(g.kv_cache_spec, "page_size_bytes", None)} for g in c.kv_cache_groups]}})
        return res

    for tag in ("K3_swblock1024", "K4_swblock1024"):
        rec = {}
        if tag not in CF:
            KO.setdefault("sim", {})[tag] = {"error": "no run (Q not calibrated or the run failed)"}
            continue
        cfgs, ws = CF[tag]
        try:
            vcr = real_vc(KX.get("seqs") or 24)
            try:
                rec["vc"] = {"pp": vcr.parallel_config.pipeline_parallel_size, "tp": vcr.parallel_config.tensor_parallel_size,
                             "async_scheduling": vcr.scheduler_config.async_scheduling, "use_v2_model_runner": bool(vcr.use_v2_model_runner),
                             "max_concurrent_batches": vcr.max_concurrent_batches, "max_in_flight_tokens": vcr.max_in_flight_tokens,
                             "max_num_batched_tokens": vcr.scheduler_config.max_num_batched_tokens, "max_num_seqs": vcr.scheduler_config.max_num_seqs,
                             "prefix_match_unit": vcr.cache_config.prefix_match_unit, "max_model_len": vcr.model_config.max_model_len,
                             "num_prefill_lookahead_tokens_no_spec": vcr.num_prefill_lookahead_tokens}
            except BaseException as e:
                rec["vc_error"] = f"{type(e).__name__}: {e}"[:300]
            sched = KU.generate_scheduler_kv_cache_config(cfgs)
            groups = sched.kv_cache_groups
            part = [g.kv_cache_spec.block_size for g in groups if g.kv_cache_spec.prefix_cacheable]
            rec["participating_block_sizes"] = part
            rec["engine_block_before"] = vcr.cache_config.block_size
            # v1/engine/core.py:434-448: cache_config.block_size = min over the prefix-cacheable groups (the SW 1024)
            vcr.cache_config.block_size = min(part if part else [g.kv_cache_spec.block_size for g in groups])
            rec["engine_block"] = vcr.cache_config.block_size
            try:                           # core.py:451 update_kv_cache_capacity = the boot max-concurrency line
                KU.update_kv_cache_capacity(vcr, sched)
                rec["mc_boot"], rec["kv_size_tokens"] = vcr.cache_config.kv_cache_max_concurrency, vcr.cache_config.kv_cache_size_tokens
                rec["mc_how"] = "update_kv_cache_capacity"
            except BaseException as e:
                rec["mc_boot"] = KU.get_max_concurrency_for_kv_cache_config(vcr, sched)
                rec["mc_how"] = f"get_max_concurrency_for_kv_cache_config (update_kv_cache_capacity: {type(e).__name__}: {e})"[:300]
            try:
                vcr.validate_block_size()
                rec["validate_block_size"] = "ok"
            except BaseException as e:
                rec["validate_block_size"] = f"{type(e).__name__}: {e}"[:300]
            sbs, hbs = KU.resolve_kv_cache_block_sizes(sched, vcr)
            rec["scheduler_block"], rec["hash_block"] = sbs, hbs
            try:
                rec["cache_hit_alignment"] = KU.resolve_cache_hit_alignment_tokens(sched, vcr, sbs, hbs)
            except BaseException as e:
                rec["cache_hit_alignment"] = f"{type(e).__name__}: {e}"[:300]
            rec["pool"] = sched.num_blocks
            rec["sched_groups"] = len(groups)
            parts, fixed, full, shared = f_parts(vcr, sched)
            rec.update(groups=parts, F_groups=fixed, full_blocks_per_request=full, shared_blocks=shared)
            rec["sw_admission_blocks"] = [p["sw_admission_blocks"] for p in parts if "sw_admission_blocks" in p]
            rec["mamba_blocks"] = sum(p.get("required", 0) for p in parts if p["spec"] == "MambaSpec")
            mcb = rec["mc_boot"]
            rec["F_mc"] = (math.ceil(round(sched.num_blocks / mcb - vcr.model_config.max_model_len / 4096, 3))
                           if isinstance(mcb, (int, float)) and mcb > 0 else None)
            rec["mc_by_seqs"] = {}
            for s_ in (21, 24):
                v_ = real_vc(s_)
                v_.cache_config.block_size = rec["engine_block"]
                rec["mc_by_seqs"][str(s_)] = KU.get_max_concurrency_for_kv_cache_config(v_, sched)
            rec["layout_ranks"] = layout_ranks(cfgs, ws)
        except BaseException as e:
            rec.update(error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
        KO.setdefault("sim", {})[tag] = rec
    out["note"] = ("target specs = the NB4T GPU record''', "kvprobe k100 extras")

    # ---- WEIGHTPROBE (R12 on the klee100 target ModelConfig, item 5) + the drafter scan (item 6)
    REAL = _k100_sub(REAL, '''    with set_current_vllm_config(dvc), set_default_torch_dtype(sc.draft_model_config.dtype), torch.device("meta"):
''', '''    try:                                       # E27 K100: what initialize_model does before the class is built
        from vllm.model_executor.model_loader.utils import configure_quant_config
        if dvc.quant_config is not None:
            configure_quant_config(dvc.quant_config, DFlash2Qwen3ForCausalLM)
        out["k100_configure_quant_config"] = type(dvc.quant_config).__name__
    except BaseException as e:
        out["k100_configure_quant_config"] = f"error {type(e).__name__}: {e}"[:300]
    with set_current_vllm_config(dvc), set_default_torch_dtype(sc.draft_model_config.dtype), torch.device("meta"):
''', "weightprobe configure_quant_config")
    REAL = _k100_sub(REAL, '''    if not out["ok"] and not out["false_raise_suspect"] and "load_weights_error" in out:
        out["ok"] = False
''', '''    if not out["ok"] and not out["false_raise_suspect"] and "load_weights_error" in out:
        out["ok"] = False
    KO = {}                                    # E27 K100: klee100 target quantization vs the drafter's, its linears, the scan
    try:
        KO["target_quantization"] = getattr(mc, "quantization", None)
        KO["target_quant_config"] = type(getattr(vc, "quant_config", None)).__name__
        KO["draft_quantization"] = getattr(sc.draft_model_config, "quantization", None)
        KO["draft_vllm_quant_config"] = type(getattr(dvc, "quant_config", None)).__name__
        from vllm.model_executor.layers.linear import LinearBase
        lin = {n_: type(getattr(mod, "quant_method", None)).__name__ for n_, mod in m.named_modules() if isinstance(mod, LinearBase)}
        KO["linears"] = len(lin)
        KO["linear_methods"] = sorted(set(lin.values()))
        KO["linears_not_unquantized"] = sorted(n_ for n_, v in lin.items() if v != "UnquantizedLinearMethod")[:12]
        qa = {}
        for n_, mod in m.named_modules():
            if getattr(mod, "quant_method", None) is not None:
                qa[type(mod.quant_method).__name__] = qa.get(type(mod.quant_method).__name__, 0) + 1
        KO["quant_methods_all_modules"] = qa
    except BaseException as e:
        KO["quant_error"] = f"{type(e).__name__}: {e}"[:400]
    try:
        lc = json.loads(chk[0].split("E27_DFLASH", 1)[1]) if chk else {}
        KO["load_check"] = {k: lc.get(k) for k in ("expected", "loaded", "skipped", "dropped")}
    except BaseException as e:
        KO["load_check_error"] = f"{type(e).__name__}: {e}"[:300]
    try:
        import nb1_worker_ext as X            # the notebook's real worker extension (written to the Merlin tree by apply_60490)
        KO["ext_file"] = X.__file__
        rows = X._k100_scan(m, "drafter")
        KO["scan_rows"] = len(rows)
        KO["scan_list"] = rows[:64]
    except BaseException as e:
        KO["scan_error"] = f"{type(e).__name__}: {e}"[:400]
    out["k100"] = KO
''', "weightprobe k100")

    # ---- the argv inputs of the KVPROBE simulation (the D arm's own argv)
    REAL = _k100_sub(REAL, '''(K / "q_fixture.json").write_text(json.dumps(FIX))
''', '''_xa = next((plans[k_].get("argv") for k_ in ("D3s", "D4s") if (plans.get(k_) or {}).get("argv")), None) or []
_lv = G["T"].last_value if G else (lambda *a: None)
K100_KX = {"argv_from": next((k_ for k_ in ("D3s", "D4s") if (plans.get(k_) or {}).get("argv")), None), "seqs": 24,
           "max_num_batched_tokens": _lv(_xa, "--max-num-batched-tokens"), "prefix_match_unit": _lv(_xa, "--prefix-match-unit"),
           "max_model_len": _lv(_xa, "--max-model-len"), "pp": _lv(_xa, "--pipeline-parallel-size"), "tp": _lv(_xa, "--tensor-parallel-size")}
(K / "q_fixture.json").write_text(json.dumps(FIX))
''', "kx")
    REAL = _k100_sub(REAL, '''"__QPIN__", "__DPIN__", str(G["CKPT"])], name="r9_kv_cache_configs"''',
                     '''"__QPIN__", "__DPIN__", str(G["CKPT"]), json.dumps(K100_KX)], name="r9_kv_cache_configs"''', "R9 argv")

    # ---- the kit-side items 1-6, 8, 9 (after R9-R12, before the revert)
    REAL = _k100_sub(REAL, '''    log(nm, json.dumps(r.get("json") or r.get("tail"), default=str)[:1500])
save()
''', '''    log(nm, json.dumps(r.get("json") or r.get("tail"), default=str)[:1500])
save()

# ==================================================================== E27 K100 record-and-check items (this kit only)
import struct, inspect, ast as _ast
K100_OK, K100_REC = {}, {}
K100_SEQS, K100_PIN_B, K100_SW = 24, 7933526016, ("K3_swblock1024", "K4_swblock1024")
r9j = R9.get("json") or {}
r9k = r9j.get("k100") or {}
r12k = (R12.get("json") or {}).get("k100") or {}
CK = Path(str(G["CKPT"])) if G else None
_isint = lambda v: isinstance(v, int) and not isinstance(v, bool)        # noqa: E731


def _run9(t):
    return (r9j.get("runs") or {}).get(t) or {}


def _page9(t, i):
    p = _run9(t).get("pages") or []
    return (p[i].get("pool_bytes_per_block") or p[i].get("tensor_page_max")) if i < len(p) else None


def _minb9(t):
    nb = _run9(t).get("num_blocks")
    return min(nb) if nb else None


def _hdr(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), n


def k100_item(key, fn):
    try:
        c = fn()
    except Exception as e:
        c = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-1500:]}
    c["ok"] = c.get("ok") is True
    REP["checks"][key] = c
    log(key, json.dumps(c, default=str)[:1500])
    save()
    return c


def k100_r9():
    """item 1: Q at 7,933,526,016 B -> 480 blocks (+-2; the GPU's 479 usable = 480 - the null block); D-SW K3 / K4 ~540 (+-3);
    drafter page 10,485,760, PP1 page 14,155,776, PP0 page 14,680,064."""
    gq = []
    for w in FIX["q_workers"]["ranks"]:
        kc = w.get("kv_cache_config") or {}
        gq.append(max(t["size"] for t in kc["tensors"]) // kc["num_blocks"] if kc.get("tensors") and kc.get("num_blocks") else None)
    c = {"q_blocks": _minb9("Q"), "q_target": 480, "q_gpu_usable": 479, "q_gpu_num_blocks": [w.get("num_gpu_blocks") for w in FIX["q_workers"]["ranks"]],
         "q_pages": [_page9("Q", i) for i in range(4)], "q_gpu_pages": gq, "q_calibrated": r9j.get("q_calibrated"), "pin": K100_PIN_B}
    c["q_pages_equal_gpu"] = c["q_pages"] == gq
    for t in K100_SW:
        sw = [g for g in _run9(t).get("groups") or [] if "SlidingWindowSpec" in (g.get("inner") or [])]
        c[t] = {"blocks": _minb9(t), "drafter_page": sw[0].get("page_size_bytes") if len(sw) == 1 else None, "pp0_page": _page9(t, 0),
                "pp1_page": _page9(t, 2), "groups_pp1": len(_run9(t).get("groups") or []), "error": _run9(t).get("error")}
        c[t]["predicted"] = {"blocks_about_540": _isint(c[t]["blocks"]) and abs(c[t]["blocks"] - 540) <= 3,
                             "pp1_page_14155776": c[t]["pp1_page"] == 14155776, "role": "record-only capacity prediction (research A)"}
        c[t]["ok"] = (_isint(c[t]["blocks"]) and c[t]["blocks"] > 0 and not c[t]["error"] and c[t]["drafter_page"] == 10485760
                      and c[t]["pp0_page"] == 14680064)
    c["ok"] = _isint(c["q_blocks"]) and abs(c["q_blocks"] - 480) <= 2 and all(c[t]["ok"] for t in K100_SW)
    return c


def k100_f():
    """item 2: D-SW F (Mamba + SW admission blocks) with the real max_in_flight_tokens (PP2 + V2 async); max concurrency
    at 131,072 tokens for max_num_seqs 21 / 24."""
    c = {"kx": r9k.get("kx")}
    for t in K100_SW:
        s = (r9k.get("sim") or {}).get(t) or {}
        c[t] = {k: s.get(k) for k in ("F_groups", "F_mc", "sw_admission_blocks", "mamba_blocks", "full_blocks_per_request", "shared_blocks",
                                      "mc_boot", "mc_by_seqs", "vc", "groups", "error")}
        c[t]["ok"] = _isint(s.get("F_groups")) and 10 <= s["F_groups"] <= 14
    c["ok"] = all(c[t]["ok"] for t in K100_SW)
    return c


def k100_blockdrop():
    """item 3: core.py's engine block drop (cache_config.block_size -> 1024, scheduler block = LCM 4096, hash unit = the
    argv's prefix-match-unit 64), then nbdf_lib.level_for_band at the target block 4096 (MIXED_KV_BLOCK_SIZES) for 40k / 90k
    / 116k on the D-SW pool."""
    N_ = G["N"]
    sig = inspect.signature(N_.level_for_band).parameters
    kw = {"seqs": K100_SEQS} if "seqs" in sig else {}
    pmu = int((r9k.get("kx") or {}).get("prefix_match_unit") or 64)
    c = {"level_for_band_params": list(sig), "seqs_passed": kw.get("seqs"), "hash_expected": pmu}
    for t in K100_SW:
        s = (r9k.get("sim") or {}).get(t) or {}
        lay = N_.kv_layout_readback({"ranks": s.get("layout_ranks")}) if s.get("layout_ranks") else None
        mc = (s.get("mc_by_seqs") or {}).get(str(K100_SEQS))
        lv = {}
        for band in ("40k", "90k", "116k"):
            l_, info = N_.level_for_band(band, N_.BAND_TOKENS[band], s.get("pool"), mc, s.get("engine_block"), lay, **kw)
            lv[band] = {"level": l_, "info": info}
        c[t] = {k: s.get(k) for k in ("engine_block_before", "participating_block_sizes", "engine_block", "scheduler_block", "hash_block",
                                      "cache_hit_alignment", "validate_block_size", "pool", "sched_groups", "error")}
        c[t].update(mc_seqs24=mc, layout_block_sizes=(lay or {}).get("block_sizes"), levels=lv)
        c[t]["ok"] = (s.get("engine_block") == 1024 and s.get("scheduler_block") == 4096 and s.get("hash_block") == pmu == 64
                      and all(_isint(v["level"]) and v["level"] >= 1 and (v["info"] or {}).get("mixed") == "MIXED_KV_BLOCK_SIZES"
                              and (v["info"] or {}).get("source") != "CAPACITY_UNVERIFIED" for v in lv.values()))
    c["ok"] = all(c[t]["ok"] for t in K100_SW)
    return c


def k100_bytes():
    """item 4 (headers only): the klee100 mtp.* tensor bytes freed per PP1 card (routed experts / EP2, the rest / TP2, norms /
    gates / hyper-connection replicated) vs the drafter's model.safetensors (embedding, lm_head = released for the shared
    target head, weights / TP2; fc (ReplicatedLinear, qwen3_dflash.py:495), norms, conv, selector replicated)."""
    idx = json.loads((CK / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    by = {}
    for n in (x for x in wm if x.startswith("mtp.")):
        by.setdefault(wm[n], []).append(n)
    cls, cnt, miss = {"routed_ep2": 0, "tp2": 0, "replicated": 0}, {"routed_ep2": 0, "tp2": 0, "replicated": 0}, []
    # replicated on every card: norms, the router gate, the shared-expert gate, the hyper-connection ReplicatedLinears
    # (hyperconnection.py:118), the QSA indexer's ReplicatedLinear index_qk_proj (indexer_qsa.py:135); fc_* ColumnParallel (mtp.py:259)
    REPL = ("norm", "mlp.gate.weight", "shared_expert_gate", "hyper_connection", "indexer.index_qk_proj")
    for f, ns in sorted(by.items()):
        h, _ = _hdr(CK / f)
        for n in ns:
            e = h.get(n)
            if not e:
                miss.append(n)
                continue
            k_ = "routed_ep2" if ".mlp.experts." in n else "replicated" if any(x in n for x in REPL) else "tp2"
            cls[k_] += e["data_offsets"][1] - e["data_offsets"][0]
            cnt[k_] += 1
    mtp_total = sum(cls.values())
    freed = cls["routed_ep2"] / 2 + cls["tp2"] / 2 + cls["replicated"]
    dd = Path(str(G["DRAFTER"].get("dir")))
    dh, _ = _hdr(dd / "model.safetensors")
    dcls, dcnt = {"embed": 0, "lm_head": 0, "replicated": 0, "tp2": 0}, {"embed": 0, "lm_head": 0, "replicated": 0, "tp2": 0}
    for n, e in dh.items():
        if n == "__metadata__":
            continue
        k_ = ("embed" if n.endswith("embed_tokens.weight") else "lm_head" if n.startswith("lm_head") else
              "replicated" if (len(e["shape"]) <= 1 or n.startswith(("fc.", "model.fc."))
                               or any(x in n for x in ("norm", "candidate_selector", "conv", "codebook"))) else "tp2")
        dcls[k_] += e["data_offsets"][1] - e["data_offsets"][0]
        dcnt[k_] += 1
    added = dcls["tp2"] / 2 + dcls["replicated"]
    emb = dcls["embed"] / 2
    G_ = 2 ** 30
    pin = (G.get("K100_PIN") or {}).get("mtp_tensor_bytes")
    c = {"mtp_files": sorted(by), "mtp_tensors": sum(cnt.values()), "mtp_missing_in_header": miss[:8], "mtp_bytes": mtp_total,
         "mtp_bytes_pin": pin, "mtp_bytes_equal_pin": mtp_total == pin, "mtp_classes": cls, "mtp_class_tensors": cnt,
         "mtp_freed_per_pp1_card_gib": round(freed / G_, 4), "drafter_file_bytes": (dd / "model.safetensors").stat().st_size,
         "drafter_tensors": sum(dcnt.values()), "drafter_classes": dcls, "drafter_class_tensors": dcnt,
         "drafter_weights_gib": round((dcls["tp2"] + dcls["replicated"]) / G_, 4),
         "drafter_added_per_pp1_card_gib_no_embed": round(added / G_, 4), "drafter_embed_per_pp1_card_gib": round(emb / G_, 4),
         "net_per_pp1_card_gib_no_embed": round((freed - added) / G_, 4), "net_per_pp1_card_gib_with_embed": round((freed - added - emb) / G_, 4),
         "nbdf_lib_plan_mib": {"DRAFTER_GPU_MIB": getattr(G["N"], "DRAFTER_GPU_MIB", None), "EMBED_GPU_MIB": getattr(G["N"], "EMBED_GPU_MIB", None),
                               "MTP_FREED_MIB": getattr(G["N"], "MTP_FREED_MIB", None)},
         "split_rule": ("MTP: routed experts / EP2, attention / shared expert / fc_* / TP2, norms / router gate / hyper-connection / "
                        "indexer index_qk_proj replicated; drafter: embed / TP2 (vocab parallel), lm_head released, fc / conv / selector / "
                        "norms replicated, the rest / TP2 (headers only)")}
    c["ok"] = c["mtp_tensors"] > 0 and mtp_total > 0 and not miss and c["drafter_tensors"] > 0 and dcls["tp2"] + dcls["replicated"] > 0
    return c


def k100_r12():
    """item 5: R12 on the klee100 target ModelConfig: the target quantized (inc / auto-round), the dflash drafter's
    quantization None, every drafter linear UnquantizedLinearMethod, draft_load_check 67 = 67."""
    lc = r12k.get("load_check") or {}
    c = {k: r12k.get(k) for k in ("target_quantization", "target_quant_config", "draft_quantization", "draft_vllm_quant_config", "linears",
                                  "linear_methods", "linears_not_unquantized", "quant_methods_all_modules", "quant_error", "load_check_error")}
    c.update(load_check=lc, r12_ok=(R12.get("json") or {}).get("ok"), configure_quant_config=(R12.get("json") or {}).get("k100_configure_quant_config"),
             r12_error=(R12.get("json") or {}).get("error"))
    c["ok"] = (c["r12_ok"] is True and c["target_quantization"] not in (None, "") and c["draft_quantization"] is None
               and _isint(c["linears"]) and c["linears"] > 0 and c["linear_methods"] == ["UnquantizedLinearMethod"]
               and not c["linears_not_unquantized"] and lc.get("expected") == lc.get("loaded") == 67 and not lc.get("skipped"))
    return c


def k100_scan_map():
    """item 6: the REAL worker extension's _k100_scan on the meta-built DFlash2 drafter -> no rows; method_map_check(a
    synthetic 48-layer klee100 target (4 ranks, 25 / 23) + the drafter rows on PP1, need_mtp False) -> GPU_METHOD_MAP_PASS."""
    KL_ = G["KL"]
    rows = r12k.get("scan_list") if isinstance(r12k.get("scan_list"), list) else None
    exp, stl = KL_.k100_expected_map(), KL_.stage_layers(KL_.PP_PARTITION)
    res = []
    for r in range(4):
        st = r // 2
        ex = [{"name": f"language_model.model.layers.{L}.mlp.experts", "module": "FusedMoE", "method": KL_.ROUTED_METHODS[0], "src": "model",
               "layer": L, "mtp": False, "bits": exp[L][0], "group": exp[L][1], "source": "synthetic", "bits_conflict": False}
              for L in sorted(stl[st])]
        if st == 1:
            ex += list(rows or [])
        res.append({"rank": r, "experts": ex, "errors": [], "drafter_path": "model_runner.get_draft_model()" if st == 1 else "none"})
    mm = KL_.method_map_check(res, need_mtp=False)
    tree_ = Path(eng["kv"]["MERLIN_TREE"]) if eng else None
    ext_f = tree_ / "nb1_worker_ext.py" if tree_ else None
    ext_sha = hashlib.sha256(ext_f.read_bytes()).hexdigest() if ext_f and ext_f.is_file() else None
    want = hashlib.sha256(G["NB1_EXT_SRC"].encode()).hexdigest() if G.get("NB1_EXT_SRC") else None
    c = {"scan_rows": r12k.get("scan_rows"), "scan_error": r12k.get("scan_error"), "ext_file": r12k.get("ext_file"), "ext_sha256": ext_sha,
         "ext_sha_equals_notebook": ext_sha is not None and ext_sha == want, "map_verdict": mm.get("verdict"), "map_why": mm.get("why"),
         "layers_matching": mm.get("layers_matching")}
    c["ok"] = (rows == [] and c["scan_rows"] == 0 and not c["scan_error"] and c["ext_sha_equals_notebook"]
               and mm.get("verdict") == KL_.MAP_PASS and mm.get("ok") is True)
    return c


def k100_tokenizer():
    """item 8: the klee100 tokenizer's highest added-token id <= 248076, vocab 248,320, and row 248077 of the target's
    embed_tokens (read from its shard by offset) = the drafter pin (nbdf_lib DRAFTER_PINS mask_row)."""
    tj = json.loads((CK / "tokenizer.json").read_text(encoding="utf-8"))
    tc = json.loads((CK / "tokenizer_config.json").read_text(encoding="utf-8"))
    cj = json.loads((CK / "config.json").read_text(encoding="utf-8"))
    ids = [int(t["id"]) for t in tj.get("added_tokens") or []] + [int(k) for k in (tc.get("added_tokens_decoder") or {})]
    vocab = (cj.get("text_config") or {}).get("vocab_size") or cj.get("vocab_size")
    wm = json.loads((CK / "model.safetensors.index.json").read_text())["weight_map"]
    names = [n for n in wm if n.endswith("embed_tokens.weight") and not n.startswith("mtp.") and "language_model" in n] or \\
            [n for n in wm if n.endswith("embed_tokens.weight") and not n.startswith("mtp.")]
    pin = G["N"].DRAFTER_PINS["mask_row"]
    c = {"added_tokens": len(ids), "max_added_id": max(ids) if ids else None, "vocab_size": vocab, "embed_names": names[:4],
         "mask_row_pin": pin["sha256"], "mask_row_id": pin.get("row", G["N"].MASK_TOKEN_ID)}
    if len(names) == 1:
        f = wm[names[0]]
        h, hl = _hdr(CK / f)
        e = h[names[0]]
        row = int(e["shape"][1]) * 2
        with open(CK / f, "rb") as fh:
            fh.seek(8 + hl + e["data_offsets"][0] + c["mask_row_id"] * row)
            raw = fh.read(row)
        c.update(shard=f, dtype=e["dtype"], shape=e["shape"], mask_row_sha256=hashlib.sha256(raw).hexdigest(),
                 mask_row_norm=round(G["N"].bf16_row_norm(raw), 8), norm_pin=pin.get("norm"))
    c["ok"] = (_isint(c["max_added_id"]) and c["max_added_id"] <= 248076 and vocab == 248320 and c.get("dtype") == "BF16"
               and list(c.get("shape") or []) == [248320, 2560] and c.get("mask_row_sha256") == pin["sha256"]
               and str(pin["sha256"]).startswith("8606") and str(pin["sha256"]).endswith("1a19"))
    return c


def k100_captures():
    """item 9: seqs 24: every decode batch b = 1 .. 24 x (1 + K) runs in a FULL uniform-decode graph for Q (MTP K3) / D3s /
    D4s (cudagraph_utils.py:332-357 rounding mirrored); the three arms' argv (--max-num-seqs 24, captures, KV bytes, spec);
    the Q-defer pack views at [n <= 24, 3 / 4] (nb4_ppack nb4_pack_views)."""
    tree_ = Path(eng["kv"]["MERLIN_TREE"])
    rs = (tree_ / "tools/merlin/registry.py").read_text()
    keep = [_ast.get_source_segment(rs, n) for n in _ast.parse(rs).body
            if getattr(n, "name", None) == "capture_sizes_for" or (isinstance(n, _ast.Assign) and any(getattr(t_, "id", "") in ("_LADDER", "MTP_K") for t_ in n.targets))]
    ns_ = {}
    exec("\\n".join(keep), ns_)
    exp_caps = {"Q": ns_["capture_sizes_for"](K100_SEQS, 3), "D3s": [4 * b for b in range(1, K100_SEQS + 1)], "D4s": [5 * b for b in range(1, K100_SEQS + 1)]}

    def graphs(caps, seqs, q):
        caps = sorted(set(int(x) for x in caps))
        out_ = set()
        for x in caps:
            rt = -(-x // q) * q
            if rt > seqs * q or rt > caps[-1] or rt // q > seqs:
                continue
            out_.add(rt // q)
        return sorted(out_)
    c, lv = {"expected_captures": exp_caps}, G["T"].last_value
    for k in ("Q", "D3s", "D4s"):
        a = G["ARMS"].get(k) or {}
        q = int(a.get("k") or 0) + 1
        g = graphs(a.get("captures") or [], K100_SEQS, q)
        unc = [b for b in range(1, K100_SEQS + 1) if not any(x >= b for x in g)]
        x = (plans.get(k) or {}).get("argv") or []
        try:
            cm = json.loads(lv(x, "--compilation-config") or "{}").get("cudagraph_mode")
        except Exception:
            cm = None
        bad = []
        if a.get("captures") != exp_caps[k]: bad.append(f"arm captures {a.get('captures')}")
        if G["N"].captures_from_argv(x) != exp_caps[k]: bad.append(f"argv captures {G['N'].captures_from_argv(x)}")
        if lv(x, "--max-num-seqs") != str(K100_SEQS): bad.append(f"seqs {lv(x, '--max-num-seqs')}")
        if lv(x, "--kv-cache-memory-bytes") != str(K100_PIN_B) or a.get("kv_bytes") != K100_PIN_B: bad.append("kv bytes")
        _want = dict(a.get("spec") or {}, draft_lm_head_quant=a["draft_head"]) if a.get("draft_head") and a.get("method") != "dflash" else a.get("spec")
        if G["T"].spec_from_argv(x) != _want: bad.append(f"spec {G['T'].spec_from_argv(x)} != {_want}")
        sp = a.get("spec") or {}
        if sp.get("method") != ("mtp" if k == "Q" else "dflash") or sp.get("num_speculative_tokens") != {"Q": 3, "D3s": 3, "D4s": 4}[k]:
            bad.append(f"spec method / K {sp.get('method')} {sp.get('num_speculative_tokens')}")
        if cm not in ("FULL_AND_PIECEWISE", "FULL", "FULL_DECODE_ONLY"): bad.append(f"cudagraph_mode {cm}")
        c[k] = {"k": q - 1, "captures": a.get("captures"), "graph_reqs": g, "uncovered_b": unc,
                "padded_b": [b for b in range(1, K100_SEQS + 1) if b not in g and b not in unc], "cudagraph_mode": cm,
                "lib_coverage": G["N"].coverage(a.get("captures") or [], K100_SEQS, q - 1).get("covers_all"), "argv_bad": bad}
        c[k]["ok"] = not unc and not bad and c[k]["lib_coverage"] is True
    import torch
    src = G["P"].PATCHES["vllm/v1/worker/gpu/pp_utils.py"][3]
    seg, keep_ = [], False
    for ln in src.splitlines():
        if ln.startswith("+def nb4_pack_views("):
            keep_ = True
        elif keep_ and (not ln.startswith("+") or ln.startswith(("+def ", "+class ", "+_"))):
            break
        if keep_:
            seg.append(ln[1:] if ln.startswith("+") else ln)
    nsv = {"torch": torch}
    exec("\\n".join(seg), nsv)
    pv, badv = nsv["nb4_pack_views"], []
    for kk in (3, 4):
        for n in range(1, K100_SEQS + 1):
            flat = torch.arange(n * (2 * kk + 3), dtype=torch.int64)
            a_, b_, c_ = pv(flat, n, kk)
            if not (tuple(a_.shape) == (n, kk + 1) and tuple(b_.shape) == (2, n) and tuple(c_.shape) == (n, kk)
                    and a_.is_contiguous() and b_.is_contiguous() and c_.is_contiguous()
                    and a_.numel() + b_.numel() + c_.numel() == flat.numel()
                    and c_.data_ptr() == flat.data_ptr() + 8 * (n * (kk + 1) + 2 * n)
                    and torch.equal(torch.cat([a_.reshape(-1), b_.reshape(-1), c_.reshape(-1)]), flat)):
                badv.append([n, kk])
    c["pack_views"] = {"cases": 2 * K100_SEQS, "bad": badv[:8]}
    c["ok"] = all(c[k]["ok"] for k in ("Q", "D3s", "D4s")) and not badv
    return c


for _key, _fn, _vk in (("K100_SETUP_CHK", lambda: dict(K100_SETUP, ok=K100_SETUP.get("ok") is True and K100_SETUP.get("sidecar_exists") is True),
                        "k100_setup_scales_sidecar_ok"),
                       ("K100_R9", k100_r9, "k100_r9_q480_dsw_kv_configs_ok"), ("K100_F", k100_f, "k100_dsw_F_10_14"),
                       ("K100_BLOCKDROP", k100_blockdrop, "k100_block_drop_level_for_band_ok"),
                       ("K100_BYTES", k100_bytes, "k100_mtp_vs_drafter_bytes_found"),
                       ("K100_R12", k100_r12, "k100_r12_klee100_draft_unquantized_67"),
                       ("K100_SCAN_MAP", k100_scan_map, "k100_scan_empty_method_map_pass"),
                       ("K100_TOKENIZER", k100_tokenizer, "k100_tokenizer_mask_row_ok"),
                       ("K100_CAPTURES", k100_captures, "k100_captures_argv_packviews_seqs24_ok")):
    K100_OK[_vk] = k100_item(_key, _fn)["ok"]
REP["checks"]["K100_R9_IDX"] = dict(r9k.get("r9_idx") or {"error": "no R9_IDX record"}, record_only=True)
K100_REC["r9_idx_blocks"] = {k: (r9k.get("r9_idx") or {}).get(k) for k in ("q_blocks", "q_idxfp8_blocks", "specs_changed", "error")}
save()
''', "k100 kit items")

    # ---- item 7 (after the revert) + the verdict keys
    REAL = _k100_sub(REAL, '''
verdict({"cell0_identity_submission"''', '''
C7 = {"ran_in_k100": True, "tree": str(tree), "patch_60490_ok": R3.get("ok") is True, "seam_apply_ok": R3S.get("apply_ok") is True,
      "seam_ring": R3S.get("ring"), "seam_combined_valid": (R3S.get("combined") or {}).get("valid"), "seam_pristine_after_revert": R3S.get("pristine"),
      "ppack_after_apply": R3P.get("after_apply"), "ppack_after_revert": R3P.get("after_revert")}
C7["ok"] = (C7["patch_60490_ok"] and C7["seam_apply_ok"] and C7["seam_ring"] is True and C7["seam_combined_valid"] is True
            and C7["seam_pristine_after_revert"] is True and C7["ppack_after_apply"] == "applied" and C7["ppack_after_revert"] == "not_applied")
REP["checks"]["K100_SEAM"] = C7
K100_OK["k100_seam_ppack_60490_apply_verify_revert_ok"] = C7["ok"]
_names = {G["ARMS"][k]["name"]: k for k in ARMS5}
_cli = R4.get("cli_flags") or {}
K100_OK["k100_help_flag_gate_real_ok"] = (R4.get("help_real") is True and set(_cli) >= set(_names)
                                          and all((_cli.get(n) or {}).get("ok") is True for n in _names))
_pl = R4.get("plans") or {}
STEPS = {"merge": bool((R1.get("merge") or {}).get("ok")), "merlin_install": R2.get("ok") is True, "patch_60490": R3.get("ok") is True,
         "k100_setup_sidecar": K100_SETUP.get("ok") is True and K100_SETUP.get("sidecar_exists") is True,
         "sha_gate": bool((R1.get("sha") or {}).get("ok")),
         "ppack_pseq_before_arm_q": (R3P.get("before_arm_q") or [None])[0] is True,
         "seam_sseq_apply": R3S.get("apply_ok") is True,
         "d_arm_sseq_check_pseq_before_arm": all((_pl.get(k) or {}).get("sseq_check", [None])[0] is True
                                                 and (_pl.get(k) or {}).get("pseq_before_arm", [None])[0] is True for k in ARMS5 if k != "Q"),
         "tree_check": all(((_pl.get(k) or {}).get("tree_check") or {}).get("ok") for k in ARMS5),
         "q_i8_draft_head_build_env": (_pl.get("Q") or {}).get("build_env") == ({"DRAFT_HEAD": "int8"} if (_pl.get("Q") or {}).get("q_i8", {}).get("on") else {}),
         "k6_before_launch": all(((_pl.get(k) or {}).get("k6_before_launch") or {}).get("ok") is True for k in ARMS5),
         "host_phase": R4.get("host_phase_ok") is True,
         "help_flag_gate": K100_OK["k100_help_flag_gate_real_ok"],
         "nbdf_launch_check": R4.get("launch_check_ok") is True,
         "qsa_kv_scales_env": all(str((_pl.get(k) or {}).get("qsa_scales_env") or "").endswith("qsa_kv_scales_262k.json") for k in ARMS5),
         "spawn_env": R4.get("spawn_env_ok") is True, "argv_spec_captures_seqs": R5.get("ok") is True}
REP["checks"]["K100_STEPS"] = STEPS
K100_OK["k100_notebook_launch_steps_all_ok"] = all(v is True for v in STEPS.values())
save()
verdict({"cell0_identity_submission"''', "item 7")
    REAL = _k100_sub(REAL, '''"drafter_loaded_weights_ok": (R12.get("json") or {}).get("ok") is True},''',
                     '''"drafter_loaded_weights_ok": (R12.get("json") or {}).get("ok") is True, **K100_OK},''', "verdict keys")
    REAL = _k100_sub(REAL, '''recorded={"patched_model_import": R11.get("json"),''',
                     '''recorded={**K100_REC, "patched_model_import": R11.get("json"),''', "verdict recorded")


KITS = {"units": {"body": UNITS, "title": "Q27 E27 NBDF K100 Kit Units" if K100 else "Q27 E27 NBDF Kit Units", "models": [], "datasets": []},
        "real": {"body": REAL, "title": "Q27 E27 NBDF K100 Kit Real" if K100 else "Q27 E27 NBDF Kit Real", "models": MODELS,
                 "datasets": list(FACTS.get("dataset_sources") or [])}}


def build(name):
    spec = KITS[name]
    code = HEAD + spec["body"]
    for k, v in (("__EXPORTBLOB__", EXPORT_BLOB), ("__UNITSBLOB__", UNITS_BLOB), ("__FIXBLOB__", FIXTURE_BLOB), ("__NBSHA__", NB_SHA),
                 ("__NAME__", name), ("__SRCSHAS__", json.dumps(SRC_SHAS)), ("__KITARMS__", json.dumps(list(KIT_ARMS))),
                 ("__QBLOCKS__", str(KV_Q_BLOCKS)), ("__QPIN__", str(KV_Q_PIN)), ("__DPIN__", str(KV_D_PIN)),
                 ("__VARIANT__", "k100" if K100 else "w4")):
        code = code.replace(k, v)
    compile(code, f"<kit {name}>", "exec")
    slug = f"q27-e27-nbdf-k100-kit-{name}" if K100 else f"q27-e27-nbdf-kit-{name}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert len(data) < 400_000, (name, len(data))
    assert b"\r" not in data.replace(b"\\r", b"")
    out = ROOT / f"{slug}__{OWNER}"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.py"):
        assert f.name not in ("argparse.py", "json.py", "typing.py"), f
    (out / f"{slug}.ipynb").write_bytes(data)
    meta = {"id": f"{OWNER}/{slug}", "title": spec["title"], "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": spec["datasets"], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": spec["models"], "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| NBDF", NB_SHA[:8], flush=True)
    return out, code


if __name__ == "__main__":
    for n in KITS:
        build(n)
