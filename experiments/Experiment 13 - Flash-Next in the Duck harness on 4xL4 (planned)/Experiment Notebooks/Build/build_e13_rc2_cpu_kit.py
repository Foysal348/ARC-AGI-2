"""E13 RC2 Kaggle CPU kit (user 2026-10-31: "CPU first"): on the pinned Py3.12 image, no GPU, foysalemonshanto.

Real checks (the parts the local harness cannot do):
  R1 the RC2 notebook's own code on the REAL mounts (RC2_UNIT + RC2_CPU_KIT): inputs at the exact paths, the merge (sizes),
     the full background sha256 of all 27 merged files (~124 GB), the prompt render + pinned hashes;
  R2 the real Merlin offline install (merlin_install());
  R3 merlin_fn_server() for BF16 and FP8 = Merlin build() + host_phase() (now fatal in RC2b) + `vllm serve --help=all`
     against every flag of the final argv; no server is started.
Harness checks on this image: rc2_unit_tests.py (34 fault injections) and three fake-server flows (main, no-Merlin,
rerun) on a fake input tree made from the real part-1 small files.
Verdict ALL OK = R1 inputs + merge + sha256 + R2 install + R3 host_phase both + units 34/34 + 3 flows. help_check is
reported (a CPU-only vLLM may not print --help=all: recorded, not part of ALL OK).

Run: python build_e13_rc2_cpu_kit.py -> D:/ARC AGI 2/_scratch/push_e13/q27-e13-rc2-cpu-kit__foysalemonshanto/
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E13 = HERE.parents[1]
E23 = Path(r"D:/ARC AGI 2/experiments/Experiment 23 - Six levers vLLM 0.32 (planned)")
EXPORT = (HERE / "rc2_code_export.py").read_text(encoding="utf-8")
UNITS = (HERE / "rc2_unit_tests.py").read_text(encoding="utf-8")
FAKESRV = (E23 / "src" / "harness" / "fake_vllm_server.py").read_text(encoding="utf-8")
PARTS = (E13 / "prep" / "merlin_fn_ckpt" / "kernels" / "parts.json").read_text(encoding="utf-8")
NB_SHA = hashlib.sha256((HERE.parent / "Q27-XHIGH-E13_FLASHNEXT_RC2.ipynb").read_bytes()).hexdigest()
OUT = Path(r"D:/ARC AGI 2/_scratch/push_e13/q27-e13-rc2-cpu-kit__foysalemonshanto")
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

MAKE_FAKE = r'''
import hashlib, json, shutil, sys
from pathlib import Path
INP_REAL, INP = Path(sys.argv[1]), Path(sys.argv[2])
PARTS = json.loads(Path(sys.argv[3]).read_text())
MD_R = INP_REAL / "models/foysalemonshanto"
P1 = MD_R / "qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1"
shutil.rmtree(INP, ignore_errors=True)
MD = INP / "models/foysalemonshanto"
PD = {k: MD / f"qwen3-8-flash-next-w4a16-merlin-p{k}/transformers/default/1" for k in range(1, 6)}
d1 = PD[1]; d1.mkdir(parents=True)
man_real = json.loads((P1 / "q27_fn_merlin_part1_manifest.json").read_text())
small = {n: v for n, v in man_real["files"].items() if not n.endswith(".safetensors")}
for n in small:
    shutil.copyfile(P1 / n, d1 / n)
idx = json.loads((d1 / "model.safetensors.index.json").read_text())
ple = set(PARTS["ple_from_overlay"])
shards = sorted(set(idx["weight_map"].values()) - ple - set(small))
def fake(path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (n * 997).to_bytes(8, "little") * 16
    path.write_bytes(data)
    return [len(data), hashlib.sha256(data).hexdigest()]
groups = [shards[i::5] for i in range(5)]
man1 = dict(man_real, files=dict(small))
for i, n in enumerate(groups[0]):
    man1["files"][n] = fake(d1 / n, i)
(d1 / "q27_fn_merlin_part1_manifest.json").write_text(json.dumps(man1))
for k in range(2, 6):
    man = {"what": f"fake part {k}", "part": k, "parts": 5, "model_subdir": ".",
           "files": {n: fake(PD[k] / n, 100 * k + i) for i, n in enumerate(groups[k - 1])}}
    (PD[k] / f"q27_fn_merlin_part{k}_manifest.json").write_text(json.dumps(man))
pd = MD / "qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1"
for i, n in enumerate(sorted(ple)):
    fake(pd / n, 9000 + i)
mp = MD / "e23-merlin-env/other/v1/1/merlin_offline_tar.bin"
mp.parent.mkdir(parents=True); mp.write_bytes(b"")
c = INP / "competitions/arc-prize-2026-arc-agi-2"; c.mkdir(parents=True)
ev = INP_REAL / "competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
shutil.copyfile(ev, c / "arc-agi_evaluation_challenges.json"); shutil.copyfile(ev, c / "arc-agi_test_challenges.json")
# the no-Merlin variant: part 3 missing, an Intel tokenizer (template only in tokenizer_config), a fake wheelhouse
nm = Path(str(INP) + "_nomerlin"); shutil.rmtree(nm, ignore_errors=True); shutil.copytree(INP, nm)
shutil.rmtree(nm / "models/foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p3")
intel = nm / "models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"; intel.mkdir(parents=True)
for n in ("tokenizer.json", "tokenizer_config.json", "config.json"):
    shutil.copyfile(d1 / n, intel / n)
tc = json.loads((intel / "tokenizer_config.json").read_text()); tc["chat_template"] = (d1 / "chat_template.jinja").read_text()
(intel / "tokenizer_config.json").write_text(json.dumps(tc))
wh = nm / "datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"; wh.mkdir(parents=True)
(wh / "install_wheelhouse.py").write_text("print('fake')\n")
print("FAKE_TREE_OK shards", len(shards), "small", len(small))
'''

CODE = r'''# E13 RC2 CPU kit: real mounts + real Merlin install / build / host_phase / --help, then the RC2 harness on this image
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib
from pathlib import Path
T0 = time.time()
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")          # env overrides: local dry run only
K = Path(os.environ.get("KIT_TMP", "/tmp/rc2kit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": "E13 RC2 CPU kit", "notebook_sha256": "__NBSHA__", "checks": {}}
def save():
    (W / "rc2_cpu_kit_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
for name, src in (("rc2_code_export.py", __EXPORT__), ("rc2_unit_tests.py", __UNITS__), ("fake_vllm_server.py", __FAKESRV__),
                  ("make_fake.py", __MAKEFAKE__), ("parts.json", __PARTS__)):
    (K / name).write_text(src, encoding="utf-8")
REP["checks"]["export_sha256"] = hashlib.sha256((K / "rc2_code_export.py").read_bytes()).hexdigest()
REP["python"] = sys.version

# ---- R1-R3: the notebook code on the real mounts (no GPU)
R1 = {}
try:
    os.environ.update(RC2_UNIT="1", RC2_CPU_KIT="1", E23_OUT=str(K / "real_out"), E23_TMP=str(K / "real_tmp"),
                      RC2_SUBMISSION=str(K / "real_out/submission.json"))
    os.environ.pop("E23_FAKE_SERVER", None); os.environ["E23_INPUT"] = INPUT_ROOT
    (K / "real_tmp").mkdir()
    t = time.time()
    G = runpy.run_path(str(K / "rc2_code_export.py"), run_name="rc2real")["gated"].__globals__
    R = G["R"]
    R1.update(load_sec=round(time.time() - t), inputs=R["inputs"], merge=R["sections"]["A0"]["merge"],
              prompt_render={k: v for k, v in R["prompt_render"].items() if "text" not in k}, host=R["sections"]["A0"]["host"],
              host_mem=R["sections"]["A0"]["host_mem"])
    log("R1 inputs", json.dumps(R["inputs"])[:600]); log("R1 merge", json.dumps(R1["merge"])[:600])
    REP["checks"]["R1"] = R1; save()
    t = time.time()
    try:
        eng = G["merlin_install"]()
        R1["install"] = {"ok": True, "sec": round(time.time() - t), "kv": {k: v for k, v in eng["kv"].items()}, "cuda_home": eng["cuda_home"]}
    except Exception as e:
        R1["install"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:800], "tb": traceback.format_exc()[-1500:]}
    log("R2 install", json.dumps(R1["install"])[:600]); save()
    R1["boots"] = {}
    for name, kv, port, kvb in (("kit_bf16", "bf16", 8000, G["KV_BF16"]), ("kit_fp8", "fp8", 8001, G["KV_FP8"])):
        b = {}
        try:
            s = G["merlin_fn_server"](name, kv, port, kvb)
            b.update(ok=True, host_phase=G["HOST_PHASE"].get(name), help=G["HELP"].get(name), argv=s.argv,
                     plan=R["setup"].get("merlin_plans", {}).get(name), server_started=s.proc is not None)
        except Exception as e:
            b.update(ok=False, error=f"{type(e).__name__}: {e}"[:1200], host_phase=G["HOST_PHASE"].get(name),
                     tb=traceback.format_exc()[-1500:])
        R1["boots"][name] = b
        log("R3", name, json.dumps({k: v for k, v in b.items() if k not in ("argv", "plan", "tb")}, default=str)[:900]); save()
    t = time.time()
    ok, why = G["sha_gate"](4 * 3600)
    R1["sha"] = {"ok": ok, "why": why, "wait_sec": round(time.time() - t), "detail": R["setup"].get("sha256")}
    log("R1 sha", json.dumps(R1["sha"])[:600])
except Exception as e:
    R1["fatal"] = f"{type(e).__name__}: {e}"[:1000]; R1["tb"] = traceback.format_exc()[-2500:]
    log("R1 FATAL", R1["fatal"])
REP["checks"]["R1"] = R1; save()

# ---- harness on this image: fake tree from the real part-1 small files, units, three flows
H = {}
def sh(name, args, env=None, timeout=3600):
    t = time.time()
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})}, cwd=str(K))
    (W / f"kit_{name}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
    return {"rc": r.returncode, "sec": round(time.time() - t), "tail": (r.stdout + r.stderr)[-1500:]}
for k in ("RC2_UNIT", "RC2_CPU_KIT", "E23_OUT", "E23_TMP", "RC2_SUBMISSION", "E23_INPUT"):
    os.environ.pop(k, None)
H["fake_tree"] = sh("fake_tree", [sys.executable, str(K / "make_fake.py"), INPUT_ROOT, str(K / "fin"), str(K / "parts.json")])
H["units"] = sh("units", [sys.executable, str(K / "rc2_unit_tests.py"), str(K / "fin"), str(K / "unit")])
try:
    H["unit_results"] = json.loads((K / "unit" / "unit_results.json").read_text())
except Exception as e:
    H["unit_results"] = repr(e)
log("units", H["units"]["rc"], H["units"]["tail"][-300:]); REP["checks"]["harness"] = H; save()
H["flows"] = {}
for case, inp, extra in (("main", "fin", {}), ("nomerlin", "fin_nomerlin", {}), ("rerun", "fin", {"KAGGLE_IS_COMPETITION_RERUN": "1"})):
    o = K / f"out_{case}"; shutil.rmtree(o, ignore_errors=True); o.mkdir(); (K / f"tmp_{case}").mkdir(exist_ok=True)
    env = dict(E23_FAKE_SERVER=str(K / "fake_vllm_server.py"), FAKE_SPEED="20000", E23_OUT=str(o), E23_TMP=str(K / f"tmp_{case}"),
               E23_INPUT=str(K / inp), RC2_SUBMISSION=str(o / "submission.json"), RC2_SHM=str(K / f"shm_{case}"),
               RC2_BUDGET_SEC="3000", E23_WIN_SEC="5", E23_EST_SCALE="0.01", **extra)
    (K / f"shm_{case}").mkdir(exist_ok=True)
    r = sh("flow_" + case, [sys.executable, str(K / "rc2_code_export.py")], env, 2400)
    try:
        Rf = json.loads((o / "rc2_results.json").read_text()) if case != "rerun" else {}
        r["steps"] = {f"{s}/{n}": [v.get("status"), v.get("correctness"), v.get("fp8_gate")] for s, st in Rf.get("sections", {}).items()
                      for n, v in st.items() if isinstance(v, dict) and "status" in v}
        r["source"] = (Rf.get("prompt_render") or {}).get("source")
        r["submission_tasks"] = len(json.loads((o / "submission.json").read_text())) if (o / "submission.json").exists() else None
    except Exception as e:
        r["parse_error"] = repr(e)
    H["flows"][case] = r
    log("flow", case, json.dumps({k: v for k, v in r.items() if k != "tail"})[:700]); save()
REP["checks"]["harness"] = H

# ---- verdict
B = R1.get("boots") or {}
m = H["flows"]
ok = {
    "inputs_found": bool((R1.get("inputs") or {}).get("merlin_inputs_ok")),
    "merge_ok": bool((R1.get("merge") or {}).get("ok")),
    "sha256_all_ok": bool((R1.get("sha") or {}).get("ok")),
    "prompt_hashes_ok": bool((R1.get("prompt_render") or {}).get("hashes_ok")),
    "merlin_install_ok": bool((R1.get("install") or {}).get("ok")),
    "host_phase_bf16_ok": (B.get("kit_bf16") or {}).get("host_phase") == "ok",
    "host_phase_fp8_ok": (B.get("kit_fp8") or {}).get("host_phase") == "ok",
    "no_server_started": not any((b or {}).get("server_started") for b in B.values()),
    "units_34_of_34": H["units"]["rc"] == 0 and isinstance(H["unit_results"], list) and len(H["unit_results"]) == 34
                      and all(x["ok"] for x in H["unit_results"]),
    "flow_main": m["main"]["rc"] == 0 and m["main"].get("steps", {}).get("A3/merlin_fp8", [None])[0] == "DONE"
                 and m["main"]["steps"].get("A1/merlin_bf16", [None, None])[1] == "CLEAN",
    "flow_nomerlin": m["nomerlin"]["rc"] == 0 and m["nomerlin"].get("source") == "intel_fallback"
                     and m["nomerlin"].get("steps", {}).get("F/stock_fn", [None, None])[1] == "CLEAN",
    "flow_rerun": m["rerun"]["rc"] == 0 and m["rerun"].get("submission_tasks") == 120,
}
help_ = {n: (b or {}).get("help") for n, b in B.items()}
REP["verdict"] = {"checks": ok, "ALL_OK": all(ok.values()), "help_check (reported)": help_, "minutes": round((time.time() - T0) / 60, 1)}
save()
print("VERDICT", json.dumps(REP["verdict"], default=str), flush=True)
print("ALL OK" if REP["verdict"]["ALL_OK"] else "NOT ALL OK", flush=True)
'''

code = CODE
for k, v in (("__EXPORT__", json.dumps(EXPORT)), ("__UNITS__", json.dumps(UNITS)), ("__FAKESRV__", json.dumps(FAKESRV)),
             ("__MAKEFAKE__", json.dumps(MAKE_FAKE)), ("__PARTS__", json.dumps(PARTS)), ("__NBSHA__", NB_SHA)):
    code = code.replace(k, v)
compile(code, "<kit>", "exec")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.mkdir(parents=True, exist_ok=True)
slug = "q27-e13-rc2-cpu-kit"
data = json.dumps(nb, indent=1)
assert len(data.encode()) < 400_000, len(data)
(OUT / f"{slug}.ipynb").write_text(data, encoding="utf-8", newline="\n")
(OUT / "kernel-metadata.json").write_text(json.dumps({
    "id": f"foysalemonshanto/{slug}", "title": "Q27 E13 RC2 CPU Kit", "code_file": f"{slug}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
    "model_sources": [f"foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p{k}/Transformers/default/1" for k in range(1, 6)] +
                     ["foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
                      "foysalemonshanto/e23-merlin-env/Other/v1/1"],
    "docker_image": IMAGE, "machine_shape": "None"}, indent=2), encoding="utf-8", newline="\n")
print("kit", OUT.name, f"{len(data) / 1024:.0f} kB", "| RC2 notebook", NB_SHA[:8],
      "| kit sha", hashlib.sha256(data.encode()).hexdigest()[:8])
