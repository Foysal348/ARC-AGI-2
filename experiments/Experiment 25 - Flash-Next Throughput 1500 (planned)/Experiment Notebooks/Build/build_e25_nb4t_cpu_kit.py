"""E25 NB4T Kaggle CPU kits (pinned Py3.12 image, no GPU, no internet, private; account foysalemonshanto). Pattern: the NB4S
kits (build_e25_nb4s_cpu_kit.py). Each kernel is fail-soft, writes its JSON report and prints exactly one line
`@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`.

  units  q27-e25-nb4t-kit-units: nb4t_unit_tests.py on Py3.12 (T1-T10: arm plan + launch check incl. the real NB4S QD launch,
         env / config read-back on the real NB4S QD log lines, capacity facts, parity-aware comparison, speed / budget,
         verdicts incl. the real NB4S records re-judged, PatchSeq + the REAL nb4_ppack apply / revert on copies of Merlin
         ad42e37's three files, the CELL flow with a stub boot runner, notebook facts, the rerun -> identity submission) +
         nb4_ppack_tests.py (rev 2). The export is rebuilt in the kit from the notebook cells (the same bytes, half the blob).
  real   q27-e25-nb4t-kit-real (the 7 RC2 Models + ARC): NB4T's own code (RC2_UNIT + RC2_CPU_KIT) on the REAL mounts: CELL 0
         (the identity submission of the real test file), inputs, merge, full sha256, prompts; the real Merlin offline install;
         #60490 + the worker extension on the INSTALLED tree; nb4_ppack rev 2 applied; launch.build() + host_phase() + the NB4T
         launch check for ALL 5 arms (Q / I8 / P26 / P27 / C12) on the patched tree, each plan's env read back in a spawned child
         of a venv process; the argv fields of every arm (I8: Merlin's own build() puts "draft_lm_head_quant":"int8" into
         --speculative-config; C12: the last --max-num-seqs is 12; P26 / P27: the argv equals Q's, the env carries the
         partition); every PLE layer on PP rank 0 for 25,23 / 26,22 / 27,21 (the merged checkpoint's config.json); the
         installed Merlin launch.py / draft_lm_head_quant.py / speculator.py == the source the flags were read from; the
         extension (nb4s_env) in the venv; nb4_ppack reverted (pre-shas verified), #60490 / extension restored; host gates.
         Recorded only (this CPU box cannot import vllm._C, as in the NB4S kits): vLLM's own argparse, `--help=all`, the
         #60490 qsa import, the QuantizedDraftLMHead import.

Run: python build_e25_nb4t_cpu_kit.py -> D:/ARC AGI 2/_scratch/push_e25/q27-e25-nb4t-kit-{units,real}__foysalemonshanto/
"""
import base64
import hashlib
import json
import lzma
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nb4_ppack as PPK  # noqa: E402  (the patched-file list: PPK.PATCHES)
NB_PATH = HERE.parent / "Q27-XHIGH-E25_NB4T_LEVERS.ipynb"
NB_SHA = hashlib.sha256(NB_PATH.read_bytes()).hexdigest()
FACTS = json.loads((HERE / "nb4t_build_facts.json").read_text(encoding="utf-8"))
assert FACTS["sha256"] == NB_SHA, "nb4t_build_facts.json is not for this notebook: rebuild"
EXPORT = (HERE / "nb4t_code_export.py").read_text(encoding="utf-8")
_nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
assert "\n\n".join("".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code") == EXPORT, "export != notebook cells"
MERLIN_SRC = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
NB1_RESULTS = Path(r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb1-capacity-r2/nb1/nb1_results.json")
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
OWNER = "foysalemonshanto"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
RC2_MODELS = ([f"foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p{k}/Transformers/default/1" for k in range(1, 6)] +
              ["foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
               "foysalemonshanto/e23-merlin-env/Other/v1/1"])
SRC_FILES = ("tools/merlin/launch.py", "vllm/model_executor/layers/draft_lm_head_quant.py",
             "vllm/v1/worker/gpu/spec_decode/speculator.py", "vllm/distributed/utils.py")
SRC_SHAS = {rel: hashlib.sha256((MERLIN_SRC / rel).read_bytes()).hexdigest() for rel in SRC_FILES}


def blob(files):
    """{relative path: text} -> lzma + base64 (keeps the kit < 400 kB)."""
    return base64.b64encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


_R = json.loads(NB1_RESULTS.read_text(encoding="utf-8"))
_mp = _R["setup"]["merlin_plans"]["nb1_b3_fp8_pin100"]
NB1_B3_PLAN = {"setup": {"merlin_plans": {"nb1_b3_fp8_pin100": {"env_keys": _mp["env_keys"], "argv_tail": _mp["argv_tail"]}}}}
UNIT_FILES = {f"build/{n}": (HERE / n).read_text(encoding="utf-8") for n in
              ("nb4t_lib.py", "nb4t_unit_tests.py", "nb4s_lib.py", "nb4_timeline.py", "nb4_ppack.py", "nb4_ppack_tests.py",
               "nb4t_build_facts.json", "nb4t_fixture_nb4s_gpu_2026-11-01.json")}
UNIT_FILES["Q27-XHIGH-E25_NB4T_LEVERS.ipynb"] = NB_PATH.read_text(encoding="utf-8")
UNIT_FILES["nb1_b3_plan.json"] = json.dumps(NB1_B3_PLAN)
for rel, (pre, *_r) in PPK.PATCHES.items():                 # rev 2: pp_utils.py, model_runner.py, gpu_worker.py
    UNIT_FILES["merlin/" + rel] = (MERLIN_SRC / rel).read_bytes().decode("utf-8")
    assert hashlib.sha256(UNIT_FILES["merlin/" + rel].encode("utf-8")).hexdigest() == pre, rel
for k, v in UNIT_FILES.items():
    assert "\r" not in v, f"CR in {k}"
UNITS_BLOB = blob(UNIT_FILES)
EXPORT_BLOB = blob({"nb4t_code_export.py": EXPORT})

HEAD = r'''# E25 NB4T CPU kit __NAME__ (pinned image, no GPU): see the builder docstring for the checks and the verdict rule
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib, re, base64, lzma
from pathlib import Path
T0 = time.time()
KIT = "__NAME__"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")          # env overrides: local dry run only
K = Path(os.environ.get("KIT_TMP", "/tmp/nb4tkit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": f"E25 NB4T CPU kit {KIT}", "notebook_sha256": "__NBSHA__", "python": sys.version, "checks": {}}
def save():
    (W / f"nb4t_kit_{KIT}_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def unpack(b64):
    return json.loads(lzma.decompress(base64.b64decode(b64)).decode("utf-8"))
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
# ---------------------------------------------------------------- U1: nb4t_unit_tests.py on this image (Py3.12)
U1 = {}
try:
    F = unpack("__UNITSBLOB__")
    B = K / "kit"
    for rel, txt in F.items():
        (B / rel).parent.mkdir(parents=True, exist_ok=True)
        (B / rel).write_text(txt, encoding="utf-8", newline="\n")
    nbk = json.loads(F["Q27-XHIGH-E25_NB4T_LEVERS.ipynb"])
    (B / "build/nb4t_code_export.py").write_text("\n\n".join("".join(c["source"]) for c in nbk["cells"] if c["cell_type"] == "code"),
                                                 encoding="utf-8", newline="\n")
    U1["files"] = {rel: hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12] for rel, txt in F.items()}
    U1["notebook_sha_in_kit"] = hashlib.sha256(F["Q27-XHIGH-E25_NB4T_LEVERS.ipynb"].encode("utf-8")).hexdigest()
    for k_ in list(os.environ):
        if k_.startswith(("NB1_", "E23_", "RC2_", "NB4_", "Q27X_")) or k_ in ("KAGGLE_IS_COMPETITION_RERUN", "CUDA_DEVICE_MAX_CONNECTIONS",
                                                                             "VLLM_PP_LAYER_PARTITION", "DRAFT_HEAD"):
            os.environ.pop(k_, None)
    r = sh("units", [sys.executable, str(B / "build/nb4t_unit_tests.py"), "--merlin-src", str(B / "merlin"),
                     "--fixture", str(B / "build/nb4t_fixture_nb4s_gpu_2026-11-01.json"), "--nb1-results", str(B / "nb1_b3_plan.json")],
           timeout=1800, cwd=B / "build")
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
    U1["units"] = {"rc": r["rc"], "sec": r["sec"], "summary": m.group(0) if m else None, "fails": [l[:300] for l in lines if l.startswith("FAIL ")][:40],
                   "skips": [l for l in lines if l.startswith("SKIP ")]}
    U1["units"]["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0
    U1["units"]["real_patch_ran"] = not any("SKIP T7" in l for l in lines)
    U1["units"]["real_fixture_ran"] = not any("SKIP fixture" in l or "SKIP T1 real NB4S" in l or "SKIP T6 real" in l for l in lines)
    log("units", json.dumps(U1["units"])[:1500])
    r = sh("ppack", [sys.executable, str(B / "build/nb4_ppack_tests.py"), "--merlin-src", str(B / "merlin")], timeout=1800, cwd=B / "build")
    lines = r["_out"].splitlines()
    js = re.search(r'"checks": (\d+),\s*"pass": (\d+),\s*"fail": (\d+)', r["_out"])
    U1["ppack"] = {"rc": r["rc"], "sec": r["sec"], "summary": js.group(0).replace(chr(10), " ") if js else None,
                   "fails": [l[:300] for l in lines if l.startswith("FAIL ")][:40]}
    U1["ppack"]["all_pass"] = bool(js) and js.group(1) == js.group(2) and js.group(3) == "0" and r["rc"] == 0 and "ALL PASS" in r["_out"]
    log("ppack", json.dumps(U1["ppack"])[:1500])
except Exception as e:
    U1["fatal"] = f"{type(e).__name__}: {e}"[:800]; U1["tb"] = traceback.format_exc()[-1500:]
REP["checks"]["U1"] = U1
verdict({"units_all_pass": (U1.get("units") or {}).get("all_pass") is True,
         "real_ppack_apply_revert_ran": (U1.get("units") or {}).get("real_patch_ran") is True,
         "real_nb4s_fixture_ran": (U1.get("units") or {}).get("real_fixture_ran") is True,
         "kit_notebook_is_the_built_one": U1.get("notebook_sha_in_kit") == "__NBSHA__",
         "ppack_tests_all_pass": (U1.get("ppack") or {}).get("all_pass") is True},
        recorded={"units": (U1.get("units") or {}).get("summary"), "skips": (U1.get("units") or {}).get("skips"),
                  "ppack": (U1.get("ppack") or {}).get("summary")})
'''

REAL = r'''
# ---------------------------------------------------------------- R1: NB4T code on the real mounts (no boot); CELL 0 first
FILES = unpack("__EXPORTBLOB__")
(K / "nb4t_code_export.py").write_text(FILES["nb4t_code_export.py"], encoding="utf-8", newline="\n")
REP["export_sha256"] = hashlib.sha256(FILES["nb4t_code_export.py"].encode("utf-8")).hexdigest()
ARMS5 = ("Q", "I8", "P26", "P27", "C12")
SRC_SHAS = json.loads(r"""__SRCSHAS__""")
R0, R1, R2, R3, R3P, R4, R5, R6, R7, R8, R9 = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
G = None
try:
    os.environ.update(RC2_UNIT="1", RC2_CPU_KIT="1", E23_OUT=str(K / "out"), E23_TMP=str(K / "tmp"), E23_INPUT=INPUT_ROOT,
                      RC2_SHM=str(K / "shm"), Q27X_WORKING=str(K / "work0"))
    for k_ in ("E23_FAKE_SERVER", "KAGGLE_IS_COMPETITION_RERUN", "DRAFT_HEAD", "VLLM_PP_LAYER_PARTITION", "NB4_PPDEFER", "NB4_PPACK"):
        os.environ.pop(k_, None)
    for d in ("out", "tmp", "shm", "work0"):
        (K / d).mkdir(exist_ok=True)
    t = time.time()
    G = runpy.run_path(str(K / "nb4t_code_export.py"), run_name="nb4tkit")["nb4t_main"].__globals__
    R = G["R"]
    sub = json.loads((K / "work0" / "submission.json").read_text())
    tc = G["test_challenges"]
    R0.update(tasks=len(tc), sub_tasks=len(sub), arc_root=str(G["ARC_ROOT"]),
              identity=all(len(sub[t_]) == len(v["test"]) and all(a["attempt_1"] == q["input"] == a["attempt_2"]
                                                                 for a, q in zip(sub[t_], v["test"])) for t_, v in tc.items()) and set(sub) == set(tc))
    R0["ok"] = R0["identity"] and R0["tasks"] > 0
    R1.update(load_sec=round(time.time() - t), inputs=R["inputs"], merge=R["sections"]["A0"]["merge"],
              rc2_oracle_prompts_match=G["NB1"].get("rc2_oracle_prompts_match"), rc2_rows_loaded=sorted(k for k in G["ROWS"] if k.startswith("rc2_")),
              arms=list(G["ARMS"]), embed_shas={k: G["NB4T"].get(k) for k in ("lib_sha256", "timeline_sha256", "ppack_sha256")},
              prompts={b: len(v) for b, v in G["PROMPTS"].items()})
    log("R0 CELL 0", json.dumps(R0)); log("R1 inputs", json.dumps(R["inputs"])[:500]); log("R1 merge", json.dumps(R1["merge"])[:300])
    save()
    t = time.time()
    ok, why = G["sha_gate"](4 * 3600)
    R1["sha"] = {"ok": ok, "why": why, "wait_sec": round(time.time() - t), "detail": R["setup"].get("sha256")}
    log("R1 sha", json.dumps(R1["sha"])[:300])
    ext = G["ensure_prompts"](12)
    R1["prompts_12"] = {"lens": ext, "distinct_8k": len({json.dumps(p) for p in G["PROMPTS"]["8k"]}),
                        "distinct_40k": len({json.dumps(p) for p in G["PROMPTS"]["40k"]}),
                        "len_8k": sorted({len(p) for p in G["PROMPTS"]["8k"]}), "len_40k": sorted({len(p) for p in G["PROMPTS"]["40k"]})}
    R1["prompts_12_ok"] = (ext == {"8k": 12, "40k": 12} and R1["prompts_12"]["distinct_8k"] == 12 and R1["prompts_12"]["distinct_40k"] == 12
                           and R1["prompts_12"]["len_8k"] == [8192] and R1["prompts_12"]["len_40k"] == [40000])
except Exception as e:
    R1["fatal"] = f"{type(e).__name__}: {e}"[:1000]; R1["tb"] = traceback.format_exc()[-2500:]
    log("R1 FATAL", R1["fatal"])
REP["checks"]["R0"] = R0; REP["checks"]["R1"] = R1; save()

# ---------------------------------------------------------------- R2: Merlin offline install + the flag sources
eng = None
try:
    t = time.time()
    eng = G["merlin_install"]()
    R2.update(ok=True, sec=round(time.time() - t), kv=dict(eng["kv"]), cuda_home=eng["cuda_home"])
    tree0 = Path(eng["kv"]["MERLIN_TREE"])
    R2["src_shas"] = {rel: hashlib.sha256((tree0 / rel).read_bytes()).hexdigest() if (tree0 / rel).is_file() else None for rel in SRC_SHAS}
    R2["src_equal"] = R2["src_shas"] == SRC_SHAS
except Exception as e:
    R2.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R2 install", json.dumps({k: v for k, v in R2.items() if k != "tb"})[:900])
REP["checks"]["R2"] = R2; save()


def venv_py(code, args=(), name="probe", timeout=900, env_extra=None, base_env=None):
    """A probe inside the Merlin venv with the tree on PYTHONPATH (what Merlin's build() exports for the server)."""
    tree, py = eng["kv"]["MERLIN_TREE"], eng["kv"]["MERLIN_VENV"] + "/bin/python"
    env = dict(base_env) if base_env is not None else {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_PRELOAD")}
    if base_env is None:
        env.update(PYTHONPATH=tree, CUDA_HOME=eng["cuda_home"], VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   PATH=f"{eng['kv']['MERLIN_VENV']}/bin:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}")
        if eng["kv"].get("MERLIN_SHIM"):
            env["LD_PRELOAD"] = eng["kv"]["MERLIN_SHIM"]
    env.update(env_extra or {})
    (K / f"{name}.py").write_text(code)
    try:
        r = subprocess.run([py, str(K / f"{name}.py"), *args], capture_output=True, text=True, timeout=timeout, env=env, cwd=str(K))
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out)
    js = next((json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("@JSON ")), None)
    return {"rc": rc, "json": js, "tail": out[-1200:]}

IMPORT_QSA = "import json\ntry:\n    import vllm.models.qwen4_exp.nvidia.qsa as m\n    print('@JSON ' + json.dumps({'ok': True, 'file': m.__file__, 'has_topk': hasattr(m, '_TOPK_WORKSPACE_BYTES')}))\nexcept BaseException as e:\n    print('@JSON ' + json.dumps({'ok': False, 'error': repr(e)[:400]}))\n"
IMPORT_PP = r"""
import json
out = {}
try:
    import vllm.v1.worker.gpu.pp_utils as p
    out.update(ok=True, pp=p.__file__, pack_views=hasattr(p, "nb4_pack_views"), handler=hasattr(p, "PPHandler"),
               flush=hasattr(getattr(p, "PPHandler", None), "flush_deferred_receive"),
               final=hasattr(getattr(p, "PPHandler", None), "nb4_log_final"),
               stat_keys=list(getattr(p, "_NB4_STAT_KEYS", ())))
except BaseException as e:
    out.update(ok=False, error=repr(e)[:600])
try:
    import inspect
    import vllm.v1.worker.gpu_worker as w
    out.update(gw_ok=True, gw=w.__file__, gw_profile_flush='_nb4_flush("rpc")' in inspect.getsource(w.Worker.profile),
               gw_shutdown_flush='_nb4_flush("rpc")' in inspect.getsource(w.Worker.shutdown))
except BaseException as e:
    out.update(gw_ok=False, gw_error=repr(e)[:400])
print("@JSON " + json.dumps(out))
"""
SPAWN_ENV = r"""
import json, multiprocessing as mp, os, sys
KEYS = sys.argv[1].split(",")
def child(q):
    q.put({k: os.environ.get(k) for k in KEYS})
if __name__ == "__main__":
    ctx = mp.get_context("spawn")              # vLLM's multiproc executor spawns its workers
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(q,))
    p.start(); r = q.get(timeout=120); p.join(60)
    print("@JSON " + json.dumps({"parent": {k: os.environ.get(k) for k in KEYS}, "child": r}))
"""
HEADPROBE = r"""
import json
out = {}
try:
    import vllm.model_executor.layers.draft_lm_head_quant as m
    out.update(import_ok=True, file=m.__file__, quant_types=sorted(getattr(m, "_QUANT_TYPES", {})), group=getattr(m, "_GROUP_SIZE", None))
except BaseException as e:
    out.update(import_ok=False, error=repr(e)[:400])
try:
    from vllm.config.speculative import SpeculativeConfig
    import typing
    out["spec_field"] = str(typing.get_type_hints(SpeculativeConfig).get("draft_lm_head_quant"))
except BaseException as e:
    out["spec_field_error"] = repr(e)[:300]
try:
    import torch
    out["gptq_marlin_gemm_op"] = hasattr(torch.ops._C, "gptq_marlin_gemm")
    out["gptq_marlin_repack_op"] = hasattr(torch.ops._C, "gptq_marlin_repack")
except BaseException as e:
    out["ops_error"] = repr(e)[:300]
print("@JSON " + json.dumps(out, default=str))
"""
PPIDX = r"""
import json, os, sys
out = {}
for part in sys.argv[1].split(";"):
    os.environ["VLLM_PP_LAYER_PARTITION"] = part
    try:
        import importlib, vllm.envs as E
        importlib.reload(E)
        from vllm.distributed import utils as U
        importlib.reload(U)
        out[part] = [U.get_pp_indices(48, r, 2) for r in (0, 1)]
    except BaseException as e:
        out[part] = {"error": repr(e)[:300]}
print("@JSON " + json.dumps(out))
"""

# ---------------------------------------------------------------- R3: #60490 + the extension on the INSTALLED tree
orig, qsa_file, tree = None, None, None
try:
    tree = Path(eng["kv"]["MERLIN_TREE"])
    qsa_file = tree / G["P60490_REL"]
    orig = qsa_file.read_bytes()
    R3["pre_sha"] = hashlib.sha256(orig).hexdigest()
    rec = G["apply_60490"]()
    R3["apply"] = {k: rec.get(k) for k in ("applied", "sha_after", "marker", "py_compile", "vllm_origin", "worker_ext")}
    R3["import_after_recorded"] = venv_py(IMPORT_QSA, name="r3_import_after")          # recorded: no vllm._C on a CPU box
    R3["ok"] = (R3["pre_sha"] == G["P60490_PRE"] and rec.get("applied") == "applied" and rec.get("sha_after") == G["P60490_POST"]
                and (rec.get("py_compile") or {}).get("rc") == 0 and bool(rec.get("marker")))
except Exception as e:
    R3.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R3 #60490", json.dumps({k: v for k, v in R3.items() if k != "tb"}, default=str)[:1000])
REP["checks"]["R3"] = R3; save()

# ---------------------------------------------------------------- R4: the patch, then build() + host_phase() + the launch check for all 5 arms
plans = {}


def plan_one(key):
    a = G["ARMS"][key]
    nm = a["name"]
    G["NB1_EXTRA"][nm], G["NB1_ENV"][nm], G["NB1_KV"][nm] = list(a["extra"]), dict(a["env"]), a["kv_bytes"]
    G["NB4T_BUILD_ENV"][nm] = dict(a["build_env"])
    G["ARM_OF"][nm] = a
    b = {"arm": key, "tree_state": G["P"].verify_ppack_patch(tree)["state"]}
    try:
        s = G["merlin_fn_server"](nm, "fp8", a["port"], a["kv_bytes"])            # the NB4T wrapper: never-do + the plan check
        b.update(ok=True, host_phase=G["HOST_PHASE"].get(nm), help=G["HELP"].get(nm), argv=s.argv,
                 launch_check=G["N"].launch_check(a, s.env, s.argv, a["kv_bytes"]), server_started=s.proc is not None,
                 spec=G["N"].spec_from_argv(s.argv), seqs=G["N"].last_value(s.argv, "--max-num-seqs"),
                 env={k: s.env.get(k) for k in G["N"].WATCHED + ("KV_BYTES", "VLLM_SERVER_DEV_MODE", "P2P", "NCCL_P2P_DISABLE", "DRAFT_HEAD",
                                                                  "VLLM_USE_V2_MODEL_RUNNER", "PYTHONPATH")})
        sp = venv_py(SPAWN_ENV, [",".join(G["N"].WATCHED)], name=f"r4_spawn_{key.lower()}", base_env=dict(s.env))
        b["spawn_env"] = sp["json"] or {"error": sp["tail"]}
        b["spawn_ok"] = (isinstance(sp["json"], dict) and sp["json"]["child"] == a["expect"] and sp["json"]["parent"] == a["expect"])
    except Exception as e:
        b.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], host_phase=G["HOST_PHASE"].get(nm))
    plans[key] = b
    log("R4", key, json.dumps({k: v for k, v in b.items() if k not in ("argv", "help")}, default=str)[:900]); save()


try:
    R3P["before"] = G["P"].verify_ppack_patch(tree)
    R3P["pre_shas"] = {rel: v["sha"] for rel, v in R3P["before"]["files"].items()}
    R3P["apply"] = G["_ppack_apply"]()                   # the notebook's own call (P.apply_ppack_patch, Merlin venv python)
    R3P["after_apply"] = G["P"].verify_ppack_patch(tree)
    R3P["import_patched"] = venv_py(IMPORT_PP, name="r3p_import_patched")
    for k in ARMS5:
        plan_one(k)
except Exception as e:
    R4["fatal"] = f"{type(e).__name__}: {e}"[:800]; R4["tb"] = traceback.format_exc()[-1500:]
R4["plans"] = plans
R4["host_phase_ok"] = len(plans) == 5 and all(p.get("host_phase") == "ok" for p in plans.values())
R4["launch_check_ok"] = len(plans) == 5 and all(p.get("ok") and p.get("launch_check") == [] for p in plans.values())
R4["spawn_env_ok"] = len(plans) == 5 and all(p.get("spawn_ok") is True for p in plans.values())
R4["tree_states"] = {k: p.get("tree_state") for k, p in plans.items()}
REP["checks"]["R4"] = R4; save()

# ---------------------------------------------------------------- R5: the argv fields of every arm (own parse; vLLM argparse recorded)
def strip_port(argv):
    a, out, i = [str(x) for x in argv], [], 0
    while i < len(a):
        if a[i] == "--port" and i + 1 < len(a):
            i += 2
            continue
        out.append(a[i]); i += 1
    return out


try:
    av = {k: strip_port(v["argv"]) for k, v in plans.items() if v.get("argv")}
    q = av.get("Q") or []
    sq = G["N"].spec_from_argv(q) or {}
    fields = {}
    for k in ARMS5:
        x, bad = av.get(k) or [], []
        sp = G["N"].spec_from_argv(x) or {}
        if not x: bad.append("no argv")
        if "--language-model-only" not in x: bad.append("--language-model-only")
        if G["N"].last_value(x, "--max-num-seqs") != ("12" if k == "C12" else "8"): bad.append(f"seqs {G['N'].last_value(x, '--max-num-seqs')}")
        if G["N"].last_value(x, "--kv-cache-memory-bytes") != "3553580524": bad.append("kv bytes")
        if G["N"].last_value(x, "--kv-cache-dtype") != "fp8_e4m3": bad.append("kv dtype")
        if G["N"].last_value(x, "--max-model-len") != "131072": bad.append("max-model-len")
        if G["N"].last_value(x, "--worker-extension-cls") != "nb1_worker_ext.NB1WorkerExt": bad.append("worker ext")
        if "torch" not in str(G["N"].last_value(x, "--profiler-config")): bad.append("profiler")
        caps = json.loads(G["N"].last_value(x, "--compilation-config") or "{}").get("cudagraph_capture_sizes") or []
        if not caps or max(caps) != 32: bad.append(f"captures {caps}")
        if any(f in x for f in ("--no-async-scheduling", "--no-enable-prefix-caching")): bad.append("never-do flag")
        if sp.get("method") != "mtp" or sp.get("num_speculative_tokens") != 3 or sp.get("draft_sample_method") != "probabilistic": bad.append(f"spec {sp}")
        if sp.get("draft_lm_head_quant") != ("int8" if k == "I8" else None): bad.append(f"draft head {sp.get('draft_lm_head_quant')}")
        if k in ("P26", "P27") and x != q: bad.append("argv differs from Q")
        if k == "I8" and [y for y in x if not y.startswith('{"method":"mtp"')] != [y for y in q if not y.startswith('{"method":"mtp"')]:
            bad.append("argv differs from Q beyond the spec")
        if k == "C12" and x != q + ["--max-num-seqs", "12"]: bad.append("argv != Q + --max-num-seqs 12")
        fields[k] = bad
    R5["field_check"] = fields
    R5["specs"] = {k: G["N"].spec_from_argv(v) for k, v in av.items()}
    R5["partition_env"] = {k: (plans[k].get("env") or {}).get("VLLM_PP_LAYER_PARTITION") for k in plans}
    R5["draft_head_env"] = {k: (plans[k].get("env") or {}).get("DRAFT_HEAD") for k in plans}
    R5["ok"] = (len(fields) == 5 and all(not v for v in fields.values())
                and R5["partition_env"] == {"Q": "25,23", "I8": "25,23", "P26": "26,22", "P27": "27,21", "C12": "25,23"})
except Exception as e:
    R5.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R5 argv", json.dumps({k: R5.get(k) for k in ("field_check", "partition_env", "draft_head_env")})[:1500])
REP["checks"]["R5"] = R5; save()

# ---------------------------------------------------------------- R8: partition facts (PLE on PP rank 0) + recorded engine probes
try:
    cfg = json.loads((Path(G["CKPT"]) / "config.json").read_text())
    tc_ = cfg.get("text_config") or cfg
    ple = [int(i) - 1 for i in tc_.get("ple_layer_ids") or []]
    nl = int(tc_.get("num_hidden_layers"))
    R8["num_hidden_layers"], R8["ple_idx"] = nl, ple
    R8["ple_on_rank0"] = {p: bool(ple) and max(ple) < int(p.split(",")[0]) and min(ple) >= 0 for p in ("25,23", "26,22", "27,21")}
    R8["sums_ok"] = all(sum(int(x) for x in p.split(",")) == nl for p in ("25,23", "26,22", "27,21"))
    R8["ok"] = nl == 48 and all(R8["ple_on_rank0"].values()) and R8["sums_ok"]
    R8["pp_indices_recorded"] = venv_py(PPIDX, ["25,23;26,22;27,21"], name="r8_pp_indices")["json"]
    R8["head_probe_recorded"] = venv_py(HEADPROBE, name="r8_head_probe")
except Exception as e:
    R8.update(ok=False, error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1200:])
log("R8 partition", json.dumps({k: v for k, v in R8.items() if k != "head_probe_recorded"}, default=str)[:1200])
REP["checks"]["R8"] = R8; save()

# ---------------------------------------------------------------- R6: the worker extension inside the venv (patched tree)
EXTPROBE = r"""
import json, os
out = {}
try:
    import nb1_worker_ext as X
    out["import"] = X.__file__
    from vllm.utils.import_utils import resolve_obj_by_qualname
    out["resolve"] = resolve_obj_by_qualname("nb1_worker_ext.NB1WorkerExt") is X.NB1WorkerExt
    try:
        from vllm.v1.worker.gpu_worker import Worker
        out["conflicts"] = [a for a in dir(X.NB1WorkerExt) if not a.startswith("__") and hasattr(Worker, a)]
    except BaseException as e:
        out["worker_import_error"] = repr(e)[:300]
    class D(X.NB1WorkerExt):
        pass
    out["nb4s_env"] = D().nb4s_env("NB4_PPDEFER,VLLM_PP_LAYER_PARTITION,NB4_PPACK")
    out["cache_info_keys"] = sorted(D().nb1_cache_info())
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {e}"[:500]
print("@JSON " + json.dumps(out, default=str))
"""
try:
    r = venv_py(EXTPROBE, name="r6_ext", env_extra={"NB4_PPDEFER": "1", "VLLM_PP_LAYER_PARTITION": "26,22"})
    R6 = r["json"] or {"error": r["tail"]}
    R6["ok"] = (bool(R6.get("import")) and R6.get("resolve") is True and R6.get("conflicts", []) == []
                and (R6.get("nb4s_env") or {}).get("env") == {"NB4_PPDEFER": "1", "VLLM_PP_LAYER_PARTITION": "26,22"})
except Exception as e:
    R6 = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}
log("R6 ext", json.dumps(R6, default=str)[:900])
REP["checks"]["R6"] = R6; save()

# ---------------------------------------------------------------- restore: nb4_ppack reverted (pre-shas), #60490 + extension removed
try:
    R3P["revert"] = G["_ppack_revert"]()                 # the notebook's own call: revert + verify the pre-shas
    R3P["after_revert"] = G["P"].verify_ppack_patch(tree)
    R3P["import_reverted"] = venv_py(IMPORT_PP, name="r3p_import_reverted")
    ip = (R3P.get("import_patched") or {}).get("json") or {}
    ir = (R3P.get("import_reverted") or {}).get("json") or {}
    R3P["ok"] = ((R3P.get("apply") or {}).get("ok") is True and R3P["after_apply"]["state"] == "applied"
                 and ip.get("ok") is True and ip.get("pack_views") is True and ip.get("flush") is True
                 and ip.get("final") is True and "flush_rpc" in (ip.get("stat_keys") or [])
                 and ip.get("gw_ok") is True and ip.get("gw_profile_flush") is True and ip.get("gw_shutdown_flush") is True
                 and R3P["after_revert"]["state"] == "not_applied"
                 and {rel: v["sha"] for rel, v in R3P["after_revert"]["files"].items()} == R3P["pre_shas"]
                 and R3P["pre_shas"] == {rel: p[0] for rel, p in G["P"].PATCHES.items()} and len(R3P["pre_shas"]) == 3
                 and ir.get("ok") is True and ir.get("pack_views") is False
                 and ir.get("gw_ok") is True and ir.get("gw_profile_flush") is False)
except Exception as e:
    R3P.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R3P ppack", json.dumps({k: (v if k not in ("apply", "revert") else {x: (v or {}).get(x) for x in ("applied", "reverted", "ok")})
                             for k, v in R3P.items() if k != "tb"}, default=str)[:1500])
REP["checks"]["R3P"] = R3P; save()
try:
    if orig is not None:
        qsa_file.write_bytes(orig)
        for pyc in (qsa_file.parent / "__pycache__").glob("qsa.*.pyc"):
            pyc.unlink()
        ext = tree / "nb1_worker_ext.py"
        if ext.exists():
            ext.unlink()
        R3["restored_sha"] = hashlib.sha256(qsa_file.read_bytes()).hexdigest()
        R3["restored_ok"] = R3["restored_sha"] == G["P60490_PRE"] and not ext.exists()
except Exception as e:
    R3["restored_ok"] = False; R3["restore_error"] = repr(e)[:300]
REP["checks"]["R3"] = R3; save()

# ---------------------------------------------------------------- R7: host gates on the real cgroup (recorded; this CPU box is ~30 GiB)
try:
    R7["engine_procs"] = G["clear_engine_procs"]()
    R7["drop"] = G["drop_page_cache"]()
    R7["host_ram_gate"] = G["host_ram_gate"]("kit")
except Exception as e:
    R7["error"] = f"{type(e).__name__}: {e}"[:600]; R7["tb"] = traceback.format_exc()[-1200:]
log("R7 host", json.dumps({k: v for k, v in R7.items() if k in ("engine_procs", "drop", "error")}, default=str)[:900])
REP["checks"]["R7"] = R7
verdict({"cell0_identity_submission": R0.get("ok") is True, "inputs_found": bool((R1.get("inputs") or {}).get("merlin_inputs_ok")),
         "merge_ok": bool((R1.get("merge") or {}).get("ok")), "sha256_all_ok": bool((R1.get("sha") or {}).get("ok")),
         "arms_q_i8_p26_p27_c12": R1.get("arms") == list(ARMS5), "rc2_oracle_rows_loaded": "rc2_fp8" in (R1.get("rc2_rows_loaded") or []),
         "prompts_12_distinct": R1.get("prompts_12_ok") is True,
         "merlin_install_ok": R2.get("ok") is True, "merlin_flag_sources_equal": R2.get("src_equal") is True,
         "patch_60490_applied_verified": R3.get("ok") is True, "tree_restored": R3.get("restored_ok") is True,
         "host_phase_all_5_ok": R4.get("host_phase_ok") is True, "launch_check_all_5_ok": R4.get("launch_check_ok") is True,
         "env_in_spawned_child_all_5": R4.get("spawn_env_ok") is True,
         "all_plans_on_patched_tree": R4.get("tree_states") == {k: "applied" for k in ARMS5},
         "argv_fields_ok": R5.get("ok") is True, "ple_on_rank0_all_partitions": R8.get("ok") is True,
         "worker_extension_ok": R6.get("ok") is True, "ppack_apply_import_revert_ok": R3P.get("ok") is True,
         "no_engine_procs": (R7.get("engine_procs") or {}).get("ok") is True, "page_cache_drop_ran": (R7.get("drop") or {}).get("available") is True},
        recorded={"help_check": {k: v.get("help") for k, v in plans.items()}, "host_ram_gate": (R7.get("host_ram_gate") or {}).get("why"),
                  "env": {k: v.get("env") for k, v in plans.items()}, "spawn_env": {k: v.get("spawn_env") for k, v in plans.items()},
                  "qsa_import_after_60490": ((R3.get("import_after_recorded") or {}).get("json")),
                  "pp_indices": R8.get("pp_indices_recorded"), "draft_head_probe": (R8.get("head_probe_recorded") or {}).get("json")})
'''

KITS = {"units": {"body": UNITS, "title": "Q27 E25 NB4T Kit Units", "models": []},
        "real": {"body": REAL, "title": "Q27 E25 NB4T Kit Real", "models": RC2_MODELS}}


def build(name):
    spec = KITS[name]
    code = HEAD + spec["body"]
    for k, v in (("__EXPORTBLOB__", EXPORT_BLOB), ("__UNITSBLOB__", UNITS_BLOB), ("__NBSHA__", NB_SHA), ("__NAME__", name),
                 ("__SRCSHAS__", json.dumps(SRC_SHAS))):
        code = code.replace(k, v)
    compile(code, f"<kit {name}>", "exec")
    slug = f"q27-e25-nb4t-kit-{name}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert len(data) < 400_000, (name, len(data))
    assert b"\r" not in data.replace(b"\\r", b"")
    out = ROOT / f"{slug}__{OWNER}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_bytes(data)
    meta = {"id": f"{OWNER}/{slug}", "title": spec["title"], "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": spec["models"], "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| NB4T", NB_SHA[:8], flush=True)
    return out, code


if __name__ == "__main__":
    for n in KITS:
        build(n)
