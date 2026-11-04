"""E25 NB4S Kaggle CPU kits (pinned Py3.12 image, no GPU, no internet, private; account foysalemonshanto). Pattern: the NB1 R2
kits (build_e25_nb1_cpu_kit.py, ran ALL OK). Each kernel is fail-soft, writes its JSON report and prints exactly one line
`@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`.

  units  q27-e25-nb4s-kit-units: nb4s_unit_tests.py on Py3.12 (T1-T9: arm plan, never-do, env read-back, /metrics parity,
         synthetic stall traces, speed / budget, per-arm verdicts, PatchSeq + the REAL nb4_ppack apply / revert on copies of
         Merlin ad42e37's three patched files, the CELL flow with a stub boot runner, notebook facts) + nb4_ppack_tests.py
         (rev 2: patch, simulator equivalence / ordering, F1 shutdown, F2 sync RPCs, F3 draft race, R1 read-back). The
         real NB1 R2 B1 traces are not shipped (17 MB): that test runs locally only.
  real   q27-e25-nb4s-kit-real (the 7 RC2 Models + ARC): NB4S's own code (RC2_UNIT + RC2_CPU_KIT) on the REAL mounts: inputs,
         merge, full sha256, prompts; the real Merlin offline install; #60490 + the extended worker extension on the INSTALLED
         tree; launch.build() + host_phase() + `vllm serve --help=all` for B and S on the pristine tree; nb4_ppack rev 2 applied
         on the installed tree (pp_utils.py + model_runner.py + gpu_worker.py: strict pre / post sha, py_compile in the Merlin
         venv, import of the patched modules incl. the patched Worker), then
         host_phase + help for QP and QD; vLLM's own argparse on all 4 argvs with field checks; each plan's env checked by
         launch_check AND read back in a spawned child of a venv process (the workers' path); the extension (nb4s_env) in
         the venv; nb4_ppack reverted (pre-shas verified), #60490 / extension restored; host gates recorded.

Run: python build_e25_nb4s_cpu_kit.py -> D:/ARC AGI 2/_scratch/push_e25/q27-e25-nb4s-kit-{units,real}__foysalemonshanto/
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
NB_PATH = HERE.parent / "Q27-XHIGH-E25_NB4S_STALL.ipynb"
NB_SHA = hashlib.sha256(NB_PATH.read_bytes()).hexdigest()
EXPORT = (HERE / "nb4s_code_export.py").read_text(encoding="utf-8")
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


def blob(files):
    """{relative path: text} -> lzma + base64 (the notebook and its export share most bytes; keeps the kit < 400 kB)."""
    return base64.b64encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


_R = json.loads(NB1_RESULTS.read_text(encoding="utf-8"))
_mp = _R["setup"]["merlin_plans"]["nb1_b3_fp8_pin100"]
NB1_B3_PLAN = {"setup": {"merlin_plans": {"nb1_b3_fp8_pin100": {"env_keys": _mp["env_keys"], "argv_tail": _mp["argv_tail"]}}}}
UNIT_FILES = {f"build/{n}": (HERE / n).read_text(encoding="utf-8") for n in
              ("nb4s_lib.py", "nb4s_unit_tests.py", "nb4_timeline.py", "nb4_ppack.py", "nb4_ppack_tests.py", "nb4s_code_export.py")}
UNIT_FILES["Q27-XHIGH-E25_NB4S_STALL.ipynb"] = NB_PATH.read_text(encoding="utf-8")
UNIT_FILES["nb1_b3_plan.json"] = json.dumps(NB1_B3_PLAN)
for rel, (pre, *_r) in PPK.PATCHES.items():                 # rev 2: pp_utils.py, model_runner.py, gpu_worker.py
    UNIT_FILES["merlin/" + rel] = (MERLIN_SRC / rel).read_bytes().decode("utf-8")
    assert hashlib.sha256(UNIT_FILES["merlin/" + rel].encode("utf-8")).hexdigest() == pre, rel
UNITS_BLOB = blob(UNIT_FILES)
EXPORT_BLOB = blob({"nb4s_code_export.py": EXPORT})

HEAD = r'''# E25 NB4S CPU kit __NAME__ (pinned image, no GPU): see the builder docstring for the checks and the verdict rule
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib, re, base64, lzma
from pathlib import Path
T0 = time.time()
KIT = "__NAME__"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")          # env overrides: local dry run only
K = Path(os.environ.get("KIT_TMP", "/tmp/nb4skit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": f"E25 NB4S CPU kit {KIT}", "notebook_sha256": "__NBSHA__", "python": sys.version, "checks": {}}
def save():
    (W / f"nb4s_kit_{KIT}_report.json").write_text(json.dumps(REP, indent=1, default=str))
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
# ---------------------------------------------------------------- U1: nb4s_unit_tests.py on this image (Py3.12)
U1 = {}
try:
    F = unpack("__UNITSBLOB__")
    B = K / "kit"
    for rel, txt in F.items():
        (B / rel).parent.mkdir(parents=True, exist_ok=True)
        (B / rel).write_text(txt, encoding="utf-8", newline="\n")
    U1["files"] = {rel: hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12] for rel, txt in F.items()}
    for k_ in list(os.environ):
        if k_.startswith(("NB1_", "E23_", "RC2_", "NB4_")) or k_ in ("KAGGLE_IS_COMPETITION_RERUN", "CUDA_DEVICE_MAX_CONNECTIONS"):
            os.environ.pop(k_, None)
    r = sh("units", [sys.executable, str(B / "build/nb4s_unit_tests.py"), "--b1-traces", str(K / "no_b1_traces"),
                     "--merlin-src", str(B / "merlin"), "--nb1-results", str(B / "nb1_b3_plan.json")], timeout=1800, cwd=B / "build")
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
    U1["units"] = {"rc": r["rc"], "sec": r["sec"], "summary": m.group(0) if m else None, "fails": [l[:300] for l in lines if l.startswith("FAIL ")][:40],
                   "skips": [l for l in lines if l.startswith("SKIP ")]}
    U1["units"]["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0
    U1["units"]["real_patch_ran"] = not any("SKIP T7" in l for l in lines)
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
         "ppack_tests_all_pass": (U1.get("ppack") or {}).get("all_pass") is True},
        recorded={"units": (U1.get("units") or {}).get("summary"), "skips": (U1.get("units") or {}).get("skips"),
                  "ppack": (U1.get("ppack") or {}).get("summary")})
'''

REAL = r'''
# ---------------------------------------------------------------- R1: NB4S code on the real mounts (no boot)
FILES = unpack("__EXPORTBLOB__")
(K / "nb4s_code_export.py").write_text(FILES["nb4s_code_export.py"], encoding="utf-8", newline="\n")
REP["export_sha256"] = hashlib.sha256(FILES["nb4s_code_export.py"].encode("utf-8")).hexdigest()
R1, R2, R3, R3P, R4, R5, R6, R6E, R7 = {}, {}, {}, {}, {}, {}, {}, {}, {}
G = None
try:
    os.environ.update(RC2_UNIT="1", RC2_CPU_KIT="1", E23_OUT=str(K / "out"), E23_TMP=str(K / "tmp"), E23_INPUT=INPUT_ROOT,
                      RC2_SUBMISSION=str(K / "out/submission.json"), RC2_SHM=str(K / "shm"))
    os.environ.pop("E23_FAKE_SERVER", None)
    for d in ("out", "tmp", "shm"):
        (K / d).mkdir(exist_ok=True)
    t = time.time()
    G = runpy.run_path(str(K / "nb4s_code_export.py"), run_name="nb4skit")["nb4s_main"].__globals__
    R = G["R"]
    R1.update(load_sec=round(time.time() - t), inputs=R["inputs"], merge=R["sections"]["A0"]["merge"],
              rc2_oracle_prompts_match=G["NB1"].get("rc2_oracle_prompts_match"), rc2_rows_loaded=sorted(k for k in G["ROWS"] if k.startswith("rc2_")),
              arms=sorted(G["ARMS"]), embed_shas={k: G["NB4S"].get(k) for k in ("lib_sha256", "timeline_sha256", "ppack_sha256")},
              cell6_dropped=len((G["NB4S"].get("cell6_dropped") or {}).get("names") or []))
    log("R1 inputs", json.dumps(R["inputs"])[:500]); log("R1 merge", json.dumps(R1["merge"])[:300])
    save()
    t = time.time()
    ok, why = G["sha_gate"](4 * 3600)
    R1["sha"] = {"ok": ok, "why": why, "wait_sec": round(time.time() - t), "detail": R["setup"].get("sha256")}
    log("R1 sha", json.dumps(R1["sha"])[:300])
except Exception as e:
    R1["fatal"] = f"{type(e).__name__}: {e}"[:1000]; R1["tb"] = traceback.format_exc()[-2500:]
    log("R1 FATAL", R1["fatal"])
REP["checks"]["R1"] = R1; save()

# ---------------------------------------------------------------- R2: Merlin offline install
eng = None
try:
    t = time.time()
    eng = G["merlin_install"]()
    R2.update(ok=True, sec=round(time.time() - t), kv=dict(eng["kv"]), cuda_home=eng["cuda_home"])
except Exception as e:
    R2.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R2 install", json.dumps({k: v for k, v in R2.items() if k != "tb"})[:600])
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
    import vllm.v1.worker.gpu.model_runner as m        # recorded only (a CPU box may lack what the GPU runner imports)
    out.update(mr_ok=True, mr=m.__file__)
except BaseException as e:
    out.update(mr_ok=False, mr_error=repr(e)[:400])
try:
    import inspect
    import vllm.v1.worker.gpu_worker as w              # rev 2 (F2): the patched Worker imports; its source holds the flush
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

# ---------------------------------------------------------------- R3: #60490 + the extended extension on the INSTALLED tree
orig, qsa_file, tree = None, None, None
try:
    tree = Path(eng["kv"]["MERLIN_TREE"])
    qsa_file = tree / G["P60490_REL"]
    orig = qsa_file.read_bytes()
    R3["pre_sha"] = hashlib.sha256(orig).hexdigest()
    rec = G["apply_60490"]()
    R3["apply"] = {k: rec.get(k) for k in ("applied", "sha_after", "marker", "py_compile", "vllm_origin", "worker_ext")}
    R3["import_after"] = venv_py(IMPORT_QSA, name="r3_import_after")
    ia = R3["import_after"]["json"] or {}
    R3["ok"] = (R3["pre_sha"] == G["P60490_PRE"] and rec.get("ok") is True and rec.get("applied") == "applied"
                and rec.get("sha_after") == G["P60490_POST"] and ia.get("ok") is True and ia.get("has_topk") is True)
except Exception as e:
    R3.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R3 #60490", json.dumps({k: v for k, v in R3.items() if k != "tb"}, default=str)[:1000])
REP["checks"]["R3"] = R3; save()

# ---------------------------------------------------------------- R4 + R3P: build() + host_phase() + --help for all 4 plans
plans = {}


def plan_one(key):
    a = G["ARMS"][key]
    nm = a["name"]
    G["NB1_EXTRA"][nm], G["NB1_ENV"][nm], G["NB1_KV"][nm] = list(a["extra"]), dict(a["env"]), a["kv_bytes"]
    G["ARM_OF"][nm] = a
    b = {"arm": key, "tree_state": G["P"].verify_ppack_patch(tree)["state"]}
    try:
        s = G["merlin_fn_server"](nm, "fp8", a["port"], a["kv_bytes"])            # the NB4S wrapper: never-do + watched keys
        b.update(ok=True, host_phase=G["HOST_PHASE"].get(nm), help=G["HELP"].get(nm), argv=s.argv,
                 launch_check=G["N"].launch_check(a, s.env, s.argv), server_started=s.proc is not None,
                 env={k: s.env.get(k) for k in G["N"].WATCHED + ("KV_BYTES", "VLLM_SERVER_DEV_MODE", "P2P", "NCCL_P2P_DISABLE",
                                                                  "VLLM_USE_V2_MODEL_RUNNER", "VLLM_PP_LAYER_PARTITION", "PYTHONPATH")})
        sp = venv_py(SPAWN_ENV, [",".join(G["N"].WATCHED)], name=f"r4_spawn_{key.lower()}", base_env=dict(s.env))
        b["spawn_env"] = sp["json"] or {"error": sp["tail"]}
        b["spawn_ok"] = (isinstance(sp["json"], dict) and sp["json"]["child"] == a["expect"] and sp["json"]["parent"] == a["expect"])
    except Exception as e:
        b.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], host_phase=G["HOST_PHASE"].get(nm))
    plans[key] = b
    log("R4", key, json.dumps({k: v for k, v in b.items() if k not in ("argv", "help")}, default=str)[:700]); save()


try:
    for k in ("B", "S"):
        plan_one(k)
    R3P["before"] = G["P"].verify_ppack_patch(tree)
    R3P["pre_shas"] = {rel: v["sha"] for rel, v in R3P["before"]["files"].items()}
    R3P["apply"] = G["_ppack_apply"]()                   # the notebook's own call (P.apply_ppack_patch, Merlin venv python)
    R3P["after_apply"] = G["P"].verify_ppack_patch(tree)
    R3P["import_patched"] = venv_py(IMPORT_PP, name="r3p_import_patched")
    for k in ("QP", "QD"):
        plan_one(k)
except Exception as e:
    R4["fatal"] = f"{type(e).__name__}: {e}"[:800]; R4["tb"] = traceback.format_exc()[-1500:]
R4["plans"] = plans
R4["host_phase_ok"] = len(plans) == 4 and all(p.get("host_phase") == "ok" for p in plans.values())
R4["launch_check_ok"] = len(plans) == 4 and all(p.get("ok") and p.get("launch_check") == [] for p in plans.values())
R4["spawn_env_ok"] = len(plans) == 4 and all(p.get("spawn_ok") is True for p in plans.values())
R4["tree_states"] = {k: p.get("tree_state") for k, p in plans.items()}
REP["checks"]["R4"] = R4; save()

# ---------------------------------------------------------------- R5: vLLM's own argparse on the 4 exact argvs (field checks)
ARGPARSE = r"""
import json, sys
argvs = json.load(open(sys.argv[1]))
out = {}
try:
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.engine.arg_utils import AsyncEngineArgs
except BaseException as e:
    print("@JSON " + json.dumps({"import_error": repr(e)[:400]})); raise SystemExit(0)
def g(o, k):
    v = getattr(o, k, None)
    return v if isinstance(v, (int, float, str, bool, type(None))) else (v if isinstance(v, (list, dict)) else str(v))
for label, argv in argvs.items():
    r = {}
    try:
        a = argv[argv.index("serve") + 1:]
        p = make_arg_parser(FlexibleArgumentParser())
        ns = p.parse_args(a)
        if getattr(ns, "model_tag", None):
            ns.model = ns.model_tag
        ea = AsyncEngineArgs.from_cli_args(ns)
        cc = ea.compilation_config
        caps = getattr(cc, "cudagraph_capture_sizes", None) if not isinstance(cc, dict) else cc.get("cudagraph_capture_sizes")
        pc = ea.profiler_config
        r = {k: g(ea, k) for k in ("language_model_only", "mamba_ssm_cache_dtype", "max_num_seqs", "max_num_batched_tokens",
                                    "kv_cache_memory_bytes", "kv_cache_dtype", "worker_extension_cls", "max_model_len", "block_size",
                                    "enable_prefix_caching", "async_scheduling", "pipeline_parallel_size", "tensor_parallel_size")}
        r["capture_sizes"] = list(caps) if caps else None
        r["profiler"] = getattr(pc, "profiler", None) if not isinstance(pc, dict) else pc.get("profiler")
        r["speculative_config"] = str(ea.speculative_config)[:200]
        r["ok"] = True
    except BaseException as e:
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
    out[label] = r
print("@JSON " + json.dumps(out, default=str))
"""
try:
    argvs = {k: v["argv"] for k, v in plans.items() if v.get("argv")}
    (K / "argvs.json").write_text(json.dumps(argvs))
    r = venv_py(ARGPARSE, [str(K / "argvs.json")], name="r5_argparse")
    R5["parsed"] = r["json"] or {"error": r["tail"]}
    fields = {}
    for lab in ("B", "S", "QP", "QD"):
        x = (R5["parsed"] or {}).get(lab) or {}
        bad = []
        if x.get("ok") is not True: bad.append("parse")
        if x.get("language_model_only") is not True: bad.append("language_model_only")
        if x.get("max_num_seqs") != 8: bad.append(f"max_num_seqs {x.get('max_num_seqs')}")
        if not x.get("capture_sizes") or max(x["capture_sizes"]) != 32: bad.append(f"captures {x.get('capture_sizes')}")
        if x.get("kv_cache_memory_bytes") != 3553580524: bad.append(f"kv {x.get('kv_cache_memory_bytes')}")
        if x.get("worker_extension_cls") != "nb1_worker_ext.NB1WorkerExt": bad.append("worker_extension_cls")
        if x.get("profiler") != "torch": bad.append(f"profiler {x.get('profiler')}")
        if x.get("max_model_len") != 131072: bad.append(f"max_model_len {x.get('max_model_len')}")
        if x.get("kv_cache_dtype") != "fp8_e4m3": bad.append("kv_cache_dtype")
        if x.get("enable_prefix_caching") is False: bad.append("prefix caching off")
        if x.get("async_scheduling") is False: bad.append("async scheduling off")
        if "mtp" not in str(x.get("speculative_config")).lower(): bad.append("speculative_config")
        fields[lab] = bad
    R5["field_check"] = fields
    R5["identical_argvs"] = len({json.dumps(v) for v in argvs.values()}) == 1 if len(argvs) == 4 else False
    R5["ok"] = len(fields) == 4 and all(not v for v in fields.values()) and R5["identical_argvs"]
except Exception as e:
    R5.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R5 argparse", json.dumps({k: R5.get(k) for k in ("field_check", "identical_argvs")})[:1200])
REP["checks"]["R5"] = R5; save()

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
    out["nb4s_env"] = D().nb4s_env("CUDA_DEVICE_MAX_CONNECTIONS,NB4_PPACK,NB4_PPDEFER")
    out["cache_info_keys"] = sorted(D().nb1_cache_info())
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {e}"[:500]
print("@JSON " + json.dumps(out, default=str))
"""
try:
    r = venv_py(EXTPROBE, name="r6_ext", env_extra={"CUDA_DEVICE_MAX_CONNECTIONS": "32", "NB4_PPACK": "1"})
    R6 = r["json"] or {"error": r["tail"]}
    R6["ok"] = (bool(R6.get("import")) and R6.get("resolve") is True and R6.get("conflicts", []) == []
                and (R6.get("nb4s_env") or {}).get("env") == {"CUDA_DEVICE_MAX_CONNECTIONS": "32", "NB4_PPACK": "1"})
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
verdict({"inputs_found": bool((R1.get("inputs") or {}).get("merlin_inputs_ok")), "merge_ok": bool((R1.get("merge") or {}).get("ok")),
         "sha256_all_ok": bool((R1.get("sha") or {}).get("ok")), "arms_b_s_qp_qd": R1.get("arms") == ["B", "QD", "QP", "S"],
         "rc2_oracle_rows_loaded": "rc2_fp8" in (R1.get("rc2_rows_loaded") or []),
         "merlin_install_ok": R2.get("ok") is True, "patch_60490_applied_verified": R3.get("ok") is True,
         "tree_restored": R3.get("restored_ok") is True, "host_phase_all_4_ok": R4.get("host_phase_ok") is True,
         "launch_check_all_4_ok": R4.get("launch_check_ok") is True, "env_in_spawned_child_all_4": R4.get("spawn_env_ok") is True,
         "q_plans_on_patched_tree": R4.get("tree_states") == {"B": "not_applied", "S": "not_applied", "QP": "applied", "QD": "applied"},
         "argparse_fields_ok": R5.get("ok") is True, "worker_extension_ok": R6.get("ok") is True,
         "ppack_apply_import_revert_ok": R3P.get("ok") is True,
         "no_engine_procs": (R7.get("engine_procs") or {}).get("ok") is True, "page_cache_drop_ran": (R7.get("drop") or {}).get("available") is True},
        recorded={"help_check": {k: v.get("help") for k, v in plans.items()}, "host_ram_gate": (R7.get("host_ram_gate") or {}).get("why"),
                  "env": {k: v.get("env") for k, v in plans.items()}, "spawn_env": {k: v.get("spawn_env") for k, v in plans.items()}})
'''

KITS = {"units": {"body": UNITS, "title": "Q27 E25 NB4S Kit Units", "models": []},
        "real": {"body": REAL, "title": "Q27 E25 NB4S Kit Real", "models": RC2_MODELS}}


def build(name):
    spec = KITS[name]
    code = HEAD + spec["body"]
    for k, v in (("__EXPORTBLOB__", EXPORT_BLOB), ("__UNITSBLOB__", UNITS_BLOB), ("__NBSHA__", NB_SHA), ("__NAME__", name)):
        code = code.replace(k, v)
    compile(code, f"<kit {name}>", "exec")
    slug = f"q27-e25-nb4s-kit-{name}"
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
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| NB4S", NB_SHA[:8], flush=True)
    return out, code


if __name__ == "__main__":
    for n in KITS:
        build(n)
