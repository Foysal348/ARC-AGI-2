"""E25 NB1 Kaggle CPU kits (user GO 2026-10-31: two parallel CPU kernels, pinned Py3.12 image, no GPU, no internet, private).
Pattern: build_e25_nb3_cpu_kit.py / Experiment 13 build_e13_rc2_cpu_kit.py (both ran ALL OK). Each kernel is fail-soft,
writes its JSON report and prints exactly one line `@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`.

  K1 q27-e25-nb1-kit-real  (foysalemonshanto; the 7 RC2 Models + ARC): NB1's own code (RC2_UNIT + RC2_CPU_KIT) on the REAL
     mounts: inputs, merge, full sha256, prompt render hashes, RC2's recorded oracle rows vs this notebook's prompt hashes;
     the real Merlin offline install; the #60490 applier on the INSTALLED tree (pre a23d5f90 / post 88fe0a8f / marker /
     py_compile in the Merlin venv / vllm + extension origin) and an import of the patched module; while patched:
     launch.build() + host_phase() + `vllm serve --help=all` for B1, B2 and every B3 pin step (on B2 and on B1), then vLLM's
     own argparse (make_arg_parser + AsyncEngineArgs.from_cli_args) on those exact argvs with field checks; the worker
     extension in the venv (import, resolve_obj_by_qualname, no Worker attribute clash, read-only methods on CPU, router
     wrap, dev routes); the tree restored to Merlin's sha; the host gates on the real cgroup (recorded).
  K2 q27-e25-nb1-kit-units (hosen42; part 1's small files + ARC): nb1_unit_tests.py on Py3.12 + nb1_local_e2e.py (main /
     e23fake / patchfail / b1fail / rerun) on a fake tree made from the real part-1 small files (RC2 kit's MAKE_FAKE).

Run: python build_e25_nb1_cpu_kit.py  ->  D:/ARC AGI 2/_scratch/push_e25/q27-e25-nb1-kit-{real__foysalemonshanto,units__hosen42}/
Local dry run (paths overridden by KIT_INPUT / KIT_TMP / KIT_WORK): python build_e25_nb1_cpu_kit.py --dry
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
E13 = Path(r"D:/ARC AGI 2/experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)")
RC2_OUT = Path(r"D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2")
MERLIN_QSA = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src/vllm/models/qwen4_exp/nvidia/qsa.py")
E23_FAKE = Path(r"D:/ARC AGI 2/experiments/Experiment 23 - Six levers vLLM 0.32 (planned)/src/harness/fake_vllm_server.py")
NB_PATH = HERE.parent / "Q27-XHIGH-E25_NB1_CAPACITY_R2.ipynb"     # R2 (2026-10-31): the R1 file ran on Kaggle (sha 47c8e3ae)
NB_SHA = hashlib.sha256(NB_PATH.read_bytes()).hexdigest()
EXPORT = (HERE / "nb1_code_export.py").read_text(encoding="utf-8")
_nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
assert "\n\n".join("".join(c["source"]) for c in _nb["cells"] if c["cell_type"] == "code") == EXPORT, "export != notebook cells"
PARTS = (E13 / "prep" / "merlin_fn_ckpt" / "kernels" / "parts.json").read_text(encoding="utf-8")
RC2_KIT = (E13 / "Experiment Notebooks" / "Build" / "build_e13_rc2_cpu_kit.py").read_text(encoding="utf-8").replace("\r\n", "\n")
MAKE_FAKE = RC2_KIT[RC2_KIT.index("MAKE_FAKE = r'''") + len("MAKE_FAKE = r'''"):RC2_KIT.index("'''\n\nCODE = r'''")]
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
RC2_MODELS = ([f"foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p{k}/Transformers/default/1" for k in range(1, 6)] +
              ["foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
               "foysalemonshanto/e23-merlin-env/Other/v1/1"])


def blob(files):
    """{relative path: text} -> zlib + base64 (keeps each kit far below the 400 kB commit limit)."""
    return base64.b64encode(zlib.compress(json.dumps(files).encode("utf-8"), 9)).decode("ascii")


EXPORT_BLOB = blob({"nb1_code_export.py": EXPORT})
UNIT_FILES = {"nb1_code_export.py": EXPORT}
for n in ("nb1_unit_tests.py", "nb1_local_e2e.py", "nb1_fake_server.py", "nb1_fixture_b1_gpu_2026-10-31.json"):   # R2: + the real B1 record
    UNIT_FILES[n] = (HERE / n).read_text(encoding="utf-8")
UNIT_FILES.update({"rc2/rc2_results.json": (RC2_OUT / "rc2_results.json").read_text(encoding="utf-8"),
                   "rc2/logs/server_merlin_fp8.log": (RC2_OUT / "logs/server_merlin_fp8.log").read_text(encoding="utf-8", errors="replace"),
                   "rc2/logs/server_merlin_bf16.log": (RC2_OUT / "logs/server_merlin_bf16.log").read_text(encoding="utf-8", errors="replace"),
                   "rc2_code_export.py": (E13 / "Experiment Notebooks/Build/rc2_code_export.py").read_text(encoding="utf-8").replace("\r\n", "\n"),
                   "merlin/qsa.py": MERLIN_QSA.read_bytes().decode("utf-8"), "e23/fake_vllm_server.py": E23_FAKE.read_text(encoding="utf-8"),
                   "make_fake.py": MAKE_FAKE, "parts.json": PARTS})
assert hashlib.sha256(UNIT_FILES["merlin/qsa.py"].encode()).hexdigest() == "a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653"
UNITS_BLOB = blob(UNIT_FILES)

HEAD = r'''# E25 NB1 CPU kit __NAME__ (pinned image, no GPU): see the builder docstring for the checks and the verdict rule
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib, re, base64, zlib
from pathlib import Path
T0 = time.time()
KIT = "__NAME__"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")          # env overrides: local dry run only
K = Path(os.environ.get("KIT_TMP", "/tmp/nb1kit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": f"E25 NB1 CPU kit {KIT}", "notebook_sha256": "__NBSHA__", "python": sys.version, "checks": {}}
def save():
    (W / f"nb1_kit_{KIT}_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def unpack(b64):
    return json.loads(zlib.decompress(base64.b64decode(b64)).decode("utf-8"))
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

K1 = r'''
# ---------------------------------------------------------------- R1: NB1 code on the real mounts (no boot)
FILES = unpack("__EXPORTBLOB__")
(K / "nb1_code_export.py").write_text(FILES["nb1_code_export.py"], encoding="utf-8", newline="\n")
REP["export_sha256"] = hashlib.sha256(FILES["nb1_code_export.py"].encode("utf-8")).hexdigest()
R1, R2, R3, R4, R5, R6, R7 = {}, {}, {}, {}, {}, {}, {}
G = None
try:
    os.environ.update(RC2_UNIT="1", RC2_CPU_KIT="1", E23_OUT=str(K / "out"), E23_TMP=str(K / "tmp"), E23_INPUT=INPUT_ROOT,
                      RC2_SUBMISSION=str(K / "out/submission.json"), RC2_SHM=str(K / "shm"))
    os.environ.pop("E23_FAKE_SERVER", None)
    for d in ("out", "tmp", "shm"):
        (K / d).mkdir(exist_ok=True)
    t = time.time()
    G = runpy.run_path(str(K / "nb1_code_export.py"), run_name="nb1kit")["nb1_main"].__globals__
    R = G["R"]
    R1.update(load_sec=round(time.time() - t), inputs=R["inputs"], merge=R["sections"]["A0"]["merge"],
              prompt_render={k: v for k, v in R["prompt_render"].items() if "text" not in k},
              rc2_oracle_prompts_match=G["NB1"].get("rc2_oracle_prompts_match"), rc2_rows_loaded=sorted(k for k in G["ROWS"] if k.startswith("rc2_")),
              prompts=R.get("prompts"), host=R["sections"]["A0"]["host"])
    log("R1 inputs", json.dumps(R["inputs"])[:500]); log("R1 merge", json.dumps(R1["merge"])[:300])
    log("R1 prompts", json.dumps({k: R1[k] for k in ("rc2_oracle_prompts_match", "rc2_rows_loaded")}))
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

def venv_py(code, args=(), name="probe", timeout=900):
    """A probe inside the Merlin venv with the tree on PYTHONPATH (what Merlin's build() exports for the server)."""
    tree, py = eng["kv"]["MERLIN_TREE"], eng["kv"]["MERLIN_VENV"] + "/bin/python"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_PRELOAD")}
    env.update(PYTHONPATH=tree, CUDA_HOME=eng["cuda_home"], VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PATH=f"{eng['kv']['MERLIN_VENV']}/bin:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}")
    if eng["kv"].get("MERLIN_SHIM"):                 # the glibc 2.38 shim Merlin's server runs with (merlin_fn_server)
        env["LD_PRELOAD"] = eng["kv"]["MERLIN_SHIM"]
    (K / f"{name}.py").write_text(code)
    try:
        r = subprocess.run([py, str(K / f"{name}.py"), *args], capture_output=True, text=True, timeout=timeout, env=env, cwd=str(K))
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out)
    js = next((json.loads(l.split(" ", 1)[1]) for l in out.splitlines() if l.startswith("@JSON ")), None)
    return {"rc": rc, "json": js, "tail": out[-1200:]}

IMPORT_QSA = "import json\ntry:\n    import vllm.models.qwen4_exp.nvidia.qsa as m\n    print('@JSON ' + json.dumps({'ok': True, 'file': m.__file__, 'has_cdiv': hasattr(m, 'cdiv'), 'has_topk': hasattr(m, '_TOPK_WORKSPACE_BYTES')}))\nexcept BaseException as e:\n    print('@JSON ' + json.dumps({'ok': False, 'error': repr(e)[:400]}))\n"

# ---------------------------------------------------------------- R3: the #60490 patch on the INSTALLED tree (restored at the end)
orig, qsa_file = None, None
try:
    tree = Path(eng["kv"]["MERLIN_TREE"])
    qsa_file = tree / G["P60490_REL"]
    orig = qsa_file.read_bytes()
    R3["pre_sha"] = hashlib.sha256(orig).hexdigest()
    R3["import_before"] = venv_py(IMPORT_QSA, name="r3_import_before")
    rec = G["apply_60490"]()
    R3["apply"] = rec
    R3["import_after"] = venv_py(IMPORT_QSA, name="r3_import_after")
    ib, ia = (R3["import_before"]["json"] or {}), (R3["import_after"]["json"] or {})
    R3["ok"] = (R3["pre_sha"] == G["P60490_PRE"] and rec.get("ok") is True and rec.get("applied") == "applied" and rec.get("marker") is True
                and rec.get("sha_after") == G["P60490_POST"] and (rec.get("py_compile") or {}).get("rc") == 0
                and (ia.get("ok") is True or ib.get("ok") is not True) and (ia.get("has_topk") is True or ia.get("ok") is not True))
except Exception as e:
    R3.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R3 patch", json.dumps({k: (v if k != "apply" else {x: v.get(x) for x in ("applied", "sha_after", "marker", "py_compile", "vllm_origin")})
                            for k, v in R3.items() if k not in ("tb",)}, default=str)[:1200])
REP["checks"]["R3"] = R3; save()

# ---------------------------------------------------------------- R4: build() + host_phase() + --help for every final argv (patched tree)
plans = {}
try:
    B = G["BOOTS"]
    todo = [("B1", B["B1"]["name"], B["B1"]["extra"], B["B1"]["env"], B["B1"]["kv_bytes"], B["B1"]["port"]),
            ("B2", B["B2"]["name"], B["B2"]["extra"], B["B2"]["env"], B["B2"]["kv_bytes"], B["B2"]["port"])]
    for base, extra, env in (("B2", G["B2_EXTRA"], {**G["DIAG_ENV"], "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}), ("B1", G["FP8_EXTRA"], dict(G["DIAG_ENV"]))):
        for gib in G["PIN_STEPS_GIB"]:
            todo.append((f"B3_on_{base}_+{gib}", f"nb1_b3_fp8_pin{int(round(gib * 100))}_{base.lower()}", extra, env,
                         G["KV_FP8"] + int(gib * 2 ** 30), 8003))
    for label, nm, extra, env, kvb, port in todo:
        G["NB1_EXTRA"][nm], G["NB1_ENV"][nm], G["NB1_KV"][nm] = list(extra), dict(env), kvb
        b = {"label": label, "kv_bytes": kvb}
        try:
            s = G["merlin_fn_server"](nm, "fp8", port, kvb)
            b.update(ok=True, host_phase=G["HOST_PHASE"].get(nm), help=G["HELP"].get(nm), argv=s.argv,
                     env={k: s.env.get(k) for k in ("KV_BYTES", "VLLM_SERVER_DEV_MODE", "NCCL_DEBUG", "MERLIN_SHARE_DRAFT_WORKSPACE",
                                                     "VLLM_RECOVERSSM_COMMIT_GRAPH", "PYTHONPATH", "VLLM_PP_LAYER_PARTITION")},
                     server_started=s.proc is not None)
        except Exception as e:
            b.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], host_phase=G["HOST_PHASE"].get(nm))
        plans[label] = b
        log("R4", label, json.dumps({k: v for k, v in b.items() if k not in ("argv", "help")}, default=str)[:500]); save()
except Exception as e:
    R4["fatal"] = f"{type(e).__name__}: {e}"[:800]; R4["tb"] = traceback.format_exc()[-1500:]
R4["plans"] = plans
R4["host_phase_ok"] = bool(plans) and all(p.get("host_phase") == "ok" for p in plans.values())
REP["checks"]["R4"] = R4; save()

# ---------------------------------------------------------------- R5: vLLM's own argparse on the exact argvs (field checks)
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
                                    "mamba_cache_mode", "enable_prefix_caching", "tensor_parallel_size", "pipeline_parallel_size",
                                    "enable_expert_parallel", "disable_custom_all_reduce", "enable_prompt_tokens_details")}
        r["capture_sizes"] = list(caps) if caps else None
        r["profiler"] = getattr(pc, "profiler", None) if not isinstance(pc, dict) else pc.get("profiler")
        r["profiler_dir"] = getattr(pc, "torch_profiler_dir", None) if not isinstance(pc, dict) else pc.get("torch_profiler_dir")
        r["speculative_config"] = str(ea.speculative_config)[:200]
        r["default_chat_template_kwargs"] = str(getattr(ns, "default_chat_template_kwargs", None))[:120]
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
    want = {"B1": {"language_model_only": True, "max_num_seqs": 8, "cap_max": 32, "ssm": ("auto", None), "kv": G["KV_FP8"]},
            "B2": {"language_model_only": True, "max_num_seqs": 16, "cap_max": 64, "ssm": ("bfloat16",), "kv": G["KV_FP8"]}}
    for lab, p in plans.items():
        if lab.startswith("B3_on_B2"):
            want[lab] = {"language_model_only": True, "max_num_seqs": 16, "cap_max": 64, "ssm": ("bfloat16",), "kv": p["kv_bytes"]}
        elif lab.startswith("B3_on_B1"):
            want[lab] = {"language_model_only": True, "max_num_seqs": 8, "cap_max": 32, "ssm": ("auto", None), "kv": p["kv_bytes"]}
    fields = {}
    for lab, w in want.items():
        x = (R5["parsed"] or {}).get(lab) or {}
        bad = []
        if x.get("ok") is not True: bad.append("parse")
        if x.get("language_model_only") is not w["language_model_only"]: bad.append("language_model_only")
        if x.get("max_num_seqs") != w["max_num_seqs"]: bad.append(f"max_num_seqs {x.get('max_num_seqs')}")
        if not x.get("capture_sizes") or max(x["capture_sizes"]) != w["cap_max"]: bad.append(f"captures {x.get('capture_sizes')}")
        if x.get("mamba_ssm_cache_dtype") not in w["ssm"]: bad.append(f"ssm {x.get('mamba_ssm_cache_dtype')}")
        if x.get("kv_cache_memory_bytes") != w["kv"]: bad.append(f"kv {x.get('kv_cache_memory_bytes')}")
        if x.get("worker_extension_cls") != "nb1_worker_ext.NB1WorkerExt": bad.append("worker_extension_cls")
        if x.get("profiler") != "torch": bad.append(f"profiler {x.get('profiler')}")
        if x.get("max_model_len") != 131072: bad.append(f"max_model_len {x.get('max_model_len')}")
        if x.get("kv_cache_dtype") != "fp8_e4m3": bad.append("kv_cache_dtype")
        fields[lab] = bad
    R5["field_check"] = fields
    R5["ok"] = bool(fields) and all(not v for v in fields.values())
except Exception as e:
    R5.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
log("R5 argparse", json.dumps(R5.get("field_check"))[:1200])
REP["checks"]["R5"] = R5; save()

# ---------------------------------------------------------------- R6: the worker extension inside the venv (patched tree)
EXTPROBE = r"""
import json
out = {}
try:
    import nb1_worker_ext as X
    out["import"] = X.__file__
    from vllm.utils.import_utils import resolve_obj_by_qualname
    out["resolve"] = resolve_obj_by_qualname("nb1_worker_ext.NB1WorkerExt") is X.NB1WorkerExt
    try:
        from vllm.v1.worker.gpu_worker import Worker
        out["conflicts"] = [a for a in dir(X.NB1WorkerExt) if not a.startswith("__") and hasattr(Worker, a)]
        out["worker_methods"] = {m: hasattr(Worker, m) for m in ("set_recoverssm_commit_mode", "get_recoverssm_commit_stats", "get_kv_cache_spec")}
    except BaseException as e:
        out["worker_import_error"] = repr(e)[:300]
    class D(X.NB1WorkerExt):
        pass
    info = D().nb1_cache_info()
    out["cache_info_dummy_keys"] = sorted(info)
    out["cache_info_json"] = len(json.dumps(info)) > 0
    out["routing_start"] = D().nb1_routing("start", "5")
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
    out["router_wrapped"] = FusedMoERouter.select_experts.__code__.co_filename.endswith("nb1_worker_ext.py")
    out["routing_stop"] = D().nb1_routing("stop", "")
    routes = []
    for mod in ("vllm.entrypoints.serve.dev.rpc.api_router", "vllm.entrypoints.serve.dev.cache.api_router",
                "vllm.entrypoints.serve.profile.api_router"):
        try:
            import importlib
            routes += [r.path for r in importlib.import_module(mod).router.routes]
        except BaseException as e:
            routes.append(f"{mod}: {e!r}"[:120])
    out["dev_routes"] = routes
except BaseException as e:
    out["error"] = f"{type(e).__name__}: {e}"[:500]
print("@JSON " + json.dumps(out, default=str))
"""
try:
    r = venv_py(EXTPROBE, name="r6_ext")
    R6 = r["json"] or {"error": r["tail"]}
    R6["ok"] = (bool(R6.get("import")) and R6.get("resolve") is True and R6.get("router_wrapped") is True and R6.get("conflicts", []) == []
                and "routing_start" in R6 and not (R6.get("routing_start") or {}).get("error")
                and all(p in R6.get("dev_routes", []) for p in ("/collective_rpc", "/reset_prefix_cache", "/start_profile")))
except Exception as e:
    R6 = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}
log("R6 ext", json.dumps(R6, default=str)[:1200])
REP["checks"]["R6"] = R6; save()

# ---------------------------------------------------------------- restore the installed tree to Merlin's file
try:
    if orig is not None:
        qsa_file.write_bytes(orig)
        for pyc in (qsa_file.parent / "__pycache__").glob("qsa.*.pyc"):
            pyc.unlink()
        ext = Path(eng["kv"]["MERLIN_TREE"]) / "nb1_worker_ext.py"
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
    R7["page_cache_before"] = G["cgroup_snapshot"]("kit_before")
    R7["drop"] = G["drop_page_cache"]()
    R7["page_cache_after"] = G["cgroup_snapshot"]("kit_after")
    R7["host_ram_gate"] = G["host_ram_gate"]("kit")
    R7["profiler_tools"] = G["profiler_tools"]()
except Exception as e:
    R7["error"] = f"{type(e).__name__}: {e}"[:600]; R7["tb"] = traceback.format_exc()[-1200:]
log("R7 host", json.dumps({k: v for k, v in R7.items() if k in ("engine_procs", "drop", "error")}, default=str)[:900])
REP["checks"]["R7"] = R7
pr = R1.get("prompt_render") or {}
verdict({"inputs_found": bool((R1.get("inputs") or {}).get("merlin_inputs_ok")), "merge_ok": bool((R1.get("merge") or {}).get("ok")),
         "sha256_all_ok": bool((R1.get("sha") or {}).get("ok")), "prompt_hashes_ok": pr.get("hashes_ok") is True,
         "rc2_oracle_prompts_match": R1.get("rc2_oracle_prompts_match") == {"bf16": True, "fp8": True}
                                     and R1.get("rc2_rows_loaded") == ["rc2_bf16", "rc2_fp8"],
         "merlin_install_ok": R2.get("ok") is True, "patch_60490_applied_verified": R3.get("ok") is True,
         "tree_restored": R3.get("restored_ok") is True, "host_phase_all_ok": R4.get("host_phase_ok") is True,
         "argparse_fields_ok": R5.get("ok") is True, "worker_extension_ok": R6.get("ok") is True,
         "no_engine_procs": (R7.get("engine_procs") or {}).get("ok") is True, "page_cache_drop_ran": (R7.get("drop") or {}).get("available") is True},
        recorded={"help_check": {k: v.get("help") for k, v in plans.items()}, "host_ram_gate": (R7.get("host_ram_gate") or {}).get("why"),
                  "ncu_nsys": ((R7.get("profiler_tools") or {}).get("found")), "patched_import": (R3.get("import_after") or {}).get("json"),
                  "worker_methods": R6.get("worker_methods")})
'''

K2 = r'''
# ---------------------------------------------------------------- R8: nb1_unit_tests.py + nb1_local_e2e.py on this image (Py3.12)
R8 = {}
try:
    F = unpack("__UNITSBLOB__")
    B = K / "build"
    for rel, txt in F.items():
        (B / rel).parent.mkdir(parents=True, exist_ok=True)
        (B / rel).write_text(txt, encoding="utf-8", newline="\n")
    R8["files"] = {rel: hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12] for rel, txt in F.items() if not rel.startswith("rc2/")}
    for k_ in list(os.environ):
        if k_.startswith(("NB1_", "E23_", "RC2_")) or k_ == "KAGGLE_IS_COMPETITION_RERUN":
            os.environ.pop(k_, None)
    R8["fake_tree"] = {k: v for k, v in sh("fake_tree", [sys.executable, str(B / "make_fake.py"), INPUT_ROOT, str(K / "fin"), str(B / "parts.json")],
                                            timeout=1800).items() if k != "_out"}
    tenv = {"NB1_TEST_RC2": str(B / "rc2"), "NB1_TEST_RC2_EXPORT": str(B / "rc2_code_export.py"), "NB1_TEST_MERLIN_QSA": str(B / "merlin/qsa.py"),
            "NB1_TEST_E23_FAKE": str(B / "e23/fake_vllm_server.py")}
    r = sh("units", [sys.executable, str(B / "nb1_unit_tests.py"), str(K / "fin"), str(K / "unit")], tenv, 2400, cwd=B)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
    R8["units"] = {"rc": r["rc"], "sec": r["sec"], "summary": m.group(0) if m else None, "fails": [l for l in lines if l.startswith("FAIL ")][:40]}
    R8["units"]["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0
    log("units", json.dumps(R8["units"])[:800]); save()
    r = sh("e2e", [sys.executable, str(B / "nb1_local_e2e.py"), str(K / "fin"), str(K / "e2e")], tenv, 4 * 3600, cwd=B)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
    R8["e2e"] = {"rc": r["rc"], "sec": r["sec"], "summary": m.group(0) if m else None, "fails": [l[:300] for l in lines if l.startswith("FAIL ")][:40],
                 "flows": [l for l in lines if l.startswith("--- flow")]}
    R8["e2e"]["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0
    log("e2e", json.dumps(R8["e2e"])[:1200])
except Exception as e:
    R8["fatal"] = f"{type(e).__name__}: {e}"[:800]; R8["tb"] = traceback.format_exc()[-1500:]
REP["checks"]["R8"] = R8
verdict({"units_all_pass": (R8.get("units") or {}).get("all_pass") is True, "e2e_all_pass": (R8.get("e2e") or {}).get("all_pass") is True},
        recorded={"units": (R8.get("units") or {}).get("summary"), "e2e": (R8.get("e2e") or {}).get("summary")})
'''

KITS = {
    "real": {"body": K1, "title": "Q27 E25 NB1 Kit Real", "owner": "foysalemonshanto", "models": RC2_MODELS},
    "units": {"body": K2, "title": "Q27 E25 NB1 Kit Units", "owner": "hosen42", "models": [RC2_MODELS[0]]},
}


def build(name):
    spec = KITS[name]
    code = HEAD + spec["body"]
    for k, v in (("__EXPORTBLOB__", EXPORT_BLOB), ("__UNITSBLOB__", UNITS_BLOB), ("__NBSHA__", NB_SHA), ("__NAME__", name)):
        code = code.replace(k, v)
    compile(code, f"<kit {name}>", "exec")
    slug = f"q27-e25-nb1-kit-{name}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert len(data) < 400_000, (name, len(data))
    assert b"\r" not in data.replace(b"\\r", b"")
    out = ROOT / f"{slug}__{spec['owner']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_bytes(data)
    meta = {"id": f"{spec['owner']}/{slug}", "title": spec["title"], "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": spec["models"], "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| NB1", NB_SHA[:8], flush=True)
    return out, code


if __name__ == "__main__":
    built = {n: build(n) for n in KITS}
    if "--dry" in sys.argv:
        base = Path(os.environ.get("KIT_LOCAL_ROOT", r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/nb1_kit_dry"))
        inp = os.environ.get("KIT_LOCAL_INPUT", r"D:/ARC AGI 2/_scratch/e13_rc2_local/input")
        for n, (out, code) in built.items():
            if len(sys.argv) > 2 and n not in sys.argv[2:]:
                continue
            d = base / n
            d.mkdir(parents=True, exist_ok=True)
            (d / "kit.py").write_text(code, encoding="utf-8")
            env = {**os.environ, "KIT_INPUT": inp, "KIT_TMP": str(d / "tmp"), "KIT_WORK": str(d / "work"), "PYTHONIOENCODING": "utf-8"}
            r = subprocess.run([sys.executable, str(d / "kit.py")], capture_output=True, text=True, env=env, timeout=4 * 3600,
                               encoding="utf-8", errors="replace")
            (d / "stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
            line = next((l for l in r.stdout.splitlines() if l.startswith("@@KIT")), None)
            print(f"DRY {n}: rc {r.returncode} | {line}", flush=True)
