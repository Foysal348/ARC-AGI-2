"""E25 humming low-bit CPU kit (structural track shortlist #1, first CPU test; user GO 2026-10-31).

One Kaggle CPU kernel (pinned Py3.12 image, no GPU, no internet), fail-soft, one JSON report, exactly one line
`@@KIT humming ALL OK@@` or `@@KIT humming NOT OK: ...@@`. Verdict ALL OK = H1 + H2 + H3 (H4 recorded).

  Merlin venv  the real offline install through NB3's own code (nb3_code_export.py: merlin_install + s_plan with
               NB3_CPU_KIT=1, exactly as the q27-e25-nb3-kit-merlin kernel that passed ALL OK), so the python and env
               are the GPU ones. Checks run as subprocesses of that venv python.
  H1  `import humming` in the Merlin venv: version 0.1.17, cu13 deps (nvrtc / cccl / nvcc / runtime), NVRTC library,
      precompiled nvrtc_compile; humming's own sm_89 tuner output (Sm89Heuristics) for the Flash-Next MoE GEMMs
      (EP2: 256 local experts, top-10, H 2560, I 640; w13 1280x2560, w2 2560x640; M 8-64; W4g128 / W3g128 / W3g64 /
      W2g128 / W2g64; grouped_contiguous + indexed).                         -> humkit_h12.py
  H2  NVRTC compile without a GPU of one MoE (w13, M 32) and one dense kernel (2560x2560, M 32) at W3/W2 x g128/g64,
      BF16 activations: seconds, cubin bytes; cache dir under /kaggle/working (what a B1 cache Model would carry).
      Device info is faked with L4 spec values (FAKE_L4_DEVICE_INFO); GPU-only failures = NO_GPU_COMPILE_UNVERIFIED.
  H3  quant-method resolution without a GPU for klee100 3bpw-MTP, Haberstroh int2-mixed and our Merlin W4A16 control
      (+ klee100 with vLLM-name layer keys as a config-only fix candidate): REAL_PATCHED_PLATFORM (current_platform
      patched to a CUDA platform reporting (8, 9) before any quant module import; real from_config +
      configure_quant_config + RoutedExperts on meta) and STATIC_CUDA_DISPATCH (pure Python rule re-evaluation).
      Per MoE layer: vLLM module name, resolved bits/group, method/backend/experts class, param-implied bits vs the
      checkpoint's real bits/group (safetensors headers); one bit-width per MoE layer; klee100's mixed layout
      (W3g128 0-11 & 36-47, W2g64 12-35, MTP W4g128); linear patterns; PLE format + Merlin's ple_format_guard.
      -> humkit_h3.py.  H3 OK = the real mode ran for all candidates and the Merlin control resolves its 48 MoE
      layers to Marlin at the checkpoint's own bits (sanity of the harness) and the layout assertions are evaluated.
  H4  (recorded) exact bytes per card in TP2xPP2 (25/23) per candidate from the safetensors headers (tensor shapes x
      dtypes; sharding rules below), calibrated against RC2's measured 17.33 / 16.87 GiB, the KV blocks freed at
      15.77 MiB per 4096-token block per card vs today's 149, and resident streams at 68k / 90k (F = 6).

Inputs: ONLY small JSON files of the two HF repos and our checkpoint's revision, downloaded beforehand with plain HTTPS
into D:/ARC AGI 2/_scratch/hf_lowbit_configs/<repo>/ (config.json, quantization_config.json, index, reports, ple.json)
plus each safetensors shard's JSON header (HTTP Range: 8-byte length + header). Data only; never executed. The builder
derives the per-layer bit table, module list, PLE formats and per-bucket byte totals from them and embeds those plus
the raw small configs (lzma + base85, sha256-checked) in the kit, which runs offline.

Run:  python build_e25_humming_cpu_kit.py          -> D:/ARC AGI 2/_scratch/push_e25/q27-e25-humming-cpu-kit__foysalemonshanto/
      python build_e25_humming_cpu_kit.py --dry    -> also runs the kit locally (no Merlin payload: H1/H2 with a local
      python + the extracted humming wheel if KIT_DRY_PY / KIT_DRY_PYTHONPATH are set, H3 static, H4)
"""
import base64
import collections
import hashlib
import json
import lzma
import math
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPORT = (HERE / "nb3_code_export.py").read_text(encoding="utf-8")
H12 = (HERE / "humkit_h12.py").read_text(encoding="utf-8")
H3 = (HERE / "humkit_h3.py").read_text(encoding="utf-8")
NB_SHA = hashlib.sha256((HERE.parent / "Q27-XHIGH-E25_NB3_KERNELS.ipynb").read_bytes()).hexdigest()
HF = Path(r"D:/ARC AGI 2/_scratch/hf_lowbit_configs")
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
SLUG = "q27-e25-humming-cpu-kit"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
MODELS = ["foysalemonshanto/e23-merlin-env/Other/v1/1",
          "foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/Transformers/default/1"]
REPOS = {"klee100": "klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP",
         "haberstroh": "HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang",
         "merlin": "halt95/Qwen3.8-Flash-Next-W4A16-Merlin"}
EMBED_FILES = {"klee100": ["config.json", "build_info.json", "validation_report.json", "quality_report.json",
                           "serving_report.json"],
               "haberstroh": ["config.json", "ple/ple.json"],
               "merlin": ["config.json", "quantization_config.json"]}
DT = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I32": 4, "I64": 8, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
      "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "U32": 4}
H_, I_, E_, NL = 2560, 640, 512, 48
PARAM_SUFFIX = {"weight", "qweight", "qzeros", "scales", "g_idx", "weight_packed", "weight_scale", "weight_shape",
                "bias", "weight_zero_point", "input_scale"}
SKIP_MOD = re.compile(r"norm|conv1d|A_log|dt_bias|ngram|ple_embedding|layer_multipliers|embed_tokens|visual")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def classify(n):
    """(bucket, category) for one HF tensor name. bucket: L<i> / mtp / embed / lm_head / vision / top.
    category: routed (EP2: /2 per card), sharded (TP2: /2), replicated (both TP ranks), ple_table_host (not on GPU)."""
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(.*)$", n)
    if m:
        bucket, rest = f"L{int(m.group(1))}", m.group(2)
    elif n.startswith("mtp."):
        bucket, rest = "mtp", n[4:]
    elif n.startswith("model.visual."):
        return "vision", "sharded"
    elif n.startswith("lm_head."):
        return "lm_head", "sharded"
    elif n.startswith("model.language_model.embed_tokens"):
        return "embed", "sharded"
    else:
        return "top", "replicated"          # final norm, hyper_connection_mixer
    r = "." + rest
    if "ngram_embedding" in r or ".ple.ple_embedding." in r:
        return bucket, "ple_table_host"
    if ".mlp.experts." in r:
        return bucket, "routed"
    if re.search(r"hyper_connection|norm|\.mlp\.gate\.|shared_expert_gate|A_log|dt_bias|\.ple\.", r):
        return bucket, "replicated"
    return bucket, "sharded"


def analyze(key):
    d = HF / REPOS[key]
    man = json.loads((d / "_fetch_manifest.json").read_text())
    heads, hshas = {}, {}
    for f, meta in man["headers"].items():
        b = (d / "_headers" / (f.replace("/", "__") + ".header.json")).read_bytes()
        assert sha(b) == meta["sha256"], (key, f)
        hshas[f] = meta["sha256"]
        for name, v in json.loads(b).items():
            if name != "__metadata__":
                heads[name] = (v["dtype"], v["shape"], f)
    idx = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(idx) == set(heads), (key, len(set(idx) ^ set(heads)))
    # ---- bytes per bucket x category
    agg = collections.defaultdict(lambda: collections.defaultdict(int))
    dtc = collections.Counter()
    for n, (dt, shape, f) in heads.items():
        nb = math.prod(shape) * DT[dt]
        b, c = classify(n)
        agg[b][c] += nb
        dtc[dt] += nb
    # ---- per MoE layer bits / group from the headers (all 512 experts x 3 projections)
    def proj_bits(base, proj):
        K = H_ if proj in ("gate_proj", "up_proj") else I_
        if base + ".qweight" in heads:
            q, s = heads[base + ".qweight"][1], heads[base + ".scales"][1]
            return ("gptq", q[0] * 32 // K, K // s[0], heads[base + ".scales"][0], base + ".qzeros" in heads)
        if base + ".weight_packed" in heads:
            q, s = heads[base + ".weight_packed"][1], heads[base + ".weight_scale"][1]
            return ("ct", q[1] * 32 // K, K // s[1], heads[base + ".weight_scale"][0], False)
        if base + ".weight" in heads:
            return ("dense", DT[heads[base + ".weight"][0]] * 8, None, heads[base + ".weight"][0], False)
        return ("absent", None, None, None, False)
    moe = {}
    for L in [str(x) for x in range(NL)] + ["mtp"]:
        pre = f"model.language_model.layers.{L}.mlp.experts" if L != "mtp" else "mtp.layers.0.mlp.experts"
        cnt = collections.Counter(proj_bits(f"{pre}.{e}.{p}", p) for e in range(E_) for p in ("gate_proj", "up_proj", "down_proj"))
        top = cnt.most_common(1)[0][0]
        moe[L] = {"fmt": top[0], "bits": top[1], "group": top[2], "scale_dtype": top[3], "has_qzeros": top[4],
                  "uniform": len(cnt) == 1, "variants": [[list(k), v] for k, v in cnt.items()][:4]}
    # ---- module list for H3 (HF names; experts / ple table / norms / vision skipped)
    mods = set()
    for n in heads:
        parts = n.split(".")
        if parts[-1] not in PARAM_SUFFIX or ".mlp.experts." in n or SKIP_MOD.search(n):
            continue
        mods.add(".".join(parts[:-1]))
    # ---- PLE table format
    ple_t = {n: v for n, v in heads.items() if "ngram_embedding" in n}
    if ple_t:
        dts = collections.Counter(v[0] for v in ple_t.values())
        ple = {"container": "safetensors", "dtype": dts.most_common(1)[0][0], "dtypes": dict(dts), "tensors": len(ple_t),
               "bytes": sum(math.prod(v[1]) * DT[v[0]] for v in ple_t.values())}
    else:
        pj = d / "ple" / "ple.json"
        pi = json.loads(pj.read_text()) if pj.is_file() else {}
        ple = {"container": "raw_bin (ple/" + pi.get("file", "?") + ")", "dtype": pi.get("dtype", "?").replace("F8_E4M3", "F8_E4M3"),
               "rows": pi.get("rows"), "dim": pi.get("dim"),
               "bytes": (pi.get("rows") or 0) * (pi.get("dim") or 0) * (1 if "F8" in pi.get("dtype", "") else 2)}
    small = {p: v for p, v in man["files"].items()}
    return {"repo": REPOS[key], "rev": man["rev"], "n_tensors": len(heads), "total_bytes": sum(dtc.values()),
            "dtype_bytes": dict(dtc), "bytes": {b: dict(c) for b, c in agg.items()}, "moe_layers": moe,
            "modules": sorted(mods), "ple": ple, "header_sha256": hshas, "small_files": small,
            "index_sha256": man["files"]["model.safetensors.index.json"]["sha256"]}


def pack(b):
    return base64.b85encode(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


HEAD = r'''# E25 humming low-bit CPU kit (pinned image, no GPU): see the builder docstring for H1-H4 and the verdict rule
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib, re, base64, zlib, lzma, math, glob, ctypes
from pathlib import Path
T0 = time.time()
KIT = "humming"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")          # env overrides: local dry run only
K = Path(os.environ.get("KIT_TMP", "/tmp/humkit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": "E25 humming low-bit CPU kit", "builder_nb3_notebook_sha256": "__NBSHA__", "python": sys.version,
       "checks": {}, "failed": []}
def save():
    (W / "humkit_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
EXPORT_SRC = __EXPORT__
(K / "nb3_code_export.py").write_text(EXPORT_SRC, encoding="utf-8", newline="\n")
REP["export_sha256"] = hashlib.sha256(EXPORT_SRC.encode("utf-8")).hexdigest()
def sh(name, args, env=None, timeout=3600, cwd=None):
    t = time.time()
    try:
        r = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=timeout, env=env or dict(os.environ),
                           cwd=str(cwd or K))
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out)
    return {"rc": rc, "sec": round(time.time() - t), "tail": out[-1500:], "_out": out}
def import_nb3():
    """NB3's notebook code with NB3_UNIT=1 / NB3_CPU_KIT=1 (definitions only, no bench) on INPUT_ROOT, as kit-merlin."""
    os.environ.update(NB3_UNIT="1", NB3_CPU_KIT="1", E23_OUT=str(K / "nb3_out"), E23_TMP=str(K / "nb3_tmp"), E23_INPUT=INPUT_ROOT,
                      RC2_SUBMISSION=str(K / "nb3_out/submission.json"), RC2_SHM=str(K / "nb3_shm"))
    os.environ.pop("E23_FAKE_SERVER", None)
    for d in ("nb3_out", "nb3_tmp", "nb3_shm"):
        (K / d).mkdir(exist_ok=True)
    return runpy.run_path(str(K / "nb3_code_export.py"), run_name="nb3kit")["finish"].__globals__
def verdict(ok, recorded=None):
    REP["verdict"] = {"checks": ok, "ALL_OK": all(ok.values()), "recorded": recorded or {}, "minutes": round((time.time() - T0) / 60, 1)}
    save()
    bad = [k for k, v in ok.items() if not v]
    print("VERDICT", json.dumps(REP["verdict"], default=str)[:8000], flush=True)
    print(f"@@KIT {KIT} ALL OK@@" if not bad else f"@@KIT {KIT} NOT OK: {', '.join(bad)}@@", flush=True)
'''

BODY = r'''
# ---------------------------------------------------------------- embedded inputs (sha256-checked)
EMB_BLOBS = __EMB__
EMBD = K / "emb"
emb = {}
for rel, (want, blob) in EMB_BLOBS.items():
    try:
        raw = lzma.decompress(base64.b85decode(blob))
        (EMBD / rel).parent.mkdir(parents=True, exist_ok=True)
        (EMBD / rel).write_bytes(raw)
        emb[rel] = {"bytes": len(raw), "sha256_ok": hashlib.sha256(raw).hexdigest() == want}
    except Exception as e:
        emb[rel] = {"error": f"{type(e).__name__}: {e}"[:300], "sha256_ok": False}
for rel, src in (("h12.py", __H12__), ("h3.py", __H3__)):
    (K / rel).write_text(src, encoding="utf-8", newline="\n")
REP["embedded"] = {"files": emb, "all_ok": all(v.get("sha256_ok") for v in emb.values()),
                   "h12_sha256": hashlib.sha256((K / "h12.py").read_bytes()).hexdigest(),
                   "h3_sha256": hashlib.sha256((K / "h3.py").read_bytes()).hexdigest()}
log("embedded", json.dumps({k: v.get("sha256_ok") for k, v in emb.items()}))
DER = json.loads((EMBD / "derived" / "lowbit_derived.json").read_text())
# our checkpoint's own config from the attached p1 Model (the control); compared with the HF revision copy
P1R = Path(INPUT_ROOT) / "models/foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1"
p1 = {"dir": str(P1R)}
for n in ("config.json", "quantization_config.json"):
    f = P1R / n
    if f.is_file():
        (EMBD / "p1").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, EMBD / "p1" / n)
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        p1[n] = {"sha256": h, "same_as_hf_cec53e9f": h == DER["merlin"]["small_files"].get(n, {}).get("sha256")}
    else:
        p1[n] = "absent"
REP["p1_control"] = p1
save()

# ---------------------------------------------------------------- Merlin offline install (kit-merlin code path)
MER = {}
PY, ENV, MODE = None, None, None
try:
    G = import_nb3()
    MER["inputs"] = G["R"]["inputs"]
    G["MERLIN_OK"] = True           # the checkpoint parts are not attached (only p1's small files are read here)
    t = time.time()
    eng = G["merlin_install"]()
    MER["install"] = {"ok": True, "sec": round(time.time() - t), "kv": dict(eng["kv"]), "cuda_home": eng["cuda_home"]}
    pl = G["s_plan"]()
    MER["bench_env"] = {k: v for k, v in pl.items()}
    PY, ENV, MODE = G["BENCH"]["py"], G["bench_env"](""), "MERLIN_VENV"
    log("install", json.dumps(MER["install"])[:500])
except Exception as e:
    MER.setdefault("install", {"ok": False})
    MER["fatal"] = f"{type(e).__name__}: {e}"[:1000]
    MER["tb"] = traceback.format_exc()[-2500:]
    log("Merlin install FATAL", MER["fatal"])
    if os.environ.get("KIT_DRY_PY"):
        PY, MODE = os.environ["KIT_DRY_PY"], "DRY_LOCAL_PYTHON"
        ENV = dict(os.environ)
        if os.environ.get("KIT_DRY_PYTHONPATH"):
            ENV["PYTHONPATH"] = os.environ["KIT_DRY_PYTHONPATH"]
MER["check_python_mode"] = MODE
REP["checks"]["merlin"] = MER
save()

# libcuda: humming's driver-API imports need libcuda.so.1 to load (never to run here). The CPU image may lack it; then
# a stub libcuda.so from the toolkit is offered via LD_LIBRARY_PATH (recorded). It is used only if the first try fails.
STUB = {}
try:
    ctypes.CDLL("libcuda.so.1")
    STUB["libcuda_so_1"] = "loadable"
except OSError as e:
    STUB["libcuda_so_1"] = f"not loadable: {e}"[:200]
    cands = []
    for root in [(ENV or {}).get("CUDA_HOME", ""), "/usr/local/cuda", "/usr/local/cuda/targets/x86_64-linux"]:
        if root:
            cands += glob.glob(f"{root}/lib*/stubs/libcuda.so") + glob.glob(f"{root}/**/stubs/libcuda.so", recursive=True)
    if cands:
        d = K / "cudastub"
        d.mkdir(exist_ok=True)
        try:
            os.symlink(cands[0], d / "libcuda.so.1")
        except FileExistsError:
            pass
        STUB.update(stub=cands[0], dir=str(d))
REP["checks"]["libcuda"] = STUB
def env_for(extra=None, stub=False):
    e = dict(ENV or os.environ)
    e.update({"PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": ""}, **(extra or {}))
    if stub and STUB.get("dir"):
        e["LD_LIBRARY_PATH"] = STUB["dir"] + ":" + e.get("LD_LIBRARY_PATH", "")
    return e
def needs_stub(text):
    return "libcuda" in (text or "") and bool(STUB.get("dir"))
save()

# ---------------------------------------------------------------- H1 + H2 (humming in the Merlin venv)
H1, H2 = {}, {}
HC = W / "humming_cache"
shutil.rmtree(HC, ignore_errors=True)
if PY:
    henv = {"HUMMING_CACHE_DIR": str(HC), "HUMMING_TMP_DIR": str(K / "humming_tmp"), "HUMMING_DISABLE_PARALLEL_BUILD": "1"}
    r = sh("h12", [PY, K / "h12.py", W / "h12.json"], env_for(henv), 3600)
    used_stub = False
    if needs_stub(r["_out"] + ((W / "h12.json").read_text() if (W / "h12.json").is_file() else "")) and \
            '"import_ok": false' in ((W / "h12.json").read_text() if (W / "h12.json").is_file() else ""):
        r = sh("h12_stub", [PY, K / "h12.py", W / "h12.json"], env_for(henv, stub=True), 3600)
        used_stub = True
    try:
        res = json.loads((W / "h12.json").read_text())
    except Exception as e:
        res = {"H1": {"error": f"no h12.json: {e}", "tail": r["tail"]}, "H2": {}}
    H1, H2 = res.get("H1", {}), res.get("H2", {})
    H1.update(rc=r["rc"], sec=r["sec"], libcuda_stub_used=used_stub, fake_device=res.get("fake_device"),
              libcuda_in_venv=res.get("libcuda_so_1"), run_mode=MODE)
else:
    H1 = {"error": "no python to run humming in (Merlin install failed)"}
d = H1.get("dists") or {}
H1["ok"] = bool(H1.get("import_ok") and H1.get("version") == "0.1.17" and d.get("nvidia-cuda-nvrtc") and d.get("nvidia-cuda-cccl")
                and H1.get("nvrtc_lib") and H1.get("heuristics_class") == "Sm89Heuristics" and H1.get("heuristics_ok"))
H2["ok"] = bool(H2.get("all_compiled"))
log("H1", json.dumps({k: H1.get(k) for k in ("import_ok", "version", "dists", "nvrtc_lib", "heuristics_class", "heuristics_ok", "ok")},
                     default=str)[:1200])
log("H2", json.dumps({k: {kk: v.get(kk) for kk in ("ok", "compile_sec", "cubin_bytes", "label", "error")}
                      for k, v in (H2.get("kernels") or {}).items()}, default=str)[:1600])
REP["checks"]["H1"], REP["checks"]["H2"] = H1, H2
save()

# ---------------------------------------------------------------- H3 (resolution; real patched platform + static)
H3 = {}
if PY:
    mode = "real" if MODE == "MERLIN_VENV" else "static"
    r = sh("h3", [PY, K / "h3.py", EMBD, W / "h3.json", mode], env_for(), 2400)
    if mode == "real" and needs_stub(r["_out"]):
        r = sh("h3_stub", [PY, K / "h3.py", EMBD, W / "h3.json", mode], env_for(stub=True), 2400)
        H3["libcuda_stub_used"] = True
    try:
        H3.update(json.loads((W / "h3.json").read_text()))
    except Exception as e:
        H3["error"] = f"no h3.json: {e}"
    H3.update(rc=r["rc"], sec=r["sec"], tail=r["tail"][-600:])
else:
    r = sh("h3_sys", [sys.executable, K / "h3.py", EMBD, W / "h3.json", "static"], dict(os.environ), 1200)
    try:
        H3.update(json.loads((W / "h3.json").read_text()))
    except Exception as e:
        H3["error"] = f"no h3.json: {e}"
    H3.update(rc=r["rc"], note="Merlin venv unavailable: STATIC_CUDA_DISPATCH only (system python)")
real = H3.get("real") or {}
lay = H3.get("checkpoint_layout") or {}
H3["ok"] = bool(real.get("steps", {}).get("platform", {}).get("ok") and real.get("ran_all") and real.get("control_ok")
                and (lay.get("klee100_mixed_layout") or {}).get("ok") is not None
                and all(k in lay for k in ("klee100", "haberstroh", "merlin")))
summ = {}
for c, v in (real.get("cands") or {}).items():
    summ[c] = {"loadable_as_is": v.get("loadable_as_is"), "moe_mismatch_layers": v.get("moe_mismatch_layers"),
               "methods": v.get("methods"), "fatal": v.get("fatal"),
               "ple": {k: (v.get("ple") or {}).get(k) for k in ("as_is", "fp8_graft", "merlin_launcher_rule_launch_py_1219")}}
for c, v in ((H3.get("static") or {}).get("cands") or {}).items():
    summ.setdefault(c, {})["static"] = {"loadable_as_is": v.get("loadable_as_is"), "moe_mismatch_layers": v.get("moe_mismatch_layers")}
H3["summary"] = summ
log("H3", json.dumps(summ, default=str)[:2500])
REP["checks"]["H3"] = H3
save()

# ---------------------------------------------------------------- H4 (recorded): bytes per card, KV blocks, streams
GiB, MiB = 2 ** 30, 2 ** 20
BLOCK = 2479838700 / 150                       # 15.77 MiB per 4096-token block per card (RC2 FP8 pin / 150 blocks)
F_FIXED = 6                                    # fixed blocks per request (RC2, engine-derived)
RC2 = {"weights_gib": {"PP0": 17.33, "PP1": 16.87}, "min_free_mib": {"PP0": 1607, "PP1": 1633}, "blocks": 149}
def stage_of(b):
    if b.startswith("L"):
        return "PP0" if int(b[1:]) < 25 else "PP1"
    return {"embed": "PP0", "vision": "PP0", "mtp": "PP1", "lm_head": "PP1", "top": "PP1"}[b]
def per_card(cand, with_vision=True):
    out = {"PP0": 0.0, "PP1": 0.0}
    for b, cats in DER[cand]["bytes"].items():
        if b == "vision" and not with_vision:
            continue
        s = stage_of(b)
        out[s] += cats.get("routed", 0) / 2 + cats.get("sharded", 0) / 2 + cats.get("replicated", 0)
    return out
H4 = {"rules": "TP2xPP2+EP, partition 25/23 (PP0 = layers 0-24 + embed + vision; PP1 = 25-47 + MTP + lm_head + final norm/HC "
               "mixer). Routed experts /2 (EP2), sharded /2 (TP2), replicated x1 (HC, norms, router, shared_expert_gate, "
               "A_log, dt_bias, PLE layer projections); PLE n-gram table on host (excluded). Same rules for every candidate, "
               "so the differences are exact header bytes; the absolute level is calibrated on RC2.",
      "block_bytes": BLOCK, "F": F_FIXED, "rc2": RC2, "cands": {}}
ref = {v: per_card("merlin", v) for v in (True, False)}
H4["merlin_calibration"] = {s: {"computed_gib": round(ref[True][s] / GiB, 3), "rc2_measured_gib": RC2["weights_gib"][s],
                                "delta_gib": round(RC2["weights_gib"][s] - ref[True][s] / GiB, 3)} for s in ("PP0", "PP1")}
for cand in ("merlin", "klee100", "haberstroh"):
    rec = {"total_header_bytes": DER[cand]["total_bytes"], "dtype_bytes": DER[cand]["dtype_bytes"], "ple": DER[cand]["ple"],
           "moe_layers": {L: [v["fmt"], v["bits"], v["group"]] for L, v in DER[cand]["moe_layers"].items()}}
    for vis in (True, False):
        pc = per_card(cand, vis)
        tag = "with_vision" if vis else "language_model_only"
        freed = {s: ref[vis][s] - pc[s] for s in ("PP0", "PP1")}
        # GPU weights per card = RC2-measured weights - (Merlin computed - candidate computed) on the same rules
        est = {s: RC2["weights_gib"][s] * GiB - freed[s] for s in ("PP0", "PP1")}
        blocks_no_c5 = RC2["blocks"] + min(math.floor(freed[s] / BLOCK) for s in ("PP0", "PP1"))
        blocks_c5 = RC2["blocks"] + min(math.floor((freed[s] + (RC2["min_free_mib"][s] - 512) * MiB) / BLOCK) for s in ("PP0", "PP1"))
        streams = {}
        for L in (68000, 90000, 120000):
            per = math.ceil(L / 4096) + F_FIXED
            streams[str(L)] = {"blocks_per_request": per, "no_c5": blocks_no_c5 // per, "with_c5": blocks_c5 // per}
        rec[tag] = {"computed_gib_per_card": {s: round(pc[s] / GiB, 3) for s in pc},
                    "est_gpu_weights_gib_per_card": {s: round(est[s] / GiB, 3) for s in est},
                    "freed_vs_merlin_gib_per_card": {s: round(freed[s] / GiB, 3) for s in freed},
                    "kv_blocks_no_c5": blocks_no_c5, "kv_blocks_with_c5_pin": blocks_c5,
                    "binding_stage": min(("PP0", "PP1"), key=lambda s: freed[s] + (RC2["min_free_mib"][s] - 512) * MiB),
                    "resident_streams": streams}
    H4["cands"][cand] = rec
log("H4", json.dumps({c: {t: {k: v[t][k] for k in ("est_gpu_weights_gib_per_card", "kv_blocks_no_c5", "kv_blocks_with_c5_pin")}
                          for t in ("with_vision", "language_model_only")} for c, v in H4["cands"].items()})[:2000])
REP["checks"]["H4"] = H4
cache = (H2.get("cache_dir") or {})
verdict({"H1_humming_import_and_sm89_heuristics": H1.get("ok") is True, "H2_nvrtc_compile_sm89_w3_w2": H2.get("ok") is True,
         "H3_resolution_real_patched_platform": H3.get("ok") is True},
        recorded={"embedded_all_sha_ok": REP["embedded"]["all_ok"], "merlin_install": MER.get("install", {}).get("ok"),
                  "check_python_mode": MODE, "libcuda": STUB, "h2_cache": cache,
                  "h2_labels": {k: v.get("label") for k, v in (H2.get("kernels") or {}).items() if not v.get("ok")},
                  "h3_summary": H3.get("summary"), "klee100_mixed_layout_ok": (lay.get("klee100_mixed_layout") or {}).get("ok"),
                  "h4_kv_blocks_lm_only_with_c5": {c: v["language_model_only"]["kv_blocks_with_c5_pin"] for c, v in H4["cands"].items()}})
'''


def build():
    der = {k: analyze(k) for k in REPOS}
    # consistency facts the kit relies on
    kq = json.loads((HF / REPOS["klee100"] / "quantization_config.json").read_text())
    assert kq == json.loads((HF / REPOS["klee100"] / "config.json").read_text())["quantization_config"]
    der["notes"] = {"klee100_quantization_config_json_equals_config_quantization_config": True,
                    "derived_by": "build_e25_humming_cpu_kit.py from the downloaded headers / index / configs"}
    der_b = json.dumps(der, separators=(",", ":"), sort_keys=True).encode()
    emb = {"derived/lowbit_derived.json": [sha(der_b), pack(der_b)]}
    files_rec = {"derived/lowbit_derived.json": {"bytes": len(der_b), "sha256": sha(der_b)}}
    for key, files in EMBED_FILES.items():
        man = json.loads((HF / REPOS[key] / "_fetch_manifest.json").read_text())
        for f in files:
            b = (HF / REPOS[key] / f).read_bytes()
            assert sha(b) == man["files"][f]["sha256"] and len(b) == man["files"][f]["size"], (key, f)
            rel = f"hf/{key}/{f}"
            emb[rel] = [sha(b), pack(b)]
            files_rec[rel] = {"bytes": len(b), "sha256": sha(b), "source": f"https://huggingface.co/{REPOS[key]}/resolve/{man['rev']}/{f}"}
    code = HEAD + BODY
    for k, v in (("__EXPORT__", json.dumps(EXPORT)), ("__H12__", json.dumps(H12)), ("__H3__", json.dumps(H3)),
                 ("__EMB__", json.dumps(emb)), ("__NBSHA__", NB_SHA)):
        code = code.replace(k, v)
    compile(code, "<humming kit>", "exec")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert len(data) < 400_000, len(data)
    assert b"\r" not in data.replace(b"\\r", b"")
    out = ROOT / f"{SLUG}__foysalemonshanto"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{SLUG}.ipynb").write_bytes(data)
    meta = {"id": f"foysalemonshanto/{SLUG}", "title": "Q27 E25 Humming CPU Kit", "code_file": f"{SLUG}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": [], "kernel_sources": [],
            "competition_sources": ["arc-prize-2026-arc-agi-2"],        # NB3's import requires the ARC eval file (code line 100-102)
            "model_sources": MODELS, "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out, "|", len(data), "bytes | sha256", sha(data), flush=True)
    print("embedded:", json.dumps(files_rec, indent=1), flush=True)
    return out, code, der


if __name__ == "__main__":
    out, code, der = build()
    for c in REPOS:
        m = der[c]["moe_layers"]
        print(c, "moe:", {L: (v["fmt"], v["bits"], v["group"], v["uniform"]) for L, v in list(m.items())[:2]}, "...",
              "mtp", (m["mtp"]["fmt"], m["mtp"]["bits"], m["mtp"]["group"]), "| modules", len(der[c]["modules"]),
              "| ple", der[c]["ple"], flush=True)
    if "--dry" in sys.argv:
        base = Path(os.environ.get("KIT_LOCAL_ROOT", r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/humkit_dry"))
        inp = Path(os.environ.get("KIT_LOCAL_INPUT", str(base / "input")))
        base.mkdir(parents=True, exist_ok=True)
        (base / "kit.py").write_text(code, encoding="utf-8")
        env = {**os.environ, "KIT_INPUT": str(inp), "KIT_TMP": str(base / "tmp"), "KIT_WORK": str(base / "work")}
        r = subprocess.run([sys.executable, str(base / "kit.py")], capture_output=True, text=True, env=env, timeout=3600)
        (base / "stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        print("DRY rc", r.returncode, "|", next((l for l in r.stdout.splitlines() if l.startswith("@@KIT")), None), flush=True)
