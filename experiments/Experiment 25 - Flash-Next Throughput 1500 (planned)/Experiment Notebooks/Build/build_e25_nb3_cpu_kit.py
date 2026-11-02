"""E25 NB3 Kaggle CPU kits (user 2026-10-31: three independent CPU kernels in parallel on foysalemonshanto, pinned
Py3.12 image, no GPU, no internet). Pattern: Experiment 13 build_e13_rc2_cpu_kit.py (RC2 kit v1 ALL OK). Each kernel is
fail-soft, writes its own JSON report and prints exactly one line `@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: ...@@`.

  K1 q27-e25-nb3-kit-mounts  R1: NB3's own code (NB3_UNIT + NB3_CPU_KIT) on the REAL mounts: inputs at the exact paths,
                             merge (sizes), the full background sha256, the embedded-file sha checks + audit copies;
                             R7: layer-0 expert tensors (+ the D4 header names) from the merged dir's safetensors headers
                             (parsed directly: 8-byte length + JSON) against the names / shapes / dtypes the benches expect.
                             ALL OK = inputs + merge + sha256 + embedded files + R7.     Inputs: the 7 RC2 Models + ARC.
  K2 q27-e25-nb3-kit-merlin  R2: the real Merlin offline install (merlin_install); bench env (s_plan: the RC2 plan if
                             host_phase can run, else NB3's EXPLORATORY_ENV, allowed only under NB3_CPU_KIT, recorded); R3: imports inside the Merlin venv of
                             every Merlin API the benches call (+ the 52 MERLIN_MARLIN_MOE_CFG sweep rules parsed by Merlin's
                             own parser, the nb3 bench modules); R4: the P5 cpp_extension build (compile + .so load probe);
                             R5: ncu present / version. ALL OK = R2 (R3 / R4 / R5 recorded). Inputs: e23-merlin-env + ARC.
  K3 q27-e25-nb3-kit-units   R6: nb3_unit_tests.py on Py3.12 (fault injection, fake end-to-end, rerun) on a fake tree
                             made from the real part-1 small files. ALL OK = rc 0 and "X/X PASS". Inputs: part 1 + ARC.

Run: python build_e25_nb3_cpu_kit.py  ->  D:/ARC AGI 2/_scratch/push_e25/q27-e25-nb3-kit-{mounts,merlin,units}__foysalemonshanto/
Local dry run (paths overridden by KIT_INPUT / KIT_TMP / KIT_WORK / KIT_LOCAL_*): python build_e25_nb3_cpu_kit.py --dry
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E13 = Path(r"D:/ARC AGI 2/experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)")
EXPORT = (HERE / "nb3_code_export.py").read_text(encoding="utf-8")
UNITS = (HERE / "nb3_unit_tests.py").read_text(encoding="utf-8")
PARTS = (E13 / "prep" / "merlin_fn_ckpt" / "kernels" / "parts.json").read_text(encoding="utf-8")
NB_SHA = hashlib.sha256((HERE.parent / "Q27-XHIGH-E25_NB3_KERNELS.ipynb").read_bytes()).hexdigest()
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
RC2_MODELS = ([f"foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p{k}/Transformers/default/1" for k in range(1, 6)] +
              ["foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
               "foysalemonshanto/e23-merlin-env/Other/v1/1"])

HEAD = r'''# E25 NB3 CPU kit __NAME__ (pinned image, no GPU): see the builder docstring for the checks and the verdict rule
import os, sys, json, time, subprocess, shutil, runpy, traceback, hashlib, re, base64, zlib
from pathlib import Path
T0 = time.time()
KIT = "__NAME__"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")          # env overrides: local dry run only
K = Path(os.environ.get("KIT_TMP", "/tmp/nb3kit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
REP = {"what": f"E25 NB3 CPU kit {KIT}", "notebook_sha256": "__NBSHA__", "python": sys.version, "checks": {}, "failed": []}
def save():
    (W / f"nb3_kit_{KIT}_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
EXPORT_SRC = __EXPORT__
(K / "nb3_code_export.py").write_text(EXPORT_SRC, encoding="utf-8", newline="\n")
REP["export_sha256"] = hashlib.sha256(EXPORT_SRC.encode("utf-8")).hexdigest()
def embedded_files():
    """The NB3 bench files exactly as the notebook carries them (zlib + base64 blob inside the export)."""
    m = re.search(r"b64decode\('([A-Za-z0-9+/=]+)'\)", EXPORT_SRC)
    return json.loads(zlib.decompress(base64.b64decode(m.group(1))).decode("utf-8"))
def sh(name, args, env=None, timeout=3600, cwd=None):
    t = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})}, cwd=str(cwd or K))
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out)
    return {"rc": rc, "sec": round(time.time() - t), "tail": out[-1500:], "_out": out}
def import_nb3(real=True, extra=None):
    """NB3's notebook code with NB3_UNIT=1 (definitions + A0 merge + the embedded files; no bench) on INPUT_ROOT."""
    os.environ.update(NB3_UNIT="1", NB3_CPU_KIT="1", E23_OUT=str(K / "nb3_out"), E23_TMP=str(K / "nb3_tmp"), E23_INPUT=INPUT_ROOT,
                      RC2_SUBMISSION=str(K / "nb3_out/submission.json"), RC2_SHM=str(K / "nb3_shm"), **(extra or {}))
    os.environ.pop("E23_FAKE_SERVER", None)
    for d in ("nb3_out", "nb3_tmp", "nb3_shm"):
        (K / d).mkdir(exist_ok=True)
    return runpy.run_path(str(K / "nb3_code_export.py"), run_name="nb3kit")["finish"].__globals__
def verdict(ok, recorded=None):
    REP["verdict"] = {"checks": ok, "ALL_OK": all(ok.values()), "recorded": recorded or {}, "minutes": round((time.time() - T0) / 60, 1)}
    save()
    bad = [k for k, v in ok.items() if not v]
    print("VERDICT", json.dumps(REP["verdict"], default=str)[:6000], flush=True)
    print(f"@@KIT {KIT} ALL OK@@" if not bad else f"@@KIT {KIT} NOT OK: {', '.join(bad)}@@", flush=True)
'''

K1 = r'''
# ---------------------------------------------------------------- R1: NB3 code on the real mounts
R1, R7 = {}, {}
try:
    t = time.time()
    G = import_nb3()
    R = G["R"]
    R1.update(load_sec=round(time.time() - t), inputs=R["inputs"], merge=R["sections"]["A0"]["merge"], host_mem=R["sections"]["A0"]["host_mem"],
              routing_file=R["setup"].get("routing_file"), nb3_grid=R.get("nb3_grid"))
    shas = R["setup"].get("nb3_files") or {}
    copies = {k: (K / "nb3_out" / "nb3_files" / k).is_file() and hashlib.sha256((K / "nb3_out" / "nb3_files" / k).read_bytes()).hexdigest() == v
              for k, v in shas.items()}
    R1["embedded"] = {"files": len(shas), "audit_copies_ok": copies, "same_as_blob": sorted(embedded_files()) == sorted(shas)}
    log("R1 inputs", json.dumps(R["inputs"])[:500]); log("R1 merge", json.dumps(R1["merge"])[:400]); save()
    t = time.time()
    ok, why = G["sha_gate"](3600)
    R1["sha"] = {"ok": ok, "why": why, "wait_sec": round(time.time() - t), "detail": R["setup"].get("sha256")}
    R1["page_cache_drop"] = G["drop_ckpt_cache"]()
    log("R1 sha", json.dumps(R1["sha"])[:400])
    CK = G["CKPT"]
except Exception as e:
    R1["fatal"] = f"{type(e).__name__}: {e}"[:1000]; R1["tb"] = traceback.format_exc()[-2500:]; CK = None
    log("R1 FATAL", R1["fatal"])
REP["checks"]["R1"] = R1; save()

# ---------------------------------------------------------------- R7: safetensors headers of the merged dir
def st_header(path):
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        return json.loads(fh.read(n)), 8 + n
def st_tensor_bytes(path, meta, base):
    a, b = meta["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(base + a)
        return fh.read(b - a)
try:
    cfg = json.loads((CK / "config.json").read_text())
    tc = cfg.get("text_config") or cfg
    wq = cfg["quantization_config"]["config_groups"]["group_0"]["weights"]
    E, H, I, bits, G_ = int(tc["num_experts"]), int(tc["hidden_size"]), int(tc["moe_intermediate_size"]), int(wq["num_bits"]), int(wq["group_size"])
    R7["config"] = {"E": E, "topk": tc.get("num_experts_per_tok"), "H": H, "I": I, "bits": bits, "group": G_, "symmetric": wq.get("symmetric"),
                    "type": wq.get("type"), "hidden_dtype": tc.get("dtype")}
    wm = json.loads((CK / "model.safetensors.index.json").read_text())["weight_map"]
    pre = "model.language_model.layers.0.mlp.experts."
    heads, bad, scale_dtypes, files = {}, [], set(), set()
    want = {"gate_proj": (I, H), "up_proj": (I, H), "down_proj": (H, I)}
    for e in range(E):
        for proj, (n0, n1) in want.items():
            for kind, shape, dts in (("weight_packed", [n0, n1 * bits // 32], ("I32",)), ("weight_scale", [n0, n1 // G_], ("BF16", "F16", "F32")),
                                     ("weight_shape", [2], ("I64", "I32"))):
                name = f"{pre}{e}.{proj}.{kind}"
                f = wm.get(name)
                if f is None:
                    bad.append([name, "absent from index"]); continue
                files.add(f)
                if f not in heads:
                    heads[f] = st_header(CK / f)
                meta = heads[f][0].get(name)
                if meta is None:
                    bad.append([name, f"not in {f} header"]); continue
                if meta["shape"] != shape or meta["dtype"] not in dts:
                    bad.append([name, meta["shape"], meta["dtype"], "want", shape, dts])
                if kind == "weight_scale":
                    scale_dtypes.add(meta["dtype"])
                if kind == "weight_shape" and e in (0, 255, 256, 511) and not bad:
                    raw = st_tensor_bytes(CK / f, meta, heads[f][1])
                    w = 8 if meta["dtype"] == "I64" else 4
                    vals = [int.from_bytes(raw[i:i + w], "little") for i in range(0, len(raw), w)]
                    if vals != [n0, n1]:
                        bad.append([name, "value", vals, "want", [n0, n1]])
    R7["experts"] = {"names_checked": E * 9, "bad": bad[:20], "n_bad": len(bad), "files": sorted(files), "scale_dtypes": sorted(scale_dtypes)}
    d4 = {}
    for name in ("model.language_model.layers.0.mlp.gate.weight", "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
                 "model.language_model.layers.0.mlp.shared_expert.down_proj.weight", "model.language_model.layers.0.mlp.shared_expert_gate.weight",
                 "model.language_model.layers.0.linear_attn.in_proj_a.weight", "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                 "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
                 "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight", "lm_head.weight") + tuple(
            sorted(n for n in wm if n.startswith("model.language_model.layers.3.self_attn.") and n.endswith(".weight"))[:16]):
        f = wm.get(name)
        if f is None:
            d4[name] = "absent"; continue
        if f not in heads:
            heads[f] = st_header(CK / f)
        m = heads[f][0].get(name)
        d4[name] = {"shape": m["shape"], "dtype": m["dtype"]} if m else "not in header"
    R7["d4_headers"] = d4
    R7["ok"] = (not bad and E == 512 and H == 2560 and I == 640 and bits == 4 and G_ == 128 and wq.get("symmetric") is True
                and all(isinstance(v, dict) for v in d4.values()))
except Exception as e:
    R7.update(ok=False, error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
log("R7", json.dumps({k: v for k, v in R7.items() if k != "d4_headers"}, default=str)[:900])
REP["checks"]["R7"] = R7
emb = R1.get("embedded") or {}
verdict({"inputs_found": bool((R1.get("inputs") or {}).get("merlin_inputs_ok")), "merge_ok": bool((R1.get("merge") or {}).get("ok")),
         "sha256_all_ok": bool((R1.get("sha") or {}).get("ok")),
         "embedded_files_ok": emb.get("files") == 10 and all((emb.get("audit_copies_ok") or {}).values()) and emb.get("same_as_blob") is True,
         "R7_expert_headers_ok": R7.get("ok") is True})
'''

K2 = r'''
# ---------------------------------------------------------------- R2: Merlin offline install + the bench env
R2, R3, R4, R5 = {}, {}, {}, {}
G = None
try:
    G = import_nb3()
    R2["inputs"] = G["R"]["inputs"]
    G["MERLIN_OK"] = True              # the checkpoint parts are not attached to this kit (K1 owns them); the payload is
    t = time.time()
    eng = G["merlin_install"]()
    R2["install"] = {"ok": True, "sec": round(time.time() - t), "kv": dict(eng["kv"]), "cuda_home": eng["cuda_home"]}
    log("R2 install", json.dumps(R2["install"])[:600]); save()
    pl = G["s_plan"]()
    R2["bench_env"] = {k: v for k, v in pl.items()}
    R2["bench_env_note"] = ("no checkpoint in this kit: host_phase cannot pass here, so NB3 (NB3_CPU_KIT=1 only) uses its "
                            "EXPLORATORY_ENV; on the GPU run the same failure fails closed (N3-23); the RC2 CPU kit v1 proved "
                            "host_phase ok on the full mounts") if pl.get("plan_label") == "EXPLORATORY_ENV" else "RC2 plan"
    R2["bench_env_label"] = pl.get("plan_label")
    log("R2 bench env", str(pl.get("plan"))[:300])
except Exception as e:
    R2.setdefault("install", {"ok": False})
    R2["fatal"] = f"{type(e).__name__}: {e}"[:1000]; R2["tb"] = traceback.format_exc()[-2500:]
    log("R2 FATAL", R2["fatal"])
REP["checks"]["R2"] = R2; save()

R3_CODE = r"""
import importlib, json, os, sys, traceback
sys.path.insert(0, sys.argv[1])
out = {"python": sys.version.split()[0]}
def chk(name, attr=None, fn=None):
    key = name + ("." + attr if attr else "")
    try:
        m = importlib.import_module(name)
        v = getattr(m, attr) if attr else m
        out[key] = {"ok": True, **({"info": fn(v)} if fn else {})}
    except BaseException as e:
        out[key] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
chk("torch", fn=lambda m: m.__version__)
chk("vllm", fn=lambda m: [getattr(m, "__version__", None), m.__file__])
chk("vllm._custom_ops", "moe_wna16_marlin_gemm")
chk("vllm._custom_ops", "gptq_marlin_moe_repack")
def opreg(_):
    import torch
    return {"_moe_C.moe_wna16_marlin_gemm": hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm"),
            "_C.gptq_marlin_repack": hasattr(torch.ops._C, "gptq_marlin_repack"), "_moe_C.moe_sum": hasattr(torch.ops._moe_C, "moe_sum"),
            "_C.marlin_gemm (draft-head path)": hasattr(torch.ops._C, "marlin_gemm")}
chk("vllm._custom_ops", fn=opreg)
chk("vllm.model_executor.layers.fused_moe.experts.marlin_moe", "fused_marlin_moe")
chk("vllm.model_executor.layers.fused_moe.oracle.int_wna16", "_process_weights_marlin")
chk("vllm.model_executor.layers.fused_moe.oracle.int_wna16", "_unpack_and_dequant_int4_gptq", fn=lambda f: str(__import__("inspect").signature(f))[:400])
chk("vllm.model_executor.layers.fused_moe.oracle.int_wna16", "_process_weights_marlin", fn=lambda f: str(__import__("inspect").signature(f))[:400])
chk("vllm.model_executor.layers.fused_moe.experts.fused_humming_moe", "HummingIndexedExperts")
def hum(m):
    import nb3_humming as HMm
    return HMm.availability()
chk("humming", fn=hum)
def dhp(m):
    import torch, nb3_gemm_bench as GB
    return GB.draft_head_preflight(torch)
chk("nb3_gemm_bench", fn=dhp)
chk("vllm.model_executor.layers.fused_moe.moe_align_block_size", "moe_align_block_size")
chk("vllm.model_executor.layers.fused_moe.expert_map_manager", "determine_expert_map",
    fn=lambda f: [int(x) for x in (f(2, 0, 512)[1] >= 0).nonzero().flatten()[[0, -1]].tolist()])
chk("vllm.scalar_type", "scalar_types", fn=lambda s: s.uint4b8.id)
chk("vllm.utils.torch_utils", "current_stream")
def cfgrules(MC):
    import nb3_moe_bench as B
    rules = B.sweep_rules(MC)
    parsed = [len(MC.parse_rules(r)) for r in rules]
    return {"rules": len(rules), "all_parse": all(p == 1 for p in parsed), "compiled_small": MC._COMPILED_SMALL_M, "bps": MC.BLOCKS_PER_SM}
chk("vllm.merlin.marlin_moe_cfg", fn=cfgrules)
chk("vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm", "shape_dynamic_skinny_gemm", fn=lambda s: {"is_available": s.is_available()})
chk("vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm", "SkinnyGemmConfig")
def plans(m):
    import nb3_common as C
    p = m.QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY
    return {"caps": [list(k) for k in p if k], "has_8_9": (8, 9) in p, "sm121_shapes": len(m.QWEN4_EXP_SM121_GEMM_PLANS),
            "nb3_shapes_in_sm121": {f"{n},{k}": (n, k) in m.QWEN4_EXP_SM121_GEMM_PLANS for n, k, *_ in C.gemm_shapes()}}
chk("vllm.models.qwen4_exp.nvidia.low_latency_gemm", fn=plans)
chk("vllm.models.qwen4_exp.nvidia.low_latency_gemm", "_qwen4_exp_low_latency_gemm")
chk("vllm.model_executor.layers.draft_lm_head_quant", "QuantizedDraftLMHead")
chk("vllm.platforms", "current_platform", fn=lambda p: type(p).__name__)
for m in ("cutlass", "cutlass.cute", "quack.compile_utils", "safetensors", "numpy", "nb3_common", "nb3_moe_bench", "nb3_humming"):
    chk(m)
print("R3_JSON " + json.dumps(out, default=str))
"""
try:
    G["NB3_DIR"].mkdir(parents=True, exist_ok=True)
    (K / "r3_imports.py").write_text(R3_CODE)
    r = sh("r3_imports", [G["BENCH"]["py"], str(K / "r3_imports.py"), str(G["NB3_DIR"])], G["bench_env"](""), 900, cwd=G["NB3_DIR"])
    R3 = next((json.loads(l[8:]) for l in r["_out"].splitlines() if l.startswith("R3_JSON ")), {"error": r["tail"]})
    R3["rc"] = r["rc"]
    p5 = G["NB3_DIR"] / "p5_src"
    R3["host_ar_sources"] = {n: hashlib.sha256((p5 / n).read_bytes()).hexdigest()[:16] if (p5 / n).is_file() else None
                             for n in ("host_ar.cu", "host_ar_binding.cpp", "host_ar_check.h", "host_ar_layout.h", "build_host_ar.py")}
except Exception as e:
    R3 = {"fatal": f"{type(e).__name__}: {e}"[:600]}
log("R3", json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in R3.items()}, default=str)[:1500])
REP["checks"]["R3"] = R3; save()

# ---------------------------------------------------------------- R4: P5 cpp_extension build on this image (no GPU run)
try:
    env = G["bench_env"]("")
    tool = {}
    for name, cmd in (("nvcc", [str(Path(env.get("CUDA_HOME", "")) / "bin" / "nvcc"), "--version"]), ("gcc_system", ["gcc", "--version"]),
                      ("cc_env", [env.get("CC") or "true", "--version"])):
        try:
            q = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
            tool[name] = (q.stdout + q.stderr).strip().splitlines()[-1:] + (q.stdout.strip().splitlines()[:1])
        except Exception as e:
            tool[name] = f"{type(e).__name__}: {e}"[:200]
    R4["toolchain"] = tool
    # N3-13: the notebook's s_p5_build now requires the global SHA gate; K2 has no checkpoint (K1 verifies the sha256)
    # and the P5 build reads no weight, so the kit marks the gate explicitly (labelled) to exercise the build
    G["R"]["setup"]["sha_gate"] = {"ok": True, "why": "CPU kit K2: no checkpoint attached; the P5 build reads no weights (K1 checks sha256)"}
    R4["build"] = G["s_p5_build"]()
except Exception as e:
    R4["fatal"] = f"{type(e).__name__}: {e}"[:600]; R4["tb"] = traceback.format_exc()[-1500:]
log("R4", json.dumps(R4, default=str)[:1500])
REP["checks"]["R4"] = R4; save()

# ---------------------------------------------------------------- R5: Nsight Compute present?
try:
    code = ("import sys, json, subprocess\nsys.path.insert(0, sys.argv[1])\nimport nb3_moe_bench as B\np = B.find_ncu()\nv = None\n"
            "if p:\n    v = subprocess.run([p, '--version'], capture_output=True, text=True, timeout=120).stdout.strip()[-300:]\n"
            "print('R5_JSON ' + json.dumps({'ncu': p, 'version': v}))\n")
    r = sh("r5_ncu", [G["BENCH"]["py"], "-c", code, str(G["NB3_DIR"])], G["bench_env"](""), 300, cwd=G["NB3_DIR"])
    R5 = next((json.loads(l[8:]) for l in r["_out"].splitlines() if l.startswith("R5_JSON ")), {"error": r["tail"]})
    R5["meaning"] = ("D5 split DRAM bytes will be MEASURED by ncu (if the GPU box allows the counters)" if R5.get("ncu")
                     else "no ncu on the image: D5 split bytes will be DERIVED (distinct experts x expert bytes)")
except Exception as e:
    R5 = {"fatal": f"{type(e).__name__}: {e}"[:400]}
log("R5", json.dumps(R5)[:600])
REP["checks"]["R5"] = R5
verdict({"merlin_install_ok": bool((R2.get("install") or {}).get("ok"))},
        recorded={"R3_failed_imports": sorted(k for k, v in R3.items() if isinstance(v, dict) and v.get("ok") is False),
                  "R4_p5_build": (R4.get("build") or {}).get("status") or R4.get("fatal"), "R5_ncu": R5.get("ncu"),
                  "bench_env": (R2.get("bench_env") or {}).get("plan")})
'''

K3 = r'''
# ---------------------------------------------------------------- R6: nb3_unit_tests.py on this image (Py3.12)
R6 = {}
try:
    B = K / "build"
    (B / "nb3_export").mkdir(parents=True)
    (B / "nb3_code_export.py").write_text(EXPORT_SRC, encoding="utf-8", newline="\n")
    (B / "nb3_unit_tests.py").write_text(__UNITS__, encoding="utf-8", newline="\n")
    for rel, txt in embedded_files().items():
        (B / "nb3_export" / rel).parent.mkdir(parents=True, exist_ok=True)
        (B / "nb3_export" / rel).write_text(txt, encoding="utf-8", newline="\n")
    (K / "parts.json").write_text(__PARTS__)
    # the real part-1 small files (index, config, tokenizer, template, scales, manifest without the shards)
    P1R = Path(INPUT_ROOT) / "models/foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1"
    P1 = K / "p1_small"; P1.mkdir()
    man = json.loads((P1R / "q27_fn_merlin_part1_manifest.json").read_text())
    small = {n: v for n, v in man["files"].items() if not n.endswith(".safetensors")}
    for n in small:
        shutil.copyfile(P1R / n, P1 / n)
    (P1 / "q27_fn_merlin_part1_manifest.json").write_text(json.dumps(dict(man, files=small)))
    R6["p1_small"] = sorted(small)
    ev = Path(INPUT_ROOT) / "competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
    for k_ in list(os.environ):
        if k_.startswith(("NB3_", "E23_", "RC2_")) or k_ == "KAGGLE_IS_COMPETITION_RERUN":
            os.environ.pop(k_, None)
    r = sh("r6_units", [sys.executable, str(B / "nb3_unit_tests.py"), str(K / "work")],
           {"NB3_TEST_P1": str(P1), "NB3_TEST_PARTS": str(K / "parts.json"), "NB3_TEST_EVAL": str(ev)}, 2400, cwd=B)
    lines = r["_out"].splitlines()
    R6.update(rc=r["rc"], sec=r["sec"], fails=[l for l in lines if l.startswith("FAIL ")][:40],
              summary=next((l for l in lines if l.startswith("NB3 unit tests:")), None),
              flows=[l[:200] for l in lines if l.startswith(("PASS e2e", "FAIL e2e", "PASS rerun", "FAIL rerun"))],
              passes=sum(1 for l in lines if l.startswith("PASS ")))
    m = re.search(r"NB3 unit tests: (\d+)/(\d+) PASS", R6["summary"] or "")
    R6["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0 and not R6["fails"]
except Exception as e:
    R6.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
log("R6", json.dumps({k: v for k, v in R6.items() if k != "p1_small"})[:1500])
REP["checks"]["R6"] = R6
verdict({"units_all_pass": R6.get("all_pass") is True,
         "fake_e2e_and_rerun": bool(R6.get("flows")) and all(l.startswith("PASS") for l in R6.get("flows") or [])})
'''

KITS = {
    "mounts": {"body": K1, "title": "Q27 E25 NB3 Kit Mounts", "models": RC2_MODELS},
    "merlin": {"body": K2, "title": "Q27 E25 NB3 Kit Merlin", "models": ["foysalemonshanto/e23-merlin-env/Other/v1/1"]},
    "units": {"body": K3, "title": "Q27 E25 NB3 Kit Units", "models": [RC2_MODELS[0]]},
}


def build(name):
    spec = KITS[name]
    code = HEAD + spec["body"]
    for k, v in (("__EXPORT__", json.dumps(EXPORT)), ("__UNITS__", json.dumps(UNITS)), ("__PARTS__", json.dumps(PARTS)),
                 ("__NBSHA__", NB_SHA), ("__NAME__", name)):
        code = code.replace(k, v)
    compile(code, f"<kit {name}>", "exec")
    slug = f"q27-e25-nb3-kit-{name}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert len(data) < 400_000, (name, len(data))
    assert b"\r" not in data.replace(b"\\r", b"")
    out = ROOT / f"{slug}__foysalemonshanto"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_bytes(data)
    meta = {"id": f"foysalemonshanto/{slug}", "title": spec["title"], "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": spec["models"], "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out, "|", len(data), "bytes | sha256", hashlib.sha256(data).hexdigest(), "| NB3", NB_SHA[:8], flush=True)
    return out, code


if __name__ == "__main__":
    built = {n: build(n) for n in KITS}
    if "--dry" in sys.argv:
        # local dry run: each kit's code with local paths (fake input tree from nb3_unit_tests; real checks fail-soft here)
        base = Path(os.environ.get("KIT_LOCAL_ROOT", r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/nb3_kit_dry"))
        inp = os.environ.get("KIT_LOCAL_INPUT", r"D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/nb3_work3/input")
        for n, (out, code) in built.items():
            d = base / n
            d.mkdir(parents=True, exist_ok=True)
            (d / "kit.py").write_text(code, encoding="utf-8")
            env = {**os.environ, "KIT_INPUT": inp, "KIT_TMP": str(d / "tmp"), "KIT_WORK": str(d / "work")}
            r = subprocess.run([sys.executable, str(d / "kit.py")], capture_output=True, text=True, env=env, timeout=3600)
            (d / "stdout.txt").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
            line = next((l for l in r.stdout.splitlines() if l.startswith("@@KIT")), None)
            print(f"DRY {n}: rc {r.returncode} | {line}", flush=True)
