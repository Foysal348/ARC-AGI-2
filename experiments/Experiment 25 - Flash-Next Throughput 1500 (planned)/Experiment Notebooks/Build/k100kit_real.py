"""E25 klee100 CPU kit, REAL mode: Merlin's own quantization code in the Merlin venv, without a GPU.

vllm.platforms.current_platform is patched to a NonNvmlCudaPlatform subclass reporting capability (8, 9) and name
"NVIDIA L4" BEFORE any quantization module is imported (a CPU platform would make inc_wna16_scheme.py return
UnquantizedFusedMoEMethod / CPU paths: a false "unquantized"). libcuda.so.1 comes from the toolkit stub on
LD_LIBRARY_PATH (vllm._C_stable_libtorch needs it to import; nothing runs on a device). A single-process gloo
group (TP1 / PP1) is initialised so vLLM parameter classes can be created on the meta device.

Per config (klee100 as published, klee100 + fixkeys, the negative-test copy, the Merlin control):
  quant config  real get_quantization_config(name).from_config + configure_quant_config(Qwen4ExpForConditionalGeneration)
                for the target and configure_quant_config(Qwen4ExpMTP) for the draft (as nvidia/mtp.py:160-200)
  every module  built with a quant config (list from the core's Manifest 1): RoutedExperts on meta (EP2, 256 local
                experts) or a stub of the real linear class -> qc.get_quant_method(layer, name), the inner linear
                method, qc.config_parser.resolve(layer, name); RoutedExperts parameters (names, shapes, dtypes) for
                EVERY MoE module of klee100_fixkeys (one module per layer, released after); a failed build records its
                traceback frames (file, line, function, class of self) so the kit can tell a GPU-only humming object
                (cuInit inside humming) from a real construction failure
  linears       create_weights on meta for one module of every quantized linear kind (TP1 shapes) -> parameter names /
                shapes; choose_mp_linear_kernel for the TP2 partition shapes of every W8 kind
  PLE           Qwen4ExpPLEEmbeddingMethod.from_quant_config + ple_format_guard.check_ple_table_format (F8_E4M3, BF16)
Also: sha256 of the installed Merlin source files vs the audited tree, transformers AutoConfig parse of the fixkeys
config vs Merlin's config, and (separate invocation "r0") a best-effort full model build on meta.

usage: python k100kit_real.py IN_JSON OUT_JSON [real|r0]
"""
import copy
import gc
import hashlib
from contextlib import nullcontext as _nullctx
import json
import sys
import time
import traceback
from pathlib import Path

IN = json.loads(Path(sys.argv[1]).read_text())
OUT_PATH = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "real"
OUT = {"mode": MODE, "label": "REAL_PATCHED_PLATFORM", "steps": {}, "configs": {}}
T0 = time.time()
H, I, NL = 2560, 640, 48


def save():
    OUT["sec"] = round(time.time() - T0, 1)
    with open(OUT_PATH, "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)


def err(e):
    return f"{type(e).__name__}: {e}"[:900]


steps = OUT["steps"]
# ------------------------------------------------------------------ source identity (before any import)
try:
    roots = [Path(p) for p in IN.get("tree_roots", []) if p]
    res = {}
    for rel, want in IN["src_sha"].items():
        got = None
        for r in roots:
            f = r / rel
            if f.is_file():
                got = hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                break
        res[rel] = "IDENTICAL" if got == want else ("MISSING" if got is None else "DIFFERENT")
    steps["source_identity"] = {"roots": [str(r) for r in roots], "files": res,
                                "all_identical": all(v == "IDENTICAL" for v in res.values()),
                                "vllm_files_identical": all(v == "IDENTICAL" for k, v in res.items() if k.startswith("vllm/"))}
except BaseException as e:
    steps["source_identity"] = {"error": err(e)}
save()

# ------------------------------------------------------------------ platform patch
try:
    import torch
    import vllm.platforms as VP

    class _Proxy:
        target = None

        def __getattr__(self, n):
            if type(self).target is None:
                raise AttributeError(n)
            return getattr(type(self).target, n)

    VP._current_platform = _Proxy()
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

    _Proxy.target = FakeL4()
    VP._current_platform = _Proxy.target
    from vllm.platforms import current_platform as CP
    steps["platform"] = {"ok": bool(CP.is_cuda() and not CP.is_cpu()), "class": type(CP).__name__,
                         "capability": list(CP.get_device_capability()),
                         "note": "PATCHED: NonNvmlCudaPlatform subclass (8, 9); no GPU; dispatch only"}
except BaseException as e:
    steps["platform"] = {"ok": False, "error": err(e), "tb": traceback.format_exc()[-2000:]}
save()
if not steps["platform"].get("ok"):
    print("K100REAL " + json.dumps({"platform_ok": False}))
    sys.exit(0)

# ------------------------------------------------------------------ single-process gloo group (TP1 / PP1)
# v1: initialize_model_parallel asserted "Current vLLM config is not set" (parallel_state.py:2095 reads
# get_current_vllm_config()), so the TP group never existed and every W8 create_weights failed on get_tp_group().
# vLLM's own CPU tests do init_distributed_environment + initialize_model_parallel inside a default VllmConfig
# (the default_vllm_config fixture); world size 1 creates no device communicator (parallel_state.py:548).
try:
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.config import VllmConfig as _VC, set_current_vllm_config as _scvc
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with _scvc(_VC()):
        init_distributed_environment(world_size=1, rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", local_rank=0,
                                     backend="gloo")
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, backend="gloo")
    from vllm.distributed import parallel_state as _ps
    steps["distributed"] = {"ok": True, "backend": "gloo", "world_size": 1, "how": "inside set_current_vllm_config(VllmConfig())",
                            "tp_world": _ps.get_tp_group().world_size, "tp_rank": _ps.get_tp_group().rank_in_group,
                            "pp_world": _ps.get_pp_group().world_size}
except BaseException as e:
    steps["distributed"] = {"ok": False, "error": err(e), "tb": traceback.format_exc()[-1500:]}
save()

IMP = {}


def imp(key, fn):
    try:
        IMP[key] = fn()
        steps.setdefault("imports", {})[key] = "ok"
    except BaseException as e:
        IMP[key] = None
        steps.setdefault("imports", {})[key] = err(e)


def _q():
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.inc.inc import INCConfig
    return get_quantization_config, INCConfig


imp("quant", _q)
imp("configure", lambda: __import__("vllm.model_executor.model_loader.utils", fromlist=["x"]).configure_quant_config)


def _moe():
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
    return RoutedExperts, FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType, MoEActivation, ExpertMapManager


imp("moe", _moe)


def _lin():
    from vllm.model_executor.layers import linear as Lm
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    return Lm, ParallelLMHead


imp("linear", _lin)
imp("cg", lambda: __import__("vllm.models.qwen4_exp.nvidia.model", fromlist=["x"]).Qwen4ExpForConditionalGeneration)
imp("mtp", lambda: __import__("vllm.models.qwen4_exp.nvidia.mtp", fromlist=["x"]).Qwen4ExpMTP)
imp("qsa_lin", lambda: __import__("vllm.models.qwen4_exp.nvidia.qsa", fromlist=["x"]).Qwen4ExpQSAQKVIndexerLinear)
imp("ple", lambda: __import__("vllm.models.qwen4_exp.common.ngram_embedding", fromlist=["x"]).Qwen4ExpPLEEmbeddingMethod)
imp("guard", lambda: __import__("vllm.models.qwen4_exp.nvidia.ple_format_guard", fromlist=["x"]).check_ple_table_format)


def _vc():
    from vllm.config import VllmConfig, set_current_vllm_config
    return VllmConfig, set_current_vllm_config


imp("vc", _vc)


def _mp():
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
    from vllm.scalar_type import scalar_types
    return choose_mp_linear_kernel, MPLinearLayerConfig, scalar_types


imp("mp", _mp)
try:
    import humming  # noqa: F401
    steps["has_humming"] = getattr(humming, "__version__", True)
except BaseException as e:
    steps["has_humming"] = err(e)
save()


def build_quant(cfg, model_cls):
    get_qc, INCConfig = IMP["quant"]
    hq = copy.deepcopy(cfg["quantization_config"])
    name = INCConfig.override_quantization_method(hq, None) or hq.get("quant_method")
    qc = get_qc(name).from_config(hq)
    if model_cls is not None and IMP["configure"] is not None:
        IMP["configure"](qc, model_cls)
    return name, qc


def moe_objects():
    RoutedExperts, FMC, FMPC, RMT, MoEAct, EMM = IMP["moe"]
    pc = FMPC(tp_size=1, tp_rank=0, pcp_size=1, pcp_rank=0, dp_size=1, dp_rank=0, ep_size=2, ep_rank=0,
              sp_size=1, use_ep=True, all2all_backend="allgather_reducescatter", enable_eplb=False)
    emm = EMM(max_num_batched_tokens=8192, top_k=10, global_num_experts=512, num_redundant_experts=0,
              num_expert_group=None, moe_parallel_config=pc, placement_strategy="linear", enable_eplb=False)
    mc = FMC(num_experts=512, experts_per_token=10, hidden_dim=H, intermediate_size=I,
             num_local_experts=emm.local_num_experts, num_logical_experts=512, activation=MoEAct.SILU, device="meta",
             routing_method=RMT.Renormalize, moe_parallel_config=pc, in_dtype=torch.bfloat16)
    return RoutedExperts, mc, emm


def describe(method, rec):
    if method is None:
        rec["method"] = "None"
        return
    rec["method"] = type(method).__name__
    inner = getattr(method, "linear_method", None) or getattr(getattr(method, "scheme", None), "linear_method", None)
    sch = getattr(method, "scheme", None)
    if sch is not None:
        rec["scheme"] = type(sch).__name__
        inner = inner or getattr(sch, "linear_method", None)
    if inner is not None:
        rec["inner"] = type(inner).__name__
    for a in ("wna16_backend", "wna16_moe_backend", "experts_cls", "use_marlin", "humming_gemm_type", "quant_type", "num_bits",
              "weight_bits", "weight_schema", "humming_configs"):
        v = getattr(method, a, None)
        if v is None and getattr(method, "quant_config", None) is not None:
            v = getattr(method.quant_config, a, None)
        if v is not None:
            rec[a] = (getattr(v, "__name__", None) or str(v))[:300]


def tb_frames(e, n=25):
    """Frames of an exception: [file tail, line, function, class of `self` or None] (K1: proves where it failed)."""
    out, tb = [], e.__traceback__
    while tb is not None:
        f = tb.tb_frame
        try:
            sc = type(f.f_locals["self"]).__name__ if "self" in f.f_locals else None
        except BaseException:
            sc = None
        fn = f.f_code.co_filename.replace("\\", "/")
        out.append(["/".join(fn.split("/")[-4:]), tb.tb_lineno, f.f_code.co_name, sc])
        tb = tb.tb_next
    return out[-n:]


def frame_evidence(rec, tag, e):
    fr = tb_frames(e)
    rec[f"{tag}_frames"] = fr
    rec[f"{tag}_humming_frame"] = any("humming" in x[0].lower() for x in fr)
    rec[f"{tag}_humming_object"] = next((x[3] for x in fr if x[3] and "humming" in x[3].lower()), None)
    rec[f"{tag}_cuda_init"] = bool(CUDA_INIT_TXT.search(err(e)))


import re as _re
CUDA_INIT_TXT = _re.compile(r"cuInit|CUDA error 34|CUDA_ERROR_STUB_LIBRARY|CUDA_ERROR_NO_DEVICE|no CUDA-capable device")


def resolve_moe(qc, vname, want_params):
    """One RoutedExperts module on meta (EP2, 256 local experts): method, resolved config, and (want_params) every
    created parameter. The module is released right after (K2: one module per layer, checked, freed)."""
    rec = {}
    RoutedExperts, mc, emm = moe_objects()
    method, layer = None, None
    VllmConfig, scvc = IMP["vc"] or (None, None)
    try:
        ctx = scvc(VllmConfig()) if scvc else None
        if ctx is not None:
            ctx.__enter__()
        try:
            with torch.device("meta"):
                layer = RoutedExperts(vname, torch.bfloat16, mc, qc, emm)
            method = layer.quant_method
            rec["built"] = "RoutedExperts(meta, EP2)"
            if want_params:
                rec["params"] = {n: [list(p.shape), str(p.dtype)] for n, p in layer.named_parameters()}
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
    except BaseException as e:
        rec["build_error"] = err(e)
        frame_evidence(rec, "build", e)
        try:
            layer = RoutedExperts.__new__(RoutedExperts)
            torch.nn.Module.__init__(layer)
            layer.moe_config, layer.layer_name, layer.params_dtype = mc, vname, torch.bfloat16
            layer.hidden_size, layer.intermediate_size_per_partition = H, I
            layer.global_num_experts, layer.local_num_experts = 512, emm.local_num_experts
            method = qc.get_quant_method(layer, vname)
            rec["built"] = "RoutedExperts stub + qc.get_quant_method (NOT a real construction)"
            rec["stub_built"] = True
        except BaseException as e2:
            rec["stub_error"] = err(e2)
            frame_evidence(rec, "stub", e2)
    describe(method, rec)
    if hasattr(qc, "config_parser") and layer is not None:
        try:
            lc = qc.config_parser.resolve(layer, vname)
            rec["resolved"] = [lc.bits, lc.group_size, lc.sym]
        except BaseException as e:
            rec["resolved_error"] = err(e)
    rec["released"] = True
    del layer, method
    return rec


LIN_CLASS = {"MergedColumnParallelLinear": "MergedColumnParallelLinear", "MergedColumnParallelLinear(disable_tp)": "MergedColumnParallelLinear",
             "RowParallelLinear": "RowParallelLinear", "ColumnParallelLinear": "ColumnParallelLinear",
             "ReplicatedLinear": "ReplicatedLinear", "VisionLinear": "ColumnParallelLinear"}


def stub_linear(cls_name, name):
    Lm, ParallelLMHead = IMP["linear"]
    if cls_name == "Qwen4ExpQSAQKVIndexerLinear":
        cls = IMP["qsa_lin"] or Lm.QKVParallelLinear
    elif cls_name == "ParallelLMHead":
        cls = ParallelLMHead
    else:
        cls = getattr(Lm, LIN_CLASS.get(cls_name, "ColumnParallelLinear"))
    o = cls.__new__(cls)
    torch.nn.Module.__init__(o)
    o.input_size, o.output_size = H, H
    o.input_size_per_partition, o.output_size_per_partition = H, H
    o.output_partition_sizes, o.params_dtype, o.has_bias, o.prefix = [H], torch.bfloat16, False, name
    return o


def resolve_linear(qc, cls_name, vname):
    rec = {}
    try:
        st = stub_linear(cls_name, vname)
        rec["stub_class"] = type(st).__name__
        meth = qc.get_quant_method(st, vname)
        describe(meth, rec)
        if hasattr(qc, "config_parser"):
            lc = qc.config_parser.resolve(st, vname)
            rec["resolved"] = [lc.bits, lc.group_size, lc.sym]
    except BaseException as e:
        rec["error"] = err(e)
    return rec


def create_linear_params(qc, cls_name, vname, K, Ns):
    """quant_method.create_weights on meta with TP1 shapes -> parameter names / shapes / dtypes."""
    rec = {"K": K, "N_parts": Ns, "label": "parameter definitions on meta with a no-op weight_loader (no tensor loaded)"}
    VllmConfig, scvc = IMP["vc"] or (None, None)
    try:
        st = stub_linear(cls_name, vname)
        meth = qc.get_quant_method(st, vname)
        st.quant_method = meth
        with (scvc(VllmConfig()) if scvc else _nullctx()), torch.device("meta"):
            meth.create_weights(st, K, list(Ns), K, sum(Ns), torch.bfloat16, weight_loader=lambda *a, **k: None)
        rec["method"] = type(meth).__name__
        rec["params"] = {n: [list(p.shape), str(p.dtype)] for n, p in st.named_parameters()}
        for a in ("kernel", "_kernel"):
            kk = getattr(getattr(meth, "linear_method", meth), a, None) or getattr(st, a, None)
            if kk is not None:
                rec["kernel"] = type(kk).__name__
    except BaseException as e:
        rec["error"] = err(e)
        rec["tb"] = traceback.format_exc()[-1200:]
        frame_evidence(rec, "error", e)
    return rec


def mp_kernels():
    choose, MPC, st = IMP["mp"]
    shapes = {"qkv_proj(q+k+v+indexer)": ((H, 12288 + 512 + 512 + 1280), (H, 6144 + 256 + 256 + 640)),
              "o_proj / out_proj (row)": ((6144, H), (3072, H)), "in_proj_qkvz": ((H, 16384), (H, 8192)),
              "mtp.fc_* (col)": ((H, H), (H, 1280))}
    res = {}
    for name, (full, part) in shapes.items():
        try:
            k = choose(MPC(full_weight_shape=full, partition_weight_shape=part, weight_type=st.uint8b128,
                           act_type=torch.bfloat16, group_size=128, zero_points=False, has_g_idx=False))
            res[name] = k.__name__
        except BaseException as e:
            res[name] = f"ERROR {err(e)[:200]}"
    return res


def ple_check(qc, cfg, vprefix):
    out = {}
    Meth, guard = IMP["ple"], IMP["guard"]
    if Meth is None or guard is None:
        return {"error": "PLE modules not importable"}
    edt = (cfg.get("text_config") or {}).get("ple_embedding_dtype")
    try:
        m = Meth.from_quant_config(qc, vprefix, edt)
        out["method"] = type(m).__name__
        for tag, dt in (("F8_E4M3", torch.float8_e4m3fn), ("BF16", torch.bfloat16)):
            try:
                guard(m, dt, vprefix)
                out[f"guard_{tag}"] = "ACCEPTED"
            except BaseException as e:
                out[f"guard_{tag}"] = "REFUSED: " + err(e)[:200]
    except BaseException as e:
        out["error"] = err(e)
    out["embedding_dtype"] = edt
    return out


def run_real():
    cg, mtp = IMP.get("cg"), IMP.get("mtp")
    if IMP.get("mp"):
        try:
            OUT["mp_linear_kernel_choice_sm89_w8g128_tp2"] = mp_kernels()
        except BaseException as e:
            OUT["mp_linear_kernel_choice_sm89_w8g128_tp2"] = err(e)
    for cname, cpath in IN["configs"].items():
        cfg = json.loads(Path(cpath).read_text())
        rec = OUT["configs"].setdefault(cname, {})
        try:
            name, qt = build_quant(cfg, cg)
            _, qd = build_quant(cfg, mtp)
            rec["quant"] = {"name": name, "class": type(qt).__name__,
                            "target_blocks": getattr(qt, "block_name_to_quantize", None),
                            "draft_blocks": getattr(qd, "block_name_to_quantize", None),
                            "packed_target": getattr(qt, "packed_modules_mapping", None),
                            "packed_draft": getattr(qd, "packed_modules_mapping", None)}
        except BaseException as e:
            rec["fatal"] = err(e)
            rec["tb"] = traceback.format_exc()[-2000:]
            save()
            continue
        rows = rec.setdefault("rows", {})
        moe_params_done = False
        for side, mod, cls, qflag in IN["modules"]:
            if cname == "merlin_control" and side == "draft":
                pass
            key = f"{side}|{mod}"
            qq = qt if side == "target" else qd
            if qflag == "ple":
                rows[key] = ple_check(qt, cfg, mod)
            elif qflag is not True:
                continue
            elif cls == "RoutedExperts":
                want = cname == "klee100_fixkeys"          # K2: every MoE module (48 target layers + MTP bank)
                rows[key] = resolve_moe(qq, mod, want)
                if want:
                    gc.collect()
                moe_params_done = moe_params_done or want
            else:
                rows[key] = resolve_linear(qq, cls, mod)
            if len(rows) % 50 == 0:
                save()
        save()
        if cname == "klee100_fixkeys":
            cw = {}
            for tag, (side, cls, mod, K, Ns) in IN["create_weights"].items():
                cw[tag] = create_linear_params(qt if side == "target" else qd, cls, mod, K, Ns)
            rec["create_weights"] = cw
        save()
    OUT["ran_all"] = all(not v.get("fatal") for v in OUT["configs"].values())


def autoconfig():
    try:
        from transformers import AutoConfig
        res = {}
        for tag, d in IN.get("autoconfig_dirs", {}).items():
            try:
                c = AutoConfig.from_pretrained(d)
                tc = getattr(c, "text_config", c)
                res[tag] = {k: (getattr(tc, k, None) if not isinstance(getattr(tc, k, None), (list, tuple))
                                else sorted(set(map(str, getattr(tc, k)))) if k == "layer_types" else list(getattr(tc, k)))
                            for k in ("layer_types", "norm_topk_prob", "number_of_conv_states", "seed", "ple_embedding_dtype",
                                      "mtp_num_hidden_layers", "mtp_use_dedicated_embeddings", "num_experts", "num_hidden_layers",
                                      "full_attention_interval", "indexer_n_heads", "hc_count")}
                lt = list(getattr(tc, "layer_types", []) or [])
                res[tag]["layer_types_seq_sha256"] = hashlib.sha256(json.dumps(lt).encode()).hexdigest()
                res[tag]["config_class"] = type(c).__name__
            except BaseException as e:
                res[tag] = {"error": err(e)}
        OUT["autoconfig"] = res
    except BaseException as e:
        OUT["autoconfig"] = {"error": err(e)}


def run_r0():
    """Best-effort full model build on meta (target + draft). Not gated; any failure is recorded."""
    r0 = OUT.setdefault("r0", {})
    try:
        from vllm.engine.arg_utils import EngineArgs
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import initialize_model
        ea = EngineArgs(model=IN["r0_dir"], skip_tokenizer_init=True, tensor_parallel_size=1, max_model_len=8192,
                        enforce_eager=True, load_format="dummy", dtype="bfloat16", language_model_only=True,
                        speculative_config={"method": "mtp", "num_speculative_tokens": 3})
        cfg = ea.create_engine_config()
        r0["engine_config"] = "ok"
        with set_current_vllm_config(cfg):
            with torch.device("meta"):
                model = initialize_model(vllm_config=cfg)
        mods = {}
        for n, m in model.named_modules():
            qm = getattr(m, "quant_method", None)
            if qm is not None:
                mods[n] = type(qm).__name__
        r0["target_modules_with_quant_method"] = mods
        r0["target_ok"] = True
    except BaseException as e:
        r0["error"] = err(e)
        r0["tb"] = traceback.format_exc()[-3000:]


if MODE == "r0":
    run_r0()
else:
    if IMP.get("quant") and IMP.get("moe") and IMP.get("linear"):
        run_real()
    autoconfig()
save()
print("K100REAL " + json.dumps({"platform_ok": steps["platform"].get("ok"), "ran_all": OUT.get("ran_all"),
                                "configs": list(OUT["configs"]), "r0": (OUT.get("r0") or {}).get("target_ok")}))
