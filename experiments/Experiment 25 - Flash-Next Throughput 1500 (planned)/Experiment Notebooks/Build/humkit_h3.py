"""E25 humming CPU kit, check H3: quant-method resolution for three Flash-Next checkpoints, without a GPU.

Candidates (configs embedded in the kit, sha256-checked; no weights):
  klee100        klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP          (auto-round, auto_round:auto_gptq, mixed W3/W2)
  haberstroh     HaberstrohSystems/Qwen3.8-Flash-Next-int2-mixed-AutoRound-24GB-SGLang  (auto-round, W2 g128)
  merlin         our Merlin W4A16 control (p1 config.json from the mount; HF copy as fallback)
  klee100_fixkeys  klee100 + layer-level extra_config keys written with vLLM module names (config-only fix candidate)

Two modes, both recorded:
  REAL_PATCHED_PLATFORM  Merlin's own code in the Merlin venv. vllm.platforms.current_platform is patched to a
                         NonNvmlCudaPlatform subclass reporting capability (8, 9) and name "NVIDIA L4" BEFORE any
                         quantization module is imported (a forwarding proxy covers modules that bind it early).
                         Without this patch inc_wna16_scheme.py:163-168 returns UnquantizedFusedMoEMethod on a CPU
                         platform (and :120 the same for GPTQ linear), which would be a false "unquantized".
                         Quant configs are built with the real from_config + configure_quant_config(model class); MoE
                         layers are real RoutedExperts built on the meta device (fallback: a RoutedExperts stub with a
                         real FusedMoEConfig); linears are stubs of the real vLLM classes (no __init__).
  STATIC_CUDA_DISPATCH   pure-Python re-evaluation of the CUDA branch's rules (inc/config_parser.py _resolve_raw,
                         inc_wna16_scheme.py:23-28 / :141-174 / :247 / :276-297 / :315, qwen3_vl.py:1987-1993 mapper):
                         runs anywhere (local dry run, or when the real mode cannot import). Labelled as such.

Per candidate it records, for every MoE layer (48 + MTP), the module name vLLM uses, the resolved (bits, group), the
method / backend / experts class, the created parameter shapes and their implied bits, and compares them with the
checkpoint's actual per-layer bits/group (from the safetensors headers, embedded); one bit-width per MoE layer is
asserted from the headers and from the config. Linear modules: method and resolved bits per pattern. PLE: Merlin's
Qwen4ExpPLEEmbeddingMethod.from_quant_config + ple_format_guard.check_ple_table_format on the stored table dtype, the
FP8-graft variant, and Merlin's launcher rule (tools/merlin/launch.py:1219).

usage: python humkit_h3.py EMB_DIR OUT_JSON [real|static]
"""
import copy
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

EMB = Path(sys.argv[1])
OUT_PATH = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "real"
OUT = {"mode_requested": MODE, "static": {}, "real": {}, "inputs": {}}
T0 = time.time()


def save():
    with open(OUT_PATH, "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)


def err(e):
    return f"{type(e).__name__}: {e}"[:700]


DER = json.loads((EMB / "derived" / "lowbit_derived.json").read_text())
P1C = EMB / "p1" / "config.json"
CFG_PATHS = {"klee100": EMB / "hf" / "klee100" / "config.json",
             "haberstroh": EMB / "hf" / "haberstroh" / "config.json",
             "merlin": P1C if P1C.is_file() else EMB / "hf" / "merlin" / "config.json"}
CFGS = {k: json.loads(p.read_text()) for k, p in CFG_PATHS.items()}
OUT["inputs"] = {k: str(p) for k, p in CFG_PATHS.items()}
NL = 48
H, I = 2560, 640


def ckpt_moe(cand):
    base = "klee100" if cand.startswith("klee100") else cand
    return DER[base]["moe_layers"]


def fix_klee(cfg):
    """Config-only fix candidate: layer-level extra_config keys spelled the way vLLM names the RoutedExperts modules
    (language_model.model.layers.N.mlp.experts, mtp.layers.48.mlp.experts) so INC's _resolve_raw finds them exactly."""
    c = copy.deepcopy(cfg)
    q = c["quantization_config"]
    ec = q.setdefault("extra_config", {})
    for L, rec in ckpt_moe("klee100").items():
        if L == "mtp":
            ec[f"mtp.layers.{NL}.mlp.experts"] = {"bits": rec["bits"], "group_size": rec["group"], "sym": True,
                                                  "data_type": "int"}
        else:
            ec[f"language_model.model.layers.{L}.mlp.experts"] = {"bits": rec["bits"], "group_size": rec["group"]}
    bn = q.get("block_name_to_quantize")
    bn = bn.split(",") if isinstance(bn, str) else list(bn or [])
    q["block_name_to_quantize"] = ",".join(bn + ["mtp.layers"])
    return c


CFGS["klee100_fixkeys"] = fix_klee(CFGS["klee100"])
CANDS = ["klee100", "haberstroh", "merlin", "klee100_fixkeys"]


# ---------------------------------------------------------------- checkpoint-side assertions (headers + configs)
def layout_checks():
    res = {}
    for cand in ("klee100", "haberstroh", "merlin"):
        moe = ckpt_moe(cand)
        res[cand] = {"one_bitwidth_per_moe_layer": all(r.get("uniform") for r in moe.values()),
                     "layers": {L: [r.get("fmt"), r.get("bits"), r.get("group")] for L, r in moe.items()},
                     "non_uniform": {L: r.get("variants") for L, r in moe.items() if not r.get("uniform")}}
    # klee100's stated layout (build_info.json): W3 g128 layers 0-11 and 36-47, W2 g64 layers 12-35, MTP W4 g128
    want = {str(L): (3, 128) if (L <= 11 or L >= 36) else (2, 64) for L in range(NL)}
    want["mtp"] = (4, 128)
    got = {L: (r.get("bits"), r.get("group")) for L, r in ckpt_moe("klee100").items()}
    bad = {L: [got.get(L), w] for L, w in want.items() if got.get(L) != w}
    # the same layout read from klee100's own config (per-expert extra_config entries; default bits 2 / g64)
    q = CFGS["klee100"]["quantization_config"]
    ec = q.get("extra_config") or {}
    from_cfg = {}
    for L in range(NL):
        vals = {json.dumps(v, sort_keys=True) for k, v in ec.items()
                if k.startswith(f"model.language_model.layers.{L}.mlp.experts.")}
        if not vals:
            from_cfg[str(L)] = [q.get("bits"), q.get("group_size"), "default"]
        elif len(vals) == 1:
            v = json.loads(vals.pop())
            from_cfg[str(L)] = [v.get("bits", q.get("bits")), v.get("group_size", q.get("group_size")), "explicit"]
        else:
            from_cfg[str(L)] = ["NON_UNIFORM", sorted(vals)[:3]]
    vals = {json.dumps(v, sort_keys=True) for k, v in ec.items() if k.startswith("mtp.layers.0.mlp.experts.")}
    if len(vals) == 1:
        v = json.loads(vals.pop())
        from_cfg["mtp"] = [v.get("bits"), v.get("group_size"), "explicit"]
    else:
        from_cfg["mtp"] = ["NON_UNIFORM_OR_ABSENT", None]
    cfg_bad = {L: [from_cfg.get(L), list(w)] for L, w in want.items() if tuple(from_cfg.get(L, [None, None])[:2]) != w}
    res["klee100_mixed_layout"] = {"want": {k: list(v) for k, v in want.items()}, "headers_vs_want_bad": bad,
                                   "config_vs_want_bad": cfg_bad, "ok": not bad and not cfg_bad,
                                   "note": "not a uniform W3: 24 layers W3 g128 + 24 layers W2 g64 + MTP W4 g128"}
    res["ple_formats"] = {c: DER[c]["ple"] for c in ("klee100", "haberstroh", "merlin")}
    return res


OUT["checkpoint_layout"] = layout_checks()
save()

# ---------------------------------------------------------------- STATIC_CUDA_DISPATCH (pure Python)
_CG_PREFIX = [("model.visual.", "visual."), ("lm_head.", "language_model.lm_head."),
              ("model.language_model.", "language_model.model.")]   # qwen3_vl.py:1987-1993 (mtp.: None is dropped
_RX = set(r"*+?^$()[]{}|\\")                                         # by get_rename_mapper, utils.py:181-195)


def cg_map(n):
    for a, b in _CG_PREFIX:
        if n.startswith(a):
            return b + n[len(a):]
    return n


def norm_g(g):
    return tuple(g) if isinstance(g, list) else g


class StaticINC:
    """inc.py from_config + apply_vllm_mapper + config_parser._resolve_raw, re-evaluated (CUDA branch)."""

    def __init__(self, q, mapper):
        self.bits, self.group, self.sym = q["bits"], norm_g(q["group_size"]), q.get("sym", True)
        self.packing = q.get("packing_format") or "auto_round:auto_gptq"
        bn = q.get("block_name_to_quantize") or q.get("to_quant_block_names")
        bn = bn.split(",") if isinstance(bn, str) else bn
        ec = q.get("extra_config") or {}
        self.extra = {mapper(k): v for k, v in ec.items()} if mapper else dict(ec)
        self.blocks = [mapper(b) for b in bn] if (bn and mapper) else bn

    def _get(self, name, quantized=True):
        ec = self.extra
        dflt = (self.bits if quantized else 16, self.group if quantized else -1, self.sym if quantized else True)
        if not ec:
            return dflt
        if name in ec:
            c = ec[name]
            return (c.get("bits", dflt[0]), norm_g(c.get("group_size", dflt[1])), c.get("sym", dflt[2]))
        for pat, c in ec.items():
            if isinstance(pat, str) and any(ch in _RX for ch in pat):
                try:
                    if re.search(pat, name):
                        return (c.get("bits", dflt[0]), norm_g(c.get("group_size", dflt[1])), c.get("sym", dflt[2]))
                except re.error:
                    continue
        return dflt

    def resolve(self, cls_name, name, is_lm_head=False):
        ec = self.extra
        if ec and name in ec:
            return self._get(name), "exact"
        if ec:
            for k in ec:
                if name.endswith("." + k):
                    return self._get(k), "suffix"
        quantized = not is_lm_head
        if self.blocks:
            quantized = any(name.startswith(b) for b in self.blocks)
        if ec and "fusedmoe" in cls_name.lower():     # config_parser.py:178 (dead for RoutedExperts in 0.32)
            cfgs = {self._get(k, quantized) for k in ec if k.startswith(name)}
            if len(cfgs) == 1:
                return cfgs.pop(), "fusedmoe_subkeys"
            if cfgs:
                return ("ERROR_INCONSISTENT",), "fusedmoe_subkeys"
        return self._get(name, quantized), ("default" if quantized else "not_in_block_name_to_quantize")


def static_dispatch(bits, packing, moe):
    if bits >= 16:
        return "UnquantizedFusedMoEMethod" if moe else "UnquantizedLinearMethod"
    if bits in (2, 3, 5, 6, 7):                     # inc_wna16_scheme.py:23-28, :141-148, :169-174
        if "gptq" not in packing and "awq" not in packing:
            return "NotImplementedError(:294 humming needs gptq/awq packing)"
        return "HummingMoEMethod (humming)" if moe else "HummingLinearMethod (humming)"
    if bits in (4, 8):
        return ("AutoGPTQMoEMethod -> WNA16 oracle (Marlin first on sm_89, int_wna16.py:126-143)" if moe
                else "INCLinearMethod(INCWNA16LinearScheme) -> MPLinear kernel choice")
    return f"unsupported bits {bits}"


def run_static():
    S = {"label": "STATIC_CUDA_DISPATCH", "cands": {}}
    for cand in CANDS:
        cfg = CFGS[cand]
        q = cfg.get("quantization_config") or {}
        rec = {"quant_method": q.get("quant_method")}
        if q.get("quant_method") != "auto-round":
            rec["note"] = "compressed-tensors control: static check limited to its config groups (Merlin RC2 ran it)"
            rec["groups"] = {g: {k: v.get("weights", {}).get(k) for k in ("num_bits", "group_size", "strategy")}
                             for g, v in (q.get("config_groups") or {}).items()}
            rec["ignore"] = q.get("ignore")
            S["cands"][cand] = rec
            continue
        qt = StaticINC(q, cg_map)        # target model (configure_quant_config with the CG mapper)
        qd = StaticINC(q, None)          # draft model (Qwen4ExpMTP has no hf_to_vllm_mapper)
        qraw = StaticINC(q, None)        # contrast: HF names, no mapping
        moe, mism = {}, []
        for L in [str(x) for x in range(NL)] + ["mtp"]:
            if L == "mtp":
                vname, hname, qq = f"mtp.layers.{NL}.mlp.experts", "mtp.layers.0.mlp.experts", qd
            else:
                hname = f"model.language_model.layers.{L}.mlp.experts"
                vname, qq = cg_map(hname), qt
            r, how = qq.resolve("RoutedExperts", vname)
            r_hf, how_hf = qraw.resolve("RoutedExperts", hname)
            ck = ckpt_moe(cand).get(L, {})
            match = (r[0] == ck.get("bits") and (r[0] >= 16 or r[1] == ck.get("group")))
            moe[L] = {"vllm_name": vname, "resolved": list(r), "how": how, "method": static_dispatch(r[0], qq.packing, True),
                      "if_hf_names": [list(r_hf), how_hf], "ckpt": [ck.get("bits"), ck.get("group")], "match": match}
            if not match:
                mism.append(L)
        rec["moe"] = moe
        rec["moe_mismatch_layers"] = mism
        rec["loadable_as_is"] = not mism
        lin = {}
        for m in DER["klee100" if cand.startswith("klee100") else cand]["modules"]:
            if m.startswith("mtp."):
                vname, qq = m.replace("mtp.layers.0.", f"mtp.layers.{NL}.", 1), qd
            else:
                vname, qq = cg_map(m), qt
            r, how = qq.resolve("Linear", vname, is_lm_head=vname.endswith("lm_head"))
            pat = re.sub(r"\.\d+\.", ".N.", vname)
            k = f"{pat} -> W{r[0]} g{r[1]} [{how}] {static_dispatch(r[0], qq.packing, False)}"
            lin[k] = lin.get(k, 0) + 1
        rec["linear_patterns"] = lin
        S["cands"][cand] = rec
    return S


try:
    OUT["static"] = run_static()
except Exception as e:
    OUT["static"] = {"error": err(e), "tb": traceback.format_exc()[-2500:]}
save()
if MODE == "static":
    print("H3_JSON " + json.dumps({"mode": "static"}))
    sys.exit(0)

# ---------------------------------------------------------------- REAL_PATCHED_PLATFORM (Merlin code)
R = OUT["real"]
R["label"] = "REAL_PATCHED_PLATFORM"
steps = R.setdefault("steps", {})
try:
    import torch
    import vllm.platforms as VP

    class _Proxy:
        target = None

        def __getattr__(self, n):
            if type(self).target is None:
                raise AttributeError(n)
            return getattr(type(self).target, n)

    VP._current_platform = _Proxy()                   # binds early importers to the proxy
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
                         "mro": [c.__name__ for c in type(CP).__mro__][:5], "capability": list(CP.get_device_capability()),
                         "note": "PATCHED: current_platform = NonNvmlCudaPlatform subclass (8, 9); no GPU; dispatch only"}
except BaseException as e:
    steps["platform"] = {"ok": False, "error": err(e), "tb": traceback.format_exc()[-2000:]}
save()

IMP = {}


def imp(key, fn):
    try:
        IMP[key] = fn()
        steps.setdefault("imports", {})[key] = "ok"
    except BaseException as e:
        IMP[key] = None
        steps.setdefault("imports", {})[key] = err(e)


if steps["platform"].get("ok"):
    def _q():
        from vllm.model_executor.layers.quantization import get_quantization_config
        from vllm.model_executor.layers.quantization.inc.inc import INCConfig
        return get_quantization_config, INCConfig
    imp("quant", _q)
    imp("iws", lambda: __import__("vllm.model_executor.layers.quantization.inc.schemes.inc_wna16_scheme",
                                  fromlist=["x"]))
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
        steps["has_humming"] = True
    except BaseException as e:
        steps["has_humming"] = err(e)
save()


def build_quant(cand, model_cls):
    get_qc, INCConfig = IMP["quant"]
    hq = copy.deepcopy(CFGS[cand]["quantization_config"])
    name = INCConfig.override_quantization_method(hq, None) or hq.get("quant_method")
    qc = get_qc(name).from_config(hq)
    if model_cls is not None and IMP["configure"] is not None:
        IMP["configure"](qc, model_cls)
    return name, qc


def moe_objects(use_ep=True):
    RoutedExperts, FMC, FMPC, RMT, MoEAct, EMM = IMP["moe"]
    pc = FMPC(tp_size=1, tp_rank=0, pcp_size=1, pcp_rank=0, dp_size=1, dp_rank=0, ep_size=2 if use_ep else 1, ep_rank=0,
              sp_size=1, use_ep=use_ep, all2all_backend="allgather_reducescatter", enable_eplb=False)
    emm = EMM(max_num_batched_tokens=8192, top_k=10, global_num_experts=512, num_redundant_experts=0,
              num_expert_group=None, moe_parallel_config=pc, placement_strategy="linear", enable_eplb=False)
    mc = FMC(num_experts=512, experts_per_token=10, hidden_dim=H, intermediate_size=I,
             num_local_experts=emm.local_num_experts, num_logical_experts=512, activation=MoEAct.SILU, device="meta",
             routing_method=RMT.Renormalize, moe_parallel_config=pc, in_dtype=torch.bfloat16)
    return RoutedExperts, mc, emm


def implied_bits(params):
    """bits from a packed int32 w13 parameter: the dim that equals K * bits / 32 for K = 2560."""
    for n, (shape, dt) in params.items():
        if "w13" in n and ("qweight" in n or "weight_packed" in n or n.endswith("w13_weight")) and "int32" in dt:
            for d in shape[1:]:
                if (d * 32) % H == 0 and 2 <= d * 32 // H <= 8:
                    return d * 32 // H
        if n == "w13_weight" and "bfloat16" in dt:
            return 16
    return None


def resolve_moe(cand, qc, vname, ep=True):
    rec = {"vllm_name": vname}
    RoutedExperts, mc, emm = moe_objects(ep)
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
            rec["built"] = "RoutedExperts(meta)"
            rec["params"] = {n: [list(p.shape), str(p.dtype)] for n, p in layer.named_parameters()}
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
    except BaseException as e:
        rec["build_error"] = err(e)
        try:
            layer = RoutedExperts.__new__(RoutedExperts)
            torch.nn.Module.__init__(layer)
            layer.moe_config, layer.layer_name, layer.params_dtype = mc, vname, torch.bfloat16
            layer.hidden_size, layer.intermediate_size_per_partition = H, I
            layer.global_num_experts, layer.local_num_experts = 512, emm.local_num_experts
            method = qc.get_quant_method(layer, vname)
            rec["built"] = "RoutedExperts stub + qc.get_quant_method"
        except BaseException as e2:
            rec["stub_error"] = err(e2)
            rec["stub_tb"] = traceback.format_exc()[-1500:]
    if method is not None:
        rec["method"] = type(method).__name__
        for a in ("wna16_backend", "experts_cls", "humming_gemm_type", "quant_type", "num_bits", "weight_bits"):
            v = getattr(method, a, None)
            if v is not None:
                rec[a] = getattr(v, "__name__", None) or str(v)
    if hasattr(qc, "config_parser") and layer is not None:
        try:
            lc = qc.config_parser.resolve(layer, vname)
            rec["inc_layer_config"] = [lc.bits, lc.group_size, lc.sym, lc.quantized]
        except BaseException as e:
            rec["inc_layer_config"] = err(e)
    rec["implied_bits_from_params"] = implied_bits(rec.get("params") or {})
    return rec


def stub_linear(cls, ParallelLMHead, name):
    o = cls.__new__(cls)
    torch.nn.Module.__init__(o)
    o.input_size, o.output_size = H, H
    o.input_size_per_partition, o.output_size_per_partition = H // 2, H // 2
    o.output_partition_sizes, o.params_dtype, o.has_bias, o.prefix = [H // 2], torch.bfloat16, False, name
    return o


LIN_CLASS = [("lm_head", "ParallelLMHead"), ("qkv_proj", "QKVParallelLinear"), ("gate_up_proj", "MergedColumnParallelLinear"),
             ("in_proj_qkvz", "MergedColumnParallelLinear"), ("in_proj_ba", "MergedColumnParallelLinear"),
             ("kv_proj", "MergedColumnParallelLinear"), ("input_mix_weight_down_block_inject", "MergedColumnParallelLinear"),
             ("mlp.gate", "ReplicatedLinear"), ("shared_expert_gate", "ReplicatedLinear"), ("hyper_connection", "ReplicatedLinear"),
             ("o_proj", "RowParallelLinear"), ("down_proj", "RowParallelLinear"), ("out_proj", "RowParallelLinear")]


def fuse(vname, packed):
    for fk, subs in (packed or {}).items():
        for s in subs:
            if vname.endswith("." + s):
                return vname[: -len(s)] + fk
    return vname


def resolve_linears(cand, qt, qd, cg, mtp):
    Lm, ParallelLMHead = IMP["linear"]
    mp_t = getattr(cg, "hf_to_vllm_mapper", None)
    mp_t = mp_t.get_rename_mapper() if mp_t is not None else None
    packed_t = getattr(cg, "packed_modules_mapping", {})
    packed_d = getattr(mtp, "packed_modules_mapping", {}) if mtp is not None else {}
    pats, seen = {}, set()
    base = "klee100" if cand.startswith("klee100") else cand
    for m in DER[base]["modules"]:
        if m.startswith("mtp."):
            vname, qq, packed = m.replace("mtp.layers.0.", f"mtp.layers.{NL}.", 1), qd, packed_d
        else:
            vname = mp_t._map_name(m + ".weight") if mp_t is not None else m + ".weight"
            if vname is None:
                continue
            vname, qq, packed = vname[: -len(".weight")], qt, packed_t
        vname = fuse(vname, packed)
        if vname in seen:
            continue
        seen.add(vname)
        cls_name = next((c for k, c in LIN_CLASS if k in vname), "ColumnParallelLinear")
        cls = ParallelLMHead if cls_name == "ParallelLMHead" else getattr(Lm, cls_name)
        st = stub_linear(cls, ParallelLMHead, vname)
        try:
            meth = qq.get_quant_method(st, vname)
            mname = type(meth).__name__ if meth is not None else "None"
            extra = ""
            sch = getattr(st, "scheme", None)
            if sch is not None:
                extra = f" scheme={type(sch).__name__}"
            if hasattr(qq, "config_parser"):
                lc = qq.config_parser.resolve(st, vname)
                extra += f" W{lc.bits} g{lc.group_size}"
            k = f"{re.sub(r'[.]\d+[.]', '.N.', vname)} -> {mname}{extra}"
        except BaseException as e:
            k = f"{re.sub(r'[.]\d+[.]', '.N.', vname)} -> ERROR {err(e)[:160]}"
        pats[k] = pats.get(k, 0) + 1
    return pats


def ple_check(cand, qt, cg):
    out = {}
    Meth, guard = IMP["ple"], IMP["guard"]
    if Meth is None or guard is None:
        return {"error": "PLE modules not importable"}
    tc = CFGS[cand].get("text_config") or {}
    mp_t = cg.hf_to_vllm_mapper.get_rename_mapper()
    lid = (tc.get("ple_layer_ids") or [2])[0] - 1
    hfp = f"model.language_model.layers.{lid}.ple.ple_embedding.ngram_embedding"
    vp = mp_t._map_name(hfp + ".shard_0.weight")
    vp = vp.rsplit(".shard_0.weight", 1)[0] if vp else hfp
    base = "klee100" if cand.startswith("klee100") else cand
    fmt = DER[base]["ple"]
    stored = {"BF16": torch.bfloat16, "F16": torch.float16, "F8_E4M3": torch.float8_e4m3fn}.get(fmt.get("dtype"))
    for variant, edt in (("as_is", tc.get("ple_embedding_dtype")), ("fp8_graft", "float8_e4m3fn")):
        r = {"embedding_dtype": edt}
        try:
            m = Meth.from_quant_config(qt, vp, edt)
            r["method"] = type(m).__name__
            sd = stored if variant == "as_is" else torch.float8_e4m3fn
            r["stored_dtype"] = str(sd)
            try:
                guard(m, sd, vp)
                r["guard"] = "ACCEPTED"
            except BaseException as e:
                r["guard"] = f"REFUSED: {err(e)[:300]}"
        except BaseException as e:
            r["error"] = err(e)
        out[variant] = r
    out["checkpoint_table"] = fmt
    out["vllm_prefix"] = vp
    out["merlin_launcher_rule_launch_py_1219"] = ("PASS" if tc.get("ple_embedding_dtype") == "float8_e4m3fn"
                                                  else "FAIL (config lacks text_config.ple_embedding_dtype = float8_e4m3fn)")
    if fmt.get("container") != "safetensors":
        out["note"] = "table is not in safetensors (raw .bin): Merlin's loader cannot read it; graft Merlin's FP8 shards"
    return out


def mp_kernels():
    choose, MPC, st = IMP["mp"]
    res = {}
    for bits, g in ((2, 64), (2, 128), (3, 128), (4, 128), (8, 128)):
        t = {2: st.uint2b2, 3: st.uint3b4, 4: st.uint4b8, 8: st.uint8b128}[bits]
        try:
            k = choose(MPC(full_weight_shape=(H, 2 * H), partition_weight_shape=(H, H), weight_type=t,
                           act_type=torch.bfloat16, group_size=g, zero_points=False, has_g_idx=False))
            res[f"W{bits}g{g}"] = k.__name__
        except BaseException as e:
            res[f"W{bits}g{g}"] = f"ERROR {err(e)[:200]}"
    return res


if steps["platform"].get("ok") and IMP.get("quant") and IMP.get("moe"):
    cg, mtp = IMP.get("cg"), IMP.get("mtp")
    R["cands"] = {}
    if IMP.get("mp"):
        try:
            R["mp_linear_kernel_choice_sm89"] = mp_kernels()
        except BaseException as e:
            R["mp_linear_kernel_choice_sm89"] = err(e)
    for cand in CANDS:
        rec = {}
        try:
            name, qt = build_quant(cand, cg)
            _, qd = build_quant(cand, mtp)
            rec["quant"] = {"name": name, "class": type(qt).__name__, "min_capability": qt.get_min_capability(),
                            "configured_with": [getattr(cg, "__name__", None), getattr(mtp, "__name__", None)]}
            mp_t = cg.hf_to_vllm_mapper.get_rename_mapper() if cg is not None else None
            moe, mism = {}, []
            for L in [str(x) for x in range(NL)] + ["mtp"]:
                if L == "mtp":
                    vname, qq = f"mtp.layers.{NL}.mlp.experts", qd
                else:
                    hn = f"model.language_model.layers.{L}.mlp.experts.0.gate_proj.qweight"
                    mn = mp_t._map_name(hn) if mp_t is not None else hn
                    vname, qq = mn.split(".0.gate_proj.")[0], qt
                r = resolve_moe(cand, qq, vname)
                ck = ckpt_moe(cand).get(L, {})
                r["ckpt"] = [ck.get("fmt"), ck.get("bits"), ck.get("group")]
                rb = r.get("inc_layer_config")
                if isinstance(rb, list):
                    r["match"] = rb[0] == ck.get("bits") and (rb[0] >= 16 or rb[1] == ck.get("group"))
                else:
                    ib = r.get("implied_bits_from_params")
                    r["match"] = ib == ck.get("bits") if ib is not None else None
                if r["match"] is False:
                    mism.append(L)
                moe[L] = r
                save()
            rec["moe"] = moe
            rec["moe_mismatch_layers"] = mism
            rec["methods"] = sorted({m.get("method", "?") + "/" + str(m.get("wna16_backend", "")) for m in moe.values()})
            try:
                rec["ep_off_layer0"] = resolve_moe(cand, qt, moe["0"]["vllm_name"], ep=False)
            except BaseException as e:
                rec["ep_off_layer0"] = err(e)
            try:
                rec["linear_patterns"] = resolve_linears(cand, qt, qd, cg, mtp)
            except BaseException as e:
                rec["linear_patterns"] = {"error": err(e), "tb": traceback.format_exc()[-1500:]}
            try:
                rec["ple"] = ple_check(cand, qt, cg)
            except BaseException as e:
                rec["ple"] = {"error": err(e)}
            rec["loadable_as_is"] = not mism and all(m.get("method") for m in moe.values())
        except BaseException as e:
            rec["fatal"] = err(e)
            rec["tb"] = traceback.format_exc()[-2500:]
        R["cands"][cand] = rec
        save()
    ctl = R["cands"].get("merlin", {})
    R["control_ok"] = (not ctl.get("fatal") and not ctl.get("moe_mismatch_layers")
                       and all("Marlin" in str(m.get("experts_cls", "")) or "MARLIN" in str(m.get("wna16_backend", ""))
                               for k, m in (ctl.get("moe") or {}).items() if k != "mtp"))
    R["ran_all"] = all(not R["cands"].get(c, {}).get("fatal") for c in CANDS)
R["sec"] = round(time.time() - T0, 1)
save()
print("H3_JSON " + json.dumps({"platform_ok": steps.get("platform", {}).get("ok"), "ran_all": R.get("ran_all"),
                               "control_ok": R.get("control_ok")}))
