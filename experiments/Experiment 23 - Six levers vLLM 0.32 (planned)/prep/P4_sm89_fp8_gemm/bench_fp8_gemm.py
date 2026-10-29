#!/usr/bin/env python3
"""E23 P4 GPU microbenchmark: vLLM PR #61224 CUTLASS block-FP8 W8A8 GEMM (SM89) vs vLLM 0.32's Marlin FP8 (W8A16).

Runs with the vLLM 0.32 venv python on ONE L4 (sm_89). Same FP8 weights + 128x128 block scales for both kernels, on the
per-rank TP4 GEMM shapes of Qwen3.8-27B-FP8 (foysalemonshanto/qwen3-8-27b-fp8-repacked-v1, weight_block_size [128,128]).

  a) Marlin: exactly what MarlinFP8ScaledMMLinearKernel does in vLLM 0.32 (block_quant path):
       process_fp8_weight_block_strategy -> prepare_fp8_layer_for_marlin(layer, size_k_first=False)   (load time)
       apply_fp8_marlin_linear(input=x, weight, weight_scale_inv, workspace, size_n, size_k, bias=None) (per call)
     -> torch.ops.vllm.marlin_gemm -> torch.ops._C.marlin_gemm (b_q_type float8_e4m3fn, use_fp32_reduce=True).
     The workspace is created once (vLLM keeps a persistent per-stream workspace; None would allocate per call).
  b) CUTLASS (PR #61224, our .so): what CutlassFp8BlockScaledMMKernel.apply_weights would do on SM89 with the PR:
       per_token_group_quant_fp8(x, 128, column_major_scales=True, use_ue8m0=False)    (QuantFP8, group (1,128))
       out = empty(M, N, bf16); torch.ops.p4_sm89.cutlass_scaled_mm_blockwise_sm89_fp8(out, xq, W.t(), xs, Ws.t())
     (= ops.cutlass_scaled_mm(A, B.T, scale_a=As, scale_b=Bs.T) -> cutlass_scaled_mm_sm89 -> blockwise branch).
     Timed twice: GEMM only (pre-quantized input) and quant + GEMM (what a layer costs unless the quant is fused).
  c) cuBLAS BF16 torch.mm on the dequantized weight (context only; --no-bf16 skips it).

Order per shape: numerical match FIRST (both kernels vs an fp32 reference on the dequantized BF16 weight: Marlin vs
x @ Wdq^T, CUTLASS vs dequant(xq) @ Wdq^T, plus the end-to-end CUTLASS error vs x @ Wdq^T, plus a row-major a_scales
layout run that must reproduce the column-major result), then latency per M with CUDA events: warmup, 100 eager iters
(per-iteration events) and the same calls captured in a CUDA graph (GPU time; vLLM runs decode under CUDA graphs).
Weights are rotated over several copies so the working set exceeds the 48 MB L2 (no warm-L2 wins).

Never crashes the caller: the main process does not touch CUDA; each shape runs in a child process (timeout); every
outcome (including a child segfault) ends up in the JSON. Exit codes: 0 = all shapes ran and all gates passed,
1 = a gate failed or a shape errored (JSON has details), 2 = fatal setup error, 3 = --check-imports failed.

usage (GPU):  VPY bench_fp8_gemm.py --so /kaggle/working/p4_sm89_fp8.so --out /kaggle/working/p4_bench.json
              [--weights auto|real|synthetic] [--model-dir DIR] [--m 1,4,8,16,32,48,64,80,128] [--iters 100]
usage (CPU):  VPY bench_fp8_gemm.py --check-imports [--so ...] --out check.json   (imports + signatures, no GPU)
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

BENCH_VERSION = "p4-bench-1"
M_DEFAULT = [1, 4, 8, 16, 32, 48, 64, 80, 128]
OP_NS, OP_NAME = "p4_sm89", "cutlass_scaled_mm_blockwise_sm89_fp8"
MODEL_DIR_CANDIDATES = [
    "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1",
    "/kaggle/input/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1",
]
# Gates (numerical match). BF16 output rounding is ~2^-9 relative; a wrong scale layout or tile bug gives O(1).
GATE_REL_FRO = 1e-2        # ||out - ref||_F / ||ref||_F, each kernel vs its own exact reference
GATE_ALT_LAYOUT = 1e-6     # row-major a_scales must reproduce the column-major result
WARN_E2E_REL_FRO = 8e-2    # CUTLASS (activation-quantized) vs the W8A16 reference: informative, flagged above this

# Qwen3.8-27B text config: hidden 5120, intermediate 17408, 64 layers (48 GDN linear_attention + 16 full_attention,
# full_attention_interval 4), 24 heads x head_dim 256 with attn_output_gate (q_proj = 2 x 6144), 4 KV heads,
# GDN: 16 key heads x 128, 48 value heads x 128 (key_dim 2048, value_dim 6144), 1 MTP layer (full attention + MLP).
# Per-rank TP4 shapes of the FP8 linears (column-parallel: N/4, row-parallel: K/4). in_proj_ba, conv1d, mtp.fc,
# embed_tokens, lm_head are BF16 in this checkpoint (modules_to_not_convert) and are not FP8 GEMMs.
L_ATTN = "model.language_model.layers.3.self_attn."
L_GDN = "model.language_model.layers.0.linear_attn."
L_MLP = "model.language_model.layers.0.mlp."
SHAPES = {
    "qkv_proj": {"n": 3584, "k": 5120, "count_main": 16, "count_mtp": 1, "parallel": "column (QKVParallelLinear)",
                 "real": {"rows": [(L_ATTN + "q_proj", 0, 3072), (L_ATTN + "k_proj", 0, 256), (L_ATTN + "v_proj", 0, 256)]}},
    "o_proj+gdn_out_proj": {"n": 5120, "k": 1536, "count_main": 64, "count_mtp": 1, "parallel": "row (RowParallelLinear)",
                            "note": "16 full-attention o_proj + 48 GDN out_proj share this shape",
                            "real": {"cols": (L_ATTN + "o_proj", 0, 1536)}},
    "gdn_in_proj_qkvz": {"n": 4096, "k": 5120, "count_main": 48, "count_mtp": 0, "parallel": "column (Merged, [2048,2048,6144,6144])",
                         "real": {"rows": [(L_GDN + "in_proj_qkv", 0, 512), (L_GDN + "in_proj_qkv", 2048, 2560),
                                           (L_GDN + "in_proj_qkv", 4096, 5632), (L_GDN + "in_proj_z", 0, 1536)]}},
    "gate_up_proj": {"n": 8704, "k": 5120, "count_main": 64, "count_mtp": 1, "parallel": "column (Merged, [17408,17408])",
                     "real": {"rows": [(L_MLP + "gate_proj", 0, 4352), (L_MLP + "up_proj", 0, 4352)]}},
    "down_proj": {"n": 5120, "k": 4352, "count_main": 64, "count_mtp": 1, "parallel": "row (RowParallelLinear)",
                  "real": {"cols": (L_MLP + "down_proj", 0, 4352)}},
}
BF16_PER_RANK = {"gdn_in_proj_ba": {"n": 24, "k": 5120, "count_main": 48},
                 "mtp_fc": {"n": 1280, "k": 10240, "count_mtp": 1},
                 "lm_head": {"n": 62080, "k": 5120, "count_main": 1}}
KERNELS = ["marlin_w8a16", "cutlass_w8a8_gemm", "cutlass_w8a8_quant_gemm", "quant_only", "bf16_cublas"]


def tile_for_m(m):  # dispatch_sm89_blockwise in the PR .cu
    if m <= 16:
        return "Tile 16x128x128 Warp 16x64x128 stages 5"
    if m <= 32:
        return "Tile 32x64x128 Warp 16x32x128 stages 4"
    if m <= 64:
        return "Tile 64x128x128 Warp 32x64x128 stages 3"
    return "Tile 128x128x128 Warp 64x32x128 stages 3 swizzle 4"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def stats(xs):
    xs = sorted(float(x) for x in xs)
    n = len(xs)
    if n == 0:
        return None
    q = lambda p: xs[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {"median": round(q(0.5), 3), "mean": round(sum(xs) / n, 3), "p10": round(q(0.1), 3),
            "p90": round(q(0.9), 3), "min": round(xs[0], 3), "n": n}


# ----------------------------------------------------------------------------------------------- child (GPU, one shape)
def find_model_dir(arg):
    cands = [arg] if arg else MODEL_DIR_CANDIDATES
    for c in cands:
        if c and (Path(c) / "model.safetensors.index.json").exists():
            return c
    if not arg and Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").glob("**/qwen3-8-27b-fp8-repacked-v1/**/model.safetensors.index.json"):
            return str(p.parent)
    return None


def load_real_weights(torch, model_dir, spec, n, k):
    """Rank-0 TP4 shard of the real checkpoint tensors, laid out as vLLM's merged / QKV / row-parallel layer holds it."""
    from safetensors import safe_open
    wmap = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]

    def get(name, rows=None, cols=None):
        with safe_open(str(Path(model_dir) / wmap[name]), framework="pt", device="cpu") as h:
            sl = h.get_slice(name)
            shape = sl.get_shape()
            r = slice(*rows) if rows else slice(0, shape[0])
            c = slice(*cols) if cols else slice(0, shape[1])
            return sl[r, c], shape

    ws, ss, src = [], [], []
    if "rows" in spec:
        for base, a, b in spec["rows"]:
            assert a % 128 == 0 and b % 128 == 0
            w, wshape = get(base + ".weight", rows=(a, b))
            s, sshape = get(base + ".weight_scale_inv", rows=(a // 128, b // 128))
            assert tuple(sshape) == (math.ceil(wshape[0] / 128), math.ceil(wshape[1] / 128)), (base, wshape, sshape)
            ws.append(w), ss.append(s), src.append(f"{base} rows [{a}:{b}] of {list(wshape)}")
        w, s = torch.cat(ws, 0), torch.cat(ss, 0)
    else:
        base, a, b = spec["cols"]
        w, wshape = get(base + ".weight", cols=(a, b))
        s, sshape = get(base + ".weight_scale_inv", cols=(a // 128, b // 128))
        assert tuple(sshape) == (math.ceil(wshape[0] / 128), math.ceil(wshape[1] / 128)), (base, wshape, sshape)
        src.append(f"{base} cols [{a}:{b}] of {list(wshape)}")
    assert w.dtype == torch.float8_e4m3fn, w.dtype
    assert tuple(w.shape) == (n, k) and tuple(s.shape) == (n // 128, k // 128), (w.shape, s.shape)
    return w.contiguous(), s.float().contiguous(), src


def synthetic_weights(torch, n, k, dev, seed):
    """Random weights with per-block magnitude spread (so wrong N/K block-scale indexing shows), quantized exactly like a
    128x128 block-FP8 checkpoint: scale = amax / 448, w_fp8 = w / scale."""
    g = torch.Generator(device=dev).manual_seed(seed)
    nb, kb = n // 128, k // 128
    w = torch.randn(n, k, device=dev, dtype=torch.float32, generator=g) * 0.02
    blk = torch.exp2(torch.empty(nb, 1, kb, 1, device=dev).uniform_(-2.0, 2.0, generator=g))
    wb = w.view(nb, 128, kb, 128) * blk
    amax = wb.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    s = amax / 448.0
    wq = (wb / s[:, None, :, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(n, k)
    return wq.contiguous(), s.float().contiguous(), ["synthetic randn*0.02 x per-block 2^U(-2,2), seed %d" % seed]


def metrics(torch, out, ref):
    o = out.float()
    err = o - ref
    rn = ref.norm().clamp_min(1e-30)
    return {"finite": bool(torch.isfinite(o).all().item()), "max_abs": float(err.abs().max().item()),
            "max_abs_over_ref_max": float((err.abs().max() / ref.abs().max().clamp_min(1e-30)).item()),
            "rel_fro": float((err.norm() / rn).item()),
            "cos": float(torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item())}


def run_child(args):
    res = {"shape": args.child, "errors": [], "correctness": {}, "timing": {}}
    out_path = args.child_out

    def save():
        write_json(out_path, res)

    try:
        import torch
        spec = SHAPES[args.child]
        n, k = spec["n"], spec["k"]
        res.update({"n": n, "k": k})
        assert n % 128 == 0 and k % 128 == 0
        res["env"] = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "python": sys.version.split()[0]}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        dev = torch.device("cuda", 0)
        torch.cuda.set_device(dev)
        prop = torch.cuda.get_device_properties(dev)
        cap = torch.cuda.get_device_capability(dev)
        res["env"].update({"gpu": prop.name, "capability": list(cap), "sm_count": prop.multi_processor_count,
                           "l2_bytes": int(getattr(prop, "L2_cache_size", 0) or 0), "total_mem": int(prop.total_memory)})
        if tuple(cap) != (8, 9) and not args.allow_other_arch:
            raise RuntimeError(f"needs sm_89 (L4), got {cap}; the .so has sm_89 SASS only")
        so = Path(args.so)
        torch.ops.load_library(str(so))
        op = getattr(getattr(torch.ops, OP_NS), OP_NAME)
        res["env"]["op_schema"] = str(op.default._schema)
        import vllm
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8, process_fp8_weight_block_strategy)
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            MARLIN_MAX_BLOCKS_PER_SM, marlin_make_workspace_new)
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            apply_fp8_marlin_linear, prepare_fp8_layer_for_marlin)
        from vllm.model_executor.utils import replace_parameter
        res["env"]["vllm"] = vllm.__version__
        res["env"]["VLLM_MARLIN_USE_ATOMIC_ADD"] = os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "")
        try:  # stock vLLM 0.32 on sm_89: False -> CutlassFp8BlockScaledMMKernel is skipped and Marlin is selected
            from vllm.model_executor.layers.quantization.utils.w8a8_utils import CUTLASS_BLOCK_FP8_SUPPORTED
            res["env"]["stock_vllm_CUTLASS_BLOCK_FP8_SUPPORTED"] = bool(CUTLASS_BLOCK_FP8_SUPPORTED)
        except Exception as e:
            res["env"]["stock_vllm_CUTLASS_BLOCK_FP8_SUPPORTED"] = f"{type(e).__name__}: {e}"

        # ---- weights (same FP8 bytes + fp32 block scales for every kernel)
        model_dir = find_model_dir(args.model_dir) if args.weights in ("auto", "real") else None
        if args.weights == "real" and not model_dir:
            raise RuntimeError("--weights real but no model dir with model.safetensors.index.json was found")
        t0 = time.time()
        if model_dir:
            w_cpu, s_cpu, src = load_real_weights(torch, model_dir, spec["real"], n, k)
            W0, S0 = w_cpu.to(dev), s_cpu.to(dev)
            res["weights_source"] = {"kind": "real", "model_dir": model_dir, "tensors": src}
        else:
            W0, S0, src = synthetic_weights(torch, n, k, dev, seed=args.seed)
            res["weights_source"] = {"kind": "synthetic", "tensors": src}
        res["weights_source"]["load_sec"] = round(time.time() - t0, 1)
        Wdq = (W0.float().view(n // 128, 128, k // 128, 128) * S0[:, None, :, None]).view(n, k)  # fp32 reference weight

        l2 = res["env"]["l2_bytes"] or 48 * 2 ** 20
        rot = max(3 * l2, args.rotate_mb * 2 ** 20)
        R = min(args.max_copies, max(2, math.ceil(rot / (n * k))))
        R_bf = min(args.max_copies, max(2, math.ceil(rot / (2 * n * k))))
        res["copies"] = {"fp8_and_marlin": R, "bf16": R_bf, "rotate_bytes": rot, "weight_bytes_fp8": n * k}

        def make_marlin(W, S):  # MarlinFP8ScaledMMLinearKernel.process_weights_after_loading, block_quant branch
            layer = torch.nn.Module()
            layer.weight = torch.nn.Parameter(W.clone(), requires_grad=False)
            layer.weight_scale_inv = torch.nn.Parameter(S.clone(), requires_grad=False)
            layer.output_size_per_partition, layer.input_size_per_partition = n, k
            layer.weight_block_size = [128, 128]
            layer.orig_dtype = torch.bfloat16
            w, s = process_fp8_weight_block_strategy(layer.weight, layer.weight_scale_inv)
            replace_parameter(layer, "weight", w.data)
            replace_parameter(layer, "weight_scale_inv", s.data)
            layer.input_scale = None
            prepare_fp8_layer_for_marlin(layer, size_k_first=False, input_dtype=None)
            del layer.input_scale
            return layer.weight.data, layer.weight_scale_inv.data

        Wc = [W0] + [W0.clone() for _ in range(R - 1)]
        Sc = [S0] + [S0.clone() for _ in range(R - 1)]
        Bt = [w.t() for w in Wc]                     # [K, N] column-major view (= B.T in CutlassFp8BlockScaledMMKernel)
        Bst = [s.t() for s in Sc]                    # [K/128, N/128] view (= Bs.T)
        MW = [make_marlin(w, s) for w, s in zip(Wc, Sc)]
        res["marlin_layout"] = {"qweight": list(MW[0][0].shape), "qweight_dtype": str(MW[0][0].dtype),
                                "scales": list(MW[0][1].shape), "scales_dtype": str(MW[0][1].dtype)}
        workspace = marlin_make_workspace_new(dev, MARLIN_MAX_BLOCKS_PER_SM)
        WB = None
        if not args.no_bf16:
            wb0 = Wdq.to(torch.bfloat16)
            WB = [wb0] + [wb0.clone() for _ in range(R_bf - 1)]
        torch.cuda.synchronize()
        res["mem_after_setup_gib"] = round(torch.cuda.memory_allocated(dev) / 2 ** 30, 2)

        def quant(x):
            return per_token_group_quant_fp8(x, 128, column_major_scales=True, dtype=torch.float8_e4m3fn,
                                             use_ue8m0=False)

        def marlin(x, i):
            return apply_fp8_marlin_linear(input=x, weight=MW[i][0], weight_scale=MW[i][1], workspace=workspace,
                                           size_n=n, size_k=k, bias=None, input_dtype=None)

        def cutlass(xq, xs, i, out=None):
            if out is None:
                out = torch.empty((xq.shape[0], n), dtype=torch.bfloat16, device=dev)
            op(out, xq, Bt[i], xs, Bst[i])
            return out

        gate_ok = True
        for m in args.m:
            key = str(m)
            g = torch.Generator(device=dev).manual_seed(args.seed * 1000 + m)
            row_mag = torch.exp2(torch.empty(m, 1, device=dev).uniform_(-3.0, 3.0, generator=g))  # per-token spread
            x = (torch.randn(m, k, device=dev, dtype=torch.float32, generator=g) * row_mag).to(torch.bfloat16)
            entry = {"tile": tile_for_m(m)}
            try:  # ---------------- correctness first
                xq, xs = quant(x)
                entry["a_scales_layout"] = {"shape": list(xs.shape), "stride": list(xs.stride()), "dtype": str(xs.dtype)}
                ref16 = x.float() @ Wdq.t()
                xdq = xq.float() * xs.repeat_interleave(128, dim=1)
                ref8 = xdq @ Wdq.t()
                oc = cutlass(xq, xs, 0)
                om = marlin(x, 0)
                torch.cuda.synchronize()
                entry["marlin_vs_ref_w8a16"] = metrics(torch, om, ref16)
                entry["cutlass_vs_ref_w8a8"] = metrics(torch, oc, ref8)
                entry["cutlass_vs_ref_w8a16_e2e"] = metrics(torch, oc, ref16)
                entry["cutlass_vs_marlin"] = metrics(torch, oc, om.float())
                oc2 = cutlass(xq, xs.contiguous(), 0)       # row-major a_scales (PR claims any strides)
                torch.cuda.synchronize()
                alt = metrics(torch, oc2, oc.float())
                alt["bitwise_equal"] = bool(torch.equal(oc2, oc))
                entry["cutlass_rowmajor_scales_vs_colmajor"] = alt
                ok = (entry["marlin_vs_ref_w8a16"]["finite"] and entry["cutlass_vs_ref_w8a8"]["finite"]
                      and entry["marlin_vs_ref_w8a16"]["rel_fro"] <= GATE_REL_FRO
                      and entry["cutlass_vs_ref_w8a8"]["rel_fro"] <= GATE_REL_FRO
                      and alt["rel_fro"] <= GATE_ALT_LAYOUT)
                entry["gate_pass"] = bool(ok)
                entry["warn_e2e"] = entry["cutlass_vs_ref_w8a16_e2e"]["rel_fro"] > WARN_E2E_REL_FRO
                gate_ok &= ok
                del ref16, xdq, ref8, oc, om, oc2
            except Exception as e:
                entry["error"] = f"{type(e).__name__}: {e}"
                entry["traceback"] = traceback.format_exc()[-3000:]
                entry["gate_pass"] = False
                gate_ok = False
            res["correctness"][key] = entry
            save()
            if "error" in entry and "CUDA" in entry["error"]:
                res["errors"].append(f"M={m}: CUDA error in correctness, timing skipped: {entry['error']}")
                break

            # ---------------- latency
            xq, xs = quant(x)
            out_buf = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
            cands = {
                "marlin_w8a16": [(lambda i=i: marlin(x, i)) for i in range(R)],
                "cutlass_w8a8_gemm": [(lambda i=i: cutlass(xq, xs, i, out_buf)) for i in range(R)],
                "cutlass_w8a8_quant_gemm": [(lambda i=i: cutlass(*quant(x), i)) for i in range(R)],
                "quant_only": [lambda: quant(x)],
            }
            if WB is not None:
                cands["bf16_cublas"] = [(lambda i=i: torch.mm(x, WB[i].t())) for i in range(R_bf)]
            tm = {}
            for name, fns in cands.items():
                try:
                    tm[name] = bench(torch, fns, args.warmup, args.iters, not args.no_graph)
                    t = (tm[name].get("graph_us") or tm[name]["eager_us"])["median"]
                    if name != "quant_only":
                        tm[name]["weight_gbps_at_median"] = round(n * k * (2 if name == "bf16_cublas" else 1) / (t * 1e3), 1)
                except Exception as e:
                    tm[name] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
            res["timing"][key] = tm
            save()
            del x, xq, xs, out_buf
            torch.cuda.empty_cache()
        res["gate_pass"] = bool(gate_ok)
        res["mem_peak_gib"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
    except Exception as e:
        res["errors"].append(f"{type(e).__name__}: {e}")
        res["traceback"] = traceback.format_exc()[-4000:]
        res["gate_pass"] = False
    save()
    return 0 if res.get("gate_pass") and not res["errors"] else 1


def bench(torch, fns, warmup, iters, use_graph):
    R = len(fns)
    for i in range(max(warmup, R)):
        fns[i % R]()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        st[i].record()
        fns[i % R]()
        en[i].record()
    torch.cuda.synchronize()
    out = {"eager_us": stats([a.elapsed_time(b) * 1e3 for a, b in zip(st, en)]), "copies": R}
    if use_graph:
        try:
            calls = R * max(1, math.ceil(10 / R))
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for i in range(3 * R):
                    fns[i % R]()
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i in range(calls):
                    fns[i % R]()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            reps = max(10, math.ceil(iters / calls))
            gs = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
            ge = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
            for r in range(reps):
                gs[r].record()
                graph.replay()
                ge[r].record()
            torch.cuda.synchronize()
            out["graph_us"] = stats([a.elapsed_time(b) * 1e3 / calls for a, b in zip(gs, ge)])
            out["graph_calls_per_replay"], out["graph_replays"] = calls, reps
            del graph
        except Exception as e:
            out["graph_error"] = f"{type(e).__name__}: {e}"
    return out


# ------------------------------------------------------------------------------------------- CPU import / layout check
def check_imports(args):
    rep = {"mode": "check_imports", "bench_version": BENCH_VERSION, "checks": {}}
    ok = True

    def step(name, fn):
        nonlocal ok
        try:
            rep["checks"][name] = {"ok": True, "info": fn()}
        except Exception as e:
            ok = False
            rep["checks"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2500:]}

    import inspect
    import torch
    rep["torch"], rep["torch_cuda"] = torch.__version__, torch.version.cuda
    if args.so:
        def load():
            torch.ops.load_library(args.so)
            opx = getattr(getattr(torch.ops, OP_NS), OP_NAME)
            return str(opx.default._schema)
        step("load_so", load)

    def need(fn, params):
        sig = inspect.signature(fn)
        miss = [p for p in params if p not in sig.parameters]
        assert not miss, f"{fn.__name__} lacks {miss}; signature {sig}"
        return str(sig)

    def fp8u():
        from vllm.model_executor.layers.quantization.utils import fp8_utils as f
        return {"per_token_group_quant_fp8": need(f.per_token_group_quant_fp8, ["x", "group_size", "column_major_scales",
                                                                                "dtype", "use_ue8m0"]),
                "process_fp8_weight_block_strategy": need(f.process_fp8_weight_block_strategy, ["weight", "weight_scale"])}
    step("fp8_utils", fp8u)

    def marl():
        from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
        from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as m8
        assert hasattr(mu, "MARLIN_MAX_BLOCKS_PER_SM") and hasattr(mu, "marlin_make_workspace_new")
        return {"apply_fp8_marlin_linear": need(m8.apply_fp8_marlin_linear, ["input", "weight", "weight_scale",
                                                                             "workspace", "size_n", "size_k", "bias",
                                                                             "input_dtype"]),
                "prepare_fp8_layer_for_marlin": need(m8.prepare_fp8_layer_for_marlin, ["layer", "size_k_first",
                                                                                       "input_dtype"]),
                "marlin_make_workspace_new": need(mu.marlin_make_workspace_new, ["device", "max_blocks_per_sm"]),
                "torch.ops.vllm.marlin_gemm": hasattr(torch.ops.vllm, "marlin_gemm")}
    step("marlin_utils", marl)

    def kern():
        from vllm.model_executor.kernels.linear.scaled_mm import marlin as km
        from vllm.model_executor.kernels.linear.scaled_mm import cutlass as kc
        src_m = inspect.getsource(km.MarlinFP8ScaledMMLinearKernel)
        src_c = inspect.getsource(kc.CutlassFp8BlockScaledMMKernel)
        assert "prepare_fp8_layer_for_marlin(" in src_m and "process_fp8_weight_block_strategy(" in src_m
        assert "apply_fp8_marlin_linear(" in src_m and "column_major_scales=True" in src_c and "B.T" in src_c
        return {"MarlinFP8ScaledMMLinearKernel": "uses process_fp8_weight_block_strategy + prepare_fp8_layer_for_marlin "
                                                 "+ apply_fp8_marlin_linear (same calls as this bench)",
                "CutlassFp8BlockScaledMMKernel": "QuantFP8(column_major_scales=True) + cutlass_scaled_mm(A, B.T, As, Bs.T)"}
    step("vllm_kernels_match_bench", kern)

    def vutil():
        from vllm.model_executor.utils import replace_parameter
        return need(replace_parameter, ["layer", "param_name", "new_tensor"])
    step("replace_parameter", vutil)

    def safet():
        import safetensors
        return safetensors.__version__
    step("safetensors", safet)
    rep["all_ok"] = ok
    write_json(args.out, rep)
    print("CHECK_IMPORTS " + json.dumps({"all_ok": ok, "failed": [k for k, v in rep["checks"].items() if not v["ok"]]}))
    return 0 if ok else 3


# ---------------------------------------------------------------------------------------------- orchestrator (no CUDA)
def nvsmi():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,name,clocks.sm,clocks.max.sm,clocks.mem,power.draw,"
                            "power.limit,temperature.gpu,utilization.gpu,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip().splitlines()
    except Exception as e:
        return [f"nvidia-smi unavailable: {e}"]


def summarize(shapes_res, ms):
    """Per M: sum over the main model's FP8 linears (count x graph median per call) per kernel, and the MTP layer."""
    summ = {}
    for m in ms:
        row = {}
        for kname in ("marlin_w8a16", "cutlass_w8a8_gemm", "cutlass_w8a8_quant_gemm", "bf16_cublas"):
            tot, tot_mtp, missing = 0.0, 0.0, []
            for sname, spec in SHAPES.items():
                t = ((shapes_res.get(sname) or {}).get("timing") or {}).get(str(m), {}).get(kname) or {}
                v = (t.get("graph_us") or t.get("eager_us") or {}).get("median")
                if v is None:
                    missing.append(sname)
                    continue
                tot += spec["count_main"] * v
                tot_mtp += spec["count_mtp"] * v
            row[kname] = {"main_model_us": round(tot, 1), "mtp_layer_us": round(tot_mtp, 1), "missing": missing}
        mar, cq, cg = (row[x]["main_model_us"] for x in ("marlin_w8a16", "cutlass_w8a8_quant_gemm", "cutlass_w8a8_gemm"))
        complete = not row["marlin_w8a16"]["missing"] and not row["cutlass_w8a8_quant_gemm"]["missing"]
        row["speedup_marlin_over_cutlass_quant_gemm"] = round(mar / cq, 3) if complete and cq else None
        row["speedup_marlin_over_cutlass_gemm_only"] = round(mar / cg, 3) if complete and cg else None
        row["saved_us_per_forward_quant_gemm"] = round(mar - cq, 1) if complete else None
        summ[str(m)] = row
    return summ


def orchestrate(args):
    t0 = time.time()
    rep = {"bench_version": BENCH_VERSION, "argv": sys.argv[1:], "so": args.so, "m": args.m, "iters": args.iters,
           "warmup": args.warmup, "graph": not args.no_graph, "weights": args.weights, "gates": {
               "rel_fro_each_kernel_vs_own_ref": GATE_REL_FRO, "alt_layout_rel_fro": GATE_ALT_LAYOUT,
               "warn_e2e_rel_fro": WARN_E2E_REL_FRO},
           "shapes_spec": {k: {kk: vv for kk, vv in v.items() if kk != "real"} for k, v in SHAPES.items()},
           "bf16_not_fp8_per_rank": BF16_PER_RANK, "nvidia_smi_before": nvsmi(), "shapes": {}, "errors": []}
    if not args.so or not Path(args.so).exists():
        rep["errors"].append(f"fatal: --so {args.so!r} does not exist")
        rep["all_ok"] = False
        write_json(args.out, rep)
        print("P4_BENCH " + json.dumps({"all_ok": False, "fatal": rep["errors"][-1]}))
        return 2
    tmpdir = Path(args.out).with_suffix("").as_posix() + "_parts"
    Path(tmpdir).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    names = list(SHAPES) if args.shapes == "all" else [s for s in args.shapes.split(",") if s]
    for sname in names:
        if sname not in SHAPES:
            rep["errors"].append(f"unknown shape {sname}")
            continue
        part = Path(tmpdir) / f"{sname}.json"
        if part.exists():
            part.unlink()
        cmd = [sys.executable, os.path.abspath(__file__), "--child", sname, "--child-out", str(part), "--so", args.so,
               "--m", ",".join(map(str, args.m)), "--iters", str(args.iters), "--warmup", str(args.warmup),
               "--weights", args.weights, "--seed", str(args.seed), "--rotate-mb", str(args.rotate_mb),
               "--max-copies", str(args.max_copies)]
        for flag in ("no_graph", "no_bf16", "allow_other_arch"):
            if getattr(args, flag):
                cmd.append("--" + flag.replace("_", "-"))
        if args.model_dir:
            cmd += ["--model-dir", args.model_dir]
        ts = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout_per_shape, env=env)
            rc, tail = r.returncode, (r.stdout + r.stderr)[-3000:]
        except subprocess.TimeoutExpired as e:
            rc, tail = "timeout", f"timeout after {args.timeout_per_shape}s: {str(e.stdout or '')[-1500:]}"
        try:
            sres = json.loads(part.read_text(encoding="utf-8"))
        except Exception:
            sres = {"shape": sname, "errors": ["child wrote no JSON"], "gate_pass": False}
        sres["child_rc"], sres["child_sec"] = rc, round(time.time() - ts, 1)
        if rc != 0:
            sres["child_tail"] = tail
            if rc not in (0, 1):
                sres.setdefault("errors", []).append(f"child exit {rc} (crash/timeout)")
        rep["shapes"][sname] = sres
        write_json(args.out, rep)
        print(f"[p4-bench] {sname}: rc={rc} gate_pass={sres.get('gate_pass')} errors={sres.get('errors')} "
              f"{sres['child_sec']}s", flush=True)
    rep["summary_per_m"] = summarize(rep["shapes"], args.m)
    rep["nvidia_smi_after"] = nvsmi()
    rep["all_ok"] = bool(rep["shapes"]) and not rep["errors"] and all(
        s.get("gate_pass") and not s.get("errors") for s in rep["shapes"].values())
    rep["sec"] = round(time.time() - t0, 1)
    write_json(args.out, rep)
    print("P4_BENCH " + json.dumps({"all_ok": rep["all_ok"], "sec": rep["sec"], "speedup_marlin_over_cutlass_quant_gemm":
                                    {m: v["speedup_marlin_over_cutlass_quant_gemm"]
                                     for m, v in rep["summary_per_m"].items()}}))
    return 0 if rep["all_ok"] else 1


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so", default="/kaggle/working/p4_sm89_fp8.so")
    ap.add_argument("--out", default="/kaggle/working/p4_bench.json")
    ap.add_argument("--m", type=lambda s: [int(x) for x in s.split(",") if x], default=M_DEFAULT)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--shapes", default="all", help="comma list of " + ",".join(SHAPES))
    ap.add_argument("--weights", choices=["auto", "real", "synthetic"], default="auto")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rotate-mb", type=int, default=144, help="min weight working set per kernel (L2 is 48 MB)")
    ap.add_argument("--max-copies", type=int, default=16)
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES for the children (default: inherit)")
    ap.add_argument("--timeout-per-shape", type=int, default=900)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--allow-other-arch", action="store_true")
    ap.add_argument("--check-imports", action="store_true")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main():
    try:
        args = parse_args()
    except SystemExit as e:
        return int(e.code or 0) if isinstance(e.code, int) else 2
    try:
        if args.child:
            return run_child(args)
        if args.check_imports:
            return check_imports(args)
        return orchestrate(args)
    except Exception as e:
        try:
            write_json(args.out if not args.child else args.child_out,
                       {"fatal": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:], "all_ok": False})
        except Exception:
            pass
        print("P4_BENCH " + json.dumps({"all_ok": False, "fatal": f"{type(e).__name__}: {e}"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
