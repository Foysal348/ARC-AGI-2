#!/usr/bin/env python3
"""E25 NB3 D4: does Merlin's low-latency skinny-GEMM plan path (CuTe DSL) compile and run on sm_89 for the Flash-Next
TP2 BF16 shapes, which plans are missing, and how fast is it against the cuBLAS fallback the engine uses today?

Merlin source (verified in _scratch/e23/p2/merlin-src):
  vllm/models/qwen4_exp/nvidia/low_latency_gemm.py: QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY has (10,3) (10,0) (9,0) (12,1)
    only (:294-300); _qwen4_exp_low_latency_gemm (:347-357) uses shape_dynamic_skinny_gemm when a plan has (N, K) and
    M = x.shape[0], else torch.nn.functional.linear; enable_qwen4_exp_low_latency_gemm (:376-412) returns early with no plans;
  vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm.py: SkinnyGemmConfig, shape_dynamic_skinny_gemm (1 <= M <= 16,
    compile on first call, PDL only when current_platform.is_arch_support_pdl(), i.e. major >= 9).
Per shape (local TP2 (N, K), nb3_common.gemm_shapes()) and M in {1, 2, 4, 8, 16}: cuBLAS (F.linear) timing + error, then
the SM121 (GB10) plan entry for that (N, K, M) if any, the skinny heuristic default (_config) and, for the biggest
shapes, a small config sweep: compile (seconds), error vs an fp32 reference, CUDA-graph timing over rotated weight
copies (cold L2), and the real dispatch path with an (8, 9) plan table patched in (the NB4 data patch, simulated).
Weights are random BF16 (timing and numerics of a GEMM do not depend on the checkpoint's values); the checkpoint's
safetensors headers are read only to record the real BF16 tensor shapes.
Timing (N3-12): eager_us and graph_us are kept apart; a failed CUDA-graph capture is graph_status
INCONCLUSIVE_CAPTURE_FAILED, never a graph number, and never a winner. No skinny config able to run on sm_89 = status
UNSUPPORTED_SM89 (diagnostic, the cuBLAS numbers kept; the other benches run on). Draft heads: a GPU-side preflight of the
Marlin linear ops (draft_head_preflight); missing ops = draft heads UNTESTED, not a D4 failure.
Exit: 0 all numeric checks pass (status DONE / UNSUPPORTED_SM89 / PARTIAL_BUDGET), 1 a check failed, 2 error.
"""
import argparse
import importlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb3_common as C  # noqa: E402

M_LIST = (1, 2, 4, 8, 16)
GATE_REL_FRO = 1e-2                      # P4's numerical-match gate for one GEMM (bf16 output rounding ~2^-9)
SWEEP_SHAPES = ((124160, 2560), (10240, 320), (336, 10240), (6656, 2560))
SWEEP_M = (8, 16)
SWEEP_CFGS = ((32, 1, 8), (32, 2, 4), (64, 1, 8), (64, 2, 8), (64, 4, 4), (128, 1, 4), (128, 2, 4), (128, 2, 8))
HEADER_NAMES = ("model.language_model.layers.0.mlp.gate.weight", "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
                "model.language_model.layers.0.mlp.shared_expert.up_proj.weight", "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
                "model.language_model.layers.0.mlp.shared_expert_gate.weight", "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                "model.language_model.layers.0.linear_attn.in_proj_b.weight",
                "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
                "model.language_model.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
                "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight", "lm_head.weight")


def header_shapes(ckpt):
    """Shapes of the checkpoint's BF16 tensors from the safetensors headers (no data read)."""
    out = {}
    try:
        wm = json.loads((Path(ckpt) / "model.safetensors.index.json").read_text())["weight_map"]
        names = list(HEADER_NAMES) + sorted(n for n in wm if n.startswith("model.language_model.layers.3.self_attn.")
                                            and n.endswith(".weight"))[:24]
        from safetensors import safe_open
        hs = {}
        for n in names:
            if n not in wm:
                out[n] = "absent from index"
                continue
            f = wm[n]
            if f not in hs:
                hs[f] = safe_open(str(Path(ckpt) / f), framework="pt", device="cpu")
            sl = hs[f].get_slice(n)
            out[n] = {"shape": list(sl.get_shape()), "dtype": str(sl.get_dtype()), "file": f}
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def cfg_dict(c):
    return {k: getattr(c, k) for k in ("num_rows", "block_size", "outputs_per_block", "k_unroll", "vector_width", "static_k")}


def cfg_valid(c, N, K):
    return N % c.outputs_per_block == 0 and K % (c.block_size * c.vector_width) == 0 and (c.static_k in (None, K))


def real(a, res, save):
    import torch
    t_start = time.time()
    left = lambda: a.budget_s - (time.time() - t_start)  # noqa: E731
    imp = res["imports"]

    def need(name, attr=None):
        try:
            m = importlib.import_module(name)
            v = getattr(m, attr) if attr else m
            imp[f"{name}{'.' + attr if attr else ''}"] = "ok"
            return v
        except Exception as e:  # noqa: BLE001
            imp[f"{name}{'.' + attr if attr else ''}"] = f"{type(e).__name__}: {e}"[:400]
            raise
    current_stream = need("vllm.utils.torch_utils", "current_stream")
    SK = need("vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm")
    LLG = need("vllm.models.qwen4_exp.nvidia.low_latency_gemm")
    platform = need("vllm.platforms", "current_platform")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    current_stream()
    cap = tuple(torch.cuda.get_device_capability(dev))
    plans = LLG.QWEN4_EXP_GEMM_PLANS_BY_CAPABILITY
    sm121 = LLG.QWEN4_EXP_SM121_GEMM_PLANS
    avail = SK.shape_dynamic_skinny_gemm.is_available()
    res["plans"] = {"capability": list(cap), "has_plan_for_capability": cap in plans,
                    "capabilities_with_plans": [list(k) for k in plans if k is not None],
                    "skinny_available": avail, "pdl": bool(platform.is_arch_support_pdl()),
                    "sm121_shapes": sorted(f"{n},{k}" for n, k in sm121),
                    "missing_for_flash_next_tp2": [f"{n},{k}" for n, k, *_ in C.gemm_shapes()
                                                   if (n, k) not in plans.get(cap, {})],
                    "sm121_has": {f"{n},{k}": sorted(sm121.get((n, k), {})) for n, k, *_ in C.gemm_shapes()}}
    res["env"] = {"gpu": torch.cuda.get_device_name(dev), "torch": torch.__version__,
                  "cutlass_dsl": imp.get("cutlass") or None, "cvd": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        import cutlass  # noqa: F401
        res["env"]["cutlass_version"] = getattr(cutlass, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        res["env"]["cutlass_version"] = f"import failed: {e}"[:200]
    res["header_shapes"] = header_shapes(a.ckpt) if a.ckpt else {}
    save()
    l2 = int(getattr(torch.cuda.get_device_properties(dev), "L2_cache_size", 0) or 48 * 2 ** 20)
    skinny = SK.shape_dynamic_skinny_gemm
    all_ok = True
    shapes = C.gemm_shapes()
    for N, K, label, n_t, _ in shapes:
        if left() < 60:
            res["errors"].append(f"budget stop before shape {N},{K}")
            res["budget_stop"] = True
            break
        key = f"{N},{K}"
        s = {"label": label, "N": N, "K": K, "calls_stage1_per_tick": n_t, "M": {}}
        wbytes = N * K * 2
        ncopies = max(1, min(64, math.ceil(max(3 * l2, a.rotate_mb * 2 ** 20) / wbytes)))
        g = torch.Generator(device=dev).manual_seed(N * 7 + K)
        W = [(torch.randn(N, K, generator=g, device=dev, dtype=torch.float32) * 0.02).to(torch.bfloat16) for _ in range(ncopies)]
        s["copies"] = ncopies
        for m in M_LIST:
            if left() < 40:
                s["budget_stop"] = True
                break
            r = {"configs": []}
            x = torch.randn(m, K, generator=g, device=dev, dtype=torch.float32).to(torch.bfloat16)
            ref = x.float() @ W[0].float().T
            try:
                out = torch.nn.functional.linear(x, W[0])
                cm = C.metrics(torch, out, ref)
                tm = C.bench(torch, [lambda w=w: torch.nn.functional.linear(x, w) for w in W], a.warmup, a.iters, True)
                t = C.timing(tm)                      # N3-12: graph and eager never substituted
                r["cublas"] = {"ref": cm, "ok": cm["finite"] and cm["rel_fro"] <= GATE_REL_FRO, "graph_us": t["graph_us"],
                               "eager_us": t["eager_us"], "graph_status": t["graph_status"], "graph_error": t["graph_error"]}
                all_ok &= r["cublas"]["ok"]
            except Exception as e:  # noqa: BLE001
                r["cublas"] = {"error": f"{type(e).__name__}: {e}"[:300], "ok": False}
                all_ok = False
            cands = []
            p = sm121.get((N, K), {}).get(m)
            if p is not None:
                cands.append(("sm121_plan", p))
            try:
                h = skinny._config(m, N, K)
                if p is None or h != p:
                    cands.append(("heuristic", h))
            except Exception as e:  # noqa: BLE001
                r["heuristic_error"] = f"{type(e).__name__}: {e}"[:200]
            if (N, K) in SWEEP_SHAPES and m in SWEEP_M:
                for bs, opb, vw in SWEEP_CFGS:
                    c = SK.SkinnyGemmConfig(m, bs, opb, k_unroll=2, vector_width=vw, static_k=K)
                    if cfg_valid(c, N, K) and all(c != q for _, q in cands):
                        cands.append(("sweep", c))
            for src, c in cands:
                if left() < 30:
                    break
                cr = {"source": src, "cfg": cfg_dict(c)}
                if not avail:
                    cr["error"] = "UNSUPPORTED_SM89: cuteDSL not importable (shape_dynamic_skinny_gemm.is_available() False)"
                elif not cfg_valid(c, N, K):
                    cr["error"] = "config not valid for this shape (divisibility / static_k)"
                elif not SK.row_stride_ok(x, c):
                    cr["error"] = "row_stride_ok False"
                else:
                    try:
                        t0 = time.time()
                        out = skinny(x, W[0], c)               # compiles on first use (cute.compile)
                        torch.cuda.synchronize()
                        cr["compile_sec"] = round(time.time() - t0, 2)
                        cr["ref"] = C.metrics(torch, out, ref)
                        cr["ok"] = cr["ref"]["finite"] and cr["ref"]["rel_fro"] <= GATE_REL_FRO
                        tm = C.bench(torch, [lambda w=w: skinny(x, w, c) for w in W], a.warmup, a.iters, True)
                        t = C.timing(tm)                  # N3-12: a failed capture never becomes a graph winner
                        cr.update(graph_us=t["graph_us"], eager_us=t["eager_us"], graph_status=t["graph_status"])
                        cr["us"] = (cr["graph_us"] or {}).get("median")
                        cr["spread"] = C.stats_spread(cr["graph_us"])
                        if t["graph_error"]:
                            cr["graph_error"] = t["graph_error"]
                    except Exception as e:  # noqa: BLE001
                        cr["error"] = f"{type(e).__name__}: {e}"[:600]
                        cr["tb"] = traceback.format_exc()[-1200:]
                        try:
                            torch.cuda.synchronize()
                        except Exception as e2:  # noqa: BLE001
                            cr["fatal"] = f"{type(e2).__name__}: {e2}"[:300]
                            r["configs"].append(cr)
                            s["M"][str(m)] = r
                            res["shapes"][key] = s
                            raise
                r["configs"].append(cr)
            ok = [c for c in r["configs"] if c.get("ok") and c.get("us")]
            if any(c.get("ok") is False for c in r["configs"]):
                all_ok = False
            cub = med((r.get("cublas") or {}).get("graph_us"))
            if ok:
                b = min(ok, key=lambda c: c["us"])
                r["best"] = {"source": b["source"], "cfg": b["cfg"], "us": b["us"], "spread": b.get("spread"), "ok": True,
                             "speedup_vs_cublas": round(cub / b["us"], 3) if cub else None}
                # the real dispatch path with an (8, 9) plan entry patched in = the NB4 data patch, simulated
                bc = next(q for s_, q in cands if cfg_dict(q) == b["cfg"])
                saved = plans.get(cap)
                try:
                    plans[cap] = {(N, K): {m: bc}}
                    d = LLG._qwen4_exp_low_latency_gemm(x, W[0])
                    torch.cuda.synchronize()
                    direct = skinny(x, W[0], bc)
                    torch.cuda.synchronize()
                    r["dispatch"] = {"ok": bool(torch.equal(d, direct)), "equal_skinny": bool(torch.equal(d, direct)),
                                     "equal_cublas": bool(torch.equal(d, torch.nn.functional.linear(x, W[0])))}
                except Exception as e:  # noqa: BLE001
                    r["dispatch"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
                finally:
                    if saved is None:
                        plans.pop(cap, None)
                    else:
                        plans[cap] = saved
            else:
                r["best"] = None
            s["M"][str(m)] = r
            res["shapes"][key] = s
            save()
        del W
        torch.cuda.empty_cache()
        res["shapes"][key] = s
        save()
    res["checks"]["all_numeric_ok"] = bool(all_ok)
    res["checks"]["plan_table_restored"] = (cap in plans) == res["plans"]["has_plan_for_capability"]
    # N3-19: no skinny config could run on sm_89 at all (cuteDSL missing / every compile failed) = UNSUPPORTED_SM89, a
    # diagnostic result (cuBLAS numbers kept), never an abort: the other arms run on
    cfgs = [c for s_ in res["shapes"].values() for r_ in (s_.get("M") or {}).values() for c in r_.get("configs") or []]
    ran = [c for c in cfgs if c.get("ok") is not None]
    res["sm89_skinny"] = {"configs": len(cfgs), "ran": len(ran), "compile_or_run_errors": sum(1 for c in cfgs if c.get("error")),
                          "status": "UNSUPPORTED_SM89" if not avail or (cfgs and not ran) else "RAN",
                          "why": None if avail and ran else ("shape_dynamic_skinny_gemm.is_available() False" if not avail
                                                              else "no skinny config compiled / ran on sm_89")}
    if left() > 120:
        res["draft_heads"] = draft_heads(torch, a, dev, left)
        if res["draft_heads"].get("status") != "UNTESTED":     # missing Marlin linear ops = UNTESTED, not a D4 failure
            res["checks"]["draft_head_sanity"] = all(v.get("sanity_ok") is not False for v in
                                                     (res["draft_heads"].get("modes") or {}).values()) and not res["draft_heads"].get("error")
    else:
        res["errors"].append("draft-head section skipped: budget")
    return all_ok


DRAFT_SANITY = {"int8": 0.05, "int4": 0.25}     # rel_fro vs the BF16 head: quantization error sanity, not a pass gate


def draft_head_preflight(torch):
    """CPU kit v1: _C.gptq_marlin_repack was not registered in the CPU import. On the GPU box, check every op / helper
    the INT8 / INT4 draft-head path needs (_C.marlin_gemm under apply_gptq_marlin_linear, marlin_quantize,
    QuantizedDraftLMHead; _C.gptq_marlin_repack recorded, not needed by this path); a missing one makes the draft heads
    UNTESTED, never a D4 failure."""
    out = {"ops": {}, "helpers": {}, "missing": []}
    try:
        importlib.import_module("vllm._custom_ops")             # loads vllm._C
    except Exception as e:  # noqa: BLE001
        out["missing"].append(f"vllm._custom_ops: {type(e).__name__}: {e}"[:200])
    # Merlin source: apply_gptq_marlin_linear -> torch.ops.vllm.marlin_gemm -> torch.ops._C.marlin_gemm (REQUIRED);
    # marlin_quantize (marlin_utils_test) permutes in Python and never calls _C.gptq_marlin_repack (recorded only)
    for op, required in (("marlin_gemm", True), ("gptq_marlin_repack", False)):
        try:
            ok = hasattr(torch.ops, "_C") and hasattr(torch.ops._C, op)
        except Exception:  # noqa: BLE001
            ok = False
        out["ops"][f"_C.{op}"] = ok
        if required and not ok:
            out["missing"].append(f"torch.ops._C.{op}")
    for mod, attr in (("vllm.model_executor.layers.quantization.utils.marlin_utils", "apply_gptq_marlin_linear"),
                      ("vllm.model_executor.layers.quantization.utils.marlin_utils_test", "marlin_quantize"),
                      ("vllm.model_executor.layers.draft_lm_head_quant", "QuantizedDraftLMHead")):
        try:
            getattr(importlib.import_module(mod), attr)
            out["helpers"][f"{mod}.{attr}"] = True
        except Exception as e:  # noqa: BLE001
            out["helpers"][f"{mod}.{attr}"] = f"{type(e).__name__}: {e}"[:200]
            out["missing"].append(f"{mod}.{attr}")
    out["ok"] = not out["missing"]
    return out


def draft_heads(torch, a, dev, left):
    """Family 2, vocab heads: the MTP draft head at DRAFT_HEAD bf16 (= the BF16 TP2 shard through F.linear, what
    Merlin keeps by default) vs int8 / int4 = vllm/model_executor/layers/draft_lm_head_quant.py QuantizedDraftLMHead
    (CPU marlin_quantize of the shard, then apply_gptq_marlin_linear per call), on the (124160, 2560) shard at the
    draft row counts r = ceil(seqs / 2). The acceptance effect belongs to NB2 (D2); here only kernel time + error."""
    out = {"shape": [124160, 2560], "M": [1, 2, 4, 8, 16], "modes": {}}
    try:
        import types
        DQ = importlib.import_module("vllm.model_executor.layers.draft_lm_head_quant")
        N, K = out["shape"]
        g = torch.Generator(device=dev).manual_seed(11)
        Wb = (torch.randn(N, K, generator=g, device=dev, dtype=torch.float32) * 0.02).to(torch.bfloat16)
        xs = {m: torch.randn(m, K, generator=g, device=dev, dtype=torch.float32).to(torch.bfloat16) for m in out["M"]}
        refs = {m: xs[m].float() @ Wb.float().T for m in out["M"]}
        bf = {}
        for m in out["M"]:
            t = C.timing(C.bench(torch, [lambda m=m: torch.nn.functional.linear(xs[m], Wb)], a.warmup, a.iters, True))
            bf[str(m)] = {"graph_us": t["graph_us"], "eager_us": t["eager_us"], "graph_status": t["graph_status"]}
        out["modes"]["bf16"] = {"per_M": bf, "sanity_ok": True, "path": "torch.nn.functional.linear (cuBLAS)"}
        out["preflight"] = pf = draft_head_preflight(torch)
        if not pf["ok"]:
            out["status"] = "UNTESTED"
            out["why"] = f"Marlin W8/W4A16 linear path not available on this GPU box: {pf['missing']}"
            for mode in ("int8", "int4"):
                out["modes"][mode] = {"skipped": "UNTESTED: " + out["why"]}
            return out
        out["status"] = "DONE"
        for mode in ("int8", "int4"):
            if left() < 90:
                out["modes"][mode] = {"skipped": "budget"}
                continue
            t0 = time.time()
            fake_head = types.SimpleNamespace(weight=Wb, tp_size=2, shard_indices=None)
            h = DQ.QuantizedDraftLMHead(fake_head, mode)
            build = round(time.time() - t0, 1)
            per = {}
            worst = 0.0
            for m in out["M"]:
                y = h.quant_method.apply(h, xs[m])
                torch.cuda.synchronize()
                met = C.metrics(torch, y, refs[m])
                worst = max(worst, met["rel_fro"] if met["finite"] else 9.9)
                t = C.timing(C.bench(torch, [lambda m=m: h.quant_method.apply(h, xs[m])], a.warmup, a.iters, True))
                per[str(m)] = {"graph_us": t["graph_us"], "eager_us": t["eager_us"], "graph_status": t["graph_status"],
                               "ref_vs_bf16": met}
            out["modes"][mode] = {"per_M": per, "build_sec": build, "worst_rel_fro_vs_bf16": round(worst, 5),
                                  "sanity_ok": worst <= DRAFT_SANITY[mode],
                                  "path": "QuantizedDraftLMHead -> apply_gptq_marlin_linear (Marlin W%sA16, g128)" % mode[-1]}
            del h
            torch.cuda.empty_cache()
        del Wb
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:500]
        out["tb"] = traceback.format_exc()[-1500:]
    return out


def med(st):
    return (st or {}).get("median") if isinstance(st, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rotate-mb", type=int, default=144, help="min weight working set per kernel (L2 is 48 MB)")
    ap.add_argument("--budget-s", type=float, default=1500.0)
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    res = {"meta": {"tool": "nb3_gemm_bench.py", "fake": a.fake, "args": vars(a), "gate_rel_fro": GATE_REL_FRO,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "imports": {}, "shapes": {}, "checks": {}, "errors": []}
    save = lambda: C.write_json(a.out, res)  # noqa: E731
    t0 = time.time()
    code = 0
    try:
        if a.fake:
            fk = C.fake_gemm(faults=C.fake_faults())
            res.update({k: v for k, v in fk.items() if k in ("plans", "shapes", "checks", "draft_heads", "sm89_skinny")})
        else:
            real(a, res, save)
        res["status"] = ("PARTIAL_BUDGET" if res.get("budget_stop") else
                         "UNSUPPORTED_SM89" if (res.get("sm89_skinny") or {}).get("status") == "UNSUPPORTED_SM89" else "DONE")
        if not all(v is True for v in res["checks"].values()):
            code = 1
    except Exception as e:  # noqa: BLE001
        res["status"] = "ERROR"
        res["errors"].append(f"{type(e).__name__}: {e}"[:800])
        res["tb"] = traceback.format_exc()[-3000:]
        code = 2
    res["sec"] = round(time.time() - t0, 1)
    res["exit_code"] = code
    save()
    print("NB3_GEMM", res["status"], "exit", code, "| checks", json.dumps(res["checks"]), "| sec", res["sec"], flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
