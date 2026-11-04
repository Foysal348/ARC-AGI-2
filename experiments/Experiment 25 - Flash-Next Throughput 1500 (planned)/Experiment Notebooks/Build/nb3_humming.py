"""E25 NB3 D5 structural-track arm (C6-humming, report STRUCTURAL_TRACK_FEASIBILITY_2026-10-31.md shortlist #1): Merlin
v3.0.0 routes 2/3-bit (and 5-7-bit) AutoRound / compressed-tensors MoE to the humming kernels
(merlin-src inc_wna16_scheme.py:28 CUDA_HUMMING_SUPPORTED_BITS {2,3,5,6,7}; fused_moe/experts/fused_humming_moe.py;
humming-kernels[cu13]==0.1.17 pinned in requirements/cuda.txt:31). This module rebuilds Merlin's INDEXED humming MoE
path (HummingIndexedExperts.apply: moe_align_block_size with the EP expert_map -> humming grouped GEMM w13 (top_k) ->
SiLU-and-mul -> humming GEMM w2 (top_k 1) -> masked router-weighted sum) on the SAME layer-0 local experts as the Marlin
bench, without a model server:
  w4g128_ct  the checkpoint's own compressed-tensors INT4 tensors (weight_packed / weight_scale) through humming's
             compressed-tensors schema = the SAME weights as Marlin W4 (kernel-vs-kernel control);
  w3g128, w2g128, w3g64
             SYNTHETIC REQUANT: round-to-nearest requantization (humming quantize_weight, symmetric, bf16 group scales)
             of our fp32 dequantisation of those INT4 experts. Accuracy NOT representative: speed and bytes only.
Correctness of every arm is checked against an fp32 reference of ITS OWN quantized weights (dequantize_weight of the
packed codes), never against W4. Time per call: graph and eager kept apart (N3-12); bytes DERIVED (distinct local
experts x expert bytes at that bit width); humming JIT compile time (transform + first call) and CUDA-graph capture
recorded. Ratios vs Marlin W4 at the same T, same routing and the same number of rotated copies; structural gate
W3 <= 0.8x and W2 <= 0.6x Marlin W4 (the capacity reading: up to ~1.3x slower can still be net positive) - both recorded.
Research only: never a NB4 candidate. `import humming` failing = UNAVAILABLE (recorded, the bench continues).
"""
import copy
import json
import math
import time
import traceback

ARMS = (("w4g128_ct", 4, 128, "ct"), ("w3g128", 3, 128, "rtn"), ("w2g128", 2, 128, "rtn"), ("w3g64", 3, 64, "rtn"))
T_LIST = (16, 8, 32, 64)                 # the RC2 point first
GATE = {"w3_max_ratio": 0.8, "w2_max_ratio": 0.6, "capacity_net_positive_max_ratio": 1.3}
LABEL = ("C6-humming: research arm, SYNTHETIC REQUANT for W3 / W2 (round-to-nearest from the INT4 checkpoint), accuracy "
         "not representative; speed and bytes only; never ranked into NB4")


def availability():
    """humming + Merlin's humming glue importable in this venv? -> (ok, info)."""
    info = {}
    try:
        import humming
        info["humming_version"] = getattr(humming, "__version__", None)
        from humming.layer import HummingLayer  # noqa: F401
        from humming.utils.weight import quantize_weight, dequantize_weight  # noqa: F401
        from humming.tune import get_heuristics_config  # noqa: F401
        from humming.config import GemmType  # noqa: F401
        from humming import dtypes  # noqa: F401
        try:
            from vllm.utils.import_utils import has_humming
            info["vllm_has_humming"] = bool(has_humming())
        except Exception as e:  # noqa: BLE001
            info["vllm_has_humming"] = f"{type(e).__name__}: {e}"[:200]
        return True, info
    except Exception as e:  # noqa: BLE001
        info["error"] = f"{type(e).__name__}: {e}"[:400]
        return False, info


def gate_reading(arm, ratio):
    """Both readings of the structural gate for one arm's ratio vs Marlin W4."""
    if ratio is None:
        return {"strict": "UNMEASURED", "capacity": "UNMEASURED"}
    lim = GATE["w3_max_ratio"] if arm.startswith("w3") else GATE["w2_max_ratio"] if arm.startswith("w2") else None
    return {"strict": ("CONTROL" if lim is None else "PASS" if ratio <= lim else "FAIL"),
            "strict_limit": lim,
            "capacity": "NET_POSITIVE_POSSIBLE" if ratio <= GATE["capacity_net_positive_max_ratio"] else "TOO_SLOW",
            "capacity_limit": GATE["capacity_net_positive_max_ratio"]}


class Arm:
    """One humming MoE arm: the two HummingLayers (w13 = [gate; up], w2), its reference dequantisation, the forward."""

    def __init__(self, torch, C, spec, src, E, E_loc, topk, H, I, dev, emap, emap_cpu, moe_align, copies=2):
        from humming import dtypes
        from humming.config import GemmType
        from humming.layer import HummingLayer
        from humming.tune import get_heuristics_config
        from humming.utils.weight import quantize_weight
        self.torch, self.C, self.name, self.bits, self.group, self.kind = torch, C, spec[0], spec[1], spec[2], spec[3]
        self.E, self.E_loc, self.topk, self.H, self.I, self.dev, self.emap, self.emap_cpu, self.moe_align = (
            E, E_loc, topk, H, I, dev, emap, emap_cpu, moe_align)
        self.dtype = dtypes.DataType.from_str(f"uint{self.bits}")
        self.rec = {"arm": self.name, "bits": self.bits, "group": self.group, "weights": (
            "checkpoint compressed-tensors INT4 (same as Marlin W4)" if self.kind == "ct" else
            "SYNTHETIC REQUANT (RTN of the dequantised INT4 experts; accuracy not representative)")}
        t0 = time.time()
        layers = {}
        self.ref_w = {}                   # sublayer -> (packed codes, scale) on GPU, for the fp32 reference
        for sub, (N, K) in (("w13", (2 * I, H)), ("w2", (H, I))):
            if self.kind == "ct":
                cfg = {"quant_method": "compressed-tensors", "format": "pack-quantized", "type": "int", "num_bits": 4,
                       "strategy": "group", "group_size": self.group, "symmetric": True}
                lay = HummingLayer(shape_n=N, shape_k=K, weight_config=cfg, num_experts=E_loc, torch_dtype=torch.bfloat16)
                pk, sc = src["ct"][sub]
                ten = {"weight_packed": pk.contiguous().cpu(), "weight_scale": sc.to(torch.bfloat16).contiguous().cpu()}
                ws = getattr(lay, "weight_shape", None)
                if ws is not None and ws.shape[-1] == 2:
                    ten["weight_shape"] = torch.tensor([N, K], dtype=ws.dtype).expand(ws.shape).clone()
            else:
                cfg = {"quant_method": "humming", "dtype": f"uint{self.bits}", "group_size": self.group}
                lay = HummingLayer(shape_n=N, shape_k=K, weight_config=cfg, num_experts=E_loc, torch_dtype=torch.bfloat16)
                packs, scales = [], []
                for lo in range(0, E_loc, 16):                       # fp32 chunks of 16 experts (bounded GPU memory)
                    w = src["fp32"](sub, lo, min(E_loc, lo + 16))
                    q, s, _, _ = quantize_weight(w, dtype=self.dtype, scale_dtype=dtypes.bfloat16, group_size=self.group,
                                                 has_zero_point=False, pack=True, allow_negative_scale=True)
                    packs.append(q)
                    scales.append(s)
                    del w
                pk, sc = torch.cat(packs, 0), torch.cat(scales, 0)
                self.ref_w[sub] = (pk, sc)
                ten = {"weight": pk.cpu(), "weight_scale": sc.cpu()}
            lay.load_from_tensors(ten)
            lay = lay.to(dev)
            lay.transform()
            layers[sub] = lay
        torch.cuda.synchronize()
        self.rec["transform_sec"] = round(time.time() - t0, 2)
        self.copies = [layers] + [{k: copy.deepcopy(v) for k, v in layers.items()} for _ in range(max(0, copies - 1))]
        self.rec["copies"] = len(self.copies)
        self.rec["gpu_mem_alloc_gib_after_build"] = round(torch.cuda.memory_allocated(dev) / 2 ** 30, 2)
        self.compute = json.dumps({"use_batch_invariant": False, "use_f16_accum": False, "gemm_type": "indexed"})
        self.tuning = {sub: get_heuristics_config(layer_config=layers[sub].humming_config, use_f16_accum=False,
                                                  use_batch_invariant=False, gemm_type=GemmType.INDEXED, device=dev)
                       for sub in layers}
        self.tuning_str = {k: json.dumps(v) for k, v in self.tuning.items()}

    def _block(self, sub, vm):
        for lo, hi, cfg in self.tuning[sub]:
            if lo < vm <= hi:
                return int(cfg["block_shape"][0])
        return 64

    def expert_fp32(self, e):
        """fp32 (gate [I, H], up [I, H], down [H, I]) of THIS arm's quantized weights for global expert e."""
        torch, I = self.torch, self.I
        li = int(self.emap_cpu[e]) if self.emap_cpu is not None else e
        if self.kind == "ct":
            return self.src_ref(e)
        from humming.utils.weight import dequantize_weight
        out = {}
        for sub in ("w13", "w2"):
            pk, sc = self.ref_w[sub]
            out[sub] = dequantize_weight(pk[li:li + 1].contiguous(), sc[li:li + 1].contiguous(), None, None, self.dtype, packed=True)[0]
        return out["w13"][:I], out["w13"][I:], out["w2"]

    def make(self, cp, x, tid, tw):
        """The forward of HummingIndexedExperts.apply for one rotated copy (buffers allocated once per input)."""
        torch = self.torch
        T, topk, H, I = x.shape[0], self.topk, self.H, self.I
        vm = math.ceil(T * topk * self.E_loc / self.E)
        b13, b2 = self._block("w13", vm), self._block("w2", vm)
        local = (self.emap[tid.long()] >= 0) if self.emap is not None else torch.ones_like(tid, dtype=torch.bool)
        g13 = torch.zeros(T * topk, 2 * I, dtype=torch.bfloat16, device=x.device)
        g2 = torch.zeros(T * topk, H, dtype=torch.bfloat16, device=x.device)
        w = (tw * local).unsqueeze(-1)
        lay = self.copies[cp]

        def fn():
            # the token alignment is part of the timed forward: fused_marlin_moe (the control) aligns inside its own call
            al13 = self.moe_align(tid, b13, self.E, self.emap, ignore_invalid_experts=True)[:3]
            al2 = al13 if b2 == b13 else self.moe_align(tid, b2, self.E, self.emap, ignore_invalid_experts=True)[:3]
            lay["w13"](x, outputs=g13, sorted_ids=al13[0], expert_ids=al13[1], num_tokens_padded=al13[2], top_k=topk,
                       valid_shape_m=vm, compute_config=self.compute, tuning_config=self.tuning_str["w13"])
            act = (torch.nn.functional.silu(g13[:, :I].float()) * g13[:, I:].float()).to(torch.bfloat16)
            lay["w2"](act, outputs=g2, sorted_ids=al2[0], expert_ids=al2[1], num_tokens_padded=al2[2], top_k=1,
                      valid_shape_m=vm, compute_config=self.compute, tuning_config=self.tuning_str["w2"])
            d = torch.where(local.unsqueeze(-1), g2.view(T, topk, H).float(), torch.zeros((), device=x.device))
            return (d * w).sum(1).to(torch.bfloat16)
        return fn, {"block_w13": b13, "block_w2": b2, "valid_shape_m": vm, "alignment_in_timed_forward": True,
                    "path": "HUMMING_INDEXED_EP_RESEARCH (Merlin's EP default may be grouped_contiguous)"}


def run(torch, C, ctx, left, T_list=T_LIST, arms=ARMS, ref_ok=None, warmup=10, iters=60):
    """All humming arms, budget-bounded. ctx: E, E_loc, topk, H, I, G, dev, emap, emap_cpu, lo, hi, moe_align,
    inputs(T, mode), marlin_mk(copy_index, x, tid, tw), marlin_copies, ct {w13: (packed, scale), w2: ...} on GPU,
    expert_fp32_w4(e) (the Marlin reference experts), fp32(sub, e0, e1) -> fp32 [n, N, K] dequantised INT4 experts."""
    ok, info = availability()
    out = {"label": LABEL, "gate": GATE, "available": ok, "info": info, "arms": {}, "T": list(T_list)}
    if not ok:
        out["status"] = "UNAVAILABLE"
        return out
    E, E_loc, topk, H, I, dev = ctx["E"], ctx["E_loc"], ctx["topk"], ctx["H"], ctx["I"], ctx["dev"]
    src = {"ct": ctx["ct"], "fp32": ctx["fp32"]}
    nmc = min(2, len(ctx["marlin_copies"]))
    for spec in arms:
        if left() < 90:
            out.setdefault("budget_stop_before", []).append(spec[0])
            out["budget_stop"] = True
            continue
        rec = {"arm": spec[0], "per_T": {}}
        try:
            arm = Arm(torch, C, spec, src, E, E_loc, topk, H, I, dev, ctx["emap"], ctx["emap_cpu"], ctx["moe_align"], copies=nmc)
            arm.src_ref = ctx["expert_fp32_w4"]
            rec.update(arm.rec)
            for T in T_list:
                if left() < 45:
                    rec["budget_stop_at_T"] = T
                    out["budget_stop"] = True
                    break
                x, tid, tw, ids, _ = ctx["inputs"](T, "independent")
                dl = C.distinct(ids, ctx["lo"], ctx["hi"])
                r = {"T": T, "distinct_local": dl, "derived_bytes": dl * C.expert_bytes(H, I, spec[1], spec[2]),
                     "derived_bytes_marlin_w4": dl * C.expert_bytes(H, I, 4, ctx["G"])}
                t1 = time.time()
                fn, meta = arm.make(0, x, tid, tw)
                y = fn()
                torch.cuda.synchronize()
                r["first_call_sec"] = round(time.time() - t1, 2)      # includes the humming JIT compile for this M
                r.update(meta)
                ref = C.moe_reference(x, tid, tw, arm.expert_fp32, ctx["lo"], ctx["hi"])
                r["ref_own_weights"] = C.metrics_x(torch, y, ref)
                r["ref_ok"] = ref_ok(r["ref_own_weights"]) if ref_ok else None
                ym = ctx["marlin_mk"](0, x, tid, tw)()
                torch.cuda.synchronize()
                if spec[3] == "ct":
                    r["vs_marlin_w4_same_weights"] = C.equivalence(y, ym)     # kernel-vs-kernel control (recorded)
                th = C.timing(C.bench(torch, [arm.make(c, x, tid, tw)[0] for c in range(len(arm.copies))], warmup, iters, True))
                tm = C.timing(C.bench(torch, [ctx["marlin_mk"](c, x, tid, tw) for c in range(nmc)], warmup, iters, True))
                r.update(graph_us=th["graph_us"], eager_us=th["eager_us"], graph_status=th["graph_status"],
                         graph_error=th["graph_error"], marlin_graph_us=tm["graph_us"], marlin_eager_us=tm["eager_us"],
                         marlin_graph_status=tm["graph_status"], copies_both=nmc)
                hg, mg, he, me = (C.med(r["graph_us"]), C.med(r["marlin_graph_us"]), C.med(r["eager_us"]), C.med(r["marlin_eager_us"]))
                r["ratio_vs_marlin_w4_graph"] = round(hg / mg, 4) if hg and mg else None
                r["ratio_vs_marlin_w4_eager"] = round(he / me, 4) if he and me else None
                r["derived_gbps"] = round(r["derived_bytes"] / hg / 1e3, 1) if hg else None
                r["gate"] = gate_reading(spec[0], r["ratio_vs_marlin_w4_graph"])
                rec["per_T"][str(T)] = r
            del arm
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            rec.update(error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
            torch.cuda.synchronize()          # a sticky device fault re-raises here (the notebook policy stops on it)
            arm = None
            torch.cuda.empty_cache()
        rec["ref_ok_all_T"] = (all(v.get("ref_ok") is True for v in rec["per_T"].values()) if rec["per_T"] else None)
        out["arms"][spec[0]] = rec
    out["status"] = "PARTIAL_BUDGET" if out.get("budget_stop") else "DONE"
    out["summary"] = summary(out)
    return out


def summary(h):
    """Per arm at the RC2 point (T16) and over T: graph ratio vs Marlin W4, both gate readings, own-weight reference."""
    s = {}
    for name, rec in (h.get("arms") or {}).items():
        pt = rec.get("per_T") or {}
        r16 = pt.get("16") or {}
        ratios = {T: v.get("ratio_vs_marlin_w4_graph") for T, v in pt.items()}
        s[name] = {"ratio_T16_graph": r16.get("ratio_vs_marlin_w4_graph"), "ratio_T16_eager": r16.get("ratio_vs_marlin_w4_eager"),
                   "gate_T16": r16.get("gate"), "ratios_graph_by_T": ratios, "ref_ok_all_T": rec.get("ref_ok_all_T"),
                   "graph_capture": sorted({v.get("graph_status") for v in pt.values() if v.get("graph_status")}),
                   "transform_sec": rec.get("transform_sec"), "first_call_sec_by_T": {T: v.get("first_call_sec") for T, v in pt.items()},
                   "error": rec.get("error")}
    return s


def fake(seed=0, unavailable=False):
    """Fake-run stand-in (same schema), numbers made up for the local / CPU-kit harness."""
    if unavailable:
        return {"label": LABEL, "gate": GATE, "available": False, "info": {"error": "ModuleNotFoundError: No module named 'humming' (fake)"},
                "arms": {}, "T": list(T_LIST), "status": "UNAVAILABLE"}
    out = {"label": LABEL, "gate": GATE, "available": True, "info": {"humming_version": "0.1.17 (fake)"}, "arms": {}, "T": list(T_LIST),
           "fake": True}
    for name, bits, g, kind in ARMS:
        f = {4: 1.05, 3: 0.82, 2: 0.62}[bits] + (0.03 if g == 64 else 0.0)
        per = {}
        for T in T_LIST:
            mg = 300.0 + 6.0 * T
            per[str(T)] = {"T": T, "graph_us": {"median": round(mg * f, 2)}, "eager_us": {"median": round(mg * f * 1.4, 2)},
                           "graph_status": "PASS", "marlin_graph_us": {"median": mg}, "marlin_eager_us": {"median": mg * 1.4},
                           "ratio_vs_marlin_w4_graph": round(f, 4), "ratio_vs_marlin_w4_eager": round(f, 4), "ref_ok": True,
                           "first_call_sec": 2.0, "gate": gate_reading(name, round(f, 4))}
        out["arms"][name] = {"arm": name, "bits": bits, "group": g, "per_T": per, "ref_ok_all_T": True, "transform_sec": 3.0}
    out["status"] = "DONE"
    out["summary"] = summary(out)
    return out
