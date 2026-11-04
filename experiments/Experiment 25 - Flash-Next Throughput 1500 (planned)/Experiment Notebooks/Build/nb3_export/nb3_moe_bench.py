#!/usr/bin/env python3
"""E25 NB3 D5: Merlin's Marlin MoE (W4A16) on one L4 with REAL Flash-Next routed experts, at the decode token counts.

Runs in the Merlin venv (vLLM 0.32.0 fork, torch 2.13) with the RC2 FP8 launch env, CUDA_VISIBLE_DEVICES = one GPU.
What it does, all through Merlin's own installed code (paths verified in _scratch/e23/p2/merlin-src):
  preflight (N3-24) inspect.signature of every private Merlin entry point called positionally (_unpack_and_dequant_int4_gptq,
         _process_weights_marlin) and the keywords of fused_marlin_moe, plus a tiny known INT4 parity (CPU) of Merlin's
         dequant against the compressed-tensors math; any mismatch stops the bench before the checkpoint is read;
  load   one MoE layer's experts of the merged checkpoint (compressed-tensors pack-quantized INT4, group 128,
         symmetric; names model.language_model.layers.{L}.mlp.experts.{e}.{gate,up,down}_proj.weight_{packed,scale,shape})
         for the local experts of EP rank `--ep-rank` of 2 (expert_map_manager.determine_expert_map, linear placement:
         TP2 + --enable-expert-parallel at dp 1 = EP2 inside each PP stage);
  gate   (N3-15) the independent dequantisation of the first local expert's gate / up / down equals Merlin's own
         GPTQ-layout dequant, else INT4_LAYOUT_OR_SCALE_MISMATCH and the bench stops before any GPU sweep;
  prep   oracle/int_wna16._process_weights_marlin (N-first -> K-first, gptq_marlin_moe_repack, marlin_moe_permute_scales)
         with w13 = [gate; up]; its 6 optional outputs (zeros, input global scales, biases) must all be None here;
  first  (N3-7) T=1 and T=4 outputs vs the fp32 reference before anything else: a failure skips every sweep;
  run    experts/marlin_moe.fused_marlin_moe(..., quant_type_id = uint4b8, global_num_experts = 512, expert_map);
  check  vs an fp32 reference from the compressed-tensors tensors dequantised independently (nb3_common): finite,
         rel_fro, cos, max_abs / ref max, p99 abs (N3-3); split vs combined and tuned vs baseline by the elementwise
         BF16 ULP check (nb3_common.equivalence); an adversarial routing case (one local expert per token from a set of
         2, repeated rows) at T16 / T4;
  time   CUDA events + CUDA-graph replay over rotated weight copies (P4 bench()); eager_us and graph_us are kept
         apart, a failed capture is graph_status INCONCLUSIVE_CAPTURE_FAILED and never a graph number (N3-12);
  tune   MERLIN_MARLIN_MOE_CFG rules (vllm/merlin/marlin_moe_cfg.py), each checked vs the reference AND the baseline.
Work order (N3-16 / N3-19): nb3_common.d5_units(phase) in priority order (D5_PRIORITY); --phase core = T16 / T4 / T8
(the notebook runs it before D7 / D4), --phase rest = everything else with the EP rank 1 spot check (N3-8) and the ncu
DRAM bytes LAST (one ncu child per arm, N3-1 / N3-20: split and combined never share a profile; permission denied,
nonzero exit, no rows or a bad metric = DRAM_BYTES_UNVERIFIED). A unit that does not fit the budget ends the bench with
PARTIAL_BUDGET; every row measured before it is kept (N3-22: res["budget_stop"] is set by every stop).
Routing (ALL SYNTHETIC, no server): independent / zipf / hot / request (nb3_common.ROUTING_NOTE); NB1's real ids replayed
when attached. Structural track (C6-humming, nb3_humming): right after the T16 Marlin block, humming W4 (same weights) /
W3 / W2 (synthetic requant) MoE arms vs Marlin W4, research only (never a check of this file, never ranked).
Exit: 0 every check of this file True (status DONE or PARTIAL_BUDGET), 1 a check failed, 2 error (status ERROR).
"""
import argparse
import importlib
import inspect
import json
import math
import os
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb3_common as C  # noqa: E402
import nb3_humming as HM  # noqa: E402

T_SWEEP = C.D5_T_SWEEP
REF_GATE = {"rel_fro": 2e-2, "cos": 0.999, "max_abs_over_ref_max": 5e-2}   # 2 GEMMs with a bf16 intermediate vs fp32
MODES = C.D5_MODES
FAKE_UNIT_SEC = 1.0                  # fake clock: every unit costs 1 s of --budget-s (priority / budget tests on CPU)
PREFLIGHT_POSITIONAL = {             # called positionally below: the leading parameter names must match exactly
    "_unpack_and_dequant_int4_gptq": ("w_int32", "scale", "qzeros", "transpose_output", "output_dtype"),
    "_process_weights_marlin": ("layer", "input_dtype", "num_bits", "pack_factor", "group_size", "w13_qweight", "w2_qweight",
                                "w13_scales", "w2_scales"),
}
FUSED_KW = ("hidden_states", "w1", "w2", "bias1", "bias2", "w1_scale", "w2_scale", "topk_weights", "topk_ids", "quant_type_id",
            "global_num_experts", "expert_map")


def ref_ok(m):
    return (bool(m) and m.get("finite") is True and m.get("rel_fro", 1.0) <= REF_GATE["rel_fro"] and m.get("cos", 0.0) >= REF_GATE["cos"]
            and m.get("max_abs_over_ref_max", 1.0) <= REF_GATE["max_abs_over_ref_max"])


def heuristic_bm(M, topk, E_loc):
    """Replica of fused_marlin_moe's block-size loop (marlin_moe.py:345-350), for the align-only timing (DERIVED)."""
    for bm in (8, 16, 32, 48, 64):
        if M * topk / E_loc / bm < 0.9:
            break
    return bm


def sweep_rules(MC):
    rules = [f"bm={bm},ack=nonexact" for bm in (16, 32, 48, 64)]
    for bm in (8, 16, 32, 64):
        for tk, tn in MC.compiled_thread_configs(bm):
            for bps in MC.BLOCKS_PER_SM:
                rules.append(f"bm={bm},g={tk}x{tn}x{bps},ack=nonexact")
    return rules


def gptq_dequant_ref(torch, w_int32, scale):
    """A CPU re-implementation of Merlin's _unpack_and_dequant_int4_gptq (symmetric, transpose_output) for the fake
    path and the tests: [E, K/8, N] int32 + [E, K/g, N] scale -> [E, N, K] float32."""
    E, Kp, N = w_int32.shape
    shifts = torch.arange(8, dtype=torch.int32) * 4
    nib = (w_int32.unsqueeze(-1) >> shifts) & 0xF
    w = nib.permute(0, 1, 3, 2).reshape(E, Kp * 8, N).to(torch.int16) - 8
    gs = (Kp * 8) // scale.shape[1]
    return (w.to(torch.float32) * scale.repeat_interleave(gs, dim=1).to(torch.float32)).permute(0, 2, 1).contiguous()


def dequant_gate(torch, mine, theirs, what):
    """N3-15: the layout / scale understanding must be EXACT, else the bench stops here (no sweep on a wrong layout)."""
    eq = bool(mine.shape == theirs.shape and torch.equal(mine, theirs))
    rec = {"what": what, "equal": eq, "max_abs": float((mine - theirs).abs().max()) if mine.shape == theirs.shape else None}
    if not eq:
        raise RuntimeError(f"INT4_LAYOUT_OR_SCALE_MISMATCH: {what}: our compressed-tensors dequant != Merlin's GPTQ dequant "
                           f"(max abs {rec['max_abs']})")
    return rec


def preflight(torch, fns):
    """N3-24: signatures of the private symbols we call + a tiny known INT4 parity, before any checkpoint read."""
    out = {"signatures": {}, "problems": []}
    for name, f in fns.items():
        try:
            sig = inspect.signature(f)
        except Exception as e:  # noqa: BLE001
            out["problems"].append(f"{name}: no signature ({type(e).__name__}: {e})"[:200])
            continue
        params = list(sig.parameters)
        out["signatures"][name] = str(sig)[:700]
        want = PREFLIGHT_POSITIONAL.get(name)
        if want and tuple(params[:len(want)]) != want:
            out["problems"].append(f"{name}: leading params {params[:len(want)]} != {list(want)}")
        if name == "fused_marlin_moe":
            miss = [k for k in FUSED_KW if k not in params]
            if miss:
                out["problems"].append(f"fused_marlin_moe: keywords missing {miss}")
    try:
        g = torch.Generator().manual_seed(17)
        q = torch.randint(-8, 8, (16, 256), generator=g)
        sc = (torch.rand(16, 2, generator=g) + 0.5).to(torch.bfloat16)
        pk = C.ct_pack_int4(q)
        exact = q.float() * sc.float().repeat_interleave(128, 1)
        mine = C.ct_dequant_int4(pk, sc, 128)
        theirs = fns["_unpack_and_dequant_int4_gptq"](pk.T.contiguous().unsqueeze(0), sc.T.contiguous().unsqueeze(0), None,
                                                      True, torch.float32)[0]
        out["int4_parity"] = {"ours_exact": bool(torch.equal(mine, exact)), "merlin_exact": bool(torch.equal(theirs, exact))}
        if not all(out["int4_parity"].values()):
            out["problems"].append(f"tiny INT4 parity failed: {out['int4_parity']}")
    except Exception as e:  # noqa: BLE001
        out["problems"].append(f"tiny INT4 parity raised {type(e).__name__}: {e}"[:300])
    out["ok"] = not out["problems"]
    return out


def cpu_selftest():
    """Pure torch (CPU): CT pack -> dequant round trip and the MoE reference on a tiny synthetic layer."""
    out = {}
    try:
        import torch
        g = torch.Generator().manual_seed(3)
        q = torch.randint(-8, 8, (16, 256), generator=g)
        sc = torch.rand(16, 2, generator=g) + 0.5
        pk = C.ct_pack_int4(q)
        deq = C.ct_dequant_int4(pk, sc, 128)
        out["pack_roundtrip_exact"] = bool(torch.equal(deq, q.float() * sc.repeat_interleave(128, 1)))
        out["gptq_ref_matches_ct"] = bool(torch.equal(gptq_dequant_ref(torch, pk.T.contiguous().unsqueeze(0),
                                                                       sc.T.contiguous().unsqueeze(0))[0], deq))
        E, k, H, I = 8, 2, 256, 128
        W = {e: (torch.randn(I, H, generator=g), torch.randn(I, H, generator=g), torch.randn(H, I, generator=g)) for e in range(E)}
        x = torch.randn(3, H, generator=g)
        ids = torch.tensor([[0, 5], [5, 7], [1, 2]])
        w = torch.tensor([[0.25, 0.75], [0.5, 0.5], [1.0, 0.0]])
        ref = C.moe_reference(x, ids, w, lambda e: W[e], 0, 4)
        man = torch.zeros(3, H)
        for t in range(3):
            for j in range(k):
                e = int(ids[t, j])
                if e < 4:
                    gg, uu, dd = W[e]
                    man[t] += w[t, j] * ((torch.nn.functional.silu(x[t] @ gg.T) * (x[t] @ uu.T)) @ dd.T)
        out["moe_reference_matches_loop"] = bool(torch.allclose(ref, man, rtol=1e-5, atol=1e-4))
        out["ok"] = out["pack_roundtrip_exact"] and out["moe_reference_matches_loop"] and out["gptq_ref_matches_ct"]
    except ImportError:
        out = {"ok": None, "why": "torch not importable"}
    except Exception as e:  # noqa: BLE001
        out = {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}
    return out


def fake_replay_rows(path, which):
    """Fake mode with a routing file: the loader + the distinct-vs-uniform statistics run for real (no GPU)."""
    recs, info = C.load_routing(path)
    sel = C.select_replay(recs)
    keep = set(C.replay_sizes(sel, which))
    return recs, info, [{"T": m, "records": len(rs), "graph_us_mean": 50.0 + m,
                         "distinct_local_mean": round(sum(C.distinct(r["ids"], 0, 256) for r in rs) / len(rs), 2),
                         "distinct_total_mean": round(sum(C.distinct(r["ids"]) for r in rs) / len(rs), 2),
                         "d_uniform": round(C.d_expected(m), 2), "derived_gbps_mean": 150.0, "ref_ok": True,
                         "split_over_combined_mean": 1.2, "split_equivalent": True} for m, rs in sel.items() if m in keep]


def find_ncu():
    """Nsight Compute CLI if this box has one (PATH, CUDA_HOME, /usr/local/cuda*, /opt/nvidia)."""
    import glob
    import shutil
    c = [shutil.which("ncu")] + [os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "ncu")] + \
        sorted(glob.glob("/usr/local/cuda*/bin/ncu")) + sorted(glob.glob("/opt/nvidia/nsight-compute*/ncu"))
    return next((x for x in c if x and os.path.isfile(x) and os.access(x, os.X_OK)), None)


def run_ncu(a, ncu, T, mode, timeout):
    """ONE arm under Nsight Compute in its own child (N3-1): mode split = the two T/2 calls, combined = the T call,
    inside cudaProfilerStart/Stop; every kernel of the run summed (nb3_common.ncu_run_verdict)."""
    cmd = [ncu, "--profile-from-start", "off", "--csv", "--print-units", "base", "--target-processes", "all",
           "--metrics", ",".join(C.NCU_METRICS), sys.executable, os.path.abspath(__file__),
           "--ckpt", a.ckpt, "--out", a.out + f".ncu{T}_{mode}.json", "--layer", str(a.layer), "--ep-size", str(a.ep_size),
           "--ep-rank", str(a.ep_rank), "--copies", "1", "--seed", str(a.seed), "--ncu-child", str(T), "--ncu-mode", mode]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(60, timeout))
        out = C.ncu_run_verdict(r.returncode, r.stdout, r.stderr)
        out["sec"] = round(time.time() - t0, 1)
        if not out.get("ok"):
            out["tail"] = (r.stdout[-600:] + r.stderr[-600:])
        return out
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "status": C.NCU_UNVERIFIED, "why": f"{type(e).__name__}: {e}"[:300], "sec": round(time.time() - t0, 1)}


def run_units(res, save, units, do, left):
    """The priority executor (N3-16): a unit starts only when its reserve fits; the first that does not stops the
    bench (budget_stop, PARTIAL_BUDGET), everything measured before it is already saved."""
    res["units_planned"] = [list(u) for u in units]
    for i, u in enumerate(units):
        need = C.D5_RESERVE_S[u[0]]
        if res.get("budget_stop") or left() < need:
            res["budget_stop"] = True
            res["units_skipped"] = [list(x) for x in units[i:]]
            res["errors"].append(f"budget stop before unit {list(u)} ({round(left())} s left, reserve {need} s)")
            break
        do(u)
        res["units_done"].append(list(u))
        save()


# ------------------------------------------------------------------------------------------------------------ fake
def fake(a, res, save):
    """The same unit plan and executor on a fake clock (FAKE_UNIT_SEC per unit), fake numbers (nb3_common fakes) and
    the injected faults of NB3_FAKE_FAULTS; the dequant gate runs on CPU torch (corrupt_scale -> mismatch -> stop)."""
    faults = C.fake_faults()
    rng = __import__("random").Random(a.seed)
    clock = {"t": 0.0}
    left = lambda: a.budget_s - clock["t"]  # noqa: E731
    res["checks"]["preflight"] = True
    try:
        import torch
        g = torch.Generator().manual_seed(5)
        q = torch.randint(-8, 8, (8, 256), generator=g)
        sc = torch.rand(8, 2, generator=g) + 0.5
        pk = C.ct_pack_int4(q)
        sc_m = sc.clone()
        if "corrupt_scale" in faults:
            sc_m[3, 1] *= 1.5                     # a scale handled differently on the Merlin side
        res["dequant_crosscheck"] = dequant_gate(torch, C.ct_dequant_int4(pk, sc, 128),
                                                 gptq_dequant_ref(torch, pk.T.contiguous().unsqueeze(0), sc_m.T.contiguous().unsqueeze(0))[0],
                                                 "fake expert 0 gate_proj")
    except ImportError:
        res["dequant_crosscheck"] = {"equal": None, "why": "torch not importable (fake)"}
    res["checks"].update(load_shapes=True, dequant_crosscheck=True, first_call_gate=True)
    res["first_call_gate"] = {"1": {"ref_ok": True}, "4": {"ref_ok": True}}
    rp_path = a.routing_file if a.routing_file and Path(a.routing_file).is_file() else None
    if not rp_path:
        res["replay"] = {"available": False, "file": a.routing_file or None,
                         "why": f"real routing unavailable (no file at {a.routing_file or '(none configured)'}): synthetic only (fake run)"}
    cases = {c["C"]: c for c in C.pp2_split_cases()}
    units = [u for u in C.d5_units(a.phase, replay=bool(rp_path), ncu=False, ep1=a.ep_rank == 0, humming=a.ep_rank == 0)] if not a.spot else \
        [("base", int(t), "independent") for t in a.spot.split(",")]

    def do(u):
        clock["t"] += FAKE_UNIT_SEC
        kind, arg = u[0], u[1]
        if kind == "base":
            row = C.fake_base_row(arg, u[2], a.seed, rng)
            if "graph_error" in faults and arg == 16 and u[2] == "independent":
                row.update(graph_us=None, graph_status="INCONCLUSIVE_CAPTURE_FAILED", graph_error="RuntimeError: injected capture failure",
                           derived_gbps=None)
            res["baseline"].setdefault(str(arg), {})[u[2]] = row
        elif kind == "adv":
            res["adversarial"][str(arg)] = {**C.fake_base_row(arg, "adversarial", a.seed, rng), "reference_only": True}
        elif kind == "sweep":
            res["sweep"][str(arg)] = C.fake_sweep(arg, (res["baseline"].get(str(arg)) or {}).get("independent"), rng)
        elif kind == "split":
            res["split"][str(arg)] = C.fake_split(cases[arg], a.seed, rng)
        elif kind == "replay":
            recs, info, rows = fake_replay_rows(rp_path, arg)
            rp = res.setdefault("replay", {"available": bool(recs), "file": rp_path, "info": info,
                                           "distinct_vs_uniform": C.routing_stats(recs), "by_size": []})
            rp["by_size"] += rows
        elif kind == "humming":
            res["humming"] = HM.fake(a.seed, unavailable="humming_unavailable" in faults)
        elif kind == "ep1_spot":
            res["ep1_spot"] = {"ok": True, "rc": 0, "status": "DONE", "local_range": [256, 512], "fake": True,
                               "rows": {str(T): {"ref_ok": True, "graph_us": round(40.0 + T, 3)} for T in (4, 16)}}
    run_units(res, save, units, do, left)
    if "d5_bench_error" in faults:
        raise RuntimeError("injected D5 bench error (NB3_FAKE_FAULTS)")


# ------------------------------------------------------------------------------------------------------------ real
def ncu_child(a, res, torch, inputs, mk, copies):
    """--ncu-child T --ncu-mode split|combined: warm both forms, then profile EXACTLY one arm."""
    T = int(a.ncu_child)
    x, tid, tw, _, _ = inputs(T, "independent")
    halves = [(x[a0:b0].contiguous(), tid[a0:b0].contiguous(), tw[a0:b0].contiguous()) for a0, b0 in C.split_halves(T)]
    for _ in range(3):
        for hv in halves:
            mk(copies[0], *hv)()
        mk(copies[0], x, tid, tw)()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    if a.ncu_mode == "split":
        for hv in halves:
            mk(copies[0], *hv)()
    else:
        mk(copies[0], x, tid, tw)()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    res["ncu_child"] = {"T": T, "mode": a.ncu_mode, "done": True}
    return True


def real(a, res, save):
    t_start = time.time()
    left = lambda: a.budget_s - (time.time() - t_start)  # noqa: E731
    import torch
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
    vllm = need("vllm")
    current_stream = need("vllm.utils.torch_utils", "current_stream")
    fused_marlin_moe = need("vllm.model_executor.layers.fused_moe.experts.marlin_moe", "fused_marlin_moe")
    process_marlin = need("vllm.model_executor.layers.fused_moe.oracle.int_wna16", "_process_weights_marlin")
    merlin_dequant = need("vllm.model_executor.layers.fused_moe.oracle.int_wna16", "_unpack_and_dequant_int4_gptq")
    moe_align = need("vllm.model_executor.layers.fused_moe.moe_align_block_size", "moe_align_block_size")
    determine_expert_map = need("vllm.model_executor.layers.fused_moe.expert_map_manager", "determine_expert_map")
    scalar_types = need("vllm.scalar_type", "scalar_types")
    MC = need("vllm.merlin.marlin_moe_cfg")
    safe_open = need("safetensors", "safe_open")
    res["preflight"] = preflight(torch, {"_unpack_and_dequant_int4_gptq": merlin_dequant, "_process_weights_marlin": process_marlin,
                                         "fused_marlin_moe": fused_marlin_moe})
    res["checks"]["preflight"] = res["preflight"]["ok"]
    save()
    if not res["preflight"]["ok"]:
        raise RuntimeError(f"DEPENDENCY_PREFLIGHT_FAILED: {res['preflight']['problems'][:3]}")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    current_stream()                       # vLLM's dedicated stream (as in the engine); events + graphs run on it
    prop = torch.cuda.get_device_properties(dev)
    res["env"] = {"gpu": prop.name, "cap": list(torch.cuda.get_device_capability(dev)), "sms": prop.multi_processor_count,
                  "l2_bytes": int(getattr(prop, "L2_cache_size", 0) or 0), "torch": torch.__version__,
                  "vllm": getattr(vllm, "__version__", None), "vllm_file": getattr(vllm, "__file__", None),
                  "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"), "MERLIN_MARLIN_MOE_CFG_at_start": os.environ.get("MERLIN_MARLIN_MOE_CFG"),
                  "VLLM_MARLIN_INPUT_DTYPE": os.environ.get("VLLM_MARLIN_INPUT_DTYPE")}
    os.environ.pop("MERLIN_MARLIN_MOE_CFG", None)           # the baseline is the engine's own choice (the profile sets none)
    cfgdir = Path(vllm.__file__).parent / "model_executor/layers/fused_moe/configs"
    res["tuning_files"] = {"dir": str(cfgdir), "note": "Triton fused_moe tables; Marlin MoE does not read them",
                           "matching": sorted(p.name for p in cfgdir.glob("*.json")
                                              if ("N=640" in p.name or "N=1280" in p.name or "L4" in p.name)
                                              and ("E=256" in p.name or "E=512" in p.name or "L4" in p.name))[:40]
                           if cfgdir.is_dir() else None,
                           "merlin_marlin_moe_cfg": {"compiled_small_m": list(MC._COMPILED_SMALL_M), "compiled_large_m": list(MC._COMPILED_LARGE_M),
                                                     "blocks_per_sm": list(MC.BLOCKS_PER_SM), "block_sizes_m": list(MC.BLOCK_SIZES_M)}}
    # ---------------------------------------------------------------- checkpoint facts
    ck = Path(a.ckpt)
    cfg = json.loads((ck / "config.json").read_text())
    tc = cfg.get("text_config") or cfg
    wq = cfg["quantization_config"]["config_groups"]["group_0"]["weights"]
    E, topk, H, I = int(tc["num_experts"]), int(tc["num_experts_per_tok"]), int(tc["hidden_size"]), int(tc["moe_intermediate_size"])
    bits, G = int(wq["num_bits"]), int(wq["group_size"])
    res["model"] = {"E": E, "topk": topk, "H": H, "I": I, "bits": bits, "group": G, "symmetric": wq.get("symmetric"),
                    "strategy": wq.get("strategy"), "format": cfg["quantization_config"].get("format"), "layer": a.layer,
                    "ep_size": a.ep_size, "ep_rank": a.ep_rank, "expert_bytes": C.expert_bytes(H, I, bits, G)}
    if not (bits == 4 and G > 0 and wq.get("symmetric") is True and wq.get("type") == "int"):
        raise RuntimeError(f"unexpected expert quantization {wq}")
    E_loc, emap_cpu, _ = determine_expert_map(a.ep_size, a.ep_rank, E)
    local = [int(e) for e in (emap_cpu >= 0).nonzero().flatten().tolist()] if emap_cpu is not None else list(range(E))
    lo, hi = min(local), max(local) + 1
    res["model"].update(E_local=E_loc, local_range=[lo, hi], local_contiguous=(hi - lo == len(local)))
    wm = json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"]
    pre = f"model.language_model.layers.{a.layer}.mlp.experts."
    t0 = time.time()
    handles = {}

    def get(name):
        f = wm[name]
        if f not in handles:
            handles[f] = safe_open(str(ck / f), framework="pt", device="cpu")
        return handles[f].get_tensor(name)
    gp, gs, upk, ups, dp, ds = [], [], [], [], [], []
    shape_bad = []
    for e in local:
        for proj, lp, ls, want in (("gate_proj", gp, gs, (I, H)), ("up_proj", upk, ups, (I, H)), ("down_proj", dp, ds, (H, I))):
            pk, sc = get(f"{pre}{e}.{proj}.weight_packed"), get(f"{pre}{e}.{proj}.weight_scale")
            shp = get(f"{pre}{e}.{proj}.weight_shape").tolist() if f"{pre}{e}.{proj}.weight_shape" in wm else None
            if tuple(pk.shape) != (want[0], want[1] * bits // 32) or tuple(sc.shape) != (want[0], want[1] // G) or \
                    pk.dtype != torch.int32 or (shp is not None and [int(v) for v in shp] != list(want)):
                shape_bad.append([e, proj, list(pk.shape), str(pk.dtype), list(sc.shape), shp])
            lp.append(pk)
            ls.append(sc)
    res["load"] = {"sec": round(time.time() - t0, 1), "files": sorted(handles), "experts": len(local),
                   "scale_dtype": str(gs[0].dtype), "packed_dtype": str(gp[0].dtype), "shape_bad": shape_bad[:10]}
    res["checks"]["load_shapes"] = not shape_bad
    save()
    if shape_bad:
        raise RuntimeError(f"expert tensor shapes differ from config: {shape_bad[:3]}")
    # N3-15: independent dequantisation vs Merlin's own GPTQ-layout dequant (first local expert, all three matrices)
    res["dequant_crosscheck"] = []
    res["checks"]["dequant_crosscheck"] = False
    for what, pk, sc in (("gate_proj", gp[0], gs[0]), ("up_proj", upk[0], ups[0]), ("down_proj", dp[0], ds[0])):
        theirs = merlin_dequant(pk.T.contiguous().unsqueeze(0), sc.T.contiguous().unsqueeze(0), None, True, torch.float32)[0]
        try:
            res["dequant_crosscheck"].append(dequant_gate(torch, C.ct_dequant_int4(pk, sc, G), theirs, f"expert {lo} {what}"))
        finally:
            save()
    res["checks"]["dequant_crosscheck"] = True
    w13 = torch.stack([torch.cat([g, u], 0) for g, u in zip(gp, upk)]).to(dev)                    # [E_loc, 2I, H/8]
    s13 = torch.stack([torch.cat([g, u], 0) for g, u in zip(gs, ups)]).to(dev, torch.bfloat16)     # [E_loc, 2I, H/G]
    w2 = torch.stack(dp).to(dev)                                                                   # [E_loc, H, I/8]
    s2 = torch.stack(ds).to(dev, torch.bfloat16)                                                   # [E_loc, H, I/G]
    del gp, gs, upk, ups, dp, ds
    layer = types.SimpleNamespace(intermediate_size_per_partition=I, num_groups_w13=H // G, num_groups_w2=I // G)
    t0 = time.time()
    conv = process_marlin(layer, None, bits, 32 // bits, G, w13, w2, s13, s2)
    torch.cuda.synchronize()
    base = tuple(conv[:4])
    res["prep"] = {"sec": round(time.time() - t0, 2), "marlin_shapes": [list(t.shape) for t in base],
                   "extra_outputs_none": [v is None for v in conv[4:]], "n_outputs": len(conv)}
    res["checks"]["marlin_prep_outputs"] = len(conv) == 10 and all(v is None for v in conv[4:])
    if not res["checks"]["marlin_prep_outputs"]:
        save()
        raise RuntimeError(f"MARLIN_PREP_UNEXPECTED_OUTPUTS: {len(conv)} outputs, extras None {res['prep']['extra_outputs_none']}")
    copies = [base] + [tuple(t.clone() for t in base) for _ in range(max(0, a.copies - 1))]
    emap = emap_cpu.to(dev) if emap_cpu is not None else None
    qt = scalar_types.uint4b8.id
    res["prep"]["copies"] = len(copies)
    res["prep"]["gpu_mem_alloc_gib"] = round(torch.cuda.memory_allocated(dev) / 2 ** 30, 2)

    def expert_fp32(e):
        li = int(emap_cpu[e]) if emap_cpu is not None else e
        return (C.ct_dequant_int4(w13[li, :I], s13[li, :I], G), C.ct_dequant_int4(w13[li, I:], s13[li, I:], G),
                C.ct_dequant_int4(w2[li], s2[li], G))

    def inputs(T, mode):
        grp = C.request_group(T) if mode == "request" else 1
        ids = C.routing_ids(T, mode, E, topk, seed=a.seed, group=grp, local=(lo, hi))
        tid = torch.tensor(ids, dtype=torch.int32, device=dev)
        tw = torch.tensor(C.routing_weights(T, topk, seed=a.seed), dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(a.seed * 131 + T)
        x = torch.randn(T, H, generator=gen, device=dev, dtype=torch.float32).to(torch.bfloat16)
        return x, tid, tw, ids, grp

    def mk(cp, x, tid, tw):
        return lambda: fused_marlin_moe(hidden_states=x, w1=cp[0], w2=cp[1], bias1=None, bias2=None, w1_scale=cp[2],
                                        w2_scale=cp[3], topk_weights=tw, topk_ids=tid, quant_type_id=qt,
                                        global_num_experts=E, expert_map=emap)
    # ---------------------------------------------------------------- N3-7: first-call gate (T=1, T=4) before any sweep
    res["first_call_gate"] = {}
    for T in (1, 4):
        x, tid, tw, ids, _ = inputs(T, "independent")
        out = mk(copies[0], x, tid, tw)()
        torch.cuda.synchronize()
        m = C.metrics_x(torch, out, C.moe_reference(x, tid, tw, expert_fp32, lo, hi))
        res["first_call_gate"][str(T)] = {"ref": m, "ref_ok": ref_ok(m), "distinct_local": C.distinct(ids, lo, hi)}
    res["checks"]["first_call_gate"] = all(v["ref_ok"] for v in res["first_call_gate"].values())
    save()
    if not res["checks"]["first_call_gate"]:
        raise RuntimeError(f"FIRST_CALL_GATE_FAILED: {json.dumps(res['first_call_gate'])[:400]}")
    eb = C.expert_bytes(H, I, bits, G)
    if a.ncu_child:
        return ncu_child(a, res, torch, inputs, mk, copies)
    ncu = find_ncu() if not a.spot else None
    res["ncu"] = {"path": ncu, "used": False, "note": "one child per arm (split / combined); runs LAST"}
    rp_file = a.routing_file if a.routing_file and Path(a.routing_file).is_file() else None
    recs = []
    if rp_file:
        recs, info = C.load_routing(rp_file, E, topk)
        res["replay"] = {"available": bool(recs), "file": rp_file, "info": info,
                         "distinct_vs_uniform": C.routing_stats(recs, lo, hi, E, topk), "by_size": []}
        if not recs:
            res["replay"]["why"] = "routing file has no usable record"
    else:
        res["replay"] = {"available": False, "file": a.routing_file or None,
                         "why": f"real routing unavailable (no file at {a.routing_file or '(none configured)'}): synthetic uniform / Zipf / hot only"}
    sel = C.select_replay(recs) if recs else {}
    rules = sweep_rules(MC)
    res["sweep_rules"] = rules
    cases = {c["C"]: c for c in C.pp2_split_cases()}

    def u_base(T, mode, store):
        r = {"T": T, "mode": mode}
        try:
            x, tid, tw, ids, grp = inputs(T, mode)
            dl = C.distinct(ids, lo, hi)
            uni = mode == "independent"           # D(T) is the uniform-routing model only
            r.update(group=grp, distinct_total=C.distinct(ids), distinct_local=dl, synthetic_routing=True,
                     d_expected=round(C.d_expected(T, E, topk), 2) if uni else None,
                     d_expected_local=round(C.d_expected(T, E, topk) * E_loc / E, 2) if uni else None,
                     bytes=dl * eb, M_est=math.ceil(T * E_loc / E), heuristic_bm=heuristic_bm(math.ceil(T * E_loc / E), topk, E_loc))
            out = mk(copies[0], x, tid, tw)()
            torch.cuda.synchronize()
            ref = C.moe_reference(x, tid, tw, expert_fp32, lo, hi)
            r["ref"] = C.metrics_x(torch, out, ref)
            r["ref_ok"] = ref_ok(r["ref"])
            if store == "adversarial":
                r["reference_only"] = True
            else:
                t = C.timing(C.bench(torch, [mk(cp, x, tid, tw) for cp in copies], a.warmup, a.iters, True))
                r.update(eager_us=t["eager_us"], graph_us=t["graph_us"], graph_status=t["graph_status"])
                if t["graph_error"]:
                    r["graph_error"] = t["graph_error"]
                r["hot_graph_us"] = C.timing(C.bench(torch, [mk(copies[0], x, tid, tw)], a.warmup, a.iters, True))["graph_us"]
                bm = r["heuristic_bm"]
                al = C.timing(C.bench(torch, [lambda: moe_align(tid, bm, E, emap, ignore_invalid_experts=True)], a.warmup, a.iters, True))
                r["align_graph_us"], r["align_eager_us"] = al["graph_us"], al["eager_us"]
                us = C.med(r["graph_us"])
                r["derived_gbps"] = round(dl * eb / us / 1e3, 1) if us else None
        except Exception as e:  # noqa: BLE001
            r.update(error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:], ref_ok=False)
        if store == "adversarial":
            res["adversarial"][str(T)] = r
        else:
            res["baseline"].setdefault(str(T), {})[mode] = r

    def u_sweep(T):
        b = (res["baseline"].get(str(T)) or {}).get("independent") or {}
        sw = {"base_graph_us": C.med(b.get("graph_us")), "base_graph_status": b.get("graph_status"),
              "base_spread": C.stats_spread(b.get("graph_us")), "baseline_bm": b.get("heuristic_bm"), "rules": []}
        if b.get("error") or not b:
            sw["why"] = "no baseline row at this T"
            res["sweep"][str(T)] = sw
            return
        x, tid, tw, ids, _ = inputs(T, "independent")
        base_out = mk(copies[0], x, tid, tw)()
        torch.cuda.synchronize()
        ref = C.moe_reference(x, tid, tw, expert_fp32, lo, hi)
        key = (math.ceil(T * E_loc / E), E_loc, topk, H, I)
        for rule in rules:
            if left() < 30:
                sw["budget_stop"] = True
                res["budget_stop"] = True            # N3-22: a sweep stop is a bench budget stop
                res["errors"].append(f"sweep T={T} stopped after {len(sw['rules'])} rules (budget)")
                break
            rr = {"rule": rule}
            os.environ["MERLIN_MARLIN_MOE_CFG"] = rule
            try:
                out = mk(copies[0], x, tid, tw)()
                torch.cuda.synchronize()
                rr["decision"] = MC.shape_decisions().get(key)
                rr["vs_base"] = C.equivalence(out, base_out)
                rr["bitwise_vs_base"] = rr["vs_base"]["bitwise_equal"]
                rr["ref"] = C.metrics_x(torch, out, ref)
                rr["ref_ok"] = ref_ok(rr["ref"])
                t = C.timing(C.bench(torch, [mk(cp, x, tid, tw) for cp in copies], a.warmup, a.iters, True))
                rr.update(graph_us=t["graph_us"], eager_us=t["eager_us"], graph_status=t["graph_status"],
                          us=C.med(t["graph_us"]), spread=C.stats_spread(t["graph_us"]))
                if t["graph_error"]:
                    rr["graph_error"] = t["graph_error"]
            except Exception as e:  # noqa: BLE001
                rr["error"] = f"{type(e).__name__}: {e}"[:400]
                try:
                    torch.cuda.synchronize()
                except Exception as e2:  # noqa: BLE001 - a sticky device error ends the sweep
                    rr["fatal"] = f"{type(e2).__name__}: {e2}"[:300]
                    sw["rules"].append(rr)
                    res["sweep"][str(T)] = sw
                    raise
            finally:
                os.environ.pop("MERLIN_MARLIN_MOE_CFG", None)
            sw["rules"].append(rr)
        ok = [r for r in sw["rules"] if r.get("us") and r.get("ref_ok") and (r.get("vs_base") or {}).get("equivalent")
              and str(r.get("decision") or "").startswith("rule#")]
        best = min(ok, key=lambda r: r["us"]) if ok and sw["base_graph_us"] else None
        sw["best"] = ({**{k: best.get(k) for k in ("rule", "us", "spread", "bitwise_vs_base", "ref_ok", "decision")},
                       "equivalent_vs_base": True, "max_ulp_vs_base": (best.get("vs_base") or {}).get("max_ulp")} if best else None)
        sw["applied_rules"] = len(ok)
        res["sweep"][str(T)] = sw

    def u_split(Cc):
        case = cases[Cc]
        T, h = case["T"], case["T_half"]
        sp = dict(case)
        try:
            x, tid, tw, ids, _ = inputs(T, "independent")
            dsum = C.distinct(ids[:h], lo, hi) + C.distinct(ids[h:], lo, hi)
            dcomb = C.distinct(ids, lo, hi)
            out_a = C.split_merge(lambda xx, ii, ww: mk(copies[0], xx, ii, ww)(), x, tid, tw, T)
            out_b = mk(copies[0], x, tid, tw)()
            torch.cuda.synchronize()
            ref = C.moe_reference(x, tid, tw, expert_fp32, lo, hi)
            sp["equivalence"] = C.equivalence(out_a, out_b)
            sp["ref_split"], sp["ref_combined"] = C.metrics_x(torch, out_a, ref), C.metrics_x(torch, out_b, ref)
            sp["ref_ok_split"], sp["ref_ok_combined"] = ref_ok(sp["ref_split"]), ref_ok(sp["ref_combined"])
            halves = [(x[a0:b0].contiguous(), tid[a0:b0].contiguous(), tw[a0:b0].contiguous()) for a0, b0 in C.split_halves(T)]
            fa = [lambda cp=cp: [mk(cp, *hv)() for hv in halves] for cp in copies]
            fb = [mk(cp, x, tid, tw) for cp in copies]
            ta, tb = C.timing(C.bench(torch, fa, a.warmup, a.iters, True)), C.timing(C.bench(torch, fb, a.warmup, a.iters, True))
            sp["split_graph_us"], sp["combined_graph_us"] = ta["graph_us"], tb["graph_us"]
            sp["split_eager_us"], sp["combined_eager_us"] = ta["eager_us"], tb["eager_us"]
            sp["graph_status"] = "PASS" if ta["graph_us"] and tb["graph_us"] else "INCONCLUSIVE_CAPTURE_FAILED"
            sp["graph_error"] = ta["graph_error"] or tb["graph_error"]
            sp.update(distinct_local_split_sum=dsum, distinct_local_split_halves=[C.distinct(ids[a0:b0], lo, hi) for a0, b0 in C.split_halves(T)],
                      distinct_local_combined=dcomb, bytes_split=dsum * eb, bytes_combined=dcomb * eb, bytes_source="DERIVED")
            ua, ub = C.med(sp["split_graph_us"]), C.med(sp["combined_graph_us"])
            sp["derived_gbps_split"] = round(dsum * eb / ua / 1e3, 1) if ua else None
            sp["derived_gbps_combined"] = round(dcomb * eb / ub / 1e3, 1) if ub else None
        except Exception as e:  # noqa: BLE001
            sp.update(error=f"{type(e).__name__}: {e}"[:600], tb=traceback.format_exc()[-1500:])
        sp["ncu"] = {"ok": False, "status": C.NCU_UNVERIFIED, "why": "ncu not found" if not ncu else "pending (ncu runs last)"}
        res["split"][str(Cc)] = sp

    def u_replay(which):
        by_size = res["replay"].setdefault("by_size", [])
        for m in C.replay_sizes(sel, which):
            rs = sel[m]
            if left() < C.D5_RESERVE_S["replay"]:
                res["replay"]["budget_stop"] = True
                res["budget_stop"] = True
                res["errors"].append(f"replay ({which}) stopped before size {m} (budget)")
                return
            rows = []
            for i, rec in enumerate(rs):
                tid = torch.tensor(rec["ids"], dtype=torch.int32, device=dev)
                tw = torch.tensor(C.routing_weights(m, topk, seed=a.seed + i), dtype=torch.float32, device=dev)
                gen = torch.Generator(device=dev).manual_seed(a.seed * 977 + m + i)
                x = torch.randn(m, H, generator=gen, device=dev, dtype=torch.float32).to(torch.bfloat16)
                row = {"layer": rec["layer"], "stage": rec["stage"], "step": rec["step"], "T": m,
                       "distinct_total": C.distinct(rec["ids"]), "distinct_local": C.distinct(rec["ids"], lo, hi)}
                try:
                    t = C.timing(C.bench(torch, [mk(cp, x, tid, tw) for cp in copies], a.warmup, a.iters, True))
                    row["us"], row["graph_status"] = C.med(t["graph_us"]), t["graph_status"]
                    row["eager_us"] = C.med(t["eager_us"])
                    row["derived_gbps"] = round(row["distinct_local"] * eb / row["us"] / 1e3, 1) if row["us"] else None
                    if i == 0:
                        out_b = mk(copies[0], x, tid, tw)()
                        torch.cuda.synchronize()
                        ref = C.moe_reference(x, tid, tw, expert_fp32, lo, hi)
                        row["ref"] = C.metrics_x(torch, out_b, ref)
                        row["ref_ok"] = ref_ok(row["ref"])
                        if m >= 2:          # PP2 split-read on real ids: the record's two halves vs the record
                            out_a = C.split_merge(lambda xx, ii, ww: mk(copies[0], xx, ii, ww)(), x, tid, tw, m)
                            row["split_equivalence"] = C.equivalence(out_a, out_b)
                            hv = [(x[a0:b0].contiguous(), tid[a0:b0].contiguous(), tw[a0:b0].contiguous()) for a0, b0 in C.split_halves(m)]
                            ta = C.timing(C.bench(torch, [lambda cp=cp: [mk(cp, *h)() for h in hv] for cp in copies], a.warmup, a.iters, True))
                            ua = C.med(ta["graph_us"])
                            row["split_us"] = ua
                            row["split_over_combined"] = round(ua / row["us"], 3) if ua and row["us"] else None
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {e}"[:400]
                    row["ref_ok"] = False if i == 0 else row.get("ref_ok")
                rows.append(row)
            ok = [r for r in rows if r.get("us")]
            sc = [r["split_over_combined"] for r in rows if r.get("split_over_combined")]
            by_size.append({"T": m, "records": len(rows), "rows": rows,
                            "graph_us_mean": round(sum(r["us"] for r in ok) / len(ok), 3) if ok else None,
                            "distinct_local_mean": round(sum(r["distinct_local"] for r in rows) / len(rows), 2),
                            "distinct_total_mean": round(sum(r["distinct_total"] for r in rows) / len(rows), 2),
                            "d_uniform": round(C.d_expected(m, E, topk), 2),
                            "derived_gbps_mean": round(sum(r["derived_gbps"] for r in ok if r.get("derived_gbps")) / max(1, len(ok)), 1) if ok else None,
                            "ref_ok": rows[0].get("ref_ok") if rows else None,
                            "split_over_combined_mean": round(sum(sc) / len(sc), 3) if sc else None,
                            "split_equivalent": (rows[0].get("split_equivalence") or {}).get("equivalent") if rows else None})
            save()

    def u_ep1_spot():
        """N3-8: a small T set at EP rank 1 (experts 256..511, same file), in a child process (1 weight copy)."""
        outp = a.out + ".ep1.json"
        cmd = [sys.executable, os.path.abspath(__file__), "--ckpt", a.ckpt, "--out", outp, "--layer", str(a.layer),
               "--ep-size", str(a.ep_size), "--ep-rank", "1", "--copies", "1", "--seed", str(a.seed), "--spot", "4,16",
               "--iters", str(a.iters), "--warmup", str(a.warmup), "--budget-s", str(max(60, int(left() - 60)))]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(90, left() - 30))
            j = json.loads(Path(outp).read_text()) if Path(outp).is_file() else {}
            rows = {T: {"ref_ok": v.get("independent", {}).get("ref_ok"), "graph_us": C.med(v.get("independent", {}).get("graph_us")),
                        "rel_fro": (v.get("independent", {}).get("ref") or {}).get("rel_fro")} for T, v in (j.get("baseline") or {}).items()}
            res["ep1_spot"] = {"rc": r.returncode, "status": j.get("status"), "local_range": (j.get("model") or {}).get("local_range"),
                               "rows": rows, "first_call_gate": j.get("first_call_gate"), "sec": round(time.time() - t0, 1),
                               "ok": r.returncode == 0 and j.get("status") == "DONE" and len(rows) == 2
                                     and all(v["ref_ok"] is True for v in rows.values()),
                               "tail": None if r.returncode == 0 else (r.stdout[-600:] + r.stderr[-600:])}
        except Exception as e:  # noqa: BLE001
            res["ep1_spot"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "sec": round(time.time() - t0, 1)}

    def u_ncu(Cc):
        sp = res["split"].get(str(Cc))
        if not sp or sp.get("error"):
            return
        arms = {}
        for mode in ("split", "combined"):
            arms[mode] = run_ncu(a, ncu, cases[Cc]["T"], mode, min(300, left() / 2 - 30))
        sp["ncu"] = C.ncu_split_combined(arms["split"], arms["combined"])
        res["ncu"]["used"] = True
        if sp["ncu"]["ok"]:
            sp["bytes_source_ncu"] = "ncu dram__bytes_read.sum (split and combined profiled in separate runs)"

    def u_humming():
        """C6-humming structural arms (nb3_humming): research, never a D5 check; UNAVAILABLE when humming is absent."""
        def fp32(sub, e0, e1):
            if sub == "w13":
                return torch.stack([C.ct_dequant_int4(w13[li], s13[li], G) for li in range(e0, e1)])
            return torch.stack([C.ct_dequant_int4(w2[li], s2[li], G) for li in range(e0, e1)])
        ctx = {"E": E, "E_loc": E_loc, "topk": topk, "H": H, "I": I, "G": G, "dev": dev, "emap": emap, "emap_cpu": emap_cpu,
               "lo": lo, "hi": hi, "moe_align": moe_align, "inputs": inputs, "marlin_copies": copies,
               "marlin_mk": lambda c, x, tid, tw: mk(copies[c], x, tid, tw), "expert_fp32_w4": expert_fp32, "fp32": fp32,
               "ct": {"w13": (w13, s13), "w2": (w2, s2)}}
        t0 = time.time()
        res["humming"] = HM.run(torch, C, ctx, lambda: left() - 30, ref_ok=ref_ok, warmup=a.warmup, iters=a.iters)
        res["humming"]["sec"] = round(time.time() - t0, 1)

    def do(u):
        kind, arg = u[0], u[1]
        if kind == "humming":
            u_humming()
        elif kind == "base":
            u_base(arg, u[2], "baseline")
        elif kind == "adv":
            u_base(arg, "adversarial", "adversarial")
        elif kind == "sweep":
            u_sweep(arg)
        elif kind == "split":
            u_split(arg)
        elif kind == "replay":
            u_replay(arg)
        elif kind == "ep1_spot":
            u_ep1_spot()
        elif kind == "ncu":
            u_ncu(arg)
    if a.spot:
        units = [("base", int(t), "independent") for t in a.spot.split(",")]
    else:
        units = C.d5_units(a.phase, replay=bool(sel), ncu=bool(ncu), ep1=a.ep_rank == 0, humming=a.ep_rank == 0)
    run_units(res, save, units, do, left)
    res["gpu_max_alloc_gib"] = round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 2)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--ep-size", type=int, default=2)
    ap.add_argument("--ep-rank", type=int, default=0)
    ap.add_argument("--copies", type=int, default=6, help="rotated weight copies (6 x ~650 MB: cold L2 at every T)")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget-s", type=float, default=1500.0)
    ap.add_argument("--phase", choices=("core", "rest", "all"), default="all", help="nb3_common.d5_units priority slice")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--ncu-child", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--ncu-mode", choices=("split", "combined"), default="split", help=argparse.SUPPRESS)
    ap.add_argument("--spot", default="", help=argparse.SUPPRESS)
    ap.add_argument("--routing-file", default="", help="NB1 router top-k ids (nb3_common.ROUTING_SCHEMA); absent = synthetic only")
    a = ap.parse_args()
    res = {"meta": {"tool": "nb3_moe_bench.py", "fake": a.fake, "args": vars(a), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "ref_gate": REF_GATE, "equiv_tol": C.EQUIV_TOL, "t_sweep": list(T_SWEEP), "priority": C.D5_PRIORITY},
           "phase": a.phase if not a.spot else "spot", "imports": {}, "baseline": {}, "sweep": {}, "split": {}, "adversarial": {},
           "checks": {}, "errors": [], "units_done": [], "units_skipped": [], "cpu_selftest": cpu_selftest(),
           "routing_note": C.ROUTING_NOTE}
    save = lambda: C.write_json(a.out, res)  # noqa: E731
    t0 = time.time()
    code = 0
    try:
        if a.fake:
            fake(a, res, save)
            res["checks"]["cpu_selftest"] = res["cpu_selftest"].get("ok") is not False
        else:
            real(a, res, save)
        res["checks"] = C.d5_checks(res)
        res["status"] = "DONE" if not res.get("budget_stop") else "PARTIAL_BUDGET"
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
    print("NB3_MOE", res["status"], "exit", code, "| checks", json.dumps(res["checks"]), "| units",
          len(res["units_done"]), "done", len(res["units_skipped"]), "skipped | sec", res["sec"], flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
