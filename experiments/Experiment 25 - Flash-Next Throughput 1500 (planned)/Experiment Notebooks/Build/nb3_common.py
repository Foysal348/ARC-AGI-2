"""E25 NB3 (FN-S3a kernel bench) shared helpers: written next to the bench scripts at run time and imported by the
notebook (Kaggle Python 3.12, no torch needed) and by the benches (Merlin venv, Python 3.13 + torch).

Pure Python unless a function says "torch": token-count sets T = seqs x (1 + K) (+ the PP2 micro-batch and MTP draft
counts), the expected distinct experts D(T), routing generators, Flash-Next message sizes, the compressed-tensors INT4
pack / dequantised reference, the result tables, the ranked NB4 patch candidates and the lever scorecard
(PLAN_1500 v3.2 section 4.1). Fake-result generators (same schemas as the real benches) serve the local / CPU-kit
end-to-end run (E23_FAKE_SERVER).

The block between the P4 markers is cut verbatim from E23 prep/P4_sm89_fp8_gemm/bench_fp8_gemm.py by the builder
(stats / bench / metrics: warmup, per-iteration CUDA events, CUDA-graph replay over rotated weight copies).
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------------------------- the NB3 grid (spec)
SEQS = (1, 2, 4, 5, 8, 12, 16, 24, 32)     # resident streams (spec section 3 NB3)
KS = (1, 2, 3)                             # MTP draft length K
STEP_MS = (30.0, 39.0, 48.0)               # RC2 FP8 step_ms range (low / mid / high), the per-tick base
RC2_STEADY = {"seqs": 8, "K": 3, "tok_s": 261.8}   # RC2 FP8 8k@8 steady window (spec section 1)
RUN_OVER_STEADY = 0.9                      # PLAN section 1: run average ~0.9 x steady (queue never empties)
PP = 2
STAGE_LAYERS = (25, 23)                    # VLLM_PP_LAYER_PARTITION=25,23 (RC2 plan env)
BINDING_STAGE = 1                          # A7 section 5.2: stage 1 carries the heads + MTP drafts (the binding stage, est.)
N_LAYERS, QSA_EVERY = 48, 4                # layer i is QSA (full attention) when i % 4 == 3
TARGETS = ({"seqs": 8, "K": 3, "why": "RC2 operating point (step_ms measured here)"},
           {"seqs": 16, "K": 3, "why": "target 16 streams"},
           {"seqs": 32, "K": 3, "why": "target 32 streams"})
# RC2 FP8 per-step cost model (Experiment 13 Results/E13_RC2_GPU_v1_2026-10-31.md section 6.2, fitted on 7 STEADY windows,
# r = requests per micro-batch = C / 2): step_ms = 25.60 + 5.79 r, or 24.21 + 0.1758 D(4 r); context adds ~nothing;
# the D term read as expert streaming on the critical stage implies ~173 GB/s; a (MAL) = 3.05; 8k@8 measured 48.2 ms.
RC2_FIT = {"linear": [25.60, 5.79], "dmodel": [24.21, 0.1758], "a": 3.05, "implied_moe_gbps": 173.0, "step_ms_8_streams": 48.2,
           "source": "E13_RC2_GPU_v1_2026-10-31.md section 6.2 (est., fitted)"}
L4_PEAK_GBPS = 300.0
CONF_WEIGHT = {"medium": 0.5, "low": 0.25, "none": 0.0}   # a microbenchmark is never "high" (engine gain unconfirmed)
ADOPTION_COST = {"config": 1, "patch": 2, "research": 4}  # PLAN 4.1 score divisor


def stage_layers(stage):
    lo = 0 if stage == 0 else STAGE_LAYERS[0]
    ids = list(range(lo, lo + STAGE_LAYERS[stage]))
    qsa = sum(1 for i in ids if i % QSA_EVERY == QSA_EVERY - 1)
    return {"layers": len(ids), "qsa": qsa, "gdn": len(ids) - qsa}


def micro_requests(seqs):
    """PP2 + async interleaves two micro-batches: each stage carries ~ceil(seqs / 2) requests per tick (A7 GOOD 6)."""
    return -(-int(seqs) // PP)


def micro_T(seqs, K):
    return micro_requests(seqs) * (1 + int(K))


def t_sets(seqs=SEQS, ks=KS):
    """full = seqs x (1+K) (spec); micro = ceil(seqs/2) x (1+K) (what one PP2 stage forward really carries); draft =
    ceil(seqs/2) (one MTP draft pass per micro-batch request); all = the union, sorted."""
    full = sorted({int(s) * (1 + int(k)) for s in seqs for k in ks})
    micro = sorted({micro_T(s, k) for s in seqs for k in ks})
    draft = sorted({micro_requests(s) for s in seqs})
    return {"full": full, "micro": micro, "draft": draft, "all": sorted(set(full) | set(micro) | set(draft))}


def d_expected(T, E=512, k=10):
    """Distinct experts touched by T tokens with independent uniform top-k routing: E (1 - (1 - k/E)^T) (PLAN section 1)."""
    return E * (1.0 - (1.0 - k / E) ** int(T))


def expert_bytes(H=2560, I=640, bits=4, group=128, scale_bytes=2):
    """Bytes one routed expert streams per call: packed gate/up (2 I x H) + down (H x I) at `bits`, plus group scales."""
    packed = (2 * I * H + H * I) * bits // 8
    scales = (2 * I * (H // group) + H * (I // group)) * scale_bytes
    return packed + scales


def ar_bytes(hidden, T, dtype_bytes=2):
    """One TP2 all-reduce payload: the hidden-size block output of every token in the forward (A7 GOOD 7)."""
    return int(hidden) * int(T) * int(dtype_bytes)


ZIPF_S = 1.0          # skew of the synthetic Zipf expert popularity (p_e ~ 1 / rank^s over a fixed random ranking)
HOT_SET = 16          # extreme-hot routing: every token draws its k experts from ONE fixed set of 16 (<= 16 distinct in total)
ROUTING_NOTE = ("ALL routing here is SYNTHETIC (no server, no real hidden states): 'independent' = uniform D(T) model "
                "(the spec), 'zipf' = skewed popularity (s = %.1f), 'hot' = every token picks its 10 experts from one fixed "
                "set of %d (extreme-hot bound: <= %d distinct experts in total), 'request' = the 1+K tokens of a request share "
                "their experts (lower bound). Real per-layer routing comes from NB1's profiler window." % (ZIPF_S, HOT_SET, HOT_SET))


def hot_set(E=512, seed=0, n=HOT_SET):
    """The fixed hot expert set (the same for every T of one seed; spread over both EP halves at random)."""
    return sorted(random.Random(31337 + int(seed)).sample(range(E), n))


def _zipf_weights(E, seed):
    rng = random.Random(4242 + int(seed))
    order = list(range(E))
    rng.shuffle(order)                     # which expert is popular is arbitrary; the skew is what matters
    w = [0.0] * E
    for rank, e in enumerate(order):
        w[e] = 1.0 / (rank + 1) ** ZIPF_S
    return w


def routing_ids(T, mode="independent", E=512, k=10, seed=0, group=4, local=(0, 256)):
    """top-k expert ids per token (python lists), all SYNTHETIC. independent: k distinct experts uniformly per token
    (the D(T) upper bound, the spec's formula); zipf: k distinct experts per token drawn without replacement from a
    Zipf(s) popularity (Efraimidis-Spirakis keys u^(1/w)); request: the `group` tokens of one request (1 + K verify
    tokens) share one uniform draw (the lower bound D(T / group), A7 section 5.1); hot: k distinct experts per token
    from one fixed set of HOT_SET (the extreme-hot case: <= HOT_SET distinct experts whatever T); adversarial (N3-3, a
    reference-only case): every token has exactly ONE expert of the EP rank's `local` range, taken from a set of 2, and
    k-1 experts of the other rank (mostly-masked rows), and every odd token repeats the previous token's ids."""
    rng = random.Random(1000003 * int(seed) + 7919 * int(T) + {"independent": 0, "request": 17, "zipf": 29, "hot": 41,
                                                               "adversarial": 53}[mode])
    zw = _zipf_weights(E, seed) if mode == "zipf" else None
    hs = hot_set(E, seed) if mode == "hot" else None
    lo, hi = int(local[0]), int(local[1])
    out, cur = [], None
    for t in range(int(T)):
        if mode == "adversarial":
            if t % 2 == 0:
                far = [e for e in range(E) if not lo <= e < hi]
                cur = [lo + rng.randrange(2)] + rng.sample(far, k - 1)
                rng.shuffle(cur)
        elif mode == "hot":
            cur = rng.sample(hs, k)
        elif mode == "zipf":
            keys = sorted(range(E), key=lambda e: -(rng.random() ** (1.0 / zw[e])))
            cur = keys[:k]
        elif mode == "independent" or t % max(1, int(group)) == 0:
            cur = rng.sample(range(E), k)
        out.append(list(cur))
    return out


def routing_weights(T, k=10, seed=0):
    """Positive router weights per token, normalised to 1 (float lists)."""
    rng = random.Random(7 * int(seed) + int(T))
    out = []
    for _ in range(int(T)):
        w = [rng.random() + 0.05 for _ in range(k)]
        s = sum(w)
        out.append([v / s for v in w])
    return out


def pp2_split_cases(Cs=(8, 16, 24, 32), K=3):
    """PLAN 1.1 split-read hypothesis: C resident streams at K3 -> combined batch T = C (1+K) tokens vs the two PP2
    micro-batches of T/2 = (C/2)(1+K) tokens each, with the SAME routing ids."""
    return [{"C": int(c), "K": int(K), "T": int(c) * (1 + int(K)), "T_half": int(c) * (1 + int(K)) // 2} for c in Cs]


def split_halves(T):
    """Row ranges of the two micro-batches: [0, T/2) and [T/2, T) (the concatenation restores the combined order)."""
    h = int(T) // 2
    return [(0, h), (h, int(T))]


def split_merge(fn, x, ids, w, T):
    """torch: run fn on each half (rows of x / ids / w), concatenate in order -> the split result for the same tokens."""
    import torch
    outs = [fn(x[a:b].contiguous(), ids[a:b].contiguous(), w[a:b].contiguous()) for a, b in split_halves(T)]
    return torch.cat(outs, 0)


# N3-3: the split/combined (and baseline/tuned) gate is a REAL elementwise BF16 ULP check: every element within
# EQUIV_MAX_ULP units in the last place (distance on the ordered bf16 bit pattern), except elements whose absolute
# difference is below a cancellation floor of EQUIV_FLOOR_ULP_OF_RMS bf16 ULPs at the reference RMS magnitude (near-zero
# outputs of a sum, where the ULP distance is meaningless); plus the rel_fro / max_abs_over_ref_max norms.
EQUIV_MAX_ULP = 2
EQUIV_FLOOR_ULP_OF_RMS = 2
EQUIV_TOL = {"max_ulp": EQUIV_MAX_ULP, "floor": f"{EQUIV_FLOOR_ULP_OF_RMS} bf16 ULP of the reference RMS (cancellation floor)",
             "rel_fro": 8e-3, "max_abs_over_ref_max": 1.6e-2}


def bf16_ulp(v):
    """Size of one bf16 ULP at magnitude v (8 significant bits: 2^(floor(log2 v) - 7))."""
    v = abs(float(v))
    if not math.isfinite(v) or v <= 0.0:
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(v)) - 7)


def bf16_ordered(t):
    """torch: bf16 tensor -> int32 on a monotone integer line (sign-magnitude bit pattern -> +/- magnitude; +0 == -0),
    so |ordered(a) - ordered(b)| is the elementwise ULP distance."""
    import torch
    b = t.contiguous().view(torch.int16).to(torch.int32)
    return torch.where(b < 0, -(b & 0x7FFF), b)


def _kth_abs(d, q):
    n = d.numel()
    if not n:
        return 0.0
    return float(d.flatten().kthvalue(max(1, min(n, int(math.ceil(q * n)))))[0])


def equivalence(a, b):
    """torch: candidate (a) vs reference output (b) of the same tokens (split vs combined, tuned vs baseline): bitwise
    flag, finite, max / p99 abs, max rel, rel Frobenius and the elementwise BF16 ULP distance; 'equivalent' = bitwise,
    or (bf16 tensors) no element beyond EQUIV_MAX_ULP outside the cancellation floor and the norms within EQUIV_TOL."""
    import torch
    if tuple(a.shape) != tuple(b.shape):
        return {"bitwise_equal": False, "shape_equal": False, "equivalent": False, "shapes": [list(a.shape), list(b.shape)],
                "tol": EQUIV_TOL}
    af, bf = a.float(), b.float()
    d = (af - bf).abs()
    ref_max = float(bf.abs().max().clamp_min(1e-30)) if d.numel() else 1e-30
    finite = bool(torch.isfinite(af).all() and torch.isfinite(bf).all())
    out = {"bitwise_equal": bool(torch.equal(a, b)), "shape_equal": True, "finite": finite,
           "max_abs": float(d.max()) if d.numel() else 0.0, "p99_abs": _kth_abs(d, 0.99),
           "max_rel": float((d / bf.abs().clamp_min(1e-6)).max()) if d.numel() else 0.0,
           "rel_fro": float(d.norm() / bf.norm().clamp_min(1e-30)) if d.numel() else 0.0,
           "max_abs_over_ref_max": (float(d.max()) / ref_max) if d.numel() else 0.0, "tol": EQUIV_TOL}
    norms_ok = out["rel_fro"] <= EQUIV_TOL["rel_fro"] and out["max_abs_over_ref_max"] <= EQUIV_TOL["max_abs_over_ref_max"]
    if a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16 and d.numel() and finite:
        ud = (bf16_ordered(a) - bf16_ordered(b)).abs()
        floor = EQUIV_FLOOR_ULP_OF_RMS * bf16_ulp(float(bf.pow(2).mean().sqrt()))
        over = (ud > EQUIV_MAX_ULP) & (d > floor)
        out.update(ulp_check="bf16 elementwise", max_ulp=int(ud.max()), p99_ulp=_kth_abs(ud.float(), 0.99),
                   n_over_max_ulp=int(over.sum()), cancellation_floor_abs=floor)
        num_ok = out["n_over_max_ulp"] == 0 and norms_ok
    else:
        out["ulp_check"] = "not run (not two finite bf16 tensors): norms only"
        num_ok = norms_ok
    out["equivalent"] = out["finite"] and (out["bitwise_equal"] or num_ok)
    return out


def metrics_x(torch, out, ref):
    """P4 metrics (finite, max_abs, max_abs_over_ref_max, rel_fro, cos) + p99_abs and the reference magnitude (N3-3)."""
    m = metrics(torch, out, ref)
    err = (out.float() - ref).abs()
    m.update(p99_abs=_kth_abs(err, 0.99), ref_max=float(ref.abs().max()) if ref.numel() else 0.0, n=int(ref.numel()))
    return m


def timing(tm):
    """N3-12: a bench() result -> eager_us and graph_us kept APART (never substituted), graph_status PASS /
    INCONCLUSIVE_CAPTURE_FAILED and graph_error. Only graph_us feeds a graph winner."""
    g = tm.get("graph_us") if isinstance(tm, dict) else None
    return {"eager_us": (tm or {}).get("eager_us"), "graph_us": g, "graph_status": "PASS" if g else "INCONCLUSIVE_CAPTURE_FAILED",
            "graph_error": (tm or {}).get("graph_error")}


def parse_ncu_csv(text):
    """Nsight Compute --csv --print-units base output -> [{"id", "kernel", metric: value}] in launch order."""
    import csv
    import io
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith('"ID"')), None)
    if start is None:
        return []
    rows = {}
    for r in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        try:
            k = int(r["ID"])
            v = float(str(r.get("Metric Value", "")).replace(",", ""))
        except Exception:
            continue
        d = rows.setdefault(k, {"id": k, "kernel": (r.get("Kernel Name") or "")[:120]})
        d[r.get("Metric Name")] = d.get(r.get("Metric Name"), 0.0) + v
    return [rows[k] for k in sorted(rows)]


NCU_METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum")
NCU_UNVERIFIED = "DRAM_BYTES_UNVERIFIED"


def ncu_arm_bytes(rows, metrics=NCU_METRICS):
    """N3-1 / N3-20: ONE arm (split or combined) profiled in its OWN ncu child run -> the sum over ALL its kernels
    (whatever their number: moe_align / Marlin may launch a different count at T/2 and T). A run with no kernel, or a
    missing / non-finite / negative metric on any kernel, is never 0 bytes: ok False, DRAM_BYTES_UNVERIFIED."""
    if not rows:
        return {"ok": False, "status": NCU_UNVERIFIED, "why": "NO_KERNEL_ROWS", "kernels": 0}
    out = {"ok": True, "status": "MEASURED", "kernels": len(rows), "kernel_names": sorted({str(r.get("kernel")) for r in rows})[:8]}
    for m in metrics:
        vals = [r.get(m) for r in rows]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in vals):
            out.update(ok=False, status=NCU_UNVERIFIED, why=f"MISSING_OR_BAD_{m}")
            out[m] = None
        else:
            out[m] = float(sum(vals))
    return out


def ncu_run_verdict(rc, stdout, stderr="", metrics=NCU_METRICS):
    """One ncu child run (one arm): counter permission denied / nonzero exit / no rows / bad metric -> UNVERIFIED."""
    text = (stdout or "") + (stderr or "")
    if "ERR_NVGPUCTRPERM" in text:
        return {"ok": False, "status": NCU_UNVERIFIED, "why": "ERR_NVGPUCTRPERM: no access to the GPU performance counters", "rc": rc}
    if rc != 0:
        return {"ok": False, "status": NCU_UNVERIFIED, "why": f"NCU_EXIT_{rc}", "rc": rc}
    return {**ncu_arm_bytes(parse_ncu_csv(stdout or ""), metrics), "rc": rc}


def ncu_split_combined(split, combined, metrics=NCU_METRICS):
    """The two arms side by side; measured bytes only when BOTH arms are MEASURED (never a mixed / zero-filled pair)."""
    ok = (split or {}).get("ok") is True and (combined or {}).get("ok") is True
    out = {"ok": ok, "status": "MEASURED" if ok else NCU_UNVERIFIED, "split": split, "combined": combined,
           "why": None if ok else "; ".join(f"{n}: {(v or {}).get('why')}" for n, v in (("split", split), ("combined", combined))
                                            if not (v or {}).get("ok"))}
    for m in metrics:
        out["split_" + m] = split.get(m) if ok else None
        out["combined_" + m] = combined.get(m) if ok else None
    return out


# ------------------------------------------------------------------ real routing replay (NB1 output, PLAN v3.6)
ROUTING_FILE_NAMES = ("nb1_routing_ids.npz", "nb1_routing_ids.jsonl.gz")
ROUTING_SCHEMA = ("npz: ids int [sum of tokens, top_k] + per-record arrays layer, stage, mb_size, step (+ optional offset = "
                  "first row of the record; default cumulative mb_size); jsonl.gz: one JSON record per line {layer, stage, "
                  "mb_size, step, ids: [[top_k ids] per token]} (aliases: micro_batch_size, topk_ids)")


def load_routing(path, E=512, k=10):
    """NB1's real router top-k ids -> [{"layer", "stage", "mb_size", "step", "ids"}] (rows outside [0, E) or not k wide
    are dropped and counted). Returns (records, info)."""
    path = Path(path)
    recs, bad = [], 0
    if path.name.endswith(".npz"):
        import numpy as np
        z = np.load(str(path), allow_pickle=False)
        ids = z["ids"]
        n = len(z["mb_size"])
        off = z["offset"] if "offset" in z.files else None
        cur = 0
        for i in range(n):
            m = int(z["mb_size"][i])
            st = int(off[i]) if off is not None else cur
            rows = ids[st:st + m].tolist()
            cur = st + m
            recs.append({"layer": int(z["layer"][i]), "stage": int(z["stage"][i]), "mb_size": m, "step": int(z["step"][i]), "ids": rows})
    else:
        import gzip
        with gzip.open(str(path), "rt") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                rows = r.get("ids", r.get("topk_ids")) or []
                recs.append({"layer": int(r.get("layer", -1)), "stage": int(r.get("stage", -1)),
                             "mb_size": int(r.get("mb_size", r.get("micro_batch_size", len(rows)))), "step": int(r.get("step", -1)),
                             "ids": rows})
    out = []
    for r in recs:
        ok = [list(map(int, row)) for row in r["ids"] if len(row) == k and all(0 <= int(e) < E for e in row)]
        bad += len(r["ids"]) - len(ok)
        if ok:
            out.append({**r, "ids": ok, "mb_size": len(ok)})
    info = {"path": str(path), "records": len(out), "rows_dropped": bad, "tokens": sum(r["mb_size"] for r in out),
            "mb_sizes": sorted({r["mb_size"] for r in out}), "layers": sorted({r["layer"] for r in out}),
            "stages": sorted({r["stage"] for r in out}), "schema": ROUTING_SCHEMA}
    return out, info


def select_replay(recs, per_size=6, max_sizes=12):
    """Up to `per_size` records per micro-batch size (spread over layers / steps), the most frequent sizes first."""
    by = {}
    for r in recs:
        by.setdefault(r["mb_size"], []).append(r)
    sizes = sorted(by, key=lambda m: (-len(by[m]), m))[:max_sizes]
    out = {}
    for m in sorted(sizes):
        rs = sorted(by[m], key=lambda r: (r["layer"], r["step"]))
        stride = max(1, len(rs) // per_size)
        out[m] = rs[::stride][:per_size]
    return out


def routing_stats(recs, lo=0, hi=256, E=512, k=10):
    """Per micro-batch size: distinct experts (total / local to the EP rank) vs the uniform model D(T)."""
    by = {}
    for r in recs:
        by.setdefault(r["mb_size"], []).append(r)
    rows = []
    for m in sorted(by):
        dt = [distinct(r["ids"]) for r in by[m]]
        dl = [distinct(r["ids"], lo, hi) for r in by[m]]
        de = d_expected(m, E, k)
        rows.append({"T": m, "records": len(by[m]), "distinct_total_mean": round(sum(dt) / len(dt), 2),
                     "distinct_local_mean": round(sum(dl) / len(dl), 2), "d_uniform": round(de, 2),
                     "real_over_uniform": round(sum(dt) / len(dt) / de, 4) if de else None})
    return rows


def request_group(T, prefer=(4, 3, 2)):
    for g in prefer:
        if T % g == 0:
            return g
    return 1


def distinct(ids, lo=None, hi=None):
    s = {e for row in ids for e in row if (lo is None or e >= lo) and (hi is None or e < hi)}
    return len(s)


def stats_spread(st):
    """(p90 - p10) / median of a stats dict (P4 schema), None when missing."""
    try:
        return (st["p90"] - st["p10"]) / st["median"] if st and st.get("median") else None
    except Exception:
        return None


def med(st):
    return (st or {}).get("median") if isinstance(st, dict) else None


# ---------------------------------------------------------------------------------- compressed-tensors INT4 (torch)
def ct_pack_int4(q):
    """torch: int tensor [N, K] in [-8, 7] -> int32 [N, K // 8] the compressed-tensors 'pack-quantized' way
    (value + 8, eight consecutive K values per int32, LSB first): the layout vLLM transposes into GPTQ [K/8, N]."""
    import torch
    N, K = q.shape
    u = (q.to(torch.int64) + 8).view(N, K // 8, 8)
    shifts = torch.arange(8, dtype=torch.int64, device=q.device) * 4
    packed = (u << shifts).sum(-1)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed)
    return packed.to(torch.int32)


def ct_dequant_int4(packed, scale, group=128):
    """torch: compressed-tensors int32 [N, K/8] + scale [N, K/group] -> float32 [N, K] = (nibble - 8) x scale."""
    import torch
    N, Kp = packed.shape
    shifts = torch.arange(8, dtype=torch.int32, device=packed.device) * 4
    nib = (packed.unsqueeze(-1) >> shifts) & 0xF
    w = nib.reshape(N, Kp * 8).to(torch.float32) - 8.0
    return w * scale.to(torch.float32).repeat_interleave(group, dim=1)


def moe_reference(x, ids, w, expert_fp32, lo, hi):
    """torch fp32 reference of one EP rank's routed-expert output: sum over the token's LOCAL experts
    (lo <= e < hi) of w_tk * down(silu(gate x) * up x). expert_fp32(e) -> (gate [I, H], up [I, H], down [H, I]) fp32.
    Non-local experts contribute 0 here (the TP all-reduce adds the other rank's part)."""
    import torch
    xf = x.float()
    out = torch.zeros_like(xf)
    for e in sorted({int(v) for v in ids.flatten().tolist() if lo <= int(v) < hi}):
        t, j = (ids == e).nonzero(as_tuple=True)
        g, u, d = expert_fp32(e)
        h = torch.nn.functional.silu(xf[t] @ g.T) * (xf[t] @ u.T)
        out.index_add_(0, t, (h @ d.T) * w[t, j].float().unsqueeze(1))
    return out


# --------------------------------------------------------------------------------- P4 helpers (cut by the builder)
# __P4_HELPERS__


# ------------------------------------------------------------------------------------------------- result tables
def _gbps(nbytes, us):
    return round(nbytes / us / 1e3, 1) if isinstance(nbytes, (int, float)) and us else None


def d5_table(j):
    """D5 JSON -> rows. baseline: one row per (T, routing mode); sweep: the best MERLIN_MARLIN_MOE_CFG rule per T.
    Every time column is a CUDA-graph median (N3-12: an eager number is never shown as a graph number; graph_status
    says whether the capture worked); every GB/s from DERIVED bytes is named derived_gbps, the ncu-measured one
    dram_gbps (N3-4)."""
    base, sweep, adv = [], [], []
    for T, modes in sorted(((int(t), m) for t, m in ((j or {}).get("baseline") or {}).items()), key=lambda x: x[0]):
        for mode, r in sorted(modes.items()):
            dg = r.get("derived_gbps")
            base.append({"T": T, "mode": mode, "distinct_local": r.get("distinct_local"),
                         "d_expected_local": r.get("d_expected_local"), "graph_us": med(r.get("graph_us")),
                         "graph_status": r.get("graph_status"), "graph_error": r.get("graph_error"),
                         "hot_graph_us": med(r.get("hot_graph_us")), "eager_us": med(r.get("eager_us")),
                         "align_graph_us": med(r.get("align_graph_us")), "derived_gbps": dg,
                         "derived_gbps_vs_rc2_implied_173": round(dg / RC2_FIT["implied_moe_gbps"], 3) if dg else None,
                         "derived_gbps_vs_l4_peak_300": round(dg / L4_PEAK_GBPS, 3) if dg else None,
                         "ref_ok": r.get("ref_ok"), "spread": stats_spread(r.get("graph_us")), "error": r.get("error")})
    for T, r in sorted(((int(t), r) for t, r in ((j or {}).get("adversarial") or {}).items()), key=lambda x: x[0]):
        adv.append({"T": T, "distinct_local": r.get("distinct_local"), "ref_ok": r.get("ref_ok"),
                    "rel_fro": (r.get("ref") or {}).get("rel_fro"), "max_abs_over_ref_max": (r.get("ref") or {}).get("max_abs_over_ref_max"),
                    "error": r.get("error")})
    for T, s in sorted(((int(t), s) for t, s in ((j or {}).get("sweep") or {}).items()), key=lambda x: x[0]):
        b = s.get("best") or {}
        sweep.append({"T": T, "base_us": s.get("base_graph_us"), "base_graph_status": s.get("base_graph_status"),
                      "base_spread": s.get("base_spread"),
                      "best_rule": b.get("rule"), "best_us": b.get("us"), "best_spread": b.get("spread"),
                      "speedup": (round(s["base_graph_us"] / b["us"], 3) if b.get("us") and s.get("base_graph_us") else None),
                      "best_bitwise_vs_base": b.get("bitwise_vs_base"), "best_ref_ok": b.get("ref_ok"),
                      "best_equivalent_vs_base": b.get("equivalent_vs_base"), "best_max_ulp_vs_base": b.get("max_ulp_vs_base"),
                      "budget_stop": bool(s.get("budget_stop")),
                      "rules_tried": len(s.get("rules") or []), "rules_failed": sum(1 for r in (s.get("rules") or []) if r.get("error")),
                      "rules_graph_inconclusive": sum(1 for r in (s.get("rules") or []) if r.get("graph_status") == "INCONCLUSIVE_CAPTURE_FAILED")})
    split = []
    for c, sp in sorted(((int(c), v) for c, v in ((j or {}).get("split") or {}).items()), key=lambda x: x[0]):
        a_us, b_us = med(sp.get("split_graph_us")), med(sp.get("combined_graph_us"))
        eq = sp.get("equivalence") or {}
        nc = sp.get("ncu") or {}
        meas = nc.get("ok") is True
        split.append({"C": c, "T": sp.get("T"), "T_half": sp.get("T_half"), "split_us": a_us, "combined_us": b_us,
                      "graph_status": sp.get("graph_status"),
                      "split_over_combined": round(a_us / b_us, 3) if a_us and b_us else None,
                      "split_spread": stats_spread(sp.get("split_graph_us")), "combined_spread": stats_spread(sp.get("combined_graph_us")),
                      "distinct_local_split_sum": sp.get("distinct_local_split_sum"), "distinct_local_combined": sp.get("distinct_local_combined"),
                      "bytes_split": sp.get("bytes_split"), "bytes_combined": sp.get("bytes_combined"),
                      "bytes_source": "DERIVED (distinct local experts x expert bytes)",
                      "ncu_status": nc.get("status") or (NCU_UNVERIFIED if nc else None), "ncu_why": nc.get("why"),
                      "ncu_read_split": nc.get("split_dram__bytes_read.sum") if meas else None,
                      "ncu_read_combined": nc.get("combined_dram__bytes_read.sum") if meas else None,
                      "dram_gbps_split": _gbps(nc.get("split_dram__bytes_read.sum"), a_us) if meas else None,
                      "dram_gbps_combined": _gbps(nc.get("combined_dram__bytes_read.sum"), b_us) if meas else None,
                      "derived_gbps_split": sp.get("derived_gbps_split"), "derived_gbps_combined": sp.get("derived_gbps_combined"),
                      "bitwise_equal": eq.get("bitwise_equal"), "equivalent": eq.get("equivalent"), "max_abs": eq.get("max_abs"),
                      "p99_abs": eq.get("p99_abs"), "max_ulp": eq.get("max_ulp"), "n_over_max_ulp": eq.get("n_over_max_ulp"),
                      "max_rel": eq.get("max_rel"), "rel_fro": eq.get("rel_fro"),
                      "ref_ok_split": sp.get("ref_ok_split"), "ref_ok_combined": sp.get("ref_ok_combined"), "error": sp.get("error")})
    rp = (j or {}).get("replay") or {}
    replay = {"available": rp.get("available", False), "why": rp.get("why"), "info": rp.get("info"),
              "distinct_vs_uniform": rp.get("distinct_vs_uniform"),
              "rows": [{k: x.get(k) for k in ("T", "records", "graph_us_mean", "distinct_local_mean", "distinct_total_mean",
                                              "d_uniform", "derived_gbps_mean", "ref_ok", "split_over_combined_mean", "split_equivalent")}
                       for x in rp.get("by_size") or []]}
    return {"baseline": base, "sweep": sweep, "split": split, "replay": replay, "adversarial": adv,
            "ep1_spot": (j or {}).get("ep1_spot"), "humming": humming_table((j or {}).get("humming")),
            "units": {"done": len((j or {}).get("units_done") or []), "skipped": (j or {}).get("units_skipped") or []}}


def humming_table(h):
    """C6-humming (structural track, research): one row per arm and T; graph and eager kept apart (N3-12)."""
    if not h:
        return {"status": "NOT_RUN", "rows": []}
    rows = []
    for name, rec in (h.get("arms") or {}).items():
        for T, r in sorted(((int(t), r) for t, r in (rec.get("per_T") or {}).items()), key=lambda x: x[0]):
            rows.append({"arm": name, "bits": rec.get("bits"), "group": rec.get("group"), "T": T,
                         "graph_us": med(r.get("graph_us")), "eager_us": med(r.get("eager_us")), "graph_status": r.get("graph_status"),
                         "marlin_w4_graph_us": med(r.get("marlin_graph_us")), "marlin_w4_eager_us": med(r.get("marlin_eager_us")),
                         "ratio_vs_marlin_w4_graph": r.get("ratio_vs_marlin_w4_graph"), "ratio_vs_marlin_w4_eager": r.get("ratio_vs_marlin_w4_eager"),
                         "gate": r.get("gate"), "ref_ok_own_weights": r.get("ref_ok"), "derived_bytes": r.get("derived_bytes"),
                         "derived_bytes_marlin_w4": r.get("derived_bytes_marlin_w4"), "derived_gbps": r.get("derived_gbps"),
                         "first_call_sec": r.get("first_call_sec")})
    return {"status": h.get("status"), "available": h.get("available"), "info": h.get("info"), "label": h.get("label"),
            "gate": h.get("gate"), "summary": h.get("summary"), "rows": rows,
            "errors": {n: r.get("error") for n, r in (h.get("arms") or {}).items() if r.get("error")}}


def merge_d5(core, rest):
    """The two D5 bench files (core = T16/T4/T8 first, rest = everything else; N3-19 run order) -> one D5 JSON for
    the tables. Each file keeps its own status / checks for the eligibility rule (bench keys D5 and D5_rest)."""
    core, rest = core or {}, rest or {}
    out = {"baseline": {}, "sweep": {}, "split": {}, "adversarial": {}, "units_done": [], "units_skipped": []}
    for src in (core, rest):
        for T, modes in (src.get("baseline") or {}).items():
            out["baseline"].setdefault(T, {}).update(modes)
        for k in ("sweep", "split", "adversarial"):
            out[k].update(src.get(k) or {})
        out["units_done"] += src.get("units_done") or []
        out["units_skipped"] += src.get("units_skipped") or []
    rc, rr = core.get("replay") or {}, rest.get("replay") or {}
    out["replay"] = dict(rc or rr)
    if rc.get("available") or rr.get("available"):
        out["replay"].update(available=True, by_size=sorted((rc.get("by_size") or []) + (rr.get("by_size") or []), key=lambda b: b["T"]))
    out["ep1_spot"] = rest.get("ep1_spot") or core.get("ep1_spot")
    out["humming"] = core.get("humming") or rest.get("humming")
    out["status"] = {"core": core.get("status"), "rest": rest.get("status")}
    for k in ("model", "load", "dequant_crosscheck", "imports", "ncu", "preflight", "first_call_gate"):
        out[k] = core.get(k) if core.get(k) is not None else rest.get(k)
    return out


def _ar_group(j, name):
    return (((j or {}).get("groups") or {}).get(name) or {})


def _ar_checks(j, prefix):
    out = {}
    for g, v in ((j or {}).get("groups") or {}).items():
        for k, ok in ((v.get("rank0") or {}).get("checks") or {}).items():
            if g.startswith(prefix) or prefix == "*":
                out[f"{g}.{k}"] = ok
        for rk, o in (v.get("other_ranks") or {}).items():
            for k, ok in (o.get("checks") or {}).items():
                if g.startswith(prefix) or prefix == "*":
                    out[f"{g}.r{rk}.{k}"] = ok
    return out


# N3-14: a D7 arm PASSES only with BOTH pairs, BOTH ranks and every mandatory check of that arm True on each.
AR_PAIRS = ("0,1", "2,3")
AR_RANKS = (0, 1)
AR_MANDATORY = {
    "nccl": {"main_ab": ("n_nccl_identical_all_ranks", "n_nccl_within_rounding_bound")},
    "p5": {"main_ab": ("a_bitexact_vs_ordered_fp32_ref", "a_identical_all_ranks", "a_no_kernel_errors", "b_no_kernel_errors"),
           "main_cd": ("c_graph_replays_bitexact", "c_eager_between_replays", "c_error_path_loud")},
    "p5_int8": {"main_cd": ("d_int8_identical_all_ranks", "d_int8_within_bound_and_nonfinite_groups_nan", "d_no_kernel_errors")},
}
LL128_LABEL = "UNTESTED (not run, NCCL docs corruption risk)"     # N3-5: forced NCCL_PROTO=LL128 is never run
CONCURRENCY_MIN_OVERLAP = 0.5      # N3-17: per size, the two pairs' timed regions must overlap >= 50 % of the longer one


def _rank_checks(j, group, rank):
    g = _ar_group(j, group)
    if rank == 0:
        return (g.get("rank0") or {}).get("checks")
    o = (g.get("other_ranks") or {})
    return (o.get(str(rank)) or o.get(rank) or {}).get("checks")


def ar_arm_verdict(j, arm):
    """One pair's verdict for one arm: FAIL (a mandatory check False) > INCONCLUSIVE_MISSING_RANK (a rank recorded no
    check in a group of this arm) > INCONCLUSIVE_MISSING_CHECK (a mandatory check absent) > PASS."""
    if not j or not (j.get("groups") or {}):
        return {"verdict": "INCONCLUSIVE_MISSING_PAIR", "missing": ["no bench JSON / no groups"]}
    missing, failed = [], []
    rank_missing = False
    for group, names in AR_MANDATORY[arm].items():
        for rk in AR_RANKS:
            ch = _rank_checks(j, group, rk)
            if not ch:
                rank_missing = True
                missing.append(f"{group}.r{rk} (no checks)")
                continue
            for n in names:
                if n not in ch:
                    missing.append(f"{group}.r{rk}.{n}")
                elif ch[n] is not True:
                    failed.append(f"{group}.r{rk}.{n}")
    v = ("FAIL" if failed else "INCONCLUSIVE_MISSING_RANK" if rank_missing else
         "INCONCLUSIVE_MISSING_CHECK" if missing else "PASS")
    return {"verdict": v, "missing": missing[:12], "failed": failed[:12]}


def ar_numeric(j):
    """Correctness per arm from the bench's own checks (mandatory sets AR_MANDATORY, ranks 0 and 1): p5 = a (bit-exact
    vs ordered FP32, identical on all ranks, no kernel error) + b + c (graph replays, eager between, loud error path);
    p5_int8 = d (analytic bound); nccl = main_ab 'n' checks; ll128 = never run (UNTESTED label)."""
    out = {arm: ar_arm_verdict(j, arm)["verdict"] for arm in AR_MANDATORY}
    out["detail"] = {arm: ar_arm_verdict(j, arm) for arm in AR_MANDATORY}
    out["ll128"] = LL128_LABEL
    out["checks"] = _ar_checks(j, "*")
    return out


def arm_verdict_all_pairs(numeric, arm):
    """Both pairs required: a missing pair -> INCONCLUSIVE_MISSING_PAIR; any FAIL -> FAIL; any inconclusive -> that."""
    vs = []
    for p in AR_PAIRS:
        n = (numeric or {}).get(p)
        vs.append(n.get(arm) if isinstance(n, dict) else "INCONCLUSIVE_MISSING_PAIR")
    if "FAIL" in vs:
        return "FAIL"
    for v in ("INCONCLUSIVE_MISSING_PAIR", "INCONCLUSIVE_MISSING_RANK", "INCONCLUSIVE_MISSING_CHECK"):
        if v in vs:
            return v
    return "PASS" if all(v == "PASS" for v in vs) else "UNTESTED"


def _lat(j, nb):
    return (((_ar_group(j, "main_ab").get("rank0") or {}).get("b") or {}).get("latency") or {}).get(str(nb)) or {}


def _window(j, nb, arm):
    w = (_lat(j, nb).get("t_mono") or {}).get(arm)
    if isinstance(w, (list, tuple)) and len(w) == 2 and all(isinstance(v, (int, float)) for v in w) and w[1] > w[0]:
        return [float(w[0]), float(w[1])]
    return None


def pair_concurrency(pairs, sizes, threshold=CONCURRENCY_MIN_OVERLAP):
    """N3-17: per payload size and arm (nccl / host), the overlap of the two pairs' TIMED regions (rank-0 monotonic
    timestamps around that size's timing) over the longer one. Every size of both arms must have both windows and an
    overlap >= threshold, else CONCURRENCY_UNVERIFIED (the 'concurrent' numbers may be solo numbers)."""
    per, missing = {}, []
    for nb in sorted(sizes):
        for arm in ("nccl", "host"):
            ws = [_window((pairs or {}).get(p), nb, arm) for p in AR_PAIRS]
            if any(w is None for w in ws):
                per[f"{nb}.{arm}"] = None
                missing.append(f"{nb}.{arm}")
                continue
            inter = max(0.0, min(w[1] for w in ws) - max(w[0] for w in ws))
            per[f"{nb}.{arm}"] = round(inter / max(1e-9, max(w[1] - w[0] for w in ws)), 3)
    vals = [v for v in per.values() if v is not None]
    if not per or missing:
        st, why = "CONCURRENCY_UNVERIFIED", f"timed-region timestamps missing for {missing[:6] or 'every size'}"
    elif min(vals) < threshold:
        st, why = "CONCURRENCY_UNVERIFIED", f"min overlap {min(vals)} < {threshold}"
    else:
        st, why = "VERIFIED", None
    return {"status": st, "why": why, "min_overlap": min(vals) if vals else None, "threshold": threshold, "per_size_arm": per}


def d7_table(pairs, hidden=2560, dtype_bytes=2, solo=None):
    """pairs: {"0,1": bench_host_ar JSON, "2,3": ...} (run concurrently); solo: the (0,1) pair alone. One row per payload
    size: T, NCCL (RC2 env), P5 host-staged native + INT8 per pair, the worse pair (a TP2 collective is as slow as its
    slowest pair under PP2; a row value needs BOTH pairs) and the solo NCCL for the interference delta. LL128: UNTESTED."""
    rows, numeric = [], {}
    sizes = set()
    for p, j in (pairs or {}).items():
        numeric[p] = ar_numeric(j)
        sizes |= {int(s) for s in ((_ar_group(j, "main_ab").get("rank0") or {}).get("b") or {}).get("latency", {})}
    for nb in sorted(sizes):
        row = {"bytes": nb, "T": round(nb / (hidden * dtype_bytes), 3), "per_pair": {}}
        for p, j in (pairs or {}).items():
            lat = _lat(j, nb)
            d = ((((_ar_group(j, "main_cd").get("rank0") or {}).get("d") or {}).get("latency") or {}).get(str(nb)) or {})
            row["per_pair"][p] = {"nccl_graph": lat.get("nccl_graph"), "nccl_eager": lat.get("nccl_eager"),
                                  "p5_graph": lat.get("host_graph"), "p5_eager": lat.get("host_eager"),
                                  "p5_int8_graph": d.get("host_int8_graph")}
        for key in ("nccl_graph", "p5_graph", "p5_int8_graph"):
            vals = [row["per_pair"].get(p, {}).get(key) for p in AR_PAIRS]
            row[key] = max(vals) if all(isinstance(v, (int, float)) for v in vals) else None
        if solo:
            sl = _lat(solo, nb)
            row["solo_nccl_graph"] = sl.get("nccl_graph")
            row["solo_p5_graph"] = sl.get("host_graph")
        for key in ("p5_graph", "p5_int8_graph"):
            row["speedup_" + key] = (round(row["nccl_graph"] / row[key], 3)
                                     if row.get("nccl_graph") and row.get(key) else None)
        rows.append(row)
    overlap = None
    try:
        g = [(_ar_group(j, "main_ab").get("t0"), _ar_group(j, "main_ab").get("t1")) for j in pairs.values()]
        if len(g) == 2 and all(a and b for a, b in g):
            inter = max(0.0, min(g[0][1], g[1][1]) - max(g[0][0], g[1][0]))
            overlap = round(inter / max(1e-6, max(g[0][1] - g[0][0], g[1][1] - g[1][0])), 3)
    except Exception:
        pass
    transports = {p: (_ar_group(j, "main_ab").get("nccl_log") or {}).get("via_counts") for p, j in (pairs or {}).items()}
    return {"rows": rows, "numeric": numeric, "arms": {arm: arm_verdict_all_pairs(numeric, arm) for arm in AR_MANDATORY},
            "pairs_present": sorted(p for p, j in (pairs or {}).items() if j), "ll128": LL128_LABEL,
            "concurrency": pair_concurrency(pairs, sizes), "group_overlap_frac_info": overlap, "nccl_transports": transports}


def d4_table(j):
    rows = []
    for key, s in ((j or {}).get("shapes") or {}).items():
        for m, r in sorted(((int(m), r) for m, r in (s.get("M") or {}).items()), key=lambda x: x[0]):
            b = r.get("best") or {}
            cb = r.get("cublas") or {}
            rows.append({"shape": key, "label": s.get("label"), "M": m, "cublas_us": med(cb.get("graph_us")),
                         "cublas_graph_status": cb.get("graph_status"), "cublas_eager_us": med(cb.get("eager_us")),
                         "cublas_ok": cb.get("ok"), "best_source": b.get("source"), "best_cfg": b.get("cfg"),
                         "best_us": b.get("us"), "best_spread": b.get("spread"), "speedup": b.get("speedup_vs_cublas"),
                         "best_ok": b.get("ok"), "dispatch_ok": (r.get("dispatch") or {}).get("ok") if r.get("dispatch") else None,
                         "configs_tried": len(r.get("configs") or []),
                         "configs_graph_inconclusive": sum(1 for c in (r.get("configs") or []) if c.get("graph_status") == "INCONCLUSIVE_CAPTURE_FAILED"),
                         "compile_failures": sum(1 for c in (r.get("configs") or []) if c.get("error"))})
    dh = []
    dj = (j or {}).get("draft_heads") or {}
    for mode, v in (dj.get("modes") or {}).items():
        for m, r in sorted(((int(m), r) for m, r in (v.get("per_M") or {}).items()), key=lambda x: x[0]):
            dh.append({"mode": mode, "M": m, "us": med(r.get("graph_us")), "graph_status": r.get("graph_status"),
                       "spread": stats_spread(r.get("graph_us")),
                       "rel_fro_vs_bf16": (r.get("ref_vs_bf16") or {}).get("rel_fro"), "sanity_ok": v.get("sanity_ok")})
    return {"rows": rows, "plans": (j or {}).get("plans"), "draft_heads": dh, "sm89_skinny": (j or {}).get("sm89_skinny"),
            "draft_heads_status": dj.get("status"), "draft_heads_preflight": dj.get("preflight")}


# ------------------------------------------------------------------------------ per-tick projection (DERIVED, est.)
# Stage-1 BF16 linears that Merlin's low-latency GEMM dispatch would see (local TP2 shapes, sm_89 has no plan today):
# (N, K, label, calls per stage-1 tick at T_micro rows, calls per draft pass at r rows). DERIVED from config.json +
# vllm/models/qwen4_exp/nvidia (hyperconnection.py:100-134, qwen3_next.py:150-230 shared expert not replicated at
# TP2+EP for compressed-tensors, low_latency_gemm.py:376-412); call counts are estimates, recorded as such.
def gemm_shapes():
    s1 = stage_layers(1)
    return [
        (124160, 2560, "LM head TP2 shard (target logits)", 1, 0),
        (10240, 320, "HC up, replicated (2 HC modules per layer)", 2 * s1["layers"], 0),
        (336, 10240, "HC down + block inject merged (320+4+12 pad), replicated", 2 * s1["layers"], 0),
        (6656, 2560, "QSA fused QKV/gate TP2", s1["qsa"], 0),
        (2560, 3072, "QSA o_proj TP2", s1["qsa"], 0),
        (640, 2560, "shared expert gate/up TP2", s1["layers"], 0),
        (2560, 320, "shared expert down TP2 (no SM121 plan)", s1["layers"], 0),
        (512, 2560, "router (GateLinear: low-latency path unverified)", s1["layers"], 0),
        (1, 2560, "shared-expert gate", s1["layers"], 0),
        (48, 2560, "GDN in_proj_ba TP2 (BF16)", s1["gdn"], 0),
    ]


def tick_calls(seqs, K):
    """Stage-1 calls per tick (DERIVED): MoE / all-reduce in every target layer at T_micro rows, plus K MTP draft passes
    (1 MTP layer each) at r = ceil(seqs/2) rows; 2 TP all-reduces per layer (A7 GOOD 7)."""
    s1 = stage_layers(BINDING_STAGE)
    r = micro_requests(seqs)
    return {"T_micro": micro_T(seqs, K), "r": r, "K": int(K), "moe_target": s1["layers"], "moe_draft": int(K),
            "ar_target": 2 * s1["layers"], "ar_draft": 2 * int(K), "stage": BINDING_STAGE, "stage_layers": s1}


def rc2_step_ms(seqs):
    """RC2 fitted step (D-model) at C = seqs resident streams: 24.21 + 0.1758 D(4 r), r = C / 2 (est.)."""
    a0, a1 = RC2_FIT["dmodel"]
    return a0 + a1 * E_D(4.0 * seqs / 2.0)


def E_D(T, E=512, k=10):
    return E * (1.0 - (1.0 - k / E) ** float(T))


def steady_from_step(seqs, step_ms, a=None):
    """Throughput = C a / (2 step) (PP2: a request advances every 2 ticks), RC2 section 6.2."""
    return seqs * (a or RC2_FIT["a"]) / (2.0 * step_ms / 1000.0)


def projection(saved_ms, seqs=None):
    """Microbench-projected gain from ms saved per tick on the binding stage. Headline: the RC2 fitted step at `seqs`
    (D-model), steady = C a / (2 step) before and after; also the RC2 step_ms range 30-48 against the measured 8k@8
    261.8 tok/s: steady' = 261.8 t / (t - s). run ~ 0.9 x steady (PLAN section 1). MICROBENCH only."""
    seqs = int(seqs or RC2_STEADY["seqs"])
    out = {"inputs": {"saved_ms_per_tick": round(saved_ms, 4), "seqs": seqs, "step_ms": list(STEP_MS), "rc2_fit": RC2_FIT,
                      "rc2_steady_tok_s": RC2_STEADY["tok_s"], "rc2_point": {k: RC2_STEADY[k] for k in ("seqs", "K")},
                      "run_over_steady": RUN_OVER_STEADY,
                      "formula": ("headline: step = 24.21 + 0.1758 D(4 C/2), steady = C a / (2 step), a = 3.05, run gain = "
                                  "0.9 (steady(step - saved) - steady(step)); range: steady_new = 261.8 t / (t - saved)")},
           "label": "MICROBENCH_PROJECTED (never an engine gain until NB4 confirms)", "by_step_ms": {}}
    st = rc2_step_ms(seqs)
    if saved_ms < st:
        b, n = steady_from_step(seqs, st), steady_from_step(seqs, st - saved_ms)
        out["headline"] = {"step_ms_fit": round(st, 2), "share_of_step": round(saved_ms / st, 4), "steady_fit_tok_s": round(b, 1),
                           "steady_new_tok_s": round(n, 1), "run_tok_s_gain": round(RUN_OVER_STEADY * (n - b), 2),
                           "run_gain_frac": round((n - b) / b, 4)}
    else:
        out["headline"] = None
    out["run_tok_s_gain_headline"] = out["headline"]["run_tok_s_gain"] if out["headline"] else None
    for t in STEP_MS:
        if saved_ms >= t:
            out["by_step_ms"][str(t)] = None
            continue
        st = RC2_STEADY["tok_s"] * t / (t - saved_ms)
        out["by_step_ms"][str(t)] = {"share_of_tick": round(saved_ms / t, 4), "steady_tok_s": round(st, 1),
                                     "run_tok_s_gain": round(RUN_OVER_STEADY * (st - RC2_STEADY["tok_s"]), 2)}
    mid = out["by_step_ms"].get(str(STEP_MS[1]))
    out["run_tok_s_gain_mid"] = mid["run_tok_s_gain"] if mid else None
    out["run_gain_frac_mid"] = round(mid["run_tok_s_gain"] / (RUN_OVER_STEADY * RC2_STEADY["tok_s"]), 4) if mid else None
    return out


def _lookup(table, T, key="T"):
    for r in table:
        if r.get(key) == T:
            return r
    return None


def _confidence(base_spread, new_spread, rel_gain):
    sp = max([v for v in (base_spread, new_spread) if isinstance(v, (int, float))] or [None]) if \
        any(isinstance(v, (int, float)) for v in (base_spread, new_spread)) else None
    if rel_gain is None or rel_gain <= 0:
        return "none"
    if sp is not None and sp < 0.10 and rel_gain > 2 * sp:
        return "medium"
    return "low"


FAMILY = {"D5": "1_marlin_moe_routed_experts", "D4": "2_dense_gemm_and_vocab_heads", "D2": "2_dense_gemm_and_vocab_heads",
          "D7": "3_world2_collectives", "PP2": "1_marlin_moe_routed_experts"}
D5_CORE_T = (16, 4, 8)          # N3-16 / N3-19: the first D5 bench (core) carries the RC2-point token counts
# ----------------------------------------------------------------------------- the ONE promotion rule (N3-2/11/22)
PROMOTION_READY = "PROMOTION_READY"
BLOCKED_BENCH_FAILURE = "BLOCKED_BENCH_FAILURE"
BLOCKED_CORRECTNESS = "BLOCKED_CORRECTNESS"
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
PARTIAL_DIAGNOSTIC = "PARTIAL_DIAGNOSTIC"
ADDITIVITY_NOTE = ("microbench_saved_ms per lever: not additive under PP2 overlap (the two stages and micro-batches "
                   "overlap; a saving on one stage only counts when that stage binds); engine_confirmed_saved_ms stays "
                   "null until NB4 measures it end-to-end")
ADOPTION_RULE = ("N3-9 (NB4 rule): a GEMM-plan or P5 change is adopted only after a full-engine A/B (2 repeats) plus the "
                 "40k / 90k correctness rows; P5 additionally needs 1000+ graph replays and both-pair contention evidence "
                 "before integration")


def eligibility(c, status):
    """The ONE rule used by the NB4 ranking AND the scorecard. PROMOTION_READY only when: not info / hypothesis,
    the bench that produced the numbers has a VERIFIED status DONE (rc 0, no timeout, GPUs released, file status DONE,
    exit_code 0, every check True), correctness == "PASS" exactly (PASS_BOUND_ONLY is diagnostic), no diagnostic
    reason (concurrency unverified, ...), saved > 0. Otherwise BLOCKED_BENCH_FAILURE / PARTIAL_DIAGNOSTIC /
    BLOCKED_CORRECTNESS / DIAGNOSTIC_ONLY / NO_GAIN (raw numbers kept)."""
    if c.get("info_only"):
        return "INFO"
    if c.get("hypothesis"):
        return DIAGNOSTIC_ONLY
    bs = (status or {}).get(c.get("bench"))
    if bs != "DONE":
        return PARTIAL_DIAGNOSTIC if bs == "PARTIAL_BUDGET" else DIAGNOSTIC_ONLY if bs == "UNSUPPORTED_SM89" else BLOCKED_BENCH_FAILURE
    if c.get("correctness") != "PASS":
        return DIAGNOSTIC_ONLY if c.get("correctness") == "PASS_BOUND_ONLY" else BLOCKED_CORRECTNESS
    if c.get("diagnostic_reason"):
        return DIAGNOSTIC_ONLY
    if not (c.get("saved_ms_per_tick") or 0) > 0:
        return "NO_GAIN"
    return PROMOTION_READY


def verified_bench_status(r, res, gpu_released, allow_unknown_release=False):
    """N3-22: the parent's bench status from EVERYTHING it knows, never from the exit code alone. r = the parent's
    process record (rc, timeout); res = the bench's result file (status, exit_code, checks). DONE only when rc 0, no
    timeout, GPUs released, file status DONE, exit_code 0 and every check True. Returns (status, reasons)."""
    r, res = r or {}, res or {}
    why = []
    if r.get("timeout"):
        return "TIMEOUT", ["killed at its deadline"]
    rc, ec, fs, ch = r.get("rc"), res.get("exit_code"), res.get("status"), res.get("checks") or {}
    bad = sorted(k for k, v in ch.items() if v is not True)
    if not res:
        why.append("no result file")
    if rc != 0:
        why.append(f"rc {rc}")
    if res and ec != 0:
        why.append(f"exit_code {ec}")
    if res and fs != "DONE":
        why.append(f"file status {fs}")
    if res and not ch:
        why.append("no checks recorded")
    if bad:
        why.append(f"checks not True: {bad[:6]}")
    rel_ok = gpu_released is True or (gpu_released is None and allow_unknown_release)
    if not rel_ok:
        why.append(f"GPU release {gpu_released}")
    if not why:
        return "DONE", []
    if rc == 1 or ec == 1 or bad:
        return "CHECK_FAILED", why
    if rc != 0 or not res or ec != 0 or not ch:
        return "BENCH_ERROR", why
    if fs in ("PARTIAL_BUDGET", "UNSUPPORTED_SM89") and rel_ok:
        return fs, why
    if not rel_ok and fs == "DONE":
        return "GPU_NOT_RELEASED", why
    return "BENCH_ERROR", why


def candidates(tables, seqs=8, K=3):
    """NB4 patch / config candidates at one operating point: measured us per call at T_micro (and r for the MTP
    draft passes), calls per tick, ms saved per tick, correctness from the bench's own numeric checks, and `bench`
    = the bench whose verified status gates promotion (D5 = core T16/T4/T8, D5_rest, D7, D4)."""
    tc = tick_calls(seqs, K)
    T, r = tc["T_micro"], tc["r"]
    d5_bench = "D5" if T in D5_CORE_T and r in D5_CORE_T else "D5_rest"
    out = []
    d5 = tables.get("d5") or {}
    sw = d5.get("sweep") or []
    a, b = _lookup(sw, T), _lookup(sw, r)
    if a:
        per = (a["base_us"] - a["best_us"]) if a.get("best_us") and a.get("base_us") else None
        per_r = (b["base_us"] - b["best_us"]) if b and b.get("best_us") and b.get("base_us") else 0.0
        ok = (a.get("best_ref_ok") is True and a.get("best_equivalent_vs_base") is not False
              and (not b or (b.get("best_ref_ok") in (True, None) and b.get("best_equivalent_vs_base") is not False)))
        if per is not None:
            saved = (max(0.0, per) * tc["moe_target"] + max(0.0, per_r) * tc["moe_draft"]) / 1000.0
            out.append({"lever": "D5", "id": "D5_marlin_moe_cfg", "adoption": "config", "bench": d5_bench,
                        "what": f"MERLIN_MARLIN_MOE_CFG best rule at T={T}: {a.get('best_rule')}",
                        "T": T, "us_current": a["base_us"], "us_new": a["best_us"], "speedup": a.get("speedup"),
                        "calls_per_tick": {"T_micro": tc["moe_target"], "r": tc["moe_draft"]},
                        "saved_ms_per_tick": round(saved, 4), "correctness": "PASS" if ok else (
                            "FAIL" if a.get("best_ref_ok") is False or a.get("best_equivalent_vs_base") is False else "UNTESTED"),
                        "bitwise_vs_current": a.get("best_bitwise_vs_base"), "max_ulp_vs_current": a.get("best_max_ulp_vs_base"),
                        "confidence": _confidence(a.get("base_spread"), a.get("best_spread"),
                                                  per / a["base_us"] if per is not None and a["base_us"] else None)})
    base = [x for x in (d5.get("baseline") or []) if x.get("mode") == "independent"]
    m, mr = _lookup(base, T), _lookup(base, r)
    if m and m.get("graph_us"):
        tot = (m["graph_us"] * tc["moe_target"] + ((mr or {}).get("graph_us") or 0.0) * tc["moe_draft"]) / 1000.0
        out.append({"lever": "D5", "id": "D5_moe_term_info", "adoption": "research", "info_only": True, "bench": d5_bench,
                    "what": f"current Marlin MoE time per stage-1 tick at T={T} (independent routing, cold weights)",
                    "T": T, "us_current": m["graph_us"], "ms_per_tick": round(tot, 3), "derived_gbps": m.get("derived_gbps"),
                    "share_of_tick": {str(t): round(tot / t, 3) for t in STEP_MS}, "saved_ms_per_tick": 0.0,
                    "correctness": "PASS" if m.get("ref_ok") else ("FAIL" if m.get("ref_ok") is False else "UNTESTED"),
                    "confidence": "low"})
    sp = next((x for x in (d5.get("split") or []) if x.get("C") == seqs), None)
    if sp and sp.get("split_us") and sp.get("combined_us"):
        per = sp["split_us"] - sp["combined_us"]          # one stage-layer, both micro-batches (= one PP2 cycle of 2 ticks)
        ok = sp.get("equivalent") is True and sp.get("ref_ok_split") is True and sp.get("ref_ok_combined") is True
        out.append({"lever": "PP2", "id": "PP2_split_read", "adoption": "research", "hypothesis": True, "bench": "D5_rest",
                    "what": (f"PLAN 1.1 split-read: two PP2 micro-batches of {sp['T_half']} tokens vs one combined batch of "
                             f"{sp['T']} (same routing ids, same weights): a layout without the split would save this"),
                    "T": sp["T"], "us_current": sp["split_us"], "us_new": sp["combined_us"], "speedup": sp.get("split_over_combined"),
                    "calls_per_tick": {"stage_layers_per_cycle_of_2_ticks": tc["moe_target"]},
                    "saved_ms_per_tick": round(max(0.0, per) * tc["moe_target"] / 2.0 / 1000.0, 4),
                    "bytes": {"split": sp.get("bytes_split"), "combined": sp.get("bytes_combined"), "source": sp.get("bytes_source"),
                              "ncu_status": sp.get("ncu_status"), "ncu_read_split": sp.get("ncu_read_split"),
                              "ncu_read_combined": sp.get("ncu_read_combined")},
                    "correctness": "PASS" if ok else ("FAIL" if sp.get("equivalent") is False or sp.get("ref_ok_split") is False
                                                      or sp.get("ref_ok_combined") is False else "UNTESTED"),
                    "bitwise_equal": sp.get("bitwise_equal"),
                    "confidence": _confidence(sp.get("split_spread"), sp.get("combined_spread"), per / sp["split_us"]),
                    "note": "a hypothesis about the PP2 layout, not a patch: removing the split means another layout (research)"})
    d7 = tables.get("d7") or {}
    rows = d7.get("rows") or []
    arms = d7.get("arms") or {}
    conc = d7.get("concurrency") or {"status": "CONCURRENCY_UNVERIFIED", "why": "no D7 table"}

    def ar_row(Tq):
        best = None
        for x in rows:
            if abs((x.get("T") or 0) - Tq) < 1e-6:
                best = x
        return best
    ra, rb = ar_row(T), ar_row(r)
    for key, cid, adopt, what in (("p5_graph", "D7_p5_host_allreduce", "patch",
                                   "P5 host-staged all-reduce as the TP2 group all-reduce (custom AR integration)"),
                                  ("p5_int8_graph", "D7_p5_int8_allreduce", "patch",
                                   "P5 INT8-payload host all-reduce (needs a logprob gate before adoption)")):
        if not ra or not ra.get("nccl_graph") or not ra.get(key):
            continue
        per = ra["nccl_graph"] - ra[key]
        per_r = (rb["nccl_graph"] - rb[key]) if rb and rb.get("nccl_graph") and rb.get(key) else 0.0
        saved = (max(0.0, per) * tc["ar_target"] + max(0.0, per_r) * tc["ar_draft"]) / 1000.0
        arm = {"p5_graph": "p5", "p5_int8_graph": "p5_int8"}[key]
        corr = arms.get(arm) or "UNTESTED"
        if corr == "PASS" and arms.get("nccl") != "PASS":
            corr = "INCONCLUSIVE_NCCL_REFERENCE"     # the NCCL baseline itself was not verified on both pairs x ranks
        if arm == "p5_int8" and corr == "PASS":
            corr = "PASS_BOUND_ONLY"           # inside its analytic error bound; not bit-exact: logprob gate still owed
        out.append({"lever": "D7", "id": cid, "adoption": adopt, "what": what, "T": T, "bytes": ra["bytes"], "bench": "D7",
                    "us_current": ra["nccl_graph"], "us_new": ra[key], "speedup": ra.get("speedup_" + key),
                    "calls_per_tick": {"T_micro": tc["ar_target"], "r": tc["ar_draft"]},
                    "saved_ms_per_tick": round(saved, 4), "correctness": corr, "nccl_numeric": arms.get("nccl"),
                    "concurrency": conc.get("status"),
                    "diagnostic_reason": None if conc.get("status") == "VERIFIED" else f"CONCURRENCY_UNVERIFIED: {conc.get('why')}",
                    "confidence": "low" if per <= 0 or conc.get("status") != "VERIFIED" else "medium"})
    d4 = tables.get("d4") or {}
    for N, Kd, label, n_t, n_r in gemm_shapes():
        key = f"{N},{Kd}"
        x = next((y for y in (d4.get("rows") or []) if y["shape"] == key and y["M"] == T), None)
        if not x or not x.get("cublas_us") or not x.get("best_us"):
            continue
        per = x["cublas_us"] - x["best_us"]
        saved = max(0.0, per) * n_t / 1000.0
        if N == 124160:                           # the K draft heads (assumed the same TP2 shard, DRAFT_HEAD bf16)
            xr = next((y for y in (d4.get("rows") or []) if y["shape"] == key and y["M"] == r), None)
            if xr and xr.get("cublas_us") and xr.get("best_us"):
                saved += max(0.0, xr["cublas_us"] - xr["best_us"]) * int(K) / 1000.0
        disp = x.get("dispatch_ok")               # N3-21: the patched (8,9) dispatch must have been run and matched
        if x.get("best_ok") is False:
            corr = "FAIL"
        elif disp is False:
            corr = "FAIL_DISPATCH"
        elif disp is None:
            corr = "UNTESTED_DISPATCH"
        else:
            corr = "PASS" if x.get("best_ok") is True and disp is True else "UNTESTED"
        out.append({"lever": "D4", "id": f"D4_sm89_plan_{N}x{Kd}", "adoption": "patch", "bench": "D4",
                    "what": f"(8,9) skinny-GEMM plan entry for {label}: {x.get('best_source')} {x.get('best_cfg')}",
                    "T": T, "us_current": x["cublas_us"], "us_new": x["best_us"], "speedup": x.get("speedup"),
                    "calls_per_tick": {"T_micro": n_t, "r": (int(K) if N == 124160 else 0)},
                    "saved_ms_per_tick": round(saved, 4), "correctness": corr, "dispatch_ok": disp,
                    "confidence": _confidence(None, x.get("best_spread"), per / x["cublas_us"] if x["cublas_us"] else None)})
    dh = d4.get("draft_heads") or []
    b16 = next((y for y in dh if y["mode"] == "bf16" and y["M"] == r), None)
    for mode in ("int8", "int4"):
        q = next((y for y in dh if y["mode"] == mode and y["M"] == r), None)
        if not b16 or not q or not b16.get("us") or not q.get("us"):
            continue
        per = b16["us"] - q["us"]
        out.append({"lever": "D2", "id": f"D2_draft_head_{mode}", "adoption": "config", "bench": "D4",
                    "what": f"DRAFT_HEAD={mode} (Marlin W{mode[-1]}A16 draft head) vs bf16, K={K} draft passes at r={r} rows",
                    "T": r, "us_current": b16["us"], "us_new": q["us"], "speedup": round(b16["us"] / q["us"], 3),
                    "calls_per_tick": {"T_micro": 0, "r": int(K)}, "saved_ms_per_tick": round(max(0.0, per) * int(K) / 1000.0, 4),
                    "correctness": "UNTESTED", "kernel_sanity": q.get("sanity_ok"), "quant_rel_fro_vs_bf16": q.get("rel_fro_vs_bf16"),
                    "note": "acceptance effect is NB2's (D2); kernel-only here, never ranked from NB3",
                    "confidence": _confidence(b16.get("spread"), q.get("spread"), per / b16["us"])})
    for c in out:
        c["family"] = FAMILY[c["lever"]]
        c["seqs"], c["K"] = seqs, K
        c["microbench_saved_ms"] = c.get("saved_ms_per_tick")
        c["engine_confirmed_saved_ms"] = None
        c["projection"] = projection(c.get("saved_ms_per_tick") or 0.0, seqs)
    return out


def rank_candidates(tables, targets=TARGETS, status=None):
    """Ranked NB4 candidates: per target point, ONLY the PROMOTION_READY candidates (eligibility(), the same rule as
    the scorecard) sorted by ms saved per tick; the others are listed with their eligibility and raw numbers:
    blocked (bench failure / correctness), diagnostic (PASS_BOUND_ONLY, partial budget, concurrency unverified, ...),
    hypotheses, info, no_gain. `status` = the VERIFIED bench statuses {D5, D5_rest, D7, D4}; absent = not verified."""
    res = {"targets": [], "bench_status": dict(status or {}), "additivity": ADDITIVITY_NOTE, "adoption_rule": ADOPTION_RULE,
           "note": "MICROBENCH_PROJECTED; ms saved on the binding stage (stage 1) per tick; headline step = "
                   "RC2 fitted D-model at each stream count (24.21 + 0.1758 D(4 C/2)); RC2 measured 30-48 ms at "
                   "1-8 streams only, so 16 / 32 streams are model extrapolations (est.)",
           "floor_decomposition": {str(tg["seqs"]): floor_decomposition(tables, tg["seqs"], tg["K"]) for tg in targets}}
    for tg in targets:
        cs = candidates(tables, tg["seqs"], tg["K"])
        for c in cs:
            c["eligibility"] = eligibility(c, status)
            c["bench_status"] = (status or {}).get(c.get("bench"))
        ranked = sorted([c for c in cs if c["eligibility"] == PROMOTION_READY], key=lambda c: -c["saved_ms_per_tick"])
        for i, c in enumerate(ranked):
            c["rank"] = i + 1
        res["targets"].append({**tg, "T_micro": micro_T(tg["seqs"], tg["K"]), "ranked": ranked,
                               "blocked": [c for c in cs if c["eligibility"] in (BLOCKED_BENCH_FAILURE, BLOCKED_CORRECTNESS)],
                               "diagnostic": [c for c in cs if c["eligibility"] in (DIAGNOSTIC_ONLY, PARTIAL_DIAGNOSTIC)
                                              and not c.get("hypothesis")],
                               "hypotheses": [c for c in cs if c.get("hypothesis")],
                               "info": [c for c in cs if c.get("info_only")],
                               "no_gain": [c["id"] for c in cs if c["eligibility"] == "NO_GAIN"]})
    return res


def floor_decomposition(tables, seqs=8, K=3):
    """RC2 step = fixed floor (~24.2-25.6 ms) + MoE slope. Measured per-call microbench times x DERIVED calls per
    stage-1 tick at the operating point: (a) the floor pieces we can time (dense BF16 GEMMs + target LM head + draft
    heads, TP2 all-reduces), the remainder = attention / QSA / GDN / RecoverSSM / sampling / launch / PP hand-off (not
    benched here); (b) the MoE term vs the fitted D term 0.1758 D(4 r) and the 173 GB/s RC2 implies. MICROBENCH only."""
    tc = tick_calls(seqs, K)
    T, r = tc["T_micro"], tc["r"]
    d4 = tables.get("d4") or {}
    d7 = tables.get("d7") or {}
    d5 = tables.get("d5") or {}
    out = {"seqs": seqs, "K": K, "T_micro": T, "r": r, "stage": BINDING_STAGE, "rc2_fit": RC2_FIT,
           "fitted_step_ms": round(rc2_step_ms(seqs), 2), "fitted_floor_ms": RC2_FIT["dmodel"][0],
           "fitted_moe_term_ms": round(RC2_FIT["dmodel"][1] * E_D(4.0 * seqs / 2.0), 2), "dense": [], "unmeasured": []}
    dense_ms = 0.0
    for N, Kd, label, n_t, _ in gemm_shapes():
        x = next((y for y in (d4.get("rows") or []) if y["shape"] == f"{N},{Kd}" and y["M"] == T), None)
        if x and x.get("cublas_us"):
            ms = x["cublas_us"] * n_t / 1000.0
            dense_ms += ms
            out["dense"].append({"shape": f"{N},{Kd}", "label": label, "calls": n_t, "us_per_call_cublas": x["cublas_us"],
                                 "ms_per_tick": round(ms, 3), "best_us": x.get("best_us")})
        else:
            out["unmeasured"].append(f"{N},{Kd}@M={T}")
    dh = next((y for y in (d4.get("draft_heads") or []) if y["mode"] == "bf16" and y["M"] == r), None)
    heads_ms = (dh["us"] * int(K) / 1000.0) if dh and dh.get("us") else None
    ar = next((x for x in (d7.get("rows") or []) if abs((x.get("T") or 0) - T) < 1e-6), None)
    arr = next((x for x in (d7.get("rows") or []) if abs((x.get("T") or 0) - r) < 1e-6), None)
    ar_ms = ((ar["nccl_graph"] * tc["ar_target"] + ((arr or {}).get("nccl_graph") or 0.0) * tc["ar_draft"]) / 1000.0
             if ar and ar.get("nccl_graph") else None)
    base = [x for x in (d5.get("baseline") or []) if x.get("mode") == "independent"]
    m, mr = _lookup(base, T), _lookup(base, r)
    moe_ms = ((m["graph_us"] * tc["moe_target"] + ((mr or {}).get("graph_us") or 0.0) * tc["moe_draft"]) / 1000.0
              if m and m.get("graph_us") else None)
    timed = [v for v in (dense_ms if out["dense"] else None, heads_ms, ar_ms) if v is not None]
    out.update(dense_gemm_target_head_ms=round(dense_ms, 3) if out["dense"] else None,
               draft_heads_bf16_ms=round(heads_ms, 3) if heads_ms is not None else None,
               allreduce_nccl_ms=round(ar_ms, 3) if ar_ms is not None else None,
               floor_timed_ms=round(sum(timed), 3) if timed else None,
               floor_remainder_ms=round(RC2_FIT["dmodel"][0] - sum(timed), 3) if timed else None,
               floor_remainder_is=("attention/QSA indexer + top-k, GDN + RecoverSSM commit, sampling + logits gather, "
                                   "PP hand-off, launch gaps, stage-0/1 imbalance (not benched in NB3)"),
               moe_measured_ms=round(moe_ms, 3) if moe_ms is not None else None,
               moe_derived_gbps=m.get("derived_gbps") if m else None,
               moe_vs_fitted_term=round(moe_ms / out["fitted_moe_term_ms"], 3) if moe_ms else None,
               moe_derived_gbps_vs_rc2_implied_173=m.get("derived_gbps_vs_rc2_implied_173") if m else None,
               label="MICROBENCH_PROJECTED; calls per tick DERIVED (est.); routing synthetic (uniform)")
    return out


# ------------------------------------------------------------------------- lever scorecard (PLAN_1500 v3.2 4.1)
COST_GPU_H = {"D5": 0.5, "D7": 1.0, "D4": 1.0, "D2": 0.5, "PP2": 3.0}   # GPU hours to confirm end-to-end in NB4 (one boot + windows; est.)


def scorecard(tables, status=None):
    """One row per lever benched (D5 / D7 / D4 / D2 / PP2). measured: us per call at every T + speedup vs the current
    path; the best PROMOTION_READY candidate at the RC2 point (8 streams, K3) under the SAME eligibility() rule as the
    NB4 ranking; score = gain x (1 if PASS) x confidence weight / adoption cost (PLAN 4.1), and 0 for every row that
    is not PROMOTION_READY (BLOCKED_BENCH_FAILURE / BLOCKED_CORRECTNESS / DIAGNOSTIC_ONLY / PARTIAL_DIAGNOSTIC /
    UNTESTED / NO_GAIN: projected gain 0, the raw numbers kept under measured / diagnostic_best / candidates_all).
    `status` = the VERIFIED bench statuses {D5, D5_rest, D7, D4}. Rows exist even when a bench did not run."""
    status = status or {}
    cs = candidates(tables, RC2_STEADY["seqs"], RC2_STEADY["K"])
    for c in cs:
        c["eligibility"] = eligibility(c, status)
    rows = []
    d5, d7, d4 = tables.get("d5") or {}, tables.get("d7") or {}, tables.get("d4") or {}
    ind = [x for x in d5.get("baseline") or [] if x.get("mode") == "independent"]
    meas = {
        "D5": {"us_per_call_current_by_T": {str(x["T"]): x["graph_us"] for x in ind},
               "graph_status_by_T": {str(x["T"]): x.get("graph_status") for x in ind},
               "us_per_call_zipf_routing_by_T": {str(x["T"]): x["graph_us"] for x in d5.get("baseline") or []
                                                 if x.get("mode") == "zipf"},
               "us_per_call_request_routing_by_T": {str(x["T"]): x["graph_us"] for x in d5.get("baseline") or []
                                                    if x.get("mode") == "request"},
               "hot_routing_by_T": {str(x["T"]): {"us": x["graph_us"], "distinct_local": x["distinct_local"],
                                                  "derived_gbps": x.get("derived_gbps"), "ref_ok": x["ref_ok"]}
                                    for x in d5.get("baseline") or [] if x.get("mode") == "hot"},
               "adversarial_reference": d5.get("adversarial"), "ep1_spot": d5.get("ep1_spot"), "units": d5.get("units"),
               "real_routing_replay": d5.get("replay") or {"available": False, "why": "real routing unavailable"},
               "routing": ROUTING_NOTE,
               "derived_gbps_by_T": {str(x["T"]): x.get("derived_gbps") for x in ind},
               "derived_gbps_vs_rc2_implied_173_by_T": {str(x["T"]): x.get("derived_gbps_vs_rc2_implied_173") for x in ind},
               "best_cfg_by_T": {str(x["T"]): {"rule": x["best_rule"], "us": x["best_us"], "speedup": x["speedup"],
                                               "max_ulp_vs_base": x.get("best_max_ulp_vs_base")}
                                 for x in d5.get("sweep") or []}},
        "D7": {"us_per_call_by_T": {str(x["T"]): {k: x.get(k) for k in ("nccl_graph", "p5_graph", "p5_int8_graph",
                                                                          "solo_nccl_graph", "solo_p5_graph")}
                                    for x in d7.get("rows") or []},
               "speedup_by_T": {str(x["T"]): {k: x.get("speedup_" + k) for k in ("p5_graph", "p5_int8_graph")}
                                for x in d7.get("rows") or []},
               "arms_both_pairs_both_ranks": d7.get("arms"), "concurrency": d7.get("concurrency"), "ll128": LL128_LABEL,
               "nccl_protocol_seen": d7.get("nccl_protocols"), "nccl_transports": d7.get("nccl_transports")},
        "D4": {"plans": {k: v for k, v in (d4.get("plans") or {}).items() if k != "sm121_shapes"}, "sm89_skinny": d4.get("sm89_skinny"),
               "us_per_call_by_shape_M": {f"{x['shape']}@{x['M']}": {"cublas": x["cublas_us"], "best": x["best_us"],
                                                                     "speedup": x["speedup"], "src": x["best_source"],
                                                                     "dispatch_ok": x.get("dispatch_ok")}
                                          for x in d4.get("rows") or []}},
        "D2": {"draft_heads_status": d4.get("draft_heads_status"), "draft_heads_preflight": d4.get("draft_heads_preflight"),
               "draft_head_us_by_mode_M": {f"{x['mode']}@{x['M']}": {"us": x["us"], "rel_fro_vs_bf16": x["rel_fro_vs_bf16"]}
                                           for x in d4.get("draft_heads") or []}},
        "PP2": {"split_vs_combined_by_C": {str(x["C"]): {k: x.get(k) for k in (
            "T", "T_half", "split_us", "combined_us", "split_over_combined", "distinct_local_split_sum", "distinct_local_combined",
            "bytes_split", "bytes_combined", "bytes_source", "ncu_status", "ncu_why", "ncu_read_split", "ncu_read_combined",
            "dram_gbps_split", "dram_gbps_combined", "derived_gbps_split", "derived_gbps_combined",
            "bitwise_equal", "equivalent", "max_abs", "p99_abs", "max_ulp", "max_rel", "rel_fro")}
            for x in d5.get("split") or []}},
    }
    what = {"D5": "Marlin MoE on L4 with real Flash-Next experts (EP2 rank, 256 local experts) + MERLIN_MARLIN_MOE_CFG",
            "D7": ("world-2 all-reduce: NCCL (RC2 env: NCCL_P2P_DISABLE=1, NCCL_PROTO / NCCL_ALGO unset, NCCL picks its "
                   "protocol) vs P5 host-staged (native / INT8); LL128: " + LL128_LABEL),
            "D4": "sm_89 skinny-GEMM plan path (CuTe DSL) vs the cuBLAS fallback on Flash-Next TP2 shapes (incl. target LM head)",
            "D2": "draft head kernels: DRAFT_HEAD bf16 (cuBLAS) vs int8 / int4 (Marlin) at the draft row counts (kernel only; NB2 owns acceptance)",
            "PP2": ("PP2 split-read hypothesis (PLAN 1.1): two micro-batch MoE calls of T/2 vs one combined call of T, same "
                    "routing ids and weights, C 8/16/24/32 at K3 (HYPOTHESIS, research)")}
    lever_bench = {"D5": "D5", "D7": "D7", "D4": "D4", "D2": "D4", "PP2": "D5_rest"}
    order = (BLOCKED_BENCH_FAILURE, BLOCKED_CORRECTNESS, PARTIAL_DIAGNOSTIC, DIAGNOSTIC_ONLY, "NO_GAIN")
    for lev in ("D5", "D7", "D4", "D2", "PP2"):
        mine = [c for c in cs if c["lever"] == lev and not c.get("info_only")]
        ready = sorted([c for c in mine if c["eligibility"] == PROMOTION_READY], key=lambda c: -c["saved_ms_per_tick"])
        best = ready[0] if ready else None
        bs = status.get(lever_bench[lev])
        if best:
            promo = PROMOTION_READY
        elif lev == "PP2":
            promo = DIAGNOSTIC_ONLY
        elif not mine:
            promo = ("UNTESTED" if bs in (None, "DONE") else PARTIAL_DIAGNOSTIC if bs == "PARTIAL_BUDGET" else
                     DIAGNOSTIC_ONLY if bs == "UNSUPPORTED_SM89" else BLOCKED_BENCH_FAILURE)
        else:
            promo = next((o for o in order if any(c["eligibility"] == o for c in mine)), "UNTESTED")
        if best:
            corr = "PASS"
        elif any(c["correctness"] == "FAIL" or str(c["correctness"]).startswith("FAIL_") for c in mine):
            corr = "FAIL"
        elif mine and all(c["correctness"] == "PASS" for c in mine):
            corr = "PASS"                       # numerics passed, but not promotable (see promotion)
        elif any(c["correctness"] == "PASS_BOUND_ONLY" for c in mine):
            corr = "PASS_BOUND_ONLY"
        else:
            corr = "UNTESTED"
        diag = max(mine, key=lambda c: c.get("saved_ms_per_tick") or 0.0) if mine else None
        conf = best["confidence"] if best else "none"
        adopt = best["adoption"] if best else ("config" if lev in ("D5", "D2") else "research" if lev == "PP2" else "patch")
        gain = (best["projection"]["run_tok_s_gain_headline"] or 0.0) if best else 0.0
        score = round(gain * CONF_WEIGHT.get(conf, 0.0) / ADOPTION_COST[adopt], 3) if promo == PROMOTION_READY else 0.0
        rows.append({"lever": lev, "family": FAMILY[lev], "what": what[lev], "hypothesis": lev == "PP2",
                     "bench_status": bs, "promotion": promo,
                     "measured": meas[lev],
                     "best_candidate": {k: best.get(k) for k in ("id", "what", "T", "us_current", "us_new", "speedup",
                                                                 "calls_per_tick", "saved_ms_per_tick")} if best else None,
                     "diagnostic_best": ({k: diag.get(k) for k in ("id", "eligibility", "correctness", "saved_ms_per_tick",
                                                                   "speedup", "us_current", "us_new", "diagnostic_reason")}
                                         if diag and not best else None),
                     "projected_run_tok_s_gain": best["projection"] if best else projection(0.0),
                     "microbench_saved_ms": best["saved_ms_per_tick"] if best else 0.0,
                     "engine_confirmed_saved_ms": None, "additivity": ADDITIVITY_NOTE,
                     "correctness": corr, "confidence": conf, "confidence_weight": CONF_WEIGHT.get(conf, 0.0),
                     "adoption": adopt, "cost_gpu_h": COST_GPU_H[lev], "score": score,
                     "candidates_all": [{k: c.get(k) for k in ("id", "eligibility", "correctness", "saved_ms_per_tick", "speedup")}
                                        for c in mine],
                     "notes": ("microbench-projected: a microbenchmark gain is never counted as an engine gain until NB4 "
                               "confirms it end-to-end; RC2 point = 8 streams K3 (T_micro 16, stage 1 = 23 layers + 3 MTP "
                               "draft passes); call counts per tick are DERIVED estimates; " + ADOPTION_RULE
                               + ("; " + ROUTING_NOTE if lev == "D5" else ""))})
    hm = d5.get("humming") or {"status": "NOT_RUN", "rows": []}
    rows.append({"lever": "C6-humming", "family": "structural_track_low_bit_moe", "hypothesis": True, "research": True,
                 "what": ("humming 2/3-bit MoE (Merlin v3.0.0 INDEXED path) vs Marlin W4 on the same layer-0 local experts, "
                          "T 8-64; W3 / W2 = SYNTHETIC REQUANT (speed + bytes only); gate W3 <= 0.8x, W2 <= 0.6x Marlin W4 "
                          "(capacity reading <= 1.3x also recorded)"),
                 "bench_status": status.get("D5"), "promotion": ("UNAVAILABLE" if hm.get("status") == "UNAVAILABLE" else
                                                                 "UNTESTED" if hm.get("status") == "NOT_RUN" else DIAGNOSTIC_ONLY),
                 "measured": {"status": hm.get("status"), "available": hm.get("available"), "summary": hm.get("summary"),
                              "rows": hm.get("rows"), "errors": hm.get("errors"), "info": hm.get("info")},
                 "best_candidate": None, "diagnostic_best": None, "projected_run_tok_s_gain": projection(0.0),
                 "microbench_saved_ms": 0.0, "engine_confirmed_saved_ms": None, "additivity": ADDITIVITY_NOTE,
                 "correctness": ("PASS" if hm.get("rows") and all(r.get("ref_ok_own_weights") is True for r in hm["rows"]) else
                                 "FAIL" if any(r.get("ref_ok_own_weights") is False for r in hm.get("rows") or []) else "UNTESTED"),
                 "confidence": "none", "confidence_weight": 0.0, "adoption": "research", "cost_gpu_h": 6.0, "score": 0.0,
                 "candidates_all": [],
                 "notes": "research only (structural track), never ranked into NB4; " + (hm.get("label") or "")})
    return {"what": "E25 NB3 lever scorecard (PLAN_1500 v3.2 section 4.1)", "rc2_point": RC2_STEADY,
            "step_ms": list(STEP_MS), "rc2_cost_model": RC2_FIT, "bench_status": dict(status),
            "promotion_rule": eligibility.__doc__.strip(), "additivity": ADDITIVITY_NOTE, "adoption_rule": ADOPTION_RULE,
            "floor_decomposition_rc2_point": floor_decomposition(tables),
            "rows": rows, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def write_json(path, obj):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(p)


# ------------------------------------------------------------------ D5 priority plan (N3-16 / N3-19), shared by real + fake
D5_MODES = ("independent", "zipf", "hot", "request")     # all SYNTHETIC routing (ROUTING_NOTE); + 'adversarial' (reference only)
D5_T_SWEEP = (1, 4, 8, 12, 16, 24, 32, 48, 64)           # the PP2 micro-batch / draft token counts that matter (8..32 streams)
D5_RESERVE_S = {"base": 30, "adv": 20, "replay": 60, "sweep": 60, "humming": 120, "split": 60, "ep1_spot": 150, "ncu": 240}
D5_PRIORITY = ("1 T16: uniform baseline + oracle (fp32 reference), real-routing replay at T16, tuning sweep -> saved; "
               "1b the C6-humming structural arms (W4 control, W3 / W2 synthetic requant, T 16 / 8 / 32 / 64; research); "
               "2 T4 / T8 baseline + sweep; adversarial reference cases (T16, T4); 3 T32 / T64 uniform + Zipf + sweep; "
               "4 hot / request routing; 5 every other T and mode, the other sweep T, PP2 split cases, replay at the "
               "other sizes, EP rank 1 spot check; 6 ncu DRAM bytes (one child per arm) LAST. A unit starts only when "
               "its reserve fits the budget; the first one that does not ends the bench with PARTIAL_BUDGET, every row "
               "already measured is kept.")


def d5_units(phase="all", t_all=None, replay=False, ncu=False, ep1=True, humming=True):
    """The D5 work units in priority order. core = the first notebook bench (T16 / T4 / T8, run before D7 / D4);
    rest = the second (run last); all = both in one process (standalone)."""
    t_all = list(t_all or t_sets()["all"])
    top = (16, 4, 8, 32, 64)
    core = ([("base", 16, "independent")] + ([("replay", "core")] if replay else []) +
            [("sweep", 16), ("base", 4, "independent"), ("sweep", 4), ("base", 8, "independent"), ("sweep", 8),
             ("adv", 16), ("adv", 4)] + ([("humming", None)] if humming else []))     # humming (research) after the core Marlin data
    rest = [("base", 32, "independent"), ("base", 32, "zipf"), ("sweep", 32),
            ("base", 64, "independent"), ("base", 64, "zipf"), ("sweep", 64)]
    rest += [("base", T, m) for T in top for m in ("hot", "request")] + [("base", T, "zipf") for T in (16, 4, 8)]
    rest += [("base", T, m) for T in t_all if T not in top for m in D5_MODES]
    rest += [("sweep", T) for T in D5_T_SWEEP if T not in top and T in t_all]
    rest += [("split", c["C"]) for c in pp2_split_cases()]
    rest += ([("replay", "rest")] if replay else []) + ([("ep1_spot", None)] if ep1 else [])
    rest += [("ncu", c["C"]) for c in pp2_split_cases()] if ncu else []
    return core if phase == "core" else rest if phase == "rest" else core + rest


def replay_sizes(sizes, which):
    """Replay record sizes handled by the core bench (the core T) vs the rest bench (every other size)."""
    return sorted(m for m in sizes if (m in D5_CORE_T) == (which == "core"))


# ------------------------------------------------------------------- fakes (local / CPU-kit end-to-end only)
def _fst(v, rng):
    v = float(v)
    return {"median": round(v, 3), "mean": round(v, 3), "p10": round(v * 0.98, 3), "p90": round(v * (1.02 + rng.random() * 0.02), 3),
            "min": round(v * 0.97, 3), "n": 20}


def fake_base_row(T, mode, seed, rng, local=(0, 256)):
    ids = routing_ids(T, mode, seed=seed, group=request_group(T), local=local)
    dl = distinct(ids, *local)
    us = 40.0 + dl * 2534400 / 240e3
    return {"T": T, "mode": mode, "distinct_local": dl, "synthetic_routing": True,
            "d_expected_local": round(d_expected(T) / 2, 2) if mode == "independent" else None,
            "graph_us": _fst(us, rng), "graph_status": "PASS", "hot_graph_us": _fst(us * 0.6, rng),
            "eager_us": _fst(us * 1.4, rng), "align_graph_us": _fst(8.0, rng),
            "derived_gbps": round(dl * 2534400 / us / 1e3, 1), "ref_ok": True,
            "ref": {"finite": True, "rel_fro": 3.9e-3, "cos": 0.99999, "max_abs_over_ref_max": 6e-3, "p99_abs": 1e-3}}


def fake_sweep(T, base_row, rng):
    base = med((base_row or {}).get("graph_us"))
    if not base:
        return {"base_graph_us": None, "base_graph_status": (base_row or {}).get("graph_status"), "rules": [], "best": None,
                "why": "no graph baseline at this T"}
    best = base * (0.85 + 0.1 * rng.random())
    return {"base_graph_us": base, "base_graph_status": "PASS", "base_spread": 0.04,
            "rules": [{"rule": "bm=16,ack=nonexact", "graph_status": "PASS"}],
            "best": {"rule": "bm=16,ack=nonexact", "us": round(best, 3), "spread": 0.03, "bitwise_vs_base": False,
                     "ref_ok": True, "equivalent_vs_base": True, "max_ulp_vs_base": 1}}


def fake_split(case, seed, rng):
    T, h = case["T"], case["T_half"]
    ids = routing_ids(T, "independent", seed=seed)
    da = distinct(ids[:h], 0, 256) + distinct(ids[h:], 0, 256)
    db = distinct(ids, 0, 256)
    ua, ub = 80.0 + da * 2534400 / 240e3, 40.0 + db * 2534400 / 240e3
    return {**case, "split_graph_us": _fst(ua, rng), "combined_graph_us": _fst(ub, rng), "graph_status": "PASS",
            "distinct_local_split_sum": da, "distinct_local_combined": db,
            "bytes_split": da * 2534400, "bytes_combined": db * 2534400, "bytes_source": "DERIVED",
            "derived_gbps_split": round(da * 2534400 / ua / 1e3, 1), "derived_gbps_combined": round(db * 2534400 / ub / 1e3, 1),
            "equivalence": {"bitwise_equal": True, "equivalent": True, "max_abs": 0.0, "p99_abs": 0.0, "max_ulp": 0,
                            "n_over_max_ulp": 0, "max_rel": 0.0, "rel_fro": 0.0},
            "ref_ok_split": True, "ref_ok_combined": True, "ncu": {"ok": False, "status": NCU_UNVERIFIED, "why": "fake: no ncu"}}


def d5_checks(j):
    """The D5 file's checks from its rows (real and fake): only what this file measured is checked."""
    ch = dict(j.get("checks") or {})
    rows = [r for modes in (j.get("baseline") or {}).values() for r in modes.values()] + list((j.get("adversarial") or {}).values())
    if rows:
        ch["reference_all_T"] = all(r.get("ref_ok") is True and not r.get("error") for r in rows)
    if j.get("split"):
        ch["split_equivalent"] = all((s.get("equivalence") or {}).get("equivalent") is True and s.get("ref_ok_split") is True
                                     and s.get("ref_ok_combined") is True and not s.get("error") for s in j["split"].values())
    if (j.get("replay") or {}).get("by_size"):
        ch["replay_reference"] = all(b.get("ref_ok") is not False and b.get("split_equivalent") is not False
                                     for b in j["replay"]["by_size"])
    if j.get("ep1_spot") and not j["ep1_spot"].get("skipped"):
        ch["ep1_spot_reference"] = j["ep1_spot"].get("ok") is True
    return ch


def fake_moe(seed=0, phase="all"):
    """Every D5 unit of `phase` with fake numbers (same schema as the real bench; no budget limit)."""
    rng = random.Random(seed)
    j = {"meta": {"fake": True}, "phase": phase, "baseline": {}, "sweep": {}, "split": {}, "adversarial": {},
         "checks": {"preflight": True, "load_shapes": True, "dequant_crosscheck": True, "first_call_gate": True},
         "units_done": [], "units_skipped": [], "status": "DONE"}
    cases = {c["C"]: c for c in pp2_split_cases()}
    for u in d5_units(phase):
        kind, arg = u[0], u[1]
        if kind == "base":
            j["baseline"].setdefault(str(arg), {})[u[2]] = fake_base_row(arg, u[2], seed, rng)
        elif kind == "adv":
            j["adversarial"][str(arg)] = {**fake_base_row(arg, "adversarial", seed, rng), "reference_only": True}
        elif kind == "sweep":
            j["sweep"][str(arg)] = fake_sweep(arg, (j["baseline"].get(str(arg)) or {}).get("independent"), rng)
        elif kind == "split":
            j["split"][str(arg)] = fake_split(cases[arg], seed, rng)
        elif kind == "ep1_spot":
            j["ep1_spot"] = {"ok": True, "rc": 0, "status": "DONE", "local_range": [256, 512],
                             "rows": {str(T): {"ref_ok": True, "graph_us": round(40.0 + T, 3)} for T in (4, 16)}}
        elif kind == "humming":
            import nb3_humming
            j["humming"] = nb3_humming.fake(seed)
        j["units_done"].append(list(u))
    j["replay"] = {"available": False, "why": "real routing unavailable (fake run)"}
    j["checks"] = d5_checks(j)
    return j


def fake_faults():
    """Fault injection for the local / CPU-kit end-to-end (N3-25): NB3_FAKE_FAULTS = comma list of d5_bench_error,
    graph_error, corrupt_scale, d7_timeout, missing_rank1, d4_check_failed, missing_dispatch, d4_unsupported."""
    return {f.strip() for f in (os.environ.get("NB3_FAKE_FAULTS") or "").split(",") if f.strip()}


def fake_gemm(seed=0, faults=()):
    rng = random.Random(seed)
    faults = set(faults or ())
    unsupported = "d4_unsupported" in faults
    j = {"meta": {"fake": True, "faults": sorted(faults)}, "plans": {"capability": [8, 9], "has_plan_for_capability": False,
                                                                     "capabilities_with_plans": [[10, 3], [10, 0], [9, 0], [12, 1]],
                                                                     "skinny_available": not unsupported, "pdl": False},
         "shapes": {}, "checks": {"all_numeric_ok": True, "plan_table_restored": True}, "status": "DONE"}
    for N, K, label, _, _ in gemm_shapes():
        s = {"label": label, "N": N, "K": K, "M": {}}
        for m in (1, 2, 4, 8, 16):
            cub = 6.0 + N * K * 2 / 250e3
            us = cub * (0.7 + 0.4 * rng.random())
            r = {"cublas": {"graph_us": _fst(cub, rng), "eager_us": _fst(cub * 1.5, rng), "graph_status": "PASS", "ok": True}}
            if unsupported:
                r.update(configs=[{"source": "heuristic", "error": "UNSUPPORTED_SM89: cuteDSL not importable (fake)"}], best=None)
            else:
                r.update(configs=[{"source": "heuristic", "ok": True, "graph_status": "PASS"}],
                         best={"source": "heuristic", "cfg": {"num_rows": m}, "us": round(us, 3), "spread": 0.03,
                               "speedup_vs_cublas": round(cub / us, 3), "ok": True})
                if "missing_dispatch" not in faults:
                    r["dispatch"] = {"ok": True}
            s["M"][str(m)] = r
        j["shapes"][f"{N},{K}"] = s
    if "d4_check_failed" in faults:
        j["shapes"]["6656,2560"]["M"]["16"]["configs"].append({"source": "sweep", "ok": False, "graph_status": "PASS"})
        j["checks"]["all_numeric_ok"] = False
    if unsupported:
        j["sm89_skinny"] = {"status": "UNSUPPORTED_SM89", "why": "fake: shape_dynamic_skinny_gemm.is_available() False"}
        j["status"] = "UNSUPPORTED_SM89"
    j["draft_heads"] = {"shape": [124160, 2560], "status": "DONE", "preflight": {"ok": True, "fake": True}, "modes": {
        mode: {"sanity_ok": True, "per_M": {str(m): {"graph_us": _fst(f * (6.0 + 124160 * 2560 * 2 / 250e3), rng), "graph_status": "PASS",
                                                     "ref_vs_bf16": {"rel_fro": e}} for m in (1, 2, 4, 8, 16)}}
        for mode, f, e in (("bf16", 1.0, 0.0), ("int8", 0.62, 0.004), ("int4", 0.42, 0.07))}}
    j["checks"]["draft_head_sanity"] = True
    return j


def fake_ar(sizes, seed=0, faults=(), steps="a,b,c,d"):
    """A bench_host_ar.py-shaped parent JSON (groups main_ab / main_cd, rank-0 parts with the per-size timed-region
    monotonic windows, every mandatory check on ranks 0 and 1). The windows of every fake pair coincide (concurrent)."""
    rng = random.Random(seed)
    faults = set(faults or ())
    lat, d = {}, {}
    base = 1000.0
    for i, nb in enumerate(sizes):
        nc = 30.0 + nb / 4.0e3
        lat[str(nb)] = {"nccl_eager": round(nc * 1.3, 2), "nccl_graph": round(nc, 2), "torch_nccl_eager": round(nc * 1.4, 2),
                        "host_eager": round(nc * 0.9, 2), "host_graph": round(nc * (0.65 + 0.1 * rng.random()), 2),
                        "host_status_ok": True, "t_mono": {"nccl": [base + 2 * i, base + 2 * i + 1.0],
                                                           "host": [base + 2 * i + 1.0, base + 2 * i + 2.0]}}
        d[str(nb)] = {"host_int8_graph": round(nc * 0.5, 2), "host_native_graph": round(nc * 0.7, 2), "status_ok": True}
    t0 = time.time()
    ab = {k: True for k in AR_MANDATORY["nccl"]["main_ab"] + AR_MANDATORY["p5"]["main_ab"]}
    cd = {k: True for k in AR_MANDATORY["p5"]["main_cd"] + AR_MANDATORY["p5_int8"]["main_cd"]}
    groups = {"main_ab": {"status": "ok", "t0": t0, "t1": t0 + 60, "nccl_log": {"via_counts": {"SHM/direct/direct": 4}},
                          "rank0": {"b": {"latency": lat}, "checks": dict(ab)}, "other_ranks": {"1": {"checks": dict(ab)}}}}
    if any(s in steps for s in ("c", "d")):
        groups["main_cd"] = {"status": "ok", "rank0": {"d": {"latency": d}, "checks": dict(cd)}, "other_ranks": {"1": {"checks": dict(cd)}}}
    if "missing_rank1" in faults:
        for g in groups.values():
            g["other_ranks"] = {}
    return {"meta": {"fake": True, "sizes": sizes, "faults": sorted(faults)}, "exit_code": 0, "groups": groups,
            "summary": {"all_checks_pass": True}}


def _cli():
    """python nb3_common.py --fake-ar --out X --sizes '[...]' [--seed N] [--steps S]: the D7 stand-in when
    E23_FAKE_SERVER is set (NB3_FAKE_FAULTS d7_timeout: the JSON is written, then the process hangs past its deadline)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-ar", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="[]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", default="a,b,c,d")
    a, _ = ap.parse_known_args()
    if a.fake_ar:
        f = fake_faults()
        write_json(a.out, fake_ar(json.loads(a.sizes), a.seed, f, a.steps))
        print("FAKE_AR", a.out, flush=True)
        if "d7_timeout" in f:
            time.sleep(3600)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(_cli())
