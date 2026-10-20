"""E13 micro-preflight D: QSA sparse-decode attention parity on this GPU, before any model is loaded.

Reproduces SGLang 0.5.21 `QwenSparseAttnBackend._forward_paged_attention` after the K/V gather
(qwen_sparse_attn_backend.py:1521-1632): packed K/V scratch in q.dtype, one query row per sequence, TP4-local GQA
(Flash-Next 24 Q / 2 KV heads -> 6 Q / 1 KV per rank), head_dim 256, token top-k capacity 2051 (2048 + ratio 4 - 1).
  FA2 = classic flash_attn.flash_attn_varlen_func(..., max_seqlen_q=1, max_seqlen_k=topk, causal=True)  (the call SGLang makes)
  P1  = sparse_gqa_packed_decode_triton with -1-masked relative indices                              (the is_hip() branch)
Reference: FP32 softmax attention over the valid keys of each row. FP8 KV storage is emulated by a float8_e4m3fn round
trip of K/V before both the kernels and the reference (the backend's extraction kernel writes q.dtype scratch).
Pass rule (diagnostic): |out - ref| <= 0.01 + 0.01 * |ref| on every valid row, finite everywhere outside padding rows.
Usage: python -I mp_decode.py --out RESULT.json [--cpu-selftest]
"""
import argparse
import json
import math
import sys
import traceback

import torch

HQ, HKV, D, TOPK = 6, 1, 256, 2051
LENS = [1, 63, 64, 65, 129, 2048, 2051]
BATCHES = [1, 4, 10]
ATOL = RTOL = 0.01


def make_case(batch, lens, fp8, gen, device, pad_rows=0):
    lens = list(lens)[:batch] + [0] * pad_rows
    b = len(lens)
    q = (torch.randn(b, HQ, D, generator=gen) * 0.5).to(torch.bfloat16)
    tot = max(1, sum(lens))
    k = (torch.randn(tot, HKV, D, generator=gen) * 0.5)
    v = torch.randn(tot, HKV, D, generator=gen)
    if fp8:
        k, v = k.to(torch.float8_e4m3fn).float(), v.to(torch.float8_e4m3fn).float()
    k, v = k.to(torch.bfloat16), v.to(torch.bfloat16)
    cu_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0).tolist()), dtype=torch.int32)
    valid = torch.tensor(lens, dtype=torch.int32)
    cu_q = torch.arange(b + 1, dtype=torch.int32)
    return [t.to(device) for t in (q, k, v, cu_q, cu_k, valid)], lens


def reference(q, k, v, cu_k, lens, scale):
    out = torch.zeros(q.shape[0], HQ, D, dtype=torch.float32)
    qf, kf, vf = q.float().cpu(), k.float().cpu(), v.float().cpu()
    for b, n in enumerate(lens):
        if n == 0:
            continue
        s0 = int(cu_k[b]); kk, vv = kf[s0:s0 + n, 0], vf[s0:s0 + n, 0]           # one KV head; all 6 Q heads share it
        p = torch.softmax(qf[b] @ kk.T * scale, dim=-1)
        out[b] = p @ vv
    return out


def compare(out, ref, lens):
    out = out.float().cpu().reshape(ref.shape)
    rows = [b for b, n in enumerate(lens) if n > 0]
    if not rows:
        return {"ok": True, "max_abs": 0.0}
    o, r = out[rows], ref[rows]
    finite = bool(torch.isfinite(o).all())
    tol_ok = bool(((o - r).abs() <= ATOL + RTOL * r.abs()).all()) if finite else False
    pad = [b for b, n in enumerate(lens) if n == 0]
    return {"ok": finite and tol_ok, "finite": finite, "max_abs": float((o - r).abs().max()) if finite else None,
            "pad_rows": len(pad), "pad_rows_finite": bool(torch.isfinite(out[pad]).all()) if pad else None}


def fa2_call(fa2):
    def f(q, k, v, cu_q, cu_k, valid, scale):
        return fa2(q=q, k=k, v=v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, max_seqlen_q=1, max_seqlen_k=TOPK,
                   softmax_scale=scale, causal=True)
    return f


def p1_call(kernel):
    def f(q, k, v, cu_q, cu_k, valid, scale):
        rel = torch.arange(TOPK, dtype=torch.int32, device=q.device).expand(q.shape[0], -1)
        rel = rel.masked_fill(rel >= valid[:, None], -1).contiguous()
        return kernel(q.contiguous(), k, v, rel, cu_q, cu_k, valid, scale)
    return f


def torch_call():          # CPU self-test stand-in with the same signature
    def f(q, k, v, cu_q, cu_k, valid, scale):
        lens = valid.tolist()
        return reference(q, k, v, cu_k.cpu(), lens, scale).to(q.dtype)
    return f


def run(name, fn, device):
    res = {"impl": name, "cases": [], "fails": 0, "error": None}
    gen = torch.Generator().manual_seed(20261019)
    scale = 1.0 / math.sqrt(D)
    try:
        for fp8 in (False, True):
            for batch in BATCHES:
                for start in range(0, len(LENS)):
                    lens = [LENS[(start + i) % len(LENS)] for i in range(batch)]           # mixed lengths per batch
                    for pad in ((0, 2) if batch == 4 else (0,)):
                        (q, k, v, cu_q, cu_k, valid), L = make_case(batch, lens, fp8, gen, device, pad)
                        out = fn(q, k, v, cu_q, cu_k, valid, scale)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        c = compare(out, reference(q, k, v, cu_k.cpu(), L, scale), L)
                        c.update(fp8=fp8, batch=batch, lens=L)
                        res["cases"].append(c)
                        res["fails"] += 0 if c["ok"] else 1
    except Exception:
        res["error"] = traceback.format_exc()[-1500:]
    res["total"] = len(res["cases"])
    res["max_abs"] = max((c["max_abs"] or 0.0) for c in res["cases"]) if res["cases"] else None
    return res


def graph_replay(name, fn, device):
    if device != "cuda":
        return {"impl": name, "skipped": "no cuda"}
    out = {"impl": name, "replays": 0, "fails": 0, "error": None}
    try:
        gen = torch.Generator().manual_seed(11)
        scale = 1.0 / math.sqrt(D)
        lens = [2051, 64, 129, 1, 2048, 65, 63, 2051, 300, 1500]
        (q, k, v, cu_q, cu_k, valid), L = make_case(10, lens, True, gen, device)
        for _ in range(3):
            fn(q, k, v, cu_q, cu_k, valid, scale)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o = fn(q, k, v, cu_q, cu_k, valid, scale)
        for r in range(4):        # new data, same capacity; also permute the lengths (graph metadata changes on device)
            L2 = lens[r:] + lens[:r]
            (q2, k2, v2, _, cu2, va2), L2 = make_case(10, L2, True, gen, device)
            q.copy_(q2); k.copy_(k2); v.copy_(v2); cu_k.copy_(cu2); valid.copy_(va2)
            g.replay(); torch.cuda.synchronize()
            c = compare(o, reference(q, k, v, cu_k.cpu(), L2, scale), L2)
            out["replays"] += 1; out["fails"] += 0 if c["ok"] else 1
            out.setdefault("max_abs", []).append(c["max_abs"])
    except Exception:
        out["error"] = traceback.format_exc()[-1500:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-selftest", action="store_true")
    a = ap.parse_args()
    rep = {"device": None, "fa2": None, "p1": None, "graph": [], "errors": []}
    if a.cpu_selftest:
        rep["device"] = "cpu-selftest"
        rep["fa2"] = run("torch_reference_as_kernel", torch_call(), "cpu")

        def wrong(q, k, v, cu_q, cu_k, valid, scale):        # top-left causal: the query sees only key 0 (a classic bug)
            out = torch_call()(q, k, v, cu_q, cu_k, valid, scale).float()
            for b, n in enumerate(valid.tolist()):
                if n > 1:
                    out[b] = v[int(cu_k[b]), 0].float()
            return out.to(q.dtype)
        rep["p1"] = run("wrong_causal", wrong, "cpu")
    else:
        rep["device"] = torch.cuda.get_device_name(0) + " cap " + str(torch.cuda.get_device_capability(0))
        try:
            import flash_attn
            from flash_attn import flash_attn_varlen_func
            rep["fa2_version"] = getattr(flash_attn, "__version__", "?")
            rep["fa2"] = run("fa2", fa2_call(flash_attn_varlen_func), "cuda")
            rep["graph"].append(graph_replay("fa2", fa2_call(flash_attn_varlen_func), "cuda"))
        except Exception:
            rep["errors"].append("FA2: " + traceback.format_exc()[-1500:])
        try:
            from sglang.srt.layers.attention.qsa.sparse_attn import sparse_gqa_packed_decode_triton
            rep["p1"] = run("p1_triton_packed", p1_call(sparse_gqa_packed_decode_triton), "cuda")
            rep["graph"].append(graph_replay("p1_triton_packed", p1_call(sparse_gqa_packed_decode_triton), "cuda"))
        except Exception:
            rep["errors"].append("P1: " + traceback.format_exc()[-1500:])
    for key in ("fa2", "p1"):
        r = rep.get(key)
        if isinstance(r, dict):
            r["pass"] = r.get("error") is None and r.get("total", 0) > 0 and r.get("fails", 1) == 0
    for g in rep["graph"]:
        g["pass"] = g.get("error") is None and g.get("replays", 0) > 0 and g.get("fails", 1) == 0
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: ({x: rep[k].get(x) for x in ("pass", "fails", "total", "max_abs", "error")} if isinstance(rep[k], dict) else rep[k])
                      for k in ("fa2", "p1")}, default=str)[:1500])
    print("graph", [(g["impl"], g.get("pass"), g.get("error", "")[:200] if g.get("error") else "") for g in rep["graph"]])


if __name__ == "__main__":
    sys.exit(main())
