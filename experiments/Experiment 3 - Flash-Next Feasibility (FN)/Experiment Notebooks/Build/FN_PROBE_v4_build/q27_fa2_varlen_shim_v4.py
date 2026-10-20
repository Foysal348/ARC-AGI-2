"""Stand-in for flash_attn (FA2) `flash_attn_varlen_func` on Ada GPUs (L4, sm89).

SGLang's Qwen4-Exp sparse-attention decode fallback (qwen_sparse_attn_backend._forward_paged_attention) calls a dense
varlen kernel on the packed top-k K/V of every request: classic FA2 if importable, else flash-attn-4's CuTe interface,
which only builds on Hopper/Blackwell (FN-SMOKE v3.3 crashed there with an MLIRError on L4). This module computes the
same thing in PyTorch.

Semantics follow FA2: q [total_q, Hq, D], k/v [total_k, Hkv, D(v)], cu_seqlens_* int32 prefix sums, GQA maps query head
h to KV head h // (Hq // Hkv), `causal` aligns the query block to the END of its keys. Rows with no key give zeros.
Returns [total_q, Hq, Dv] in q's dtype. The decode branch (one query per sequence) does no host synchronisation.
"""
import torch


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0,
                           softmax_scale=None, causal=False, **kwargs):
    total_q, hq, d = q.shape
    hkv, dv = k.shape[1], v.shape[-1]
    g = hq // hkv
    b = cu_seqlens_q.numel() - 1
    scale = float(softmax_scale) if softmax_scale is not None else d ** -0.5
    if total_q == b:                     # decode: one query row per sequence -> the query sees all of its keys
        lens = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).to(torch.long)
        maxk = max(int(max_seqlen_k), 1)
        ar = torch.arange(maxk, device=q.device)
        valid = ar[None, :] < lens[:, None]                                   # [b, maxk]
        idx = cu_seqlens_k[:-1].to(torch.long)[:, None] + ar[None, :]
        idx = torch.where(valid, idx, torch.zeros_like(idx)).clamp(max=max(k.shape[0] - 1, 0))
        kk = k[idx].float()                                                   # [b, maxk, hkv, d]
        vv = v[idx].float()
        qf = q.float().reshape(b, hkv, g, d)
        s = torch.einsum("bhgd,bkhd->bhgk", qf, kk) * scale
        s = s.masked_fill(~valid[:, None, None, :], float("-inf"))
        p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
        return torch.einsum("bhgk,bkhd->bhgd", p, vv).reshape(b, hq, dv).to(q.dtype)
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()                     # general varlen (prefill-shaped)
    outs = []
    for i in range(b):
        qi = q[cq[i]:cq[i + 1]].float()
        ki = k[ck[i]:ck[i + 1]].float().repeat_interleave(g, dim=1)
        vi = v[ck[i]:ck[i + 1]].float().repeat_interleave(g, dim=1)
        lq, lk = qi.shape[0], ki.shape[0]
        s = torch.einsum("qhd,khd->hqk", qi, ki) * scale
        if causal:
            keep = torch.ones(lq, lk, dtype=torch.bool, device=q.device).tril(diagonal=lk - lq)
            s = s.masked_fill(~keep[None], float("-inf"))
        p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
        outs.append(torch.einsum("hqk,khd->qhd", p, vi))
    return torch.cat(outs).to(q.dtype) if outs else q.new_zeros(0, hq, dv)
