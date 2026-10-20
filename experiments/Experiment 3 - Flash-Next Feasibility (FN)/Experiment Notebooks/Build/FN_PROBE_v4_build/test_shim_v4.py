"""CPU check of q27_fa2_varlen_shim against a naive per-sequence, per-head reference (FA2 semantics)."""
import math, torch
from q27_fa2_varlen_shim import flash_attn_varlen_func

def ref(q, k, v, cq, ck, scale, causal):
    hq, hkv = q.shape[1], k.shape[1]; g = hq // hkv; out = torch.zeros(q.shape[0], hq, v.shape[-1], dtype=torch.float64)
    for i in range(len(cq) - 1):
        for h in range(hq):
            qi = q[cq[i]:cq[i + 1], h].double(); ki = k[ck[i]:ck[i + 1], h // g].double(); vi = v[ck[i]:ck[i + 1], h // g].double()
            lq, lk = qi.shape[0], ki.shape[0]
            for r in range(lq):
                if lk == 0: continue
                last = lk - lq + r if causal else lk - 1
                if last < 0: continue
                s = (ki[:last + 1] @ qi[r]) * scale
                p = torch.softmax(s, 0); out[cq[i] + r, h] = p @ vi[:last + 1]
    return out

torch.manual_seed(0)
for hq, hkv, d in ((8, 1, 64), (16, 4, 32), (4, 4, 16)):
    lens = [5, 0, 17, 1, 9]                                       # includes an empty top-k selection
    ck = [0]; [ck.append(ck[-1] + n) for n in lens]
    b = len(lens); cq = list(range(b + 1))
    q = torch.randn(b, hq, d, dtype=torch.bfloat16); k = torch.randn(ck[-1], hkv, d, dtype=torch.bfloat16); v = torch.randn(ck[-1], hkv, d, dtype=torch.bfloat16)
    sc = 1 / math.sqrt(d)
    o = flash_attn_varlen_func(q, k, v, torch.tensor(cq, dtype=torch.int32), torch.tensor(ck, dtype=torch.int32), 1, max(lens), softmax_scale=sc, causal=True)
    r = ref(q, k, v, cq, ck, sc, True)
    assert o.shape == (b, hq, d) and o.dtype == torch.bfloat16
    err = (o.double() - r).abs().max().item(); assert err < 3e-2, err
    assert o[1].abs().max().item() == 0.0                         # empty selection -> zeros, no NaN
    # prefill-shaped varlen with causal bottom-right alignment
    lq = [3, 1, 4]; lk = [5, 2, 4]
    cq2 = [0]; [cq2.append(cq2[-1] + n) for n in lq]; ck2 = [0]; [ck2.append(ck2[-1] + n) for n in lk]
    q2 = torch.randn(cq2[-1], hq, d, dtype=torch.bfloat16); k2 = torch.randn(ck2[-1], hkv, d, dtype=torch.bfloat16); v2 = torch.randn(ck2[-1], hkv, d, dtype=torch.bfloat16)
    for causal in (True, False):
        o2 = flash_attn_varlen_func(q2, k2, v2, torch.tensor(cq2, dtype=torch.int32), torch.tensor(ck2, dtype=torch.int32), 4, 5, softmax_scale=sc, causal=causal)
        e2 = (o2.double() - ref(q2, k2, v2, cq2, ck2, sc, causal)).abs().max().item(); assert e2 < 3e-2, (causal, e2)
    print(f"shim OK hq={hq} hkv={hkv} d={d}: decode max err {err:.4f}, prefill ok")
print("ALL SHIM TESTS PASS")
