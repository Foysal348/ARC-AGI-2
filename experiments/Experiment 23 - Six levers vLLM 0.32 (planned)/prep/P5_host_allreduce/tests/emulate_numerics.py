"""CPU emulation of the P5 kernel numerics (torch, no GPU), used to validate the bench's oracles before the GPU run.

Emulates host_ar.cu exactly at the arithmetic level:
  native: out = bf16/f32( ((x0 + x1) + x2) + x3 ) in FP32, rank order 0..W-1 (acc starts at x0)
  int8:   non-owner contributions quantized per 32-element group (amax floored at 1e-36 when > 0, scale = amax/127
          in FP32, q = rint(x * (127/amax)), clamp +-127; a group with any NaN/Inf gets scale NaN), owner adds its own
          unquantized chunk, the FP32 sum is quantized again per group, every rank outputs dequant(q) * scale rounded
          to the dtype.
Checks: native == bench.ordered_ref bit for bit; int8 passes bench.int8_eval (non-finite groups all-NaN, every other
element within bench.int8_bound); random + the bench's adversarial patterns incl. tiny, tiny37 and nan_inf; odd sizes.
Prints EMU_JSON {...}; exit 0 when all pass.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench_host_ar as B  # noqa: E402

G = 32


def chunk_bounds(n, world):
    cl = -(-(-(-n // world)) // G) * G
    return [(min(n, cl * j), min(n, cl * (j + 1))) for j in range(world)]


def quant(v):
    """v: float32 [m] with m % 32 == 0 -> (q int, scale per group)."""
    g = v.view(-1, G)
    nonfinite = (~torch.isfinite(g)).any(1)
    a = torch.where(nonfinite[:, None], torch.zeros_like(g), g).abs().amax(1)
    a = torch.where(a > 0, a.clamp_min(B.INT8_FLOOR), a)
    scale = a / 127.0
    inv = torch.where(a > 0, 127.0 / a, torch.zeros_like(a))
    q = torch.round(torch.where(nonfinite[:, None], torch.zeros_like(g), g) * inv[:, None]).clamp(-127, 127)
    scale = torch.where(nonfinite, torch.full_like(scale, float("nan")), scale)
    return q, scale


def deq(q, scale):
    return (q * scale[:, None]).reshape(-1)


def emulate(xs, mode, dt):
    world, n = len(xs), xs[0].numel()
    f = [x.float() for x in xs]
    out = torch.empty(n, dtype=torch.float32)
    for r, (cb, ce) in enumerate(chunk_bounds(n, world)):
        if ce <= cb:
            continue
        L = ce - cb
        Lp = -(-L // G) * G
        parts = []
        for j in range(world):
            v = torch.zeros(Lp)
            v[:L] = f[j][cb:ce]
            if mode == 1 and j != r:
                q, s = quant(v)
                v = deq(q, s)
            parts.append(v)
        acc = parts[0].clone()
        for v in parts[1:]:
            acc = acc + v
        if mode == 1:
            acc[L:] = 0
            q, s = quant(acc)
            acc = deq(q, s)
        out[cb:ce] = acc[:L]
    return out.to(dt)


def main():
    res = {"native_cases": 0, "native_fail": [], "int8_cases": 0, "int8_fail": [], "int8_worst_err_over_bound": 0.0}
    dev = torch.device("cpu")
    for dt in (torch.bfloat16, torch.float32):
        esz = torch.tensor([], dtype=dt).element_size()
        for nbytes in [2, 64, 1000, 4098, 65536, 245760, 983042, 983040]:
            if nbytes % esz:
                continue
            n = nbytes // esz
            for pat in B.PATTERNS:
                xs = [B.gen(torch, dev, 7 + nbytes, j, n, dt, pat) for j in range(4)]
                ref = B.ordered_ref(torch, xs, dt)
                got = emulate(xs, 0, dt)
                ok, nd = B.bit_equal(torch, got, ref)
                res["native_cases"] += 1
                if not ok:
                    res["native_fail"].append([str(dt), nbytes, pat, nd])
            for pat in B.INT8_PATTERNS:
                xs = [B.gen(torch, dev, 11 + nbytes, j, n, dt, pat) for j in range(4)]
                got = emulate(xs, 1, dt)
                ev = B.int8_eval(torch, xs, got, n, 4, dt)
                res["int8_cases"] += 1
                res["int8_worst_err_over_bound"] = max(res["int8_worst_err_over_bound"], ev["worst_err_over_bound"])
                if pat == "nan_inf":
                    res.setdefault("nan_inf_groups", 0)
                    res["nan_inf_groups"] += ev["nonfinite_groups"]
                if not ev["ok"]:
                    res["int8_fail"].append([str(dt), nbytes, pat, ev])
    # negative control: the pre-fix codec (no floor, NaN dropped / Inf -> 0*inf) must fail the new checks
    def quant_old(v):
        g = v.view(-1, G)
        a = torch.nan_to_num(g, nan=0.0).abs().amax(1)
        inv = torch.where(a > 0, 127.0 / a, torch.zeros_like(a)).float()
        q = torch.nan_to_num(torch.round(g * inv[:, None]), nan=0.0).clamp(-127, 127)
        return q, a / 127.0
    global quant
    saved = quant
    quant = quant_old
    try:
        caught = {}
        for pat in ("tiny37", "nan_inf"):
            n = 4096
            xs = [B.gen(torch, dev, 99, j, n, torch.float32, pat) for j in range(4)]
            caught[pat] = not B.int8_eval(torch, xs, emulate(xs, 1, torch.float32), n, 4, torch.float32)["ok"]
        res["old_codec_caught"] = caught
    finally:
        quant = saved
    res["all_ok"] = not res["native_fail"] and not res["int8_fail"] and all(res["old_codec_caught"].values())
    print("EMU_JSON " + json.dumps(res))
    sys.exit(0 if res["all_ok"] else 1)


if __name__ == "__main__":
    main()
