"""Hypothesis test: in SGLang 0.5.21 GPTQ-Marlin MoE under EP4 (no a2a), a -1 (non-local) expert id is computed as some
local expert instead of being skipped. Uses the RC1 fixture L1 fw1 rank1 (all 10 ids non-local, Marlin out != 0).
Fetches only the needed Intel AutoRound expert tensors from HF by HTTP range reads."""
import json, struct, sys, urllib.request, glob
import torch
REPO = "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"
IDX = json.load(open(r"D:/ARC AGI 2/_scratch/hf_intel_fn_cfg/index_hf.json"))["weight_map"]
hdr_cache = {}
def url(f): return f"https://huggingface.co/{REPO}/resolve/main/{f}"
def rng(f, a, b):
    r = urllib.request.Request(url(f), headers={"Range": f"bytes={a}-{b}", "User-Agent": "rc1"})
    return urllib.request.urlopen(r, timeout=120).read()
def header(f):
    if f not in hdr_cache:
        n = struct.unpack("<Q", rng(f, 0, 7))[0]
        hdr_cache[f] = (8 + n, json.loads(rng(f, 8, 8 + n - 1)))
    return hdr_cache[f]
DT = {"I32": torch.int32, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
def tensor(name):
    f = IDX[name]; base, h = header(f); m = h[name]; a, b = m["data_offsets"]
    raw = rng(f, base + a, base + b - 1)
    return torch.frombuffer(bytearray(raw), dtype=DT[m["dtype"]]).reshape(m["shape"]).clone()
def deq(qw, qz, sc, g=128):
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((qw.unsqueeze(1) >> sh.view(1, 8, 1)) & 0xF).reshape(-1, qw.shape[1])
    z = ((qz.unsqueeze(2) >> sh.view(1, 1, 8)) & 0xF).reshape(qz.shape[0], -1) + 1
    gi = torch.arange(q.shape[0]) // g
    return (q.float() - z[gi].float()) * sc.float()[gi]
cache = {}
def expert(L, e):
    if (L, e) not in cache:
        W = {}
        for p in ("gate_proj", "up_proj", "down_proj"):
            b = f"model.language_model.layers.{L}.mlp.experts.{e}.{p}."
            W[p] = deq(tensor(b + "qweight"), tensor(b + "qzeros"), tensor(b + "scales"))
        cache[(L, e)] = W
    return cache[(L, e)]
def mlp(L, e, x):
    W = expert(L, e)
    return ((torch.nn.functional.silu(x @ W["gate_proj"]) * (x @ W["up_proj"])) @ W["down_proj"])[0]
D = sys.argv[1]
fx = {}
for f in sorted(glob.glob(D + "/fixture_L1_*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    fx[(d["fw"], d["meta"]["moe_ep_rank"])] = d
rel = lambda a, b: float((a - b).norm() / max(float(b.norm()), 1e-9))
for (fw, r) in [(1, 1), (1, 2), (0, 0)]:
    d = fx[(fw, r)]
    x = d["hidden"].float()[0:1]; ids = d["ints"][0][0].tolist(); w = d["floats"][1][0].float().tolist(); em = d["expert_map"]
    out = d["out"].float()[0]
    lo, hi = r * 128, r * 128 + 127
    owned = [(k, g) for k, g in enumerate(ids) if lo <= g <= hi]
    y_loc = sum((mlp(1, g, x) * w[k] for k, g in owned), torch.zeros(2560))
    hyps = {"skip(-1 ignored)": y_loc}
    for name, ge in (("-1 -> local 127", hi), ("-1 -> local 0", lo)):
        extra = sum((mlp(1, ge, x) * w[k] for k, g in enumerate(ids) if not (lo <= g <= hi)), torch.zeros(2560))
        hyps[name] = y_loc + extra
    print(f"fw{fw} rank{r}: owned {len(owned)}/10 | Marlin out norm {float(out.norm()):.4f}")
    for n, y in hyps.items():
        print(f"    {n:18s} norm {float(y.norm()):.4f}  rel_err {rel(out, y):.4f}")

print("\n--- hypothesis: local index = global id % 128 (global ids fed to the local weight table)")
for (fw, r) in [(1, 1), (1, 2), (0, 0), (1, 0), (1, 3)]:
    d = fx[(fw, r)]
    x = d["hidden"].float()[0:1]; ids = d["ints"][0][0].tolist(); w = d["floats"][1][0].float().tolist()
    out = d["out"].float()[0]
    y = sum((mlp(1, r * 128 + (g % 128), x) * w[k] for k, g in enumerate(ids)), torch.zeros(2560))
    print(f"fw{fw} rank{r}: Marlin norm {float(out.norm()):.4f} | mod128 norm {float(y.norm()):.4f} rel_err {rel(out, y):.4f}")
