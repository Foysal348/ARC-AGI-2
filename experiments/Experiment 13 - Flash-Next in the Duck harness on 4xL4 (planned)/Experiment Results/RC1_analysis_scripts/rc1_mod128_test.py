"""Hypotheses for SGLang GPTQ-Marlin MoE under EP4 (RC1 fixtures, layer 1): which experts does Marlin actually compute?
H_skip: only owned experts (correct). H_mod: local index = global id % 128 for every id (dispatcher map not applied).
Tensors fetched by HTTP range reads from the Intel checkpoint, cached on disk, 16 in parallel."""
import json, struct, sys, urllib.request, glob, os
from concurrent.futures import ThreadPoolExecutor
import torch
REPO = "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"
IDX = json.load(open(r"D:/ARC AGI 2/_scratch/hf_intel_fn_cfg/index_hf.json"))["weight_map"]
CD = r"D:/ARC AGI 2/_scratch/rc1_tensor_cache"; os.makedirs(CD, exist_ok=True)
def rng(f, a, b):
    r = urllib.request.Request(f"https://huggingface.co/{REPO}/resolve/main/{f}", headers={"Range": f"bytes={a}-{b}", "User-Agent": "rc1"})
    return urllib.request.urlopen(r, timeout=120).read()
HDR = {}
def header(f):
    if f not in HDR:
        n = struct.unpack("<Q", rng(f, 0, 7))[0]; HDR[f] = (8 + n, json.loads(rng(f, 8, 8 + n - 1)))
    return HDR[f]
DT = {"I32": torch.int32, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
def tensor(name):
    p = os.path.join(CD, name.replace("/", "_") + ".pt")
    if os.path.exists(p):
        return torch.load(p)
    f = IDX[name]; base, h = header(f); m = h[name]; a, b = m["data_offsets"]
    t = torch.frombuffer(bytearray(rng(f, base + a, base + b - 1)), dtype=DT[m["dtype"]]).reshape(m["shape"]).clone()
    torch.save(t, p); return t
def names(L, e):
    return [f"model.language_model.layers.{L}.mlp.experts.{e}.{p}.{s}" for p in ("gate_proj", "up_proj", "down_proj") for s in ("qweight", "qzeros", "scales")]
def deq(qw, qz, sc, g=128):
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((qw.unsqueeze(1) >> sh.view(1, 8, 1)) & 0xF).reshape(-1, qw.shape[1])
    z = ((qz.unsqueeze(2) >> sh.view(1, 1, 8)) & 0xF).reshape(qz.shape[0], -1) + 1
    gi = torch.arange(q.shape[0]) // g
    return (q.float() - z[gi].float()) * sc.float()[gi]
def mlp(L, e, x):
    t = [tensor(n) for n in names(L, e)]
    G, U, Dn = deq(*t[0:3]), deq(*t[3:6]), deq(*t[6:9])
    return ((torch.nn.functional.silu(x @ G) * (x @ U)) @ Dn)[0]
D = sys.argv[1]
fx = {}
for f in sorted(glob.glob(D + "/fixture_L1_*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False); fx[(d["fw"], d["meta"]["moe_ep_rank"])] = d
cases = [(1, 1), (0, 0)]
need = set()
for fw, r in cases:
    for g in fx[(fw, r)]["ints"][0][0].tolist():
        need.add(g); need.add(r * 128 + g % 128)
todo = [n for e in sorted(need) for n in names(1, e)]
for f in {IDX[n] for n in todo}:
    header(f)
with ThreadPoolExecutor(16) as ex:
    list(ex.map(tensor, todo))
rel = lambda a, b: float((a - b).norm() / max(float(b.norm()), 1e-9))
for fw, r in cases:
    d = fx[(fw, r)]
    x = d["hidden"].float()[0:1]; ids = d["ints"][0][0].tolist(); w = d["floats"][1][0].float().tolist(); out = d["out"].float()[0]
    lo, hi = r * 128, r * 128 + 127
    y_skip = sum((mlp(1, g, x) * w[k] for k, g in enumerate(ids) if lo <= g <= hi), torch.zeros(2560))
    y_mod = sum((mlp(1, lo + g % 128, x) * w[k] for k, g in enumerate(ids)), torch.zeros(2560))
    print(f"fw{fw} rank{r}: owned {sum(lo <= g <= hi for g in ids)}/10 | Marlin norm {float(out.norm()):.4f}")
    print(f"   H_skip (correct)       norm {float(y_skip.norm()):.4f}  rel_err {rel(out, y_skip):.4f}")
    print(f"   H_mod (global id %128) norm {float(y_mod.norm()):.4f}  rel_err {rel(out, y_mod):.4f}  cos {float(torch.nn.functional.cosine_similarity(out, y_mod, 0)):.4f}")
