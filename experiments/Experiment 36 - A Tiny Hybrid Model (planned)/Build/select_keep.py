"""E36: build the expert keep-list from the gEAN calibration dumps.

usage: python select_keep.py <dir with gean_*.pt> <K> <out.json>
Each gean_<pid>.pt = {layer: float64 [3, E+2]} rows (gEAN sum, routed count, router-weight sum); the
TP/EP ranks hold different experts, so the files are summed. Keeps the top-K experts by gEAN per layer.
Prints, per layer, the share of gEAN mass and of routed tokens that the kept experts carry.
"""
import json
import sys
from pathlib import Path

import torch

src, K, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
E = 512
tot = {}
files = sorted(src.glob("gean_*.pt"))
assert files, f"no gean_*.pt in {src}"
for f in files:
    for L, acc in torch.load(f, map_location="cpu").items():
        tot[L] = tot.get(L, 0) + acc[:, :E]
assert sorted(tot) == list(range(48)), f"layers seen: {sorted(tot)}"
layers, stats = {}, []
for L in range(48):
    g, cnt = tot[L][0], tot[L][1]
    assert (cnt > 0).sum() > K, f"layer {L}: only {(cnt > 0).sum().item()} experts seen"
    top = torch.argsort(g, descending=True)[:K]
    keep = sorted(int(e) for e in top)
    layers[str(L)] = keep
    stats.append((L, float(g[top].sum() / g.sum()), float(cnt[top].sum() / cnt.sum()), int((cnt == 0).sum()), int(cnt.sum())))
json.dump({"num_experts": K, "source": [f.name for f in files], "layers": layers}, open(out, "w"))
print(f"files {len(files)} | K {K} -> {out}")
print("layer | gEAN kept | routed kept | experts never routed | routed slots")
for L, gm, cm, z, n in stats:
    print(f"{L:5d} | {gm:9.4f} | {cm:11.4f} | {z:20d} | {n}")
print(f"mean gEAN kept {sum(s[1] for s in stats) / 48:.4f} | min {min(s[1] for s in stats):.4f} (layer {min(stats, key=lambda s: s[1])[0]})")
