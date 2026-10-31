import torch, glob, sys
D = sys.argv[1]
fx = {}
for f in sorted(glob.glob(D + "/fixture_*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    L = 1 if "_L1_" in f else 2
    fx[(L, d["fw"], d["meta"]["moe_ep_rank"])] = d
d = fx[(1, 1, 1)]
print("L1 fw1 rank1 | int shapes", d["int_shapes"], "| float shapes", d["float_shapes"], "| out", list(d["out"].shape), d["out"].dtype)
ids = d["ints"][0]; em = d["expert_map"]
print("ids", ids.tolist(), "| local via map", [int(em[i]) for i in ids[0].tolist()])
print("out norm", float(d["out"].float().norm()), "| out[0,:6]", [round(v, 5) for v in d["out"][0, :6].float().tolist()], "| hidden norm", float(d["hidden"].float().norm()))
for L, fw in ((1, 0), (1, 1)):
    outs = [fx[(L, fw, r)]["out"].float() for r in range(4)]
    print(f"L{L} fw{fw}: per-rank out norms", [round(float(o.norm()), 4) for o in outs], "| sum norm", round(float(sum(outs).norm()), 4),
          "| max|out_r-out_0|", [round(float((o - outs[0]).abs().max()), 6) for o in outs])
print("hidden identical across ranks L1 fw0:", all(torch.equal(fx[(1,0,r)]["hidden"], fx[(1,0,0)]["hidden"]) for r in range(4)))
print("ints L1 fw0 r0:", [(list(t.shape), str(t.dtype), int(t.min()), int(t.max())) for t in fx[(1,0,0)]["ints"]])
print("floats L1 fw0 r0:", [(list(t.shape), str(t.dtype)) for t in fx[(1,0,0)]["floats"]])
print("expert_map r1:", fx[(1,0,1)]["expert_map"][120:135].tolist(), "... dtype", fx[(1,0,1)]["expert_map"].dtype, "len", fx[(1,0,1)]["expert_map"].numel())
