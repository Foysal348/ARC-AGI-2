"""RC1b: extend unit_rc1.py for the RC1 audit fixes: dispatcher expert map, in-place MoE (pre-hook copy + pre-hook
input check), non-contiguous 3D scan without a full copy, layer-2 sync env, class-level checks."""
from pathlib import Path

p = Path(r"D:/ARC AGI 2/_scratch/e13_nanprobe_test/unit_rc1.py")
s = p.read_text(encoding="utf-8")
R = [
    ('''        s.register_buffer("expert_map_cpu", torch.tensor([0, 1, 2, 3] + [-1] * 12, dtype=torch.int32))
        if SCEN == "badscale" and i == 2:
            with torch.no_grad():
                s.w13_scales[3, 1, 5] = float("nan")
    def forward(s, hidden_states, topk_output):
        y = hidden_states * topk_output.topk_weights.sum(-1, keepdim=True)
        if s.i == 2 and SCEN in ("badscale", "clean_nan"):
            y = y * float("nan")''',
     '''        s.register_buffer("expert_map_cpu", torch.tensor([0, 1, 2, 3] + [-1] * 12, dtype=torch.int32))
        import types as _t
        s.dispatcher = _t.SimpleNamespace(local_expert_mapping=torch.tensor([-1] * 8 + [0, 1, 2, 3] + [-1] * 4, dtype=torch.int32))
        s.reduce_results = False
        w2 = torch.rand(4, 16, 3)
        w2[2, 5, 1] = float("nan")
        s.w2_scales = nn.Parameter(w2.transpose(1, 2), requires_grad=False)      # non-contiguous 3D [4, 3, 16]
        if SCEN == "badscale" and i == 2:
            with torch.no_grad():
                s.w13_scales[3, 1, 5] = float("nan")
    def forward(s, hidden_states, topk_output):
        y = hidden_states * topk_output.topk_weights.sum(-1, keepdim=True)
        if s.i == 2 and SCEN in ("badscale", "clean_nan"):
            y = y * float("nan")
        if SCEN == "inplace" and s.i in (1, 2):              # SGLang inplace=True: the output overwrites the input
            with torch.no_grad():
                hidden_states.copy_(y * (float("nan") if s.i == 2 else 3.0))
            return hidden_states'''),
]
for a, b in R:
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
A = '''p, r, res, tdir = run("mutate", {"E13_L2_NO_MARLIN": "0"})'''
B = '''t2b = ((sc.get("after_postprocess") or {}).get("2") or {}).get("tensors") or {}
w2 = t2b.get("w2_scales") or {}
check("non-contiguous 3D scale scanned row by row: 1 NaN in expert row 2, contig False",
      w2.get("nonfinite") == 1 and w2.get("bad_rows") == [2] and w2.get("contig") is False and w2.get("shape") == [4, 3, 16], w2)
dm = t2b.get("dispatcher.local_expert_mapping") or {}
check("scan includes the dispatcher's EP map", dm.get("min") == -1 and dm.get("max") == 3 and dm.get("shape") == [16], dm)
if fx:
    import torch
    d0 = torch.load(tdir / [n for n in fx if "L2_fw0" in n][0], weights_only=False)
    check("fixture expert map = the dispatcher map (rank 2 block 8..11), reduce_results recorded, hidden from pre hook",
          d0["expert_map"].tolist() == [-1] * 8 + [0, 1, 2, 3] + [-1] * 4 and d0["meta"].get("reduce_results") is False
          and d0.get("hidden_from") == "pre_hook", (d0.get("expert_map"), d0.get("meta"), d0.get("hidden_from")))

p, r, res, tdir = run("inplace", {"E13_SYNC_RE": r"layers\\.2\\.mlp\\.experts$"})
nf = [x for x in r if x["ev"] == "nan_forward"]
f0 = (nf[0]["first_nonfinite"] or [{}])[0] if nf else {}
check("in-place MoE: input finiteness checked before the overwrite (inputs_bad False, origin layer 2)",
      p.returncode == 0 and f0.get("module") == "model.layers.2.mlp.experts" and f0.get("inputs_bad") is False, f0)
import torch
fxs = {f.name: torch.load(f, weights_only=False) for f in tdir.glob("fixture_L1_fw0_*.pt")}
d1 = next(iter(fxs.values()), None)
check("in-place MoE: the fixture keeps the true input (output = 3 x input x sum(w), not equal to the stored hidden)",
      d1 is not None and d1["hidden_from"] == "pre_hook" and not torch.allclose(d1["hidden"].float(), d1["out"].float()), d1 and d1["hidden_from"])
check("E13_SYNC_RE on CPU: the failed sync is logged, never a hook error",
      any(x["ev"] == "sync_error" for x in r) and not [x for x in r if x["ev"] == "hook_error"], [x["ev"] for x in r if "error" in x["ev"]][:5])

p, r, res, tdir = run("mutate", {"E13_L2_NO_MARLIN": "0"})'''
assert s.count(A) == 1
s = s.replace(A, B)
p.write_text(s, encoding="utf-8")
print("unit_rc1 extended")
