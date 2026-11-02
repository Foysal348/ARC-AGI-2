"""E27 NBDF-K100 NaN isolation test of the DFlash2 grouped convolution (local + the Py3.12 CPU units kit; torch, no vllm import).
  N0  the source: Merlin's vllm/model_executor/models/qwen3_dflash2.py at its ad42e37 pre-sha (dflash_seam.DF2_PRE); the seam's
      DF2 patch applied by dflash_seam.apply_unified gives its post-sha (DF2_POST); _grouped_conv (the original, :144-167) and the
      seam's _e27_grouped_conv_torch / patched _grouped_conv are extracted by AST (only those functions + torch / F)
  N1  the seam's torch path (DFLASH_CONV_IMPL=torch, SGLang #41736 torch.where masking): several draft blocks of 1 + K rows
      (K 3 / 4 / 7 -> block 4 / 5 / 8: the power-of-2 and the modulo position branches), the drafter's real conv geometry
      (dflash_config conv_kernel_size 2, conv_group_size 16, hidden 2560), float32 and bfloat16; NaN or +Inf injected into
      one block's rows (hidden states, delta or both): every OTHER block's output rows are bit-identical to the clean run
      (torch.equal), and the poisoned block itself is non-finite (the injection reached the conv)
  N2  the same through the patched _grouped_conv dispatcher with _E27_CONV_TORCH True (the engine's call path)
  N3  for finite inputs the seam's torch path equals Merlin's original torch fallback bit for bit (no numeric change)
  N4  negative control (recorded as a check of the test's sensitivity): Merlin's ORIGINAL CPU fallback multiplies the shifted
      taps by the mask (0 * NaN = NaN), so a NaN block leaks into the first taps - 1 rows of the NEXT block; the same
      comparison must detect that leak (else the test could not see a leak at all)
Run: python nbdf_k100_nan_tests.py [--merlin-src <dir with vllm/>] [--drafter-config <config.json>]
  -> 'FAIL <name>' lines, then '<pass>/<total> PASS'; no torch -> 'SKIP torch not importable' and '0/1 PASS' (exit 2)."""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REL = "vllm/model_executor/models/qwen3_dflash2.py"
RES = {"pass": 0, "fail": 0}


def ok(name, cond, info=""):
    if cond:
        RES["pass"] += 1
    else:
        RES["fail"] += 1
        print("FAIL", name, str(info)[:400], flush=True)


def finish():
    print(f"{RES['pass']}/{RES['pass'] + RES['fail']} PASS", flush=True)
    sys.exit(1 if RES["fail"] else 0)


ap = argparse.ArgumentParser()
ap.add_argument("--merlin-src", default=None)
ap.add_argument("--drafter-config", default=None)
a = ap.parse_args()
try:
    import torch
    import torch.nn.functional as F
except Exception as e:  # noqa: BLE001
    print(f"SKIP torch not importable: {type(e).__name__}: {e}"[:300], flush=True)
    print("0/1 PASS", flush=True)
    sys.exit(2)
import dflash_seam as DSM  # noqa: E402

cands = [Path(a.merlin_src)] if a.merlin_src else [HERE.parent / "merlin", Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")]
src_dir = next((c for c in cands if (c / REL).is_file()), None)
if src_dir is None:
    ok("N0 Merlin qwen3_dflash2.py found", False, [str(c) for c in cands])
    finish()
orig = (src_dir / REL).read_bytes().decode("utf-8")
pre, post = DSM.PATCHES[REL][0], DSM.PATCHES[REL][1]
ok("N0 original qwen3_dflash2.py at the Merlin ad42e37 pre-sha", hashlib.sha256(orig.encode("utf-8")).hexdigest() == pre)
patched = DSM.apply_unified(orig, DSM.PATCHES[REL][3])
ok("N0 seam DF2 patch -> post-sha", hashlib.sha256(patched.encode("utf-8")).hexdigest() == post)

cfg = {"dflash_config": {"conv_kernel_size": 2, "conv_group_size": 16}, "hidden_size": 2560}
for c in ([Path(a.drafter_config)] if a.drafter_config else []) + [HERE.parent / "drafter_config.json"]:
    if c.is_file():
        cfg = json.loads(c.read_text(encoding="utf-8"))
        break
TAPS, GS, HID = int(cfg["dflash_config"]["conv_kernel_size"]), int(cfg["dflash_config"]["conv_group_size"]), int(cfg["hidden_size"])
NG = HID // GS
ok("N0 conv geometry", TAPS >= 2 and HID % GS == 0, (TAPS, GS, HID))


def extract(text, names, extra=None):
    """The named top-level functions (and assignments) of a module source, executed with only torch / F (no vllm)."""
    tree = ast.parse(text)
    keep = []
    for node in tree.body:
        nm = getattr(node, "name", None)
        if nm is None and isinstance(node, ast.Assign):
            nm = next((t.id for t in node.targets if isinstance(t, ast.Name)), None)
        if nm in names:
            keep.append(ast.get_source_segment(text, node))
    ns = {"torch": torch, "F": F, "__name__": "nbdf_k100_nan"}

    def _no_fused(*_a, **_k):
        raise RuntimeError("the fused CUDA kernel must not run on CPU tensors")
    ns["dflash2_grouped_conv"] = _no_fused
    ns.update(extra or {})
    exec(compile("\n\n".join(keep), "<extracted qwen3_dflash2>", "exec"), ns)
    return ns, len(keep)


ns_o, n_o = extract(orig, {"_grouped_conv"})
ns_p, n_p = extract(patched, {"_e27_grouped_conv_torch", "_grouped_conv"}, {"_E27_CONV_TORCH": True})
ok("N0 extracted: original _grouped_conv, seam _e27_grouped_conv_torch + _grouped_conv", n_o == 1 and n_p == 2, (n_o, n_p))
orig_fn, seam_fn, seam_disp = ns_o["_grouped_conv"], ns_p["_e27_grouped_conv_torch"], ns_p["_grouped_conv"]
src_o = ast.get_source_segment(orig, next(n for n in ast.parse(orig).body if getattr(n, "name", None) == "_grouped_conv"))
ok("N0 the original CPU fallback multiplies the shifted taps by the mask (the 0 * NaN path)", "* (position >= tap)" in src_o)

N_BLOCKS, BAD = 6, 2
NEG = {}
for k in (3, 4, 7):
    bs = 1 + k
    T_ = N_BLOCKS * bs
    for dt in (torch.float32, torch.bfloat16):
        g = torch.Generator().manual_seed(1000 * k + (dt == torch.bfloat16))
        h = torch.randn(T_, HID, generator=g).to(dt)
        d = (0.1 * torch.randn(T_, TAPS, NG, generator=g)).to(dt)
        base = (0.1 * torch.randn(TAPS, HID, generator=g)).to(dt)
        rows = slice(BAD * bs, (BAD + 1) * bs)
        other = [i for i in range(T_) if not (rows.start <= i < rows.stop)]
        tag = f"K{k} block {bs} {str(dt).split('.')[-1]}"
        for fn_name, fn in (("seam torch path", seam_fn), ("seam dispatcher", seam_disp)):
            clean = fn(h, d, base, bs, NG, GS, TAPS)
            ok(f"N1 {fn_name} {tag}: clean output finite, shape [T, hidden]", tuple(clean.shape) == (T_, HID) and bool(torch.isfinite(clean).all()))
            for val in (float("nan"), float("inf")):
                for where in ("hidden", "delta", "both"):
                    h2, d2 = h.clone(), d.clone()
                    if where in ("hidden", "both"):
                        h2[rows] = val
                    if where in ("delta", "both"):
                        d2[rows] = val
                    out = fn(h2, d2, base, bs, NG, GS, TAPS)
                    same = torch.equal(out[other], clean[other])
                    bad_rows = sorted({i for i in other if not torch.equal(out[i], clean[i])})
                    ok(f"{'N1' if fn is seam_fn else 'N2'} {fn_name} {tag} {val} in {where}: every other block bit-identical",
                       same, f"differing rows {bad_rows[:8]}")
                    ok(f"{'N1' if fn is seam_fn else 'N2'} {fn_name} {tag} {val} in {where}: the poisoned block is non-finite",
                       not bool(torch.isfinite(out[rows]).all()))
        ok(f"N3 {tag}: seam torch path == Merlin original fallback on finite inputs", torch.equal(seam_fn(h, d, base, bs, NG, GS, TAPS),
                                                                                                 orig_fn(h, d, base, bs, NG, GS, TAPS)))
        h2 = h.clone()
        h2[rows] = float("nan")
        out_o, clean_o = orig_fn(h2, d, base, bs, NG, GS, TAPS), orig_fn(h, d, base, bs, NG, GS, TAPS)
        leak = sorted(i for i in other if not torch.equal(out_o[i], clean_o[i]))
        want = list(range((BAD + 1) * bs, (BAD + 1) * bs + TAPS - 1))
        NEG[tag] = leak
        ok(f"N4 negative control {tag}: Merlin's original fallback leaks the NaN block into the next block's first taps - 1 rows "
           f"and the comparison sees it", leak == want, f"leak rows {leak[:8]} want {want}")
print("NOTE N4 original-fallback leak rows (recorded):", json.dumps(NEG), flush=True)
finish()
