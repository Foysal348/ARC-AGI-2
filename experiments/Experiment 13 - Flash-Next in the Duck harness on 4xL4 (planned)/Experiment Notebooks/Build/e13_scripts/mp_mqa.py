"""E13 micro-preflight M: QSA indexer decode scores, TileLang vs SGLang's torch reference, on this GPU (no model).

Flash-Next indexer: 4 heads x head_dim 128 over a paged compressed key cache (page 64). Both implementations come from
SGLang 0.5.21 `srt/layers/attention/qsa/mqa.py` (tilelang_qsa_mqa_decode / torch_qsa_mqa_decode). Checks:
  - same -inf mask; finite scores agree within 0.02 + 0.02*|ref| (bf16 inputs, fp32 accumulate);
  - the top-512 chosen from TileLang scores carries the same value multiset (under the torch scores) as the torch top-512
    (selection-level agreement, tie-aware; this is what reaches the sparse attention);
  - repeat stability (5 runs bitwise equal) and torch-path peak memory (the B2 'TileLang off' cost).
Usage: python -I mp_mqa.py --out RESULT.json
"""
import argparse
import json
import sys
import traceback

import torch

HEADS, HD, PAGE, K = 4, 128, 64, 512


def case(batch, comp_len, gen, device):
    pages_per_row = (comp_len + PAGE - 1) // PAGE
    total_pages = batch * pages_per_row + 3
    q = torch.randn(batch, HEADS, HD, generator=gen).to(torch.bfloat16)
    kc = torch.randn(total_pages, PAGE, 1, HD, generator=gen).to(torch.bfloat16)
    perm = torch.randperm(total_pages, generator=gen)[: batch * pages_per_row].reshape(batch, pages_per_row).int()
    lens = torch.tensor([comp_len if b % 3 else max(1, comp_len // 2) for b in range(batch)], dtype=torch.int32)
    return [t.to(device) for t in (q, kc, perm, lens)], pages_per_row * PAGE


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    rep = {"has_tilelang": None, "cases": [], "fails": 0, "torch_peak_gb": None, "error": None}
    try:
        from sglang.srt.layers.attention.qsa import mqa
        rep["has_tilelang"] = bool(mqa.HAS_TILELANG)
        gen = torch.Generator().manual_seed(20261019)
        for batch, comp in ((1, 2048), (4, 8192), (10, 8192), (4, 34816), (10, 34816)):
            (q, kc, pt, lens), mml = case(batch, comp, gen, "cuda")
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
            ref = mqa.torch_qsa_mqa_decode(q, kc, pt, lens, mml); torch.cuda.synchronize()
            peak = (torch.cuda.max_memory_allocated() - base) / 1e9
            c = {"batch": batch, "comp_len": comp, "torch_peak_gb": round(peak, 3)}
            if rep["has_tilelang"]:
                outs = [mqa.tilelang_qsa_mqa_decode(q, kc, pt, lens, mml) for _ in range(5)]
                torch.cuda.synchronize()
                tl = outs[0]
                c["repeat_bitwise_equal"] = all(torch.equal(tl, o) for o in outs[1:])
                c["mask_equal"] = bool(torch.equal(torch.isinf(tl), torch.isinf(ref)))
                fin = torch.isfinite(ref)
                d = (tl[fin] - ref[fin]).abs(); c["max_abs"] = float(d.max()) if d.numel() else 0.0
                c["tol_ok"] = bool((d <= 0.02 + 0.02 * ref[fin].abs()).all())
                sel_ok = True
                for b in range(batch):
                    n = int(lens[b]); kk = min(K, n)
                    i_tl = torch.topk(tl[b, :n], kk).indices
                    got = torch.sort(ref[b, :n][i_tl])[0]; want = torch.topk(ref[b, :n], kk).values.sort()[0]
                    if not torch.allclose(got, want, atol=0.02, rtol=0.02):
                        sel_ok = False
                c["selection_ok"] = sel_ok
                c["ok"] = c["mask_equal"] and c["tol_ok"] and sel_ok and c["repeat_bitwise_equal"]
                rep["fails"] += 0 if c["ok"] else 1
            rep["cases"].append(c)
            rep["torch_peak_gb"] = max(rep["torch_peak_gb"] or 0.0, c["torch_peak_gb"])
    except Exception:
        rep["error"] = traceback.format_exc()[-1500:]
    rep["pass"] = rep["error"] is None and rep["has_tilelang"] is True and rep["fails"] == 0 and len(rep["cases"]) == 5
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("has_tilelang", "fails", "torch_peak_gb", "pass")}), (rep["error"] or "")[-300:])


if __name__ == "__main__":
    sys.exit(main())
