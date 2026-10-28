"""E13 micro-preflight T1/T2: QSA block top-k (k=512) correctness on this GPU, before any model is loaded.

T1 = SGLang 0.5.21 JIT `fast_topk` (the kernel QSA uses for decode and prefill when block_topk == 512; issue #36807 reports
silent drops on concentrated scores on sm89). T2 = FlashInfer `top_k_ragged_transform(deterministic=True, tie_break=0,
dsa_graph_safe=False)`, measured only (a switch needs the user's approval).

Oracle (tie-aware): per row, the selected set must have min(k, len) entries, be unique and in [0, len), and its value
multiset must equal the multiset of the top min(k, len) values. Exact index equality is not required (ties).
Usage: python -I mp_topk.py --out RESULT.json --fixtures DIR [--repeats 20] [--cpu-selftest]
"""
import argparse
import json
import os
import sys
import time
import traceback

import torch

K = 512
LENGTHS = [300, 2048, 8192, 16384, 24500, 30938, 34816]   # compressed blocks; 34816 = 139,264 tokens / 4; 30,938 = #36807
DISTS = ["gaussian", "uniform", "concentrated", "relu_sparse", "relu_sum4", "near_tie", "all_equal"]


def make_scores(dist, rows, length, gen, device):
    """fp32 [rows, length]. relu_* mimic the QSA indexer score: sum over heads of relu(q.k) / sqrt(d) (mqa.py)."""
    if dist == "gaussian":
        x = torch.randn(rows, length, generator=gen)
    elif dist == "uniform":
        x = torch.rand(rows, length, generator=gen)
    elif dist == "concentrated":      # most mass equal, a few large: the #36807 failure shape
        x = torch.zeros(rows, length)
        hot = torch.rand(rows, length, generator=gen) < 0.01
        x[hot] = 5.0 + torch.rand(int(hot.sum()), generator=gen)
        x += 1e-3 * (torch.rand(rows, length, generator=gen) < 0.002).float()
    elif dist == "relu_sparse":       # q.k mostly negative -> about half the blocks score exactly 0
        x = torch.relu(torch.randn(rows, length, 4, generator=gen) - 1.0).sum(-1) / 128 ** 0.5
    elif dist == "relu_sum4":
        x = torch.relu(torch.randn(rows, length, 4, generator=gen)).sum(-1) / 128 ** 0.5
    elif dist == "near_tie":
        x = torch.round(torch.randn(rows, length, generator=gen) * 4) / 4
    elif dist == "all_equal":
        x = torch.ones(rows, length)
    else:
        raise ValueError(dist)
    return x.to(device=device, dtype=torch.float32).contiguous()


def check_rows(scores, lengths, indices):
    """Return a list of per-row problems ([] = all rows correct)."""
    bad = []
    s_cpu, l_cpu, i_cpu = scores.float().cpu(), lengths.cpu(), indices.cpu()
    for b in range(s_cpu.shape[0]):
        n = int(l_cpu[b]); want = min(K, n)
        row = i_cpu[b]
        sel = row[row >= 0]
        prob = []
        if sel.numel() != want:
            prob.append(f"count {sel.numel()} != {want}")
        if sel.numel() and (int(sel.min()) < 0 or int(sel.max()) >= n):
            prob.append("index out of range")
        if torch.unique(sel).numel() != sel.numel():
            prob.append("duplicate index")
        if not prob:
            got = torch.sort(s_cpu[b, sel.long()])[0]
            ref = torch.topk(s_cpu[b, :n], want).values.sort()[0]
            if not torch.equal(got, ref):
                prob.append(f"value multiset differs (min got {float(got[0]):.6g} vs ref {float(ref[0]):.6g})")
        # (padding position is not checked: fast_topk's output order is unspecified; the count check covers missing slots)
        if prob:
            bad.append({"row": b, "len": n, "problems": prob})
    return bad


def run_impl(name, fn, device, repeats, fixtures, rows_list=(4, 10)):
    res = {"impl": name, "cases": [], "fail_cases": 0, "total_cases": 0}
    gen = torch.Generator().manual_seed(20261019)
    for rows in rows_list:
        for length in LENGTHS:
            for dist in DISTS:
                scores = make_scores(dist, rows, length, gen, device)
                # ragged: the last row is shorter (a real batch mixes context lengths)
                lens = torch.full((rows,), length, dtype=torch.int32)
                if rows > 1:
                    lens[-1] = max(1, length // 3)
                lens = lens.to(device)
                fails, sets, err = 0, set(), None
                t = time.perf_counter()
                for r in range(repeats):
                    try:
                        idx = fn(scores, lens)
                        if device == "cuda":
                            torch.cuda.synchronize()
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"[:300]
                        break
                    bad = check_rows(scores, lens, idx)
                    sets.add(tuple(sorted(idx[0][idx[0] >= 0].tolist())))
                    if bad:
                        fails += 1
                        if fails == 1 and fixtures:
                            os.makedirs(fixtures, exist_ok=True)
                            torch.save({"scores": scores.cpu(), "lengths": lens.cpu(), "indices": idx.cpu(), "impl": name,
                                        "dist": dist, "rows": rows, "length": length, "seed": 20261019, "repeat": r,
                                        "dtype": "float32", "row_starts": "zeros", "graph": False, "bad": bad[:4]},
                                       os.path.join(fixtures, f"{name}_{dist}_{rows}x{length}.pt"))
                sec = (time.perf_counter() - t) / max(1, repeats)
                case = {"rows": rows, "len": length, "dist": dist, "fail_repeats": fails, "repeats": repeats,
                        "distinct_row0_sets": len(sets), "ms_per_call": round(1000 * sec, 3), "error": err}
                if fails or err:
                    res["fail_cases"] += 1
                    case["example"] = (check_rows(scores, lens, fn(scores, lens)) if not err else None) or None
                    if case["example"]:
                        case["example"] = case["example"][:2]
                res["total_cases"] += 1
                res["cases"].append(case)
    return res


def graph_replay(name, fn, device):
    """Capture once on static buffers, change the scores in place, replay, check with the oracle."""
    if device != "cuda":
        return {"impl": name, "skipped": "no cuda"}
    out = {"impl": name, "replays": 0, "replay_failures": 0, "error": None}
    try:
        gen = torch.Generator().manual_seed(7)
        rows, length = 10, 34816
        scores = make_scores("relu_sparse", rows, length, gen, device)
        lens = torch.full((rows,), length, dtype=torch.int32, device=device)
        for _ in range(3):
            fn(scores, lens)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            idx = fn(scores, lens)
        for dist in ("gaussian", "concentrated", "relu_sparse", "near_tie"):
            scores.copy_(make_scores(dist, rows, length, gen, device))
            g.replay(); torch.cuda.synchronize()
            out["replays"] += 1
            if check_rows(scores, lens, idx):
                out["replay_failures"] += 1
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:400]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--cpu-selftest", action="store_true", help="local: check the oracle with torch.topk and a broken impl")
    a = ap.parse_args()
    report = {"device": None, "t1": None, "t2": None, "graph": [], "errors": []}
    if a.cpu_selftest:
        dev = "cpu"
        good = lambda s, l: torch.stack([torch.cat([torch.topk(s[b, :int(l[b])], min(K, int(l[b]))).indices.int(),
                                                    torch.full((K - min(K, int(l[b])),), -1, dtype=torch.int32)])
                                         for b in range(s.shape[0])])

        def broken(s, l):          # drops the last 20 selections of row 0 and duplicates one index (what #36807 looks like)
            x = good(s, l).clone(); x[0, -20:] = -1; x[1, 0] = x[1, 1]; return x
        report["device"] = "cpu-selftest"
        report["t1"] = run_impl("torch_topk_good", good, dev, 1, a.fixtures, rows_list=(4,))
        report["t2"] = run_impl("broken_impl", broken, dev, 1, a.fixtures, rows_list=(4,))
    else:
        dev = "cuda"
        report["device"] = torch.cuda.get_device_name(0) + " cap " + str(torch.cuda.get_device_capability(0))
        try:
            from sglang.kernels.ops.attention.fast_topk import fast_topk
            t1 = lambda s, l: fast_topk(s, l, topk=K, row_starts=None)
            report["t1"] = run_impl("t1_fast_topk", t1, dev, a.repeats, a.fixtures)
            report["graph"].append(graph_replay("t1_fast_topk", t1, dev))
        except Exception:
            report["errors"].append("T1: " + traceback.format_exc()[-1500:])
        try:
            import flashinfer
            fi = getattr(flashinfer, "top_k_ragged_transform", None)
            if fi is None:
                report["t2"] = {"skipped": "flashinfer has no top_k_ragged_transform"}
            else:
                def t2(s, l):
                    off = torch.zeros(s.shape[0], dtype=torch.int32, device=s.device)
                    return fi(s, off, l, K, deterministic=True, tie_break=0, dsa_graph_safe=False)
                report["t2"] = run_impl("t2_flashinfer_ragged", t2, dev, a.repeats, a.fixtures)
                report["graph"].append(graph_replay("t2_flashinfer_ragged", t2, dev))
        except Exception:
            report["errors"].append("T2: " + traceback.format_exc()[-1500:])
    for key in ("t1", "t2"):
        r = report.get(key)
        if isinstance(r, dict) and "total_cases" in r:
            r["summary"] = {"fail_cases": r["fail_cases"], "total_cases": r["total_cases"],
                            "by_dist": {d: sum(c["fail_repeats"] > 0 or bool(c["error"]) for c in r["cases"] if c["dist"] == d)
                                        for d in DISTS}}
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps({k: (v.get("summary") if isinstance(v, dict) else v) for k, v in report.items() if k in ("t1", "t2")}))
    print("graph", report["graph"], "errors", [e[:200] for e in report["errors"]])


if __name__ == "__main__":
    sys.exit(main())
