"""E12-K1 tuner: search W8A8 block-FP8 Triton configs on one L4 for our Qwen3.8-27B TP4 GEMM shapes.

It uses vLLM 0.19.0's own production kernel (`_w8a8_triton_block_scaled_mm`) with the same call as
`w8a8_triton_block_scaled_mm`. The search follows vLLM's benchmarks/kernels/benchmark_w8a8_block_fp8.py (Apache-2.0),
pruned per batch size so that it fits the time budget. For every (shape, M) it records:
- the default-config time;
- the best time and config;
- the max relative error of the best config against the default.

It writes vLLM's config file name: N=..,K=..,device_name=NVIDIA_L4,dtype=fp8_w8a8,block_shape=[128,128].json.

usage: python k1_tune_block_fp8.py --shapes 8704x5120,4096x5120 --out-dir DIR --deadline UNIX_TS --label gpu0
K1_DRYRUN=1 writes fake results without torch/vLLM (local orchestration test only).
"""
import argparse, json, os, random, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", required=True)          # "N1xK1,N2xK2"
ap.add_argument("--out-dir", required=True)
ap.add_argument("--deadline", type=float, required=True)
ap.add_argument("--label", default="gpu")
args = ap.parse_args()
SHAPES = [tuple(int(v) for v in s.split("x")) for s in args.shapes.split(",") if s]
OUT = args.out_dir
os.makedirs(OUT, exist_ok=True)
# decode first: 16 seqs x (1 + 3 MTP) = 64 tokens per verify step, 16 per MTP draft step; then prefill chunks (4096 max)
M_ORDER = [64, 16, 48, 32, 96, 128, 4096, 2048, 1024, 512, 256, 24, 8, 1536, 3072]
BLOCK = [128, 128]
DEFAULT = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 2}


def search_space(M):
    bms = [16, 32, 64] if M <= 128 else [64, 128, 256]
    groups = [1] if M <= 128 else [1, 16, 32]
    out = []
    for st in (2, 3, 4, 5):
        for bm in bms:
            for bk in (64, 128):
                for bn in (32, 64, 128, 256):
                    for nw in (4, 8):
                        for g in groups:
                            out.append({"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": g,
                                        "num_warps": nw, "num_stages": st})
    # a different order per GPU process: parallel processes then compile different configs first and reuse each
    # other's results through Triton's shared disk cache (the best config does not depend on the order)
    random.Random(f"{args.label}-{M}").shuffle(out)
    return out


def file_name(N, K, device_name):
    return f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{BLOCK[0]},{BLOCK[1]}].json"


def log(*a):
    print(f"[{args.label} {time.strftime('%H:%M:%S')}]", *a, flush=True)


if os.environ.get("K1_DRYRUN") == "1":
    for (N, K) in SHAPES:
        cfg = {str(M): dict(DEFAULT, num_stages=3) for M in M_ORDER[:6]}
        json.dump(cfg, open(os.path.join(OUT, file_name(N, K, "NVIDIA_L4")), "w"), indent=1)
        tim = {str(M): {"default_us": 100.0, "best_us": 80.0, "speedup": 1.25, "max_rel_err": 0.0, "configs_tried": 3}
               for M in M_ORDER[:6]}
        json.dump({"N": N, "K": K, "timings": tim, "complete": False}, open(os.path.join(OUT, f"timings_N{N}_K{K}.json"), "w"))
    log("DRYRUN done", SHAPES)
    sys.exit(0)

import torch
from vllm.model_executor.layers.quantization.utils.fp8_utils import _w8a8_triton_block_scaled_mm
from vllm.platforms import current_platform
from vllm.triton_utils import triton

DEVICE_NAME = current_platform.get_device_name().replace(" ", "_")
OUT_DTYPE = torch.bfloat16          # the model runs in bf16, so the block-FP8 GEMM output is bf16


def mm(A, B, As, Bs, cfg):
    M = A.numel() // A.shape[-1]
    N, K = B.shape
    C = A.new_empty(A.shape[:-1] + (N,), dtype=OUT_DTYPE)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    _w8a8_triton_block_scaled_mm[grid](A, B, C, As, Bs, M, N, K, BLOCK[0], BLOCK[1],
                                       A.stride(-2), A.stride(-1), B.stride(1), B.stride(0), C.stride(-2), C.stride(-1),
                                       As.stride(-2), As.stride(-1), Bs.stride(1), Bs.stride(0), **cfg)
    return C


def bench(A, B, As, Bs, cfg, iters=20):
    for _ in range(3):
        mm(A, B, As, Bs, cfg)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        mm(A, B, As, Bs, cfg)
    e.record(); e.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters          # microseconds per call


def inputs(M, N, K):
    fi = torch.finfo(torch.float8_e4m3fn)
    A = ((torch.rand(M, K, device="cuda") - 0.5) * 2 * fi.max).clamp(fi.min, fi.max).to(torch.float8_e4m3fn)
    B = ((torch.rand(N, K, device="cuda") - 0.5) * 2 * fi.max).clamp(fi.min, fi.max).to(torch.float8_e4m3fn).contiguous()
    As = torch.rand(M, triton.cdiv(K, BLOCK[1]), device="cuda") * 1e-2
    Bs = torch.rand(triton.cdiv(N, BLOCK[0]), triton.cdiv(K, BLOCK[1]), device="cuda") * 1e-2
    return A, B, As, Bs


log("device", DEVICE_NAME, "shapes", SHAPES, "deadline in", round(args.deadline - time.time()), "s")
for (N, K) in SHAPES:
    best_cfgs, timings = {}, {}
    complete = True
    for M in M_ORDER:
        if time.time() > args.deadline:
            complete = False; log("deadline: stop before M", M); break
        A, B, As, Bs = inputs(M, N, K)
        t_def = bench(A, B, As, Bs, DEFAULT)
        ref = mm(A, B, As, Bs, DEFAULT).float()
        best, best_t, tried, cut = DEFAULT, t_def, 0, False
        for cfg in search_space(M):
            if time.time() > args.deadline:
                cut = True; break
            try:
                t = bench(A, B, As, Bs, cfg)
            except Exception:          # OutOfResources / compile errors for oversized tiles
                continue
            tried += 1
            if t < best_t:
                best, best_t = cfg, t
        out = mm(A, B, As, Bs, best).float()
        rel = ((out - ref).abs().max() / (ref.abs().max() + 1e-9)).item()
        if rel > 1e-2:                     # a wrong kernel must never be written
            log(f"N={N} K={K} M={M}: best config differs from default by {rel:.3g}; keeping default")
            best, best_t = DEFAULT, t_def
        best_cfgs[str(M)] = best
        timings[str(M)] = {"default_us": round(t_def, 2), "best_us": round(best_t, 2), "speedup": round(t_def / best_t, 3),
                           "max_rel_err": rel, "configs_tried": tried, "cut_by_deadline": cut}
        log(f"N={N} K={K} M={M}: default {t_def:.1f} us -> best {best_t:.1f} us (x{t_def / best_t:.2f}), tried {tried}")
        del A, B, As, Bs, ref, out
        if cut:
            complete = False; break
    if best_cfgs:
        with open(os.path.join(OUT, file_name(N, K, DEVICE_NAME)), "w") as f:
            json.dump(best_cfgs, f, indent=1); f.write("\n")
    with open(os.path.join(OUT, f"timings_N{N}_K{K}.json"), "w") as f:
        json.dump({"N": N, "K": K, "device": DEVICE_NAME, "timings": timings, "complete": complete}, f, indent=1)
log("done")
