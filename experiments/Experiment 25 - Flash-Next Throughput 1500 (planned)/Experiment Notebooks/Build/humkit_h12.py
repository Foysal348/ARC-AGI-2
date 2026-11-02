"""E25 humming CPU kit, checks H1 + H2 (runs inside the Merlin venv python; no GPU).

H1  import humming (version, cu13 deps, NVRTC library, precompiled nvrtc_compile, cache dir) and the sm_89 heuristic
    entries humming's own tuner (humming.tune.get_heuristics_config) returns for the Flash-Next MoE GEMM shapes:
    EP2 local experts 256, top-10, hidden 2560, intermediate 640 (w13: N 1280 x K 2560, w2: N 2560 x K 640), M 8-64,
    for GPTQ-style symmetric W4 g128 (reference), W3 g128 / g64, W2 g128 / g64, BF16 activations, gemm types
    grouped_contiguous (what Merlin's EP path picks, fused_humming_moe.py:109-123) and indexed.
H2  NVRTC-compile (no GPU) one humming MoE kernel (w13, M 32) and one dense kernel (N 2560 x K 2560, M 32) for sm_89 at
    W3 / W2 x group 128 / 64: compile seconds, cubin bytes, cache dir (= what a B1 cache Model would carry).

The device is faked (humming's _DeviceInfo extension needs the CUDA driver): L4 spec values below, labelled
FAKE_L4_DEVICE_INFO in the report. torch.cuda.device / current_device are no-op patched (humming wraps its tuner in
torch.cuda.device(i)). HummingKernel.register_kernel (cuModuleLoad, needs a GPU) is skipped: the cubin file is the
product. A failure that needs a GPU is labelled NO_GPU_COMPILE_UNVERIFIED, an NVRTC compile error COMPILE_FAILED.

usage: python humkit_h12.py OUT_JSON
"""
import contextlib
import dataclasses
import hashlib
import json
import os
import sys
import time
import traceback
import types

OUT_PATH = sys.argv[1]
OUT = {"H1": {}, "H2": {}, "env": {k: os.environ.get(k) for k in ("HUMMING_CACHE_DIR", "HUMMING_TMP_DIR", "CUDA_HOME",
                                                                   "LD_LIBRARY_PATH", "LD_PRELOAD")}}
H1, H2 = OUT["H1"], OUT["H2"]
GPU_WORDS = ("cuda driver", "no nvidia driver", "cuinit", "cuda_error", "libcuda", "cudaerror", "device-side",
             "torch not compiled with cuda", "found no nvidia", "cuctx", "cudevice")


def save():
    with open(OUT_PATH, "w") as fh:
        json.dump(OUT, fh, indent=1, default=str)


def err(e):
    return f"{type(e).__name__}: {e}"[:900]


def label(e):
    s = str(e).lower()
    if "failed with exit code" in s:
        return "COMPILE_FAILED"
    return "NO_GPU_COMPILE_UNVERIFIED" if any(w in s for w in GPU_WORDS) else "ERROR"


# ------------------------------------------------------------------ libcuda presence (informational)
try:
    import ctypes
    ctypes.CDLL("libcuda.so.1")
    OUT["libcuda_so_1"] = "loadable"
except OSError as e:
    OUT["libcuda_so_1"] = f"not loadable: {e}"[:300]

# ------------------------------------------------------------------ H1: import + deps
t0 = time.time()
try:
    import humming
    H1["import_ok"] = True
    H1["version"] = getattr(humming, "__version__", None)
    H1["file"] = humming.__file__
except BaseException as e:
    H1.update(import_ok=False, import_error=err(e), tb=traceback.format_exc()[-2500:])
    H1["import_label"] = "NO_GPU_COMPILE_UNVERIFIED" if label(e) == "NO_GPU_COMPILE_UNVERIFIED" else "IMPORT_FAILED"
    save()
    print("H12_JSON " + json.dumps({"import_ok": False}))
    sys.exit(0)
H1["import_sec"] = round(time.time() - t0, 2)
try:
    from importlib.metadata import version
    dists = {}
    for d in ("humming-kernels", "nvidia-cuda-nvrtc", "nvidia-cuda-cccl", "nvidia-cuda-nvcc", "nvidia-cuda-runtime",
              "cuda-bindings", "torch", "triton", "jinja2", "filelock", "numpy"):
        try:
            dists[d] = version(d)
        except Exception as e:
            dists[d] = None
    H1["dists"] = dists
    req = []
    try:
        from importlib.metadata import requires
        req = requires("humming-kernels") or []
    except Exception:
        pass
    H1["humming_requires"] = req
except Exception as e:
    H1["dists_error"] = err(e)
try:
    import humming.utils.jit as hj
    import humming.utils.nvrtc as hn
    import humming.utils.cuda as hc
    H1["cache_dir"] = hj.get_humming_cache_dir()
    try:
        paths = hc.find_all_cuda_paths()
        H1["cuda_paths"] = [{k: (v if not isinstance(v, (list, dict)) else str(v)[:400]) for k, v in p.items()}
                            for p in paths][:6]
    except Exception as e:
        H1["cuda_paths_error"] = err(e)
    try:
        H1["nvrtc_lib"] = hn.get_nvrtc_library_path()
    except Exception as e:
        H1["nvrtc_lib_error"] = err(e)
    try:
        H1["nvrtc_compile_bin"] = hn.may_build_nvrtc_compile_binary()
        H1["nvrtc_compile_bin_precompiled"] = "/_native/" in H1["nvrtc_compile_bin"]
    except Exception as e:
        H1["nvrtc_compile_bin_error"] = err(e)
    try:
        from humming.jit.compiler import NVRTCCompiler
        H1["nvrtc_signature"] = NVRTCCompiler.signature()
    except Exception as e:
        H1["nvrtc_signature_error"] = err(e)
except Exception as e:
    H1["utils_error"] = err(e)
save()

# ------------------------------------------------------------------ fake L4 device info (spec values)
L4 = dict(index=0, name="NVIDIA L4", is_ppu=False, sm_count=58, max_threads_per_block=1024, max_threads_per_sm=1536,
          max_registers_per_sm=65536, sm_major=8, sm_minor=9, sm_version=89, l2_cache_size=48 * 2 ** 20,
          l1_cache_size=128 * 1024, default_smem_size=48 * 1024, max_smem_size=99 * 1024, memory_clock_khz=6251000,
          memory_bus_width=192, sm_clock_khz=2040000)
L4["l2_cache_size_mb"] = L4["l2_cache_size"] / 2 ** 20
L4["l1_cache_size_kb"] = L4["l1_cache_size"] / 1024
L4["max_smem_size_kb"] = L4["max_smem_size"] / 1024
L4["memory_bandwidth_gbps"] = L4["memory_clock_khz"] * 2 * L4["memory_bus_width"] / 8.0 / 1e6
_base = L4["sm_count"] * 1024 * L4["sm_clock_khz"] / 1e9           # device_info.cpp:157-168,246 (sm_89: 1024 ops/clk)
L4["tensorcore_tops"] = {"float16": _base / 2, "bfloat16": _base / 2, "float8e4m3": _base, "float8e5m2": _base,
                         "int8": _base * 2, "int4": _base * 4}     # device_info.cpp:176-215 (half-rate floats on 89)


class _FakeInfo:
    def __init__(self, d):
        self.__dict__.update(d)

    def print(self):
        print("FAKE_L4_DEVICE_INFO", self.__dict__)


FAKE = _FakeInfo(L4)
try:
    import torch
    import humming.utils.device as hud
    import humming.device as hd
    hud._extension = types.SimpleNamespace(_DeviceInfo=lambda index=0, *a, **k: FAKE)
    hd.current_device._value = FAKE
    torch.cuda.device = lambda *a, **k: contextlib.nullcontext()
    torch.cuda.current_device = lambda: 0
    OUT["fake_device"] = {"label": "FAKE_L4_DEVICE_INFO", "values": {k: v for k, v in L4.items()}}
except Exception as e:
    OUT["fake_device"] = {"error": err(e)}
save()

# ------------------------------------------------------------------ H1: sm_89 heuristics for our shapes
COMBOS = [(4, 128), (3, 128), (3, 64), (2, 128), (2, 64)]
SHAPES = {"w13": (1280, 2560), "w2": (2560, 640)}   # (N, K) per expert under EP2 (intermediate 640, gated: 2 x 640)
E_LOCAL = 256
MS = (8, 16, 32, 48, 64)


def make_layer_config(bits, group, n, k, num_experts):
    from humming.schema.gptq import GPTQWeightSchema
    from humming.transform import prepare_layer_config
    ws = GPTQWeightSchema(bits=bits, group_size=group, desc_act=False, sym=True).to_humming_schema(torch.bfloat16)
    # pads exactly as vLLM's humming MoE path (moe.py:_prepare_and_transform_sublayer) and dense path (linear.py)
    return prepare_layer_config(shape_n=n, shape_k=k, weight_schema=ws, input_schema=None, num_experts=num_experts,
                                pad_n_to_multiple=256, pad_k_to_multiple=128, has_bias=False, torch_dtype=torch.bfloat16,
                                device=0)


def tun_summary(c):
    keys = ("mma_type", "block_shape", "warp_shape", "num_stages", "num_ctas_per_sm", "use_stream_k", "use_warp_spec",
            "num_sms", "raster_group_m", "use_pdl")
    return {k: (list(c[k]) if isinstance(c.get(k), tuple) else c.get(k)) for k in keys if k in c}


try:
    from humming.tune import get_heuristics_class, get_heuristics_config
    H1["heuristics_class"] = get_heuristics_class(0).__name__
    heur = {}
    for bits, group in COMBOS:
        for sub, (n, k) in SHAPES.items():
            for gt in ("grouped_contiguous", "indexed"):
                key = f"W{bits}g{group}/{sub}/{gt}"
                rec = {}
                try:
                    lc = make_layer_config(bits, group, n, k, E_LOCAL)
                    rec["layer_config"] = {f: str(getattr(lc, f)) for f in ("b_dtype", "a_dtype", "bs_dtype", "shape_n", "shape_k",
                                                                            "pad_shape_n", "pad_shape_k", "num_experts",
                                                                            "weight_scale_group_size", "has_zero_point",
                                                                            "sm_version")}
                    rec["weight_nbytes_per_layer_card"] = lc.weight_nbytes
                    rec["per_m"] = {}
                    for m in MS:
                        try:
                            rec["per_m"][m] = tun_summary(get_heuristics_config(lc, shape_m=m, gemm_type=gt, device=0))
                        except Exception as e:
                            rec["per_m"][m] = {"error": err(e)}
                    try:
                        iv = get_heuristics_config(lc, gemm_type=gt, device=0)
                        rec["intervals_le_1024"] = [[a, b, tun_summary(c)] for a, b, c in iv if a <= 1024][:24]
                        rec["n_intervals"] = len(iv)
                    except Exception as e:
                        rec["intervals_error"] = err(e)
                except Exception as e:
                    rec["error"] = err(e)
                    rec["tb"] = traceback.format_exc()[-1200:]
                heur[key] = rec
        save()
    H1["heuristics"] = heur
    H1["heuristics_ok"] = all("error" not in v and all("error" not in x for x in v.get("per_m", {}).values())
                              for v in heur.values()) and len(heur) == len(COMBOS) * 4
except Exception as e:
    H1["heuristics_ok"] = False
    H1["heuristics_error"] = err(e)
    H1["heuristics_tb"] = traceback.format_exc()[-2000:]
save()

# ------------------------------------------------------------------ H2: NVRTC compile, no GPU
try:
    from humming.kernel.humming import HummingKernel
    HummingKernel.register_kernel = lambda self: None          # cuModuleLoad needs a GPU; the cubin is the product
    FIELDS = {f.name for f in dataclasses.fields(HummingKernel)}
    kernels = {}
    for bits, group in [(3, 128), (3, 64), (2, 128), (2, 64)]:
        for kind in ("moe_w13", "dense"):
            key = f"W{bits}g{group}/{kind}"
            rec = {}
            try:
                if kind == "moe_w13":
                    lc, gt = make_layer_config(bits, group, 1280, 2560, E_LOCAL), "grouped_contiguous"
                else:
                    lc, gt = make_layer_config(bits, group, 2560, 2560, 0), "dense"
                tun = get_heuristics_config(lc, shape_m=32, gemm_type=gt, device=0)
                lo = json.loads(lc.to_str())
                lo.pop("sublayer_name", None)
                cfg = {**lo, "use_f16_accum": False, "use_batch_invariant": False, "gemm_type": gt, **tun}
                cfg.pop("num_sms", None)
                dropped = sorted(k_ for k_ in cfg if k_ not in FIELDS)
                cfg = {k_: v for k_, v in cfg.items() if k_ in FIELDS}
                rec.update(tuning=tun_summary(tun), dropped_keys=dropped)
                t = time.time()
                kern = HummingKernel(**cfg)
                rec["compile_sec"] = round(time.time() - t, 2)
                fn = getattr(kern, "kernel_filename", None)
                rec["cubin"] = fn
                if fn and os.path.isfile(fn):
                    b = open(fn, "rb").read()
                    rec.update(cubin_bytes=len(b), cubin_sha256=hashlib.sha256(b).hexdigest()[:16], ok=True)
                    d = os.path.dirname(fn)
                    for lg in ("stderr.log", "stdout.log"):
                        if os.path.isfile(os.path.join(d, lg)):
                            rec[lg] = open(os.path.join(d, lg)).read()[-600:]
                else:
                    rec["ok"] = False
            except BaseException as e:
                rec.update(ok=False, error=err(e), label=label(e), tb=traceback.format_exc()[-1800:])
            kernels[key] = rec
            save()
    H2["kernels"] = kernels
    H2["all_compiled"] = len(kernels) == 8 and all(v.get("ok") for v in kernels.values())
except BaseException as e:
    H2.update(all_compiled=False, error=err(e), label=label(e), tb=traceback.format_exc()[-2000:])
try:
    cd = OUT["env"].get("HUMMING_CACHE_DIR") or H1.get("cache_dir")
    tot, n, cub = 0, 0, 0
    for root, _, files in os.walk(cd or "/nonexistent"):
        for f in files:
            n += 1
            sz = os.path.getsize(os.path.join(root, f))
            tot += sz
            cub += sz if f.endswith(".cubin") else 0
    H2["cache_dir"] = {"path": cd, "files": n, "bytes": tot, "cubin_bytes": cub}
except Exception as e:
    H2["cache_dir_error"] = err(e)
save()
print("H12_JSON " + json.dumps({"import_ok": H1.get("import_ok"), "version": H1.get("version"),
                                "heuristics_ok": H1.get("heuristics_ok"), "all_compiled": H2.get("all_compiled")}))
