"""E11-05: S49 (LB-pending Poetiq, thinking 49,152, ORDER2 + cells tie-break) ported to the E11-04 W1 engine
(SGLang 0.5.21 + RedHat INT4 W4A16 + DFlash2 drafter, 12 really running, KV 881,336 tokens, 416 tok/s at <= 32k context).

What changes against S49 (everything else is byte-identical):
  cell 0 (markdown)  E11-05 header
  CELL 1 config      E11 engine block (paths + discovery, W1 constants, give-up deadline)
  CELL 2 install     tokenizer stack only when SGLang is the engine (vLLM later, only for the fallback);
                     weight prefetch of INT4 + DFlash2; tokenizer check (INT4 vocab == S49 tokenizer vocab)
  CELL 5 engine      NEW: SGLang venv install, W1 launch with diagnosed fallbacks, caps from the server log,
                     smoke (stream + abort), vLLM-shaped shim (add_request / step / abort_request / get_metrics)
  CELL 5b            S49's own engine cell, indented under `if not ENGINE_READY` = vLLM FP8 fallback (AST-identical)
  report cell        + E11 decode summary + shim stats, server shut down at the end
The scheduler cell (Poetiq loop, drain, hard stop, KV gate, voting, submission) is not touched.

usage: python build_e11_05_s49_sglang.py
"""
import ast, hashlib, json, re, textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent
SRC_NB = Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Notebooks/"
              r"Q27-XHIGH-v1_4_4_E10_S1_49K_POETIQ_FIX_ORDER2.ipynb")
SRC_SHA = "1cae1d5941ebcdba86953f0f300fd38a053d566e96e395c038c53667dbcfffda"     # = S49 as quick-saved (push163)
NAME = "Q27-XHIGH-v1_4_4_E11_05_S49_SGLANG_INT4_DFLASH2.ipynb"


def once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"patch '{label}': expected 1 match, found {n}")
    return text.replace(old, new)


raw = SRC_NB.read_bytes()
if hashlib.sha256(raw).hexdigest() != SRC_SHA:
    raise SystemExit("S49 source notebook changed: " + hashlib.sha256(raw).hexdigest())
nb = json.loads(raw.decode("utf-8"))
cells = nb["cells"]
src = ["".join(c["source"]) for c in cells]
assert [c["cell_type"] for c in cells] == ["markdown"] + ["code"] * 8, [c["cell_type"] for c in cells]
MD, C0, C1, C2, C3, C4, C5, C6, C7 = src
assert C5.startswith("# ==================================================================== CELL 5 — engine (PIV/V3.1 C1 settings")
assert C6.startswith("# E10 — Poetiq iterative program-synthesis solver")

# ---------------------------------------------------------------- markdown
MD = ("# E11-05 · S49 Poetiq on SGLang 0.5.21 + RedHat INT4 + DFlash2 (E11-04 W1 engine)\n\n"
      "**Port, not a new method.** Same prompts, tokenizer, thinking budget 49,152, ORDER2 + cells tie-break, Poetiq loop, "
      "drain/hard-stop clock, voting and submission as S49. Only the engine changes: SGLang 0.5.21 with the RedHat INT4 "
      "(W4A16 g128) checkpoint and the BF16 DFlash2 drafter, launched exactly like E11-04 W1 (KV 881,336 tokens, 12 really "
      "running, 416 tok/s at <= 32k context vs about 255-300 for vLLM FP8 + MTP-3). A small shim gives the S49 scheduler the "
      "vLLM calls it already uses. If SGLang does not come up, the server is killed and S49's own vLLM FP8 engine runs "
      "(cell 5b, unchanged code).\n\n"
      "**Inputs:** competition data · `foysalemonshanto/qwen3-8-27b-int4-redhat` · `foysalemonshanto/qwen3-8-27b-dflash2` · "
      "dataset `foysalemonshanto/sglang-0-5-21-cu13-wheelhouse` · `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` (tokenizer + "
      "fallback) · dataset `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3` (tokenizer stack + fallback). L4 x4, internet off, "
      "pinned image.\n\n---\n\n" + MD)

# ---------------------------------------------------------------- CELL 1: config
C1 = once(C1, 'CONFIG = {k: globals()[k] for k in ["PROFILE",', r'''# ---- E11-05: engine = SGLang 0.5.21 + RedHat INT4 (W4A16 g128) + DFlash2 drafter = the E11-04 W1 launch (2026-10-18).
# MODEL_PATH (FP8) stays: its tokenizer + chat template render every prompt exactly as in S49, and it is the vLLM fallback.
ENGINE_BACKEND = os.environ.get("E11_ENGINE", "sglang").strip().lower()       # sglang (vLLM FP8 fallback) | vllm (= S49)
E11_ALLOW_VLLM_FALLBACK = os.environ.get("E11_VLLM_FALLBACK", "1") == "1"
SGL_MODEL_PATH = Path(os.environ.get("E11_INT4", "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1"))
SGL_DRAFT_PATH = Path(os.environ.get("E11_DFLASH", "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1"))
SGL_WHEELHOUSE = Path(os.environ.get("E11_SGL_WHEELHOUSE", "/kaggle/input/datasets/foysalemonshanto/sglang-0-5-21-cu13-wheelhouse"))


def _e11_discover(path, marker, pred):
    """The given path if it holds `marker`, else the first /kaggle/input folder whose `marker` json satisfies pred."""
    if (path / marker).is_file():
        return path
    root = Path("/kaggle/input")
    for p in (sorted(root.rglob(marker)) if root.exists() else []):
        try:
            if pred(p.parent, _load_json(p)):
                return p.parent
        except Exception:
            continue
    return path


SGL_MODEL_PATH = _e11_discover(SGL_MODEL_PATH, "config.json", lambda d, c: isinstance(c, dict)
                               and (c.get("quantization_config") or {}).get("quant_method") == "compressed-tensors")
SGL_DRAFT_PATH = _e11_discover(SGL_DRAFT_PATH, "config.json", lambda d, c: isinstance(c, dict)
                               and c.get("architectures") == ["DFlash2DraftModel"])
SGL_WHEELHOUSE = _e11_discover(SGL_WHEELHOUSE, "BUILD_INFO.json", lambda d, c: isinstance(c, dict)
                               and c.get("sglang") == "0.5.21" and (d / "requirements.lock").is_file())
SGL_CONTEXT_LEN = 155648            # = the E11-04 W1 launch; every request stays inside MAX_MODEL_LEN (phase1_cap)
SGL_MAX_RUNNING = 12                # E11-04 W1: 12 really running (mamba cache 56 / 4 state slots = 14 >= 12)
SGL_LAUNCH_TIMEOUT_SEC = 30 * 60    # first launch includes JIT + CUDA graphs: 567 s in E11-04
SGL_RELAUNCH_TIMEOUT_SEC = 20 * 60  # a diagnosed relaunch (E11-04: about 75 s once the JIT cache exists)
SGL_GIVE_UP_AT = min(NOTEBOOK_T0 + 75 * 60, ADMIT_DEADLINE - 5 * 60)   # no SGLang launch after this -> vLLM fallback
SGL_STREAM_IDLE_SEC = 1800          # a socket read silent for longer closes that request as 'error'

CONFIG = {k: globals()[k] for k in ["PROFILE",''', "e11 config block")
C1 = once(C1, 'CONFIG["profile_row"] = P\n', '''CONFIG["profile_row"] = P
CONFIG["e11"] = {k: (str(globals()[k]) if isinstance(globals()[k], Path) else globals()[k]) for k in (
    "ENGINE_BACKEND", "E11_ALLOW_VLLM_FALLBACK", "SGL_MODEL_PATH", "SGL_DRAFT_PATH", "SGL_WHEELHOUSE", "SGL_CONTEXT_LEN",
    "SGL_MAX_RUNNING", "SGL_LAUNCH_TIMEOUT_SEC", "SGL_RELAUNCH_TIMEOUT_SEC", "SGL_STREAM_IDLE_SEC")}
CONFIG["e11"]["give_up_after_sec"] = round(SGL_GIVE_UP_AT - NOTEBOOK_T0)
print("E11 engine:", json.dumps(CONFIG["e11"]))
''', "e11 config record")

# ---------------------------------------------------------------- CELL 2: install + tokenizer
C2 = once(C2, '''        _wfiles = sorted(str(p) for p in MODEL_PATH.glob("*.safetensors"))''',
          '''        _wdirs = (SGL_MODEL_PATH, SGL_DRAFT_PATH) if ENGINE_BACKEND == "sglang" else (MODEL_PATH,)   # E11-05
        _wfiles = sorted(str(p) for _d in _wdirs for p in Path(_d).glob("*.safetensors"))''', "prefetch dirs")
C2 = once(C2, '''    if before_stack["vllm"] != EXPECTED_RUNTIME_PACKAGES["vllm"]:
        offline_pip_install(["vllm==0.19.0"])
    else:
        offline_pip_install([f"{p}=={EXPECTED_RUNTIME_PACKAGES[p]}" for p in ("transformers", "tokenizers", "huggingface-hub")
                             if before_stack[p] != EXPECTED_RUNTIME_PACKAGES[p]])''',
          '''    _tok_pkgs = [f"{p}=={EXPECTED_RUNTIME_PACKAGES[p]}" for p in ("transformers", "tokenizers", "huggingface-hub")
                 if before_stack[p] != EXPECTED_RUNTIME_PACKAGES[p]]
    if ENGINE_BACKEND == "sglang" and before_stack["vllm"] != EXPECTED_RUNTIME_PACKAGES["vllm"]:
        # E11-05: the kernel only tokenizes (S49 stack); SGLang runs in its own venv; vLLM is installed only for the fallback
        try:
            offline_pip_install(_tok_pkgs)
        except Exception as _e:
            print("tokenizer-only install failed -> the full S49 install:", repr(_e)[:300], flush=True)
            offline_pip_install(["vllm==0.19.0"])
    elif before_stack["vllm"] != EXPECTED_RUNTIME_PACKAGES["vllm"]:
        offline_pip_install(["vllm==0.19.0"])
    else:
        offline_pip_install(_tok_pkgs)''', "install branch")
C2 = once(C2, '''    for _p, _want in EXPECTED_RUNTIME_PACKAGES.items():
        if after_stack.get(_p) != _want:''', '''    for _p, _want in EXPECTED_RUNTIME_PACKAGES.items():
        if _p == "vllm" and ENGINE_BACKEND == "sglang":
            continue                                   # E11-05: checked by the fallback cell, only if it runs
        if after_stack.get(_p) != _want:''', "parity skip vllm")
C2 = once(C2, '''print("tokenizer OK | </think> id", THINK_CLOSE_ID, "| stop text", len(STOP_IDS), "tokens")''',
          '''print("tokenizer OK | </think> id", THINK_CLOSE_ID, "| stop text", len(STOP_IDS), "tokens")

# E11-05: the SGLang server receives token ids made by this (S49) tokenizer. Check that the INT4 checkpoint has the same
# vocabulary (with added tokens) and tokenizes a rendered prompt identically; otherwise run the S49 vLLM engine.
E11_TOKENIZER_CHECK = {"ok": None, "skipped": ENGINE_BACKEND != "sglang"}
if ENGINE_BACKEND == "sglang":
    try:
        from tokenizers import Tokenizer as _HFTokenizer
        _t4 = _HFTokenizer.from_file(str(SGL_MODEL_PATH / "tokenizer.json"))
        _probe = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Example 1\\nInput:\\n0 1 2\\n3 4 5\\nOutput:\\n5 4 3\\n```python\\ndef transform(grid):\\n    return grid\\n```"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort=REASONING_EFFORT)
        _probe += "Let me think.\\n</think>\\n\\n```python\\nimport numpy as np\\n```<|im_end|>"
        _ids_s49 = tokenizer.encode(_probe, add_special_tokens=False)
        _ids_int4 = _t4.encode(_probe, add_special_tokens=False).ids
        _same_vocab = _t4.get_vocab(with_added_tokens=True) == tokenizer.backend_tokenizer.get_vocab(with_added_tokens=True)
        _int4_gen = _load_json(SGL_MODEL_PATH / "generation_config.json") if (SGL_MODEL_PATH / "generation_config.json").exists() else {}
        _e4 = _int4_gen.get("eos_token_id", [])
        _same_eos = set(_e4 if isinstance(_e4, list) else [_e4]) == EOS_IDS
        E11_TOKENIZER_CHECK = {"ok": bool(_same_vocab and _ids_s49 == _ids_int4 and _same_eos), "same_vocab": _same_vocab,
                               "same_probe_ids": _ids_s49 == _ids_int4, "probe_tokens": len(_ids_s49), "same_eos": _same_eos}
    except Exception as _e:
        E11_TOKENIZER_CHECK = {"ok": False, "error": f"{type(_e).__name__}: {_e}"[:300]}
    if not E11_TOKENIZER_CHECK["ok"]:
        print("E11 tokenizer check FAILED -> S49 vLLM FP8 engine:", E11_TOKENIZER_CHECK, flush=True)
        ENGINE_BACKEND = "vllm"
CONFIG["e11"]["tokenizer_check"] = E11_TOKENIZER_CHECK
print("E11 tokenizer check:", E11_TOKENIZER_CHECK)''', "tokenizer check")

# ---------------------------------------------------------------- CELL 5: SGLang engine + shim (new)
C5_SGL = r'''# ==================================================================== CELL 5 — engine: SGLang 0.5.21 + RedHat INT4 + DFlash2 (E11-04 W1)
# E11-05. The scheduler (next cells) is S49's, unchanged. It drives the engine with the vLLM calls it already uses
# (add_request / step / abort_request, DELTA token ids, finish 'stop' | 'length' | 'abort'); the shim below maps them onto
# SGLang's HTTP API: /generate streamed with --incremental-streaming-output (disjoint deltas, tokenizer_manager.py:2493-2515),
# /abort_request (the stream then ends with finish type 'abort'), /metrics. Launch = E11-04 W1, measured 2026-10-18.
# If SGLang does not come up (inputs, install, launch, smoke incl. abort), the server is killed and CELL 5b runs S49's vLLM.
import queue as _queue, signal as _signal, statistics as _statistics, types as _types, urllib.request as _urlreq
E11_HARNESS = os.environ.get("E11_HARNESS") == "1"            # local CPU harness only: fake server, no install, no GPU
SGL_PORT = int(os.environ.get("E11_PORT", 30000))
SGL_URL = f"http://127.0.0.1:{SGL_PORT}"
E11_SGL = {"requested": ENGINE_BACKEND, "ok": False, "stage": "not started", "inputs": {}, "install": None, "attempts": [],
           "caps": None, "launch_flags": None, "smoke": None, "error": None, "tokenizer_check": E11_TOKENIZER_CHECK}
E11_SERVER = None
ENGINE_READY = False; llm = None; engine = None; CONCURRENCY = 0
ENGINE_INFO = {"backend": None, "sglang": E11_SGL, "kv_tokens": None, "concurrency": None, "fatal": None, "load_sec": None}


class SamplingParams:
    """vLLM-shaped container. The shim sends max_tokens, temperature, top_p, top_k. The seed is not sent: SGLang 0.5.21
    uses sampling_seed only under --enable-deterministic-inference (sampling_batch_info.py:143-157), so sampling is random."""
    def __init__(self, max_tokens=16, seed=None, temperature=1.0, top_p=1.0, top_k=-1, **kw):
        self.max_tokens, self.seed = int(max_tokens), seed
        self.temperature, self.top_p, self.top_k = float(temperature), float(top_p), int(top_k)
        self.extra = kw


class RequestOutputKind:
    DELTA = "delta"


def make_params(max_tokens, seed):
    return SamplingParams(max_tokens=int(max_tokens), seed=int(seed), detokenize=False, skip_special_tokens=False,
                          output_kind=RequestOutputKind.DELTA, **SAMPLING)


class E11Fail(Exception):
    pass


def _e11_post(path, payload, timeout=30):
    req = _urlreq.Request(SGL_URL + path, data=json.dumps(payload).encode(), method="POST",
                          headers={"Content-Type": "application/json"})
    with _urlreq.urlopen(req, timeout=timeout) as r:
        return r.read()


def _e11_gpu_used():
    try:
        return [int(x) for x in subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                               capture_output=True, text=True, timeout=30).stdout.split()]
    except Exception:
        return []


def _e11_gpu_free(max_sec=180):
    t = time.perf_counter()
    while time.perf_counter() - t < max_sec:
        u = _e11_gpu_used()
        if E11_HARNESS or (u and max(u) < 1024):
            return {"free": True, "sec": round(time.perf_counter() - t), "used_mb": u}
        time.sleep(5)
    for _pat in ("sglang.launch_server", "sglang::"):
        subprocess.run(["pkill", "-9", "-f", _pat], capture_output=True)
    time.sleep(15)
    return {"free": False, "sec": round(time.perf_counter() - t), "used_mb": _e11_gpu_used(), "pkill": True}


# ---- install (= E11-03/04 C1): a venv from the wheelhouse lock, --no-deps; CUDA 13 pip runtime as CUDA_HOME; link shims
def e11_install():
    if E11_HARNESS:
        return Path(sys.executable), dict(os.environ), {"harness": True}
    shims, t0 = {}, time.perf_counter()
    venv = Path("/tmp/sglang-venv"); vpy = venv / "bin" / "python"
    # Kaggle drops '+' from dataset file names (flashinfer_jit_cache-0.6.18+cu130 -> ...0.6.18cu130); rebuild the original
    # names from SHA256SUMS as links in a find-links folder, checking the hash of every renamed wheel
    fl = Path("/tmp/e11-wheels"); fl.mkdir(parents=True, exist_ok=True)
    renamed = []
    for line in (SGL_WHEELHOUSE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, name = line.split("  ", 1); name = name.strip()
        if not name.endswith(".whl"):
            continue
        s = SGL_WHEELHOUSE / name if (SGL_WHEELHOUSE / name).exists() else SGL_WHEELHOUSE / name.replace("+", "")
        if not s.exists():
            raise E11Fail(f"wheel missing from the dataset: {name}")
        if s.name != name:
            d = hashlib.sha256()
            with open(s, "rb") as f:
                for b in iter(lambda: f.read(16 * 1024 * 1024), b""):
                    d.update(b)
            if d.hexdigest() != h:
                raise E11Fail(f"sha256 mismatch for {s.name}")
            renamed.append(f"{s.name} -> {name}")
        if not (fl / name).exists():
            (fl / name).symlink_to(s)
    shims["renamed_wheels"] = renamed
    if not vpy.exists():
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
        r = subprocess.run([sys.executable, "-m", "pip", "--python", str(vpy), "install", "--no-index", "--find-links", str(fl),
                            "--requirement", str(SGL_WHEELHOUSE / "requirements.lock"), "--no-deps", "--only-binary", ":all:",
                            "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts", "-q"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise E11Fail("offline SGLang install failed: " + (r.stderr or r.stdout)[-1500:])
    site = Path(subprocess.run([str(vpy), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                               capture_output=True, text=True, check=True).stdout.strip())
    cuda_home = site / "nvidia" / "cu13"
    # E11-02 BUILD_PROBE: the pip CUDA runtime ships lib/libcudart.so.13 only; JIT links need libcudart.so and lib64
    if not (cuda_home / "lib" / "libcudart.so").exists():
        (cuda_home / "lib" / "libcudart.so").symlink_to("libcudart.so.13"); shims["libcudart.so"] = "created"
    if not (cuda_home / "lib64").exists():
        (cuda_home / "lib64").symlink_to("lib"); shims["lib64"] = "created"
    # E10 run 1: FlashInfer's JIT link needs -lcuda; Kaggle has only libcuda.so.1
    libcuda = Path("/tmp/e11-libcuda")
    have = [d for d in ("/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs", "/usr/lib/x86_64-linux-gnu", "/usr/lib64")
            if Path(d, "libcuda.so").exists()]
    if have:
        shims["libcuda.so"] = f"exists in {have[0]}"
    else:
        real = None
        try:
            for line in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30).stdout.splitlines():
                if "libcuda.so.1" in line and "=>" in line:
                    real = line.split("=>")[-1].strip(); break
        except Exception:
            pass
        for d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
            if real is None and Path(d, "libcuda.so.1").exists():
                real = str(Path(d, "libcuda.so.1"))
        if real:
            libcuda.mkdir(parents=True, exist_ok=True)
            if (libcuda / "libcuda.so").is_symlink() or (libcuda / "libcuda.so").exists():
                (libcuda / "libcuda.so").unlink()
            (libcuda / "libcuda.so").symlink_to(Path(real).resolve()); shims["libcuda.so"] = f"shim -> {real}"
        else:
            shims["libcuda.so"] = "libcuda.so.1 NOT FOUND"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_HOME", "CUDA_PATH")}
    env.update({
        "PATH": os.pathsep.join([str(venv / "bin"), str(cuda_home / "bin"), os.environ.get("PATH", "")]),
        "CUDA_HOME": str(cuda_home), "CUDA_PATH": str(cuda_home),
        "LIBRARY_PATH": os.pathsep.join(x for x in (str(libcuda), str(cuda_home / "lib"), os.environ.get("LIBRARY_PATH", "")) if x),
        "LD_LIBRARY_PATH": os.pathsep.join(x for x in (str(cuda_home / "lib"), os.environ.get("LD_LIBRARY_PATH", "")) if x),
        "TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "4",
        "SGLANG_CACHE_DIR": "/tmp/e11-cache/sglang", "CUDA_CACHE_PATH": "/tmp/e11-cache/cuda",
        "TRITON_CACHE_DIR": "/tmp/e11-cache/triton", "TORCH_EXTENSIONS_DIR": "/tmp/e11-cache/torch_ext",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1",
    })
    shims.update(cuda_home=str(cuda_home), venv_python=str(vpy), sec=round(time.perf_counter() - t0, 1))
    return vpy, env, shims


# ---- launch: the E11-04 W1 command; one diagnosed fallback per cause (E11-04 rev 2b next_launch)
W1 = {"strategy": "extra_buffer_lazy", "mamba_cache": 56, "no_prefill_graph": True, "mem": "0.83"}
E11_OOM_MARKERS = ("out of memory", "outofmemory", "cuda error: out of memory", "not enough memory", "insufficient memory",
                   "cudaerrormemoryallocation")


def e11_flags(cfg):
    f = ["--model-path", str(SGL_MODEL_PATH), "--tp-size", "4", "--kv-cache-dtype", "fp8_e4m3",
         "--context-length", str(SGL_CONTEXT_LEN), "--max-running-requests", str(SGL_MAX_RUNNING),
         "--chunked-prefill-size", "4096", "--mamba-ssm-dtype", "float16", "--enable-metrics", "--disable-flashinfer-autotune",
         "--random-seed", str(SEED), "--host", "127.0.0.1", "--port", str(SGL_PORT), "--log-level", "info",
         "--cuda-graph-max-bs-decode", str(SGL_MAX_RUNNING), "--watchdog-timeout", "1800", "--decode-log-interval", "40",
         "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", str(SGL_DRAFT_PATH),
         "--speculative-draft-model-quantization", "unquant", "--speculative-num-draft-tokens", "8"]
    if cfg["no_prefill_graph"]:
        f += ["--disable-prefill-cuda-graph"]
    f += ["--mamba-radix-cache-strategy", cfg["strategy"], "--max-mamba-cache-size", str(cfg["mamba_cache"]),
          "--mem-fraction-static", cfg["mem"],
          # E11-05: streaming for the shim (disjoint deltas); a chunk at most every 8 tokens (output_streamer.py:427-436)
          "--incremental-streaming-output", "--stream-interval", "8"]
    return f


def e11_next(cfg, tail, used):
    """(cfg, reason) for the one diagnosed relaunch of this cause, or None. Lazy / prefill causes are read from error
    lines only (normal startup lines also say 'Mamba Cache ...')."""
    t = "\n".join(l for l in (tail or "").splitlines()
                  if any(k in l.lower() for k in ("error", "exception", "assert", "unsupported", "not supported")))[-4000:].lower()
    if any(k in (tail or "").lower() for k in E11_OOM_MARKERS) and cfg["mem"] == "0.83" and "oom" not in used:
        return dict(cfg, mem="0.78"), "oom"
    if cfg["strategy"] == "extra_buffer_lazy" and ("lazy" in t or "extra_buffer" in t or "mamba" in t) and "lazy" not in used:
        return dict(cfg, strategy="extra_buffer", mamba_cache=70), "lazy"          # 5 slots per request -> 14 >= 12
    if cfg["no_prefill_graph"] and "graph" in t and "prefill" not in used:
        return dict(cfg, no_prefill_graph=False), "prefill"
    return None


def e11_slots(cfg):
    """kv_cache_configurator.py:2232-2261 with the overlap scheduler on: extra_buffer 5, extra_buffer_lazy 4 slots."""
    return {"extra_buffer_lazy": 4, "no_buffer": 3}.get(cfg["strategy"], 5)


def e11_caps(text):
    """Effective limits from the server log (/get_server_info reports the requested values)."""
    caps = {}
    m = re.findall(r"max_running_requests is capped to (\d+) by the mamba state cache \(max_mamba_cache_size=(\d+), (\d+) state", text)
    caps["capped_to"] = min(int(a) for a, _, _ in m) if m else None
    tp0 = "\n".join(l for l in text.splitlines() if "TP0]" in l) or text
    kv = re.findall(r"KV Cache is allocated\. dtype: ([\w.]+), #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB", tp0)
    caps["kv_tokens"] = int(kv[0][1]) if kv else None                     # first TP0 line = target (second = drafter)
    mc = re.findall(r"Mamba Cache is allocated\. max_mamba_cache_size: (\d+)", tp0)
    caps["mamba_cache_size"] = int(mc[0]) if mc else None
    vb = re.findall(r"Capture target verify CUDA graph begin\..*?bs=\[([\d, ]+)\]", text)
    caps["verify_graph_bs"] = [int(x) for x in re.findall(r"\d+", vb[0])] if vb else None
    caps["prefill_graph_captured"] = "Capture target prefill CUDA graph" in text
    am = re.findall(r"avail mem=([\d.]+) GB", tp0)
    caps["avail_mem_gb_last"] = float(am[-1]) if am else None
    return caps


class E11Server:
    def __init__(self, cmd, env, tag):
        self.cmd, self.env, self.tag, self.proc, self.log_f = cmd, env, tag, None, None
        self.log_path = OUT / f"sglang_server_{tag}.log"

    def start(self):
        self.log_f = open(self.log_path, "w", encoding="utf-8")
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=self.log_f, stderr=subprocess.STDOUT,
                                     cwd=None if E11_HARNESS else "/tmp", start_new_session=True)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_text(self):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def wait_ready(self, timeout):
        while time.perf_counter() - self.t0 < timeout:
            if not self.alive():
                return False, f"server exited rc={self.proc.returncode}"
            try:
                _urlreq.urlopen(SGL_URL + "/health", timeout=10).read()
                return True, f"ready in {time.perf_counter() - self.t0:.0f}s"
            except Exception:
                time.sleep(2 if E11_HARNESS else 5)
        return False, "timeout"

    def kill(self, wait=True):
        if self.alive():
            for sig, w in ((_signal.SIGTERM, 30 if wait else 0), (getattr(_signal, "SIGKILL", _signal.SIGTERM), 30)):
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(self.proc.pid, sig)          # the whole group: TP workers, scheduler, detokenizer
                    else:
                        self.proc.kill()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=w if w else 0.1)
                    break
                except Exception:
                    continue
        if self.log_f:
            try:
                self.log_f.close()
            except Exception:
                pass
            self.log_f = None


def e11_kill_server(wait=True):
    if E11_SERVER is not None:
        E11_SERVER.kill(wait=wait)


# ---- the shim: the vLLM engine calls of the S49 scheduler on top of SGLang's HTTP API
class _E11Completion:
    __slots__ = ("token_ids", "finish_reason")

    def __init__(self, ids, reason):
        self.token_ids, self.finish_reason = ids, reason


class _E11Output:
    __slots__ = ("request_id", "outputs", "finished")

    def __init__(self, rid, ids, reason):
        self.request_id, self.outputs, self.finished = rid, [_E11Completion(ids, reason)], reason is not None


class E11SglangEngine:
    """add_request starts one streaming /generate per request in a thread; step() returns the token deltas that arrived
    (merged per request, order kept) and raises when the server died; abort_request posts /abort_request per id."""

    def __init__(self, server):
        self.server, self.q, self.aborted = server, _queue.Queue(), set()
        self.lock = threading.Lock()
        self.stats = {"added": 0, "finished": 0, "by_reason": {}, "errors": [], "prompt_tokens": 0, "cached_tokens": 0,
                      "completion_tokens": 0, "aborts_sent": 0, "abort_errors": 0, "accept_len_sum": 0.0, "accept_len_n": 0}
        self.req_log = OUT / "sglang_requests.jsonl"

    def add_request(self, rid, prompt, params):
        if not self.server.alive():
            raise RuntimeError(f"SGLang server is not running (rc={self.server.proc.returncode})")
        sp = {"max_new_tokens": int(params.max_tokens), "temperature": float(params.temperature), "top_p": float(params.top_p),
              "top_k": int(params.top_k), "skip_special_tokens": False, "no_stop_trim": True}
        body = json.dumps({"rid": rid, "input_ids": [int(x) for x in prompt["prompt_token_ids"]],
                           "sampling_params": sp, "stream": True}).encode()
        with self.lock:
            self.stats["added"] += 1
        threading.Thread(target=self._stream, args=(rid, body, time.perf_counter()), daemon=True).start()

    def _stream(self, rid, body, t0):
        reason, meta, err, n = None, {}, None, 0
        try:
            req = _urlreq.Request(SGL_URL + "/generate", data=body, method="POST", headers={"Content-Type": "application/json"})
            with _urlreq.urlopen(req, timeout=SGL_STREAM_IDLE_SEC) as r:
                for raw in r:
                    line = raw.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    d = json.loads(data)
                    if "error" in d:
                        err = str(d["error"])[:300]
                        break
                    ids = d.get("output_ids") or []
                    m = d.get("meta_info") or {}
                    fr = m.get("finish_reason")
                    if fr is not None and reason is None:
                        reason = (fr.get("type") if isinstance(fr, dict) else str(fr)) or "stop"
                        meta = m
                    n += len(ids)
                    if ids or reason is not None:
                        self.q.put((rid, list(ids), reason))
                    if reason is not None:
                        break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
        if reason is None:
            reason = "error"
            self.q.put((rid, [], "error"))
        rec = {"rid": rid, "finish": reason, "tokens": n, "sec": round(time.perf_counter() - t0, 1),
               "prompt_tokens": meta.get("prompt_tokens"), "cached_tokens": meta.get("cached_tokens"),
               "completion_tokens": meta.get("completion_tokens"), "spec_accept_length": meta.get("spec_accept_length"),
               "error": err}
        with self.lock:
            s = self.stats
            s["finished"] += 1; s["by_reason"][reason] = s["by_reason"].get(reason, 0) + 1
            s["prompt_tokens"] += int(meta.get("prompt_tokens") or 0); s["cached_tokens"] += int(meta.get("cached_tokens") or 0)
            s["completion_tokens"] += int(meta.get("completion_tokens") or 0)
            if meta.get("spec_accept_length") is not None:
                s["accept_len_sum"] += float(meta["spec_accept_length"]); s["accept_len_n"] += 1
            if err and len(s["errors"]) < 50:
                s["errors"].append(f"{rid}: {err}")
            try:
                with open(self.req_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass

    def step(self):
        if not self.server.alive():
            raise RuntimeError(f"SGLang server exited (rc={self.server.proc.returncode}); see {self.server.log_path}")
        items = []
        try:
            items.append(self.q.get(timeout=0.05))
        except _queue.Empty:
            return []
        while True:
            try:
                items.append(self.q.get_nowait())
            except _queue.Empty:
                break
        merged = {}
        for rid, ids, reason in items:
            m = merged.setdefault(rid, [[], None])
            if m[1] is None:
                m[0].extend(ids); m[1] = reason
        return [_E11Output(rid, ids, reason) for rid, (ids, reason) in merged.items()]

    def abort_request(self, ids):
        for rid in ids:
            self.aborted.add(rid)
            try:
                _e11_post("/abort_request", {"rid": rid}, timeout=30)
                with self.lock:
                    self.stats["aborts_sent"] += 1
            except Exception as e:
                with self.lock:
                    self.stats["abort_errors"] += 1
                    if len(self.stats["errors"]) < 50:
                        self.stats["errors"].append(f"abort {rid}: {type(e).__name__}: {e}"[:300])


_E11_PROM_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def e11_prom(timeout=10):
    txt = _urlreq.urlopen(SGL_URL + "/metrics", timeout=timeout).read().decode("utf-8", "replace")
    rows = []
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _E11_PROM_RE.match(line)
        if not m:
            continue
        try:
            rows.append(_types.SimpleNamespace(name=m.group(1), labels=m.group(2), value=float(m.group(3))))
        except ValueError:
            continue
    return rows


class E11SglangLLM:
    """llm.get_metrics() for the S49 KV gate: the gate reads names containing 'kv_cache_usage'. SGLang's full_token_usage
    (KV pool used, evictable cache excluded) is given that name; token_usage = max(full, mamba) is kept as it is."""

    def get_metrics(self):
        rows = e11_prom()
        full = [r.value for r in rows if r.name == "sglang:full_token_usage"]
        anyu = [r.value for r in rows if r.name == "sglang:token_usage"]
        kv = max(full) if full else (max(anyu) if anyu else None)
        if kv is not None:
            rows.append(_types.SimpleNamespace(name="e11:kv_cache_usage(sglang:full_token_usage)", labels=None, value=kv))
        return rows


_E11_SNAP_KEYS = ("num_running_reqs", "num_queue_reqs", "token_usage", "gen_throughput", "spec_accept", "cache_hit_rate",
                  "retract", "generation_tokens_total", "prompt_tokens_total", "cached_tokens_total", "kv_cache_usage",
                  "num_requests_total")


def metrics_snapshot():
    if not ENGINE_READY:
        return []
    try:
        return [{"name": r.name, "labels": r.labels, "value": r.value} for r in llm.get_metrics()
                if any(k in r.name for k in _E11_SNAP_KEYS) and not r.name.endswith(("_bucket", "_created"))]
    except Exception as e:
        return [{"error": repr(e)[:300]}]


_E11_KV_RE = re.compile(r"([A-Za-z#][A-Za-z#_ ()/\-]*?):\s*(-?\d+(?:\.\d+)?)")


def e11_decode_summary():
    """Decode rate by running level and by mean context per running request, from the server's TP0 decode lines."""
    if E11_SERVER is None:
        return None
    rows = []
    for line in E11_SERVER.log_text().splitlines():
        if "Decode batch" in line and "TP0]" in line:
            rows.append({k.strip(): float(v) for k, v in _E11_KV_RE.findall(line.split("Decode batch", 1)[1])})
    def med(v):
        return round(_statistics.median(v), 1) if v else None
    bins = {}
    for r in rows:
        n, ft, g = r.get("#running-req"), r.get("#full token"), r.get("gen throughput (token/s)")
        if not n or ft is None or g is None:
            continue
        ctx = ft / n
        k = "<10k" if ctx < 10000 else "10-20k" if ctx < 20000 else "20-40k" if ctx < 40000 else "40-60k" if ctx < 60000 else ">=60k"
        bins.setdefault(f"running={int(n)}" if n < SGL_MAX_RUNNING else "running=max", {}).setdefault(k, []).append(g)
    out = {"n_lines": len(rows),
           "running_hist": {str(int(k)): sum(1 for r in rows if r.get("#running-req") == k) for k in sorted({r.get("#running-req", 0) for r in rows})},
           "gen_tok_s_median_at_max": med([r["gen throughput (token/s)"] for r in rows
                                           if r.get("#running-req") == SGL_MAX_RUNNING and "gen throughput (token/s)" in r]),
           "accept_len_median": med([r["accept len"] for r in rows if "accept len" in r]),
           "queue_max": max((r.get("#queue-req", 0) for r in rows), default=None),
           "retract_lines": sum(1 for l in E11_SERVER.log_text().splitlines() if "retract" in l.lower()),
           "by_context": {lvl: {k: {"lines": len(v), "gen_tok_s_median": med(v)} for k, v in d.items()} for lvl, d in bins.items()}}
    try:
        write_json_atomic(OUT / "sglang_decode_summary.json", out)
    except Exception:
        pass
    return out


def e11_smoke(eng):
    """The shim's own path: (a) greedy 16 tokens, streamed deltas == completion_tokens; (b) abort mid-stream -> the stream
    ends with finish 'abort' (the drain and the hard stop rely on it to free the slot)."""
    res, t0 = {}, time.perf_counter()
    ids = tokenizer.encode("Reply with the digits 1 2 3 4 5 6 7 8:", add_special_tokens=False)
    eng.add_request("e11-smoke-a", {"prompt_token_ids": ids}, SamplingParams(max_tokens=16, temperature=0.0, top_p=1.0, top_k=-1))
    got, fin = [], None
    while fin is None and time.perf_counter() - t0 < 600:
        for o in eng.step():
            if o.request_id == "e11-smoke-a":
                got += o.outputs[0].token_ids; fin = o.outputs[0].finish_reason if o.finished else None
    time.sleep(0.5)
    logged = [json.loads(l) for l in open(eng.req_log, encoding="utf-8")] if eng.req_log.exists() else []
    meta = next((r for r in logged if r["rid"] == "e11-smoke-a"), {})
    res["a"] = {"tokens": len(got), "finish": fin, "completion_tokens": meta.get("completion_tokens"),
                "text": tokenizer.decode(got)[:80]}
    res["a"]["ok"] = bool(got) and fin in ("stop", "length") and meta.get("completion_tokens") == len(got)
    t1 = time.perf_counter()
    eng.add_request("e11-smoke-b", {"prompt_token_ids": ids}, SamplingParams(max_tokens=4096, temperature=0.6, top_p=0.95, top_k=20))
    got, fin, sent = [], None, False
    while fin is None and time.perf_counter() - t1 < 600:
        for o in eng.step():
            if o.request_id == "e11-smoke-b":
                got += o.outputs[0].token_ids; fin = o.outputs[0].finish_reason if o.finished else None
        if got and not sent:
            eng.abort_request(["e11-smoke-b"]); sent = True; t_abort = time.perf_counter()
        if sent and time.perf_counter() - t_abort > 60:
            break
    res["b"] = {"tokens": len(got), "finish": fin, "abort_sent": sent,
                "abort_to_finish_sec": round(time.perf_counter() - t_abort, 2) if sent else None}
    res["b"]["ok"] = sent and fin == "abort" and len(got) < 4096
    res["ok"] = res["a"]["ok"] and res["b"]["ok"]
    res["sec"] = round(time.perf_counter() - t0, 1)
    return res


def e11_bring_up():
    global E11_SERVER
    E11_SGL["stage"] = "inputs"
    inp = {"int4": SGL_MODEL_PATH, "dflash2": SGL_DRAFT_PATH, "wheelhouse": SGL_WHEELHOUSE}
    E11_SGL["inputs"] = {k: str(v) for k, v in inp.items()}
    missing = [k for k, v in inp.items() if not (v / ("BUILD_INFO.json" if k == "wheelhouse" else "config.json")).is_file()]
    if missing:
        raise E11Fail(f"inputs missing: {missing}")
    if (_load_json(SGL_MODEL_PATH / "config.json").get("quantization_config") or {}).get("quant_method") != "compressed-tensors":
        raise E11Fail("INT4 checkpoint is not compressed-tensors")
    E11_SGL["stage"] = "install"
    vpy, env, E11_SGL["install"] = e11_install()
    launch = [str(vpy), "-m", "sglang.launch_server"] if not E11_HARNESS else [sys.executable, os.environ["E11_FAKE_SERVER"]]
    cfg, used = dict(W1), []
    for i in range(3):
        left = SGL_GIVE_UP_AT - time.perf_counter()
        if left < 60:
            raise E11Fail("give-up deadline reached before a launch")
        E11_SGL["stage"] = f"launch {i}"
        srv = E11Server(launch + e11_flags(cfg), env, f"try{i}")
        srv.start()
        ok, why = srv.wait_ready(min(SGL_LAUNCH_TIMEOUT_SEC if i == 0 else SGL_RELAUNCH_TIMEOUT_SEC, left))
        att = {"try": i, "cfg": dict(cfg), "ready": ok, "why": why, "sec": round(time.perf_counter() - srv.t0, 1)}
        E11_SGL["attempts"].append(att)
        print(f"E11 SGLang launch {i} {cfg}: {why}", flush=True)
        if ok:
            E11_SERVER = srv
            break
        tail = srv.log_text()[-20000:]
        att["log_tail"] = tail[-3000:]
        srv.kill()
        att["gpu_free"] = _e11_gpu_free()
        nxt = e11_next(cfg, tail, used)
        if nxt is None:
            raise E11Fail(f"launch {i} failed ({why}); no diagnosed fallback")
        cfg, reason = nxt
        used.append(reason); att["retry_reason"] = reason
    else:
        raise E11Fail("3 launches failed")
    E11_SGL["launch_flags"] = e11_flags(cfg)
    caps = e11_caps(E11_SERVER.log_text())
    slots = e11_slots(cfg)
    eff = min(SGL_MAX_RUNNING, caps["mamba_cache_size"] // slots) if caps.get("mamba_cache_size") else None
    if eff is not None and caps.get("capped_to"):
        eff = min(eff, caps["capped_to"])
    caps.update(slots_per_request=slots, effective_running=eff)
    E11_SGL["caps"] = caps
    print("E11 caps:", json.dumps(caps), flush=True)
    E11_SGL["stage"] = "smoke"
    eng = E11SglangEngine(E11_SERVER)
    E11_SGL["smoke"] = e11_smoke(eng)
    print("E11 smoke:", json.dumps(E11_SGL["smoke"]), flush=True)
    if not E11_SGL["smoke"]["ok"]:
        raise E11Fail("smoke failed: " + json.dumps(E11_SGL["smoke"])[:600])
    return eng, caps


_t_engine = time.perf_counter()
if not E10_SANDBOX_SELFTEST.get("ok", False):
    E11_SGL["stage"] = "skipped: sandbox self-test failed"
elif ENGINE_BACKEND != "sglang":
    E11_SGL["stage"] = f"skipped: ENGINE_BACKEND={ENGINE_BACKEND}"
else:
    try:
        engine, _caps = e11_bring_up()
        llm = E11SglangLLM()
        ENGINE_READY = True
        # the server queues anything above its real running limit, so 12 in flight is safe even if the cap were lower
        CONCURRENCY = SGL_MAX_RUNNING
        E11_SGL.update(ok=True, stage="ready")
        ENGINE_INFO.update(backend="sglang-0.5.21-int4-dflash2-w1", kv_tokens=_caps.get("kv_tokens"),
                           effective_running=_caps.get("effective_running"))
    except Exception as _e:
        E11_SGL["error"] = f"{type(_e).__name__}: {_e}"[:3000]
        E11_SGL["traceback"] = traceback.format_exc()[-3000:]
        print("E11 SGLang engine NOT ready:", E11_SGL["stage"], "|", E11_SGL["error"][:1500], flush=True)
        e11_kill_server(); E11_SERVER = None
        engine = None; llm = None; ENGINE_READY = False; CONCURRENCY = 0
ENGINE_INFO["load_sec"] = round(time.perf_counter() - _t_engine, 1)
ENGINE_INFO["concurrency"] = int(CONCURRENCY)
print("engine:", json.dumps(ENGINE_INFO, default=str)[:3000], flush=True)
'''

# ---------------------------------------------------------------- CELL 5b: S49 engine cell as the vLLM fallback
_body = textwrap.indent(C5, "    ")
C5B = ("# ==================================================================== CELL 5b — vLLM FP8 fallback (S49's engine cell)\n"
       "# E11-05: runs only when the SGLang engine is not ready. The indented code is S49's cell 5, unchanged (AST-checked).\n"
       "if not ENGINE_READY and E11_ALLOW_VLLM_FALLBACK:\n"
       "    _e11_sgl_info = E11_SGL\n"
       "    e11_kill_server(); E11_SERVER = None\n"
       "    print('E11: SGLang engine not ready -> S49 vLLM FP8 engine', '| gpu', _e11_gpu_free(), flush=True)\n"
       "    if (os.environ.get('Q27X_SKIP_INSTALL') != '1' and E10_SANDBOX_SELFTEST.get('ok', False)\n"
       "            and pkg_version('vllm') != EXPECTED_RUNTIME_PACKAGES['vllm']):\n"
       "        offline_pip_install(['vllm==0.19.0'])\n"
       "        if pkg_version('vllm') != EXPECTED_RUNTIME_PACKAGES['vllm']:\n"
       "            raise RuntimeError(f\"runtime parity failure: vllm={pkg_version('vllm')!r}\")\n"
       + _body.rstrip() + "\n"
       "    ENGINE_INFO['backend'] = 'vllm-0.19.0-fp8-mtp (E11 fallback)' if ENGINE_READY else None\n"
       "    ENGINE_INFO['sglang'] = _e11_sgl_info\n"
       "else:\n"
       "    print('CELL 5b skipped:', 'SGLang engine ready' if ENGINE_READY else 'fallback disabled', flush=True)\n")
# the fallback body must be S49's cell 5 exactly (only indentation added)
_tree = ast.parse(C5B)
_if = next(n for n in _tree.body if isinstance(n, ast.If))
_orig = ast.parse(C5).body
_n_pre = 5                                   # _e11_sgl_info, kill, None, print, pip-if
_inner = _if.body[_n_pre:_n_pre + len(_orig)]
assert ast.dump(ast.Module(body=_inner, type_ignores=[])) == ast.dump(ast.Module(body=_orig, type_ignores=[])), \
    "fallback body differs from S49's cell 5"

# ---------------------------------------------------------------- report cell
C7 = once(C7, "    write_json_atomic(OUT/'poetiq_run_report.json',report)\n",
          "    try:   # E11-05: engine evidence (decode rate by context, shim counters incl. prefix-cache hits)\n"
          "        report['e11']={'backend':ENGINE_INFO.get('backend'),'decode_summary':e11_decode_summary(),\n"
          "                       'shim':engine.stats if isinstance(getattr(engine,'stats',None),dict) else None}\n"
          "    except Exception as _e11e:\n"
          "        report['e11']={'error':repr(_e11e)[:300]}\n"
          "    write_json_atomic(OUT/'poetiq_run_report.json',report)\n", "report e11")
C7 = once(C7, "                   OUT/'poetiq_vllm_metrics.json'):", "                   OUT/'poetiq_vllm_metrics.json',OUT/'sglang_decode_summary.json'):",
          "bundle files")
C7 = C7.rstrip() + "\ntry:   # E11-05: stop the SGLang server (no-op on the vLLM path)\n    e11_kill_server(wait=False)\nexcept Exception:\n    pass\n"

# ---------------------------------------------------------------- write
new_src = [MD, C0, C1, C2, C3, C4, C5_SGL, C5B, C6, C7]
for i, s in enumerate(new_src[1:], 1):
    compile(s, f"<cell{i}>", "exec")
assert new_src[1] == C0 and new_src[5] == C4 and new_src[8] == C6         # untouched cells
assert "'''" not in C5_SGL


def to_cell(kind, text):
    lines = text.splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


nb["cells"] = [to_cell("markdown", MD)] + [to_cell("code", s) for s in new_src[1:]]
out = OUTDIR / NAME
data = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
out.write_text(data, encoding="utf-8", newline="\n")
print("wrote", out, "sha256", hashlib.sha256(out.read_bytes()).hexdigest())
print("cells:", [len(s) for s in new_src])
