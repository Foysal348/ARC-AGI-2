"""FN-SMOKE v3 (review R1 fixes included): Intel W4A16 main shards (dfranzen, on Kaggle) + albucino FP8 PLE overlay
(foysalemonshanto) merged into one folder; overlay authoritative for every non-weight file; every shard checked by size,
header hash and full SHA256; RAM gate 75 GB + re-check right before launch; memlock logged and raised (informational);
stage-3 fixes. Launch profile unchanged from v2.

usage: python patch_v3.py   (reads make_fn_smoke_v2.py + stage1_v3.txt + manifest.json next to it, writes
make_fn_smoke_v3.py; then `python make_fn_smoke_v3.py` writes ../notebooks/FN-SMOKE-FLASHNEXT-FP8PLE-4XL4.ipynb)"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = (HERE / "make_fn_smoke_v2.py").read_text(encoding="utf-8")
MAN = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
EXPECT = {
    "shards": {n: [v["size"], v["sha256"], v["header_sha256"], v["header_len"]]
               for n, v in sorted(MAN["files"].items()) if n.endswith(".safetensors")},
    "files": {n: [v["size"], v["sha256"]] for n, v in sorted(MAN["files"].items())
              if v["role"] == "overlay" and not n.endswith(".safetensors")},
}
assert len(EXPECT["shards"]) == 25 and "tokenizer.json" in EXPECT["files"] and "config.json" in EXPECT["files"]


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:90], s.count(a))
    s = s.replace(a, b)


rep(r'DST = Path(r"D:\ARC AGI 2\Research Base Experiment\FN-SMOKE-FLASHNEXT-4XL4.ipynb")',
    'DST = Path(__file__).resolve().parent.parent / "notebooks" / "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4.ipynb"')

# ---------------------------------------------------------------- header text (review R1: stale v1/v2 wording)
i = s.index("C_HEAD = r'''")
j = s.index("'''\n", i + len("C_HEAD = r'''")) + 4
s = s[:i] + r"""C_HEAD = r'''# FN-SMOKE v3 — Qwen3.8-Flash-Next (Intel W4A16 AutoRound + FP8 PLE) on Kaggle 4×L4 for ARC-AGI-2

Load-only feasibility probe (`LOAD_ONLY = True`) of the Pennyroyal SGLang fork from dfranzen's ARC-AGI-3 Milestone-2
notebook on 4×L4 (Ada, sm89, 24 GB each). Absolute 50-min deadline; every blocking call is clamped to it.

Model folder (built in stage 1 from symlinks):
* the 15 main W4A16 shards of `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` (byte-identical to
  Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@4c67bf6);
* everything else from the overlay `foysalemonshanto/qwen38-flash-next-fp8ple-overlay` (albucino@c6af9fe): 10 FP8 PLE
  shards (RadixArk FP8 table + scale), config with `ple_embedding_dtype: float8_e4m3fn`, index, tokenizer files.
Pennyroyal allocates the pinned host PLE table directly in FP8: 51.2 GB in total (12.8 GB per TP rank).

Stages (each recorded in `fn_smoke_report.json`; the first failure stops the run):
0. environment gates: 4 GPUs, all L4, all sm89, driver ≥ 580 / CUDA 13, MemAvailable ≥ 75 GB, /tmp ≥ 25 GB;
   RLIMIT_MEMLOCK logged and its soft limit raised to the hard limit (informational: the PLE is pinned with
   mmap + cudaHostRegister, not mlock)
1. inputs: merged model folder; every overlay file SHA256-checked; every shard checked by size, safetensors-header
   SHA256 and full SHA256 against the pinned Hugging Face commits
2. install Pennyroyal offline (uv venv)
3. preflight: torch/sglang/sgl_kernel/flashinfer/triton import on all 4 GPUs (8,9); `sglang serve --help` must exit 0
   and list every flag used
4. launch, re-checking MemAvailable right before it: TP4 + EP4, Triton GDN, BF16 KV, 8K context, 1 request,
   language-model-only, no CUDA graphs, no DP-attention, no speculative decoding; MoE runner `auto`, then `triton`
5. one 64-token generation

Commit / interactive run: stages 0–5. Competition rerun (`KAGGLE_IS_COMPETITION_RERUN`): only a valid fallback
`submission.json` (this probe is not a solver).
'''
""" + s[j:]

# ---------------------------------------------------------------- config
rep('SMOKE_BUDGET_SEC = float(os.environ.get("FN_SMOKE_BUDGET", (45 if LOAD_ONLY else 80) * 60))   # whole-probe wall budget',
    'SMOKE_BUDGET_SEC = float(os.environ.get("FN_SMOKE_BUDGET", (50 if LOAD_ONLY else 85) * 60))   # whole-probe wall budget')
rep('MIN_RAM_AVAILABLE_GB = 110          # PLE pinned host table ~95.4 GB (TP4-sharded, 23.8 GB/rank) + headroom',
    'MIN_RAM_AVAILABLE_GB = float(os.environ.get("FN_MIN_RAM_GB", 75))   # FP8 PLE pinned 51.2 GB (12.8 GB/rank) + ~24 GB headroom')
rep('''MODEL_DIR = _first(["/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
                    "/kaggle/input/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
                    "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/*/*/*"])''',
    '''INTEL_DIR = os.environ.get("FN_SMOKE_INTEL") or _first([
    "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
    "/kaggle/input/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
    "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/*/*/*"])
OVERLAY_DIR = os.environ.get("FN_SMOKE_OVERLAY") or _first([
    "/kaggle/input/models/foysalemonshanto/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1",
    "/kaggle/input/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1",
    "/kaggle/input/models/foysalemonshanto/qwen38-flash-next-fp8ple-overlay/*/*/*"])
MODEL_DIR = os.environ.get("FN_SMOKE_MODEL_DST", "/tmp/fn_model")   # merged folder, built in stage 1 (symlinks)''')
rep('REPORT = {"experiment": "FN-SMOKE-FLASHNEXT-4XL4", "rerun": RERUN, "stages": {}, "stopped_at": None,\n'
    '          "paths": {"competition": COMP_DIR, "wheelhouse": WHEELHOUSE_DIR, "model": MODEL_DIR}}',
    'REPORT = {"experiment": "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4", "rerun": RERUN, "stages": {}, "stopped_at": None,\n'
    '          "paths": {"competition": COMP_DIR, "wheelhouse": WHEELHOUSE_DIR, "intel": INTEL_DIR, "overlay": OVERLAY_DIR,\n'
    '                    "model": MODEL_DIR}}')

# ---------------------------------------------------------------- stage 0: memlock (informational) + MemAvailable helper
rep('''if go():
    smi = os.environ.get("FN_SMOKE_FAKE_SMI") or sh("nvidia-smi")''',
    '''def memlock_info():
    # informational: Pennyroyal pins the PLE with mmap + cudaHostRegister (no mlock); raise soft to hard anyway,
    # the server inherits the limit from this process
    try:
        import resource
        fmt = lambda v: "unlimited" if v == resource.RLIM_INFINITY else v
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        raised = False
        if soft != hard:
            try:
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard)); raised = True
            except (ValueError, OSError):
                pass
        soft2, hard2 = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        return {"soft_before": fmt(soft), "hard": fmt(hard2), "soft_now": fmt(soft2), "raised": raised}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

def mem_available_gb():
    try:
        txt = os.environ.get("FN_SMOKE_FAKE_MEMINFO") or open("/proc/meminfo").read()
    except OSError:
        return 0.0
    for line in txt.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / 1e9
    return 0.0

if go():
    smi = os.environ.get("FN_SMOKE_FAKE_SMI") or sh("nvidia-smi")''')
rep('''          memlock=sh("ulimit -l").strip(), disk_working=sh("df -h /kaggle/working | tail -1").strip())''',
    '''          memlock=memlock_info(), disk_working=sh("df -h /kaggle/working | tail -1").strip())''')

# ---------------------------------------------------------------- stage 1
i = s.index("C_INPUTS = r'''")
j = s.index("'''\n", i + len("C_INPUTS = r'''")) + 4
s = s[:i] + (HERE / "stage1_v3.txt").read_text(encoding="utf-8").replace("__EXPECT__", json.dumps(EXPECT, separators=(",", ":"))) + s[j:]

# ---------------------------------------------------------------- stage 3 (review R1)
rep('''        stage("3_preflight", r.returncode == 0 and not missing, cuda_home=CUDA_HOME, cxx=CXX,''',
    '''        stage("3_preflight", r.returncode == 0 and h.returncode == 0 and not missing, cuda_home=CUDA_HOME, cxx=CXX,''')
rep('''        stage("3_torch_cuda", False, error=f"{type(e).__name__}: {e}"[:3000])''',
    '''        stage("3_preflight", False, error=f"{type(e).__name__}: {e}"[:3000])''')

# ---------------------------------------------------------------- stage 4: RAM re-check right before launch
rep('''# 4 x 95 GB pinned host RAM); plain TP4''', '''# 4 x 51 GB FP8 pinned host RAM); plain TP4''')
rep('''if go():
    results = []
    for name, extra in ATTEMPTS:''',
    '''if go():
    ram_now = mem_available_gb()
    if ram_now < MIN_RAM_AVAILABLE_GB:
        stage("4_launch", False, reason="MemAvailable below the gate right before launch",
              ram_available_gb=round(ram_now, 1), min_gb=MIN_RAM_AVAILABLE_GB)
if go():
    results = []
    for name, extra in ATTEMPTS:''')

(HERE / "make_fn_smoke_v3.py").write_text(s, encoding="utf-8")
print("wrote", HERE / "make_fn_smoke_v3.py")
