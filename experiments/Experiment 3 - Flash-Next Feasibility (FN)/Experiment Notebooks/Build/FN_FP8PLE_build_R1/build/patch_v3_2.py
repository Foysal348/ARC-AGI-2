"""FN-SMOKE v3.2 (review R3, each point checked against the Pennyroyal source):
1. mamba state slots: kv_cache_configurator.py:1919-1947 + constants 156-161 -> ratio = 1 with --disable-radix-cache,
   3 + 2 = 5 with extra_buffer (overlap schedule on, the default). max_num_reqs = max_mamba_cache_size // ratio, and
   <= 0 raises RuntimeError AFTER the model load. v3.1 had 4 for extra_buffer -> 0 requests. Now per attempt:
   no radix -> 2, extra_buffer -> 6 (one slot of margin each).
2. RLIMIT_MEMLOCK on Kaggle is 8 MiB hard. Rather than argue whether cudaHostRegister is charged against it, stage 3
   now registers 64 MiB and then 1 GiB of mmap'd anonymous memory with the exact Pennyroyal call/flags; failure stops
   the run before the 125 GB load (stage 3_pinned_memory).
3. attempt order: simplest first. A1 radix cache off + triton attention + triton MoE; A2 extra_buffer (interval 64) +
   triton attention + triton MoE. MoE runner auto is left for a later run (public Marlin TP4 failures).
4. GPU memory: mem-fraction-static 0.80 -> 0.90 (EP4 main weights ~17.1 GiB/GPU vs 18.0 GiB static budget at 0.80);
   context 4096, chunked prefill 1024, max prefill 2048 for the load-only probe.
usage: python patch_v3_2.py   (make_fn_smoke_v3_1.py -> make_fn_smoke_v3_2.py; run that to write the notebook)"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = (HERE / "make_fn_smoke_v3_1.py").read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:90], s.count(a))
    s = s.replace(a, b)


# ---------------------------------------------------------------- launch profile
rep('''            "--mem-fraction-static", "0.80", "--context-length", "8192", "--page-size", "64",
            "--max-running-requests", "1", "--chunked-prefill-size", "2048", "--max-prefill-tokens", "4096",
            "--mamba-ssm-dtype", "bfloat16", "--max-mamba-cache-size", "4",''',
    '''            "--mem-fraction-static", "0.90", "--context-length", "4096", "--page-size", "64",
            "--max-running-requests", "1", "--chunked-prefill-size", "1024", "--max-prefill-tokens", "2048",
            "--mamba-ssm-dtype", "bfloat16",''')
rep('''ATTEMPTS = [
    ("tp4_ep4_extrabuf_moe_auto",  ["--mamba-radix-cache-strategy", "extra_buffer", "--mamba-track-interval", "64",
                                    "--moe-runner-backend", "auto"]),
    ("tp4_ep4_noradix_all_triton", ["--disable-radix-cache", "--attention-backend", "triton",
                                    "--moe-runner-backend", "triton"]),
]''', '''# v3.2: mamba state slots per request (kv_cache_configurator.py:1919-1947): 1 with the radix cache disabled, 5 with
# extra_buffer (3 + 2, overlap schedule on); max requests = max_mamba_cache_size // slots, 0 raises after the load.
MAMBA_SLOTS_PER_REQ = {"no_radix": 1, "extra_buffer": 5}
ATTEMPTS = [
    ("tp4_ep4_noradix_all_triton",   ["--disable-radix-cache", "--attention-backend", "triton",
                                      "--moe-runner-backend", "triton", "--max-mamba-cache-size", "2"]),
    ("tp4_ep4_extrabuf_all_triton",  ["--mamba-radix-cache-strategy", "extra_buffer", "--mamba-track-interval", "64",
                                      "--attention-backend", "triton", "--moe-runner-backend", "triton",
                                      "--max-mamba-cache-size", "6"]),
]''')

# ---------------------------------------------------------------- stage 3: real pinned-memory probe
rep('''        stage("3_preflight", r.returncode == 0 and h.returncode == 0 and not missing, cuda_home=CUDA_HOME, cxx=CXX,
              probe_output=(r.stdout + r.stderr)[-3000:], missing_flags=missing, help_exit=h.returncode)''',
    '''        stage("3_preflight", r.returncode == 0 and h.returncode == 0 and not missing, cuda_home=CUDA_HOME, cxx=CXX,
              probe_output=(r.stdout + r.stderr)[-3000:], missing_flags=missing, help_exit=h.returncode)
        if REPORT["stopped_at"] is None:
            # RLIMIT_MEMLOCK is 8 MiB on Kaggle: prove that cudaHostRegister (Pennyroyal's exact call and flags) can pin
            # far more than that before the 125 GB load. The PLE needs ~12.8 GB per rank.
            pin = subprocess.run([PYTHON, "-c", PIN_PROBE], text=True, capture_output=True, env=env, timeout=clamp(600))
            stage("3_pinned_memory", pin.returncode == 0 and "PIN_PROBE_OK" in pin.stdout,
                  memlock=REPORT["stages"].get("0_environment", {}).get("memlock"),
                  output=(pin.stdout + pin.stderr)[-2500:])''')
rep('''# ==================================================================== CELL 5 — stage 3: torch + CUDA toolkit in the venv
''', '''# ==================================================================== CELL 5 — stage 3: torch + CUDA toolkit in the venv
PIN_PROBE = """
import mmap, time, torch
torch.zeros(1, device="cuda")                       # create the CUDA context on GPU 0
FLAGS = 0x01 | 0x02                                 # cudaHostRegisterPortable | Mapped, as in qwen4_exp._allocate_pinned_table
for mib in (64, 1024):
    n = mib << 20
    buf = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    raw = torch.frombuffer(buf, dtype=torch.uint8)
    t = time.time()
    err = torch.cuda.cudart().cudaHostRegister(raw.data_ptr(), n, FLAGS)
    if int(err) != 0:
        raise SystemExit(f"cudaHostRegister {mib} MiB failed: {err}")
    gpu = torch.empty(n, dtype=torch.uint8, device="cuda"); gpu.copy_(raw, non_blocking=True); torch.cuda.synchronize()
    err = torch.cuda.cudart().cudaHostUnregister(raw.data_ptr())
    if int(err) != 0:
        raise SystemExit(f"cudaHostUnregister {mib} MiB failed: {err}")
    del raw, gpu; buf.close()
    print(f"pinned {mib} MiB OK in {time.time() - t:.2f}s")
print("PIN_PROBE_OK")
"""
''')

# ---------------------------------------------------------------- report: the probe is an expected stage
rep('''_expected = ["0_environment", "1_inputs", "2_install", "3_preflight", "4_launch",''',
    '''_expected = ["0_environment", "1_inputs", "2_install", "3_preflight", "3_pinned_memory", "4_launch",''')

# ---------------------------------------------------------------- header
rep("# FN-SMOKE v3.1 — Qwen3.8-Flash-Next (Intel W4A16 AutoRound + FP8 PLE) on Kaggle 4×L4 for ARC-AGI-2",
    "# FN-SMOKE v3.2 — Qwen3.8-Flash-Next (Intel W4A16 AutoRound + FP8 PLE) on Kaggle 4×L4 for ARC-AGI-2")
rep('''3. preflight: torch/sglang/sgl_kernel/flashinfer/triton import on all 4 GPUs (8,9); `sglang serve --help` must exit 0
   and list every flag used''', '''3. preflight: torch/sglang/sgl_kernel/flashinfer/triton import on all 4 GPUs (8,9); `sglang serve --help` must exit 0
   and list every flag used; then `3_pinned_memory`: cudaHostRegister of 64 MiB and 1 GiB (Pennyroyal's call and flags),
   because RLIMIT_MEMLOCK is 8 MiB on Kaggle — failure stops the run before the model load''')
rep('''4. launch, re-checking MemAvailable right before it: TP4 + EP4, Triton GDN, BF16 KV, 8K context, 1 request,
   no CUDA graphs, no DP-attention, no speculative decoding. Attempt 1: mamba radix `extra_buffer` (track interval
   64, as dfranzen) + MoE runner `auto`; attempt 2: radix cache disabled + triton attention + triton MoE''',
    '''4. launch, re-checking MemAvailable right before it: TP4 + EP4, Triton GDN, BF16 KV, 4K context, 1 request,
   mem-fraction-static 0.90, no CUDA graphs, no DP-attention, no speculative decoding, triton attention + triton MoE.
   Attempt 1: radix cache disabled (mamba cache 2 slots; 1 needed per request). Attempt 2: mamba radix `extra_buffer`
   with track interval 64, as dfranzen (mamba cache 6 slots; 5 needed per request)''')
rep('''follow dfranzen's launcher.
''', '''follow dfranzen's launcher.

v3.2 (review R3, checked against the Pennyroyal source): mamba cache sized per attempt (v3.1's 4 slots would have
given 0 requests with extra_buffer, after the load); real pinned-memory probe; simplest attempt first; more static
GPU memory, 4K context.
''')
rep('"experiment": "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4-v3.1"', '"experiment": "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4-v3.2"')
(HERE / "make_fn_smoke_v3_2.py").write_text(s, encoding="utf-8")
print("wrote", HERE / "make_fn_smoke_v3_2.py")
