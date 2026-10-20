"""FN-SMOKE v3.3 after the second 4xL4 run (2026-10-05): stages 0-3 and the new pinned-memory probe passed
(cudaHostRegister 64 MiB + 1 GiB OK despite RLIMIT_MEMLOCK = 8 MiB). Attempt 1 died ~70 s into the load:
  gptq_kernels.py:360 create_moe_runner: assert get_moe_runner_backend().is_auto()
AutoRound W4 MoE layers use the GPTQ-Marlin MoE method, which accepts only the `auto` runner (it then picks
MoeRunnerBackend.MARLIN). There is no triton MoE path for this quantization; dfranzen uses `auto`. With EP4 every rank
keeps whole experts (intermediate 640, 640 % 128 == 0), so the public TP4 Marlin failure (160-wide shards) does not
apply.
Changes: --moe-runner-backend auto in both attempts; on a failed launch the full log tail of every attempt is printed
(the stage line is cut at 1500 characters).
usage: python patch_v3_3.py   (make_fn_smoke_v3_2.py -> make_fn_smoke_v3_3.py)"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = (HERE / "make_fn_smoke_v3_2.py").read_text(encoding="utf-8")


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:90], s.count(a))
    s = s.replace(a, b)


rep('''"--moe-runner-backend", "triton", "--max-mamba-cache-size", "2"]''', '''"--moe-runner-backend", "auto", "--max-mamba-cache-size", "2"]''')
rep('''"--attention-backend", "triton", "--moe-runner-backend", "triton",
                                      "--max-mamba-cache-size", "6"]''', '''"--attention-backend", "triton", "--moe-runner-backend", "auto",
                                      "--max-mamba-cache-size", "6"]''')
rep('''ATTEMPTS = [
    ("tp4_ep4_noradix_all_triton",''', '''# v3.3: AutoRound W4 MoE = GPTQ-Marlin MoE method, which asserts the runner is `auto` (gptq_kernels.py:360).
ATTEMPTS = [
    ("tp4_ep4_noradix_triton_attn",''')
rep('''    ("tp4_ep4_extrabuf_all_triton",''', '''    ("tp4_ep4_extrabuf_triton_attn",''')
rep('''    stage("4_launch", ok, attempts=results, gpu_mem=sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))''',
    '''    stage("4_launch", ok, attempts=results, gpu_mem=sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))
    # key lines for diagnosis (MoE runner / Marlin, weight loading, memory, errors), success or not
    try:
        KEY = re.compile(r"(?i)marlin|moe.*(runner|backend)|intermediate|load weight|weight load|memory|KV Cache|"
                         r"max_total_num_tokens|mamba|ple|pinned|error|exception|assert|ready to roll|fired up")
        key_lines = [l for l in LOG.read_text(errors="replace").splitlines() if KEY.search(l)]
        print("----- serve.log key lines (last 80) -----"); print("\\n".join(key_lines[-80:]))
    except Exception as e:
        print("key-line scan failed:", e)
    if not ok:   # the stage line is truncated; show every attempt's log tail in full
        for r in results:
            print(); print(f"----- {r.get('attempt')} (exit {r.get('exit_code')}, {r.get('sec')} s) -----")
            print(r.get("log_tail", r.get("skipped", "")))''')
rep('''if go():
    results = []
    for name, extra in ATTEMPTS:''', '''if go():
    # AutoRound W4 MoE = GPTQ-Marlin MoE method: only the `auto` runner is accepted (gptq_kernels.py:360) -> reject a
    # wrong runner here instead of ~70 s into a load
    _b = base_args(); _q = _b[_b.index("--quantization") + 1] if "--quantization" in _b else None
    _bad = [n for n, ex in ATTEMPTS if _q == "auto-round"
            and (ex[ex.index("--moe-runner-backend") + 1] if "--moe-runner-backend" in ex else "auto") != "auto"]
    if _bad:
        stage("4_launch", False, reason="AutoRound W4 MoE accepts only --moe-runner-backend auto", attempts_rejected=_bad)
if go():
    results = []
    for name, extra in ATTEMPTS:''')
rep('''triton attention + triton MoE.
   Attempt 1: radix cache disabled''', '''triton attention, MoE runner auto (Marlin; the only runner AutoRound W4 MoE accepts).
   Attempt 1: radix cache disabled''')
rep("# FN-SMOKE v3.2 — Qwen3.8-Flash-Next", "# FN-SMOKE v3.3 — Qwen3.8-Flash-Next")
rep('''GPU memory, 4K context.
''', '''GPU memory, 4K context.

v3.3 (second 4×L4 run): stages 0-3 passed and cudaHostRegister pinned 1 GiB despite the 8 MiB memlock limit. The load
stopped at `gptq_kernels.py:360` (`assert get_moe_runner_backend().is_auto()`): AutoRound W4 MoE accepts only the
`auto` runner (Marlin). Both attempts now use `auto`.
''')
rep('"experiment": "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4-v3.2"', '"experiment": "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4-v3.3"')
(HERE / "make_fn_smoke_v3_3.py").write_text(s, encoding="utf-8")
print("wrote", HERE / "make_fn_smoke_v3_3.py")
