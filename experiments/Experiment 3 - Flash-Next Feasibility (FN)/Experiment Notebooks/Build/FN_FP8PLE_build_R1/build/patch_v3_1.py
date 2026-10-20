"""FN-SMOKE v3.1 after the first 4xL4 run (2026-10-05): both launches failed in argument resolution after 15 s:
  ValueError: --language-model-only does not support ['Qwen4ExpForConditionalGeneration'].
  Supported: ['MuseGlimmerForConditionalGeneration']        (server_args.py:8469)
and the next check would have failed too: compressed QSA pins page_size=64 and the default mamba radix strategy
`no_buffer` asserts page_size == 1 (server_args.py:6267). dfranzen's launcher uses
`--mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64` (supported for Qwen4Exp when
linear_attn_backend == "triton", overrides.py:1842-1872; interval % page_size == 0).

Changes: drop --language-model-only; attempt 1 = dfranzen's radix settings (extra_buffer, track interval 64) with
MoE runner auto; attempt 2 = most conservative (radix cache disabled, triton attention, triton MoE).
usage: python patch_v3_1.py   (make_fn_smoke_v3.py -> make_fn_smoke_v3_1.py; run that to write the notebook)"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = (HERE / "make_fn_smoke_v3.py").read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:90], s.count(a))
    s = s.replace(a, b)


rep('''            "--ple-offload-embedding", "--language-model-only", "--trust-remote-code",''',
    '''            "--ple-offload-embedding", "--trust-remote-code",''')
rep('''ATTEMPTS = [
    ("tp4_ep4_triton_moe_auto",   ["--moe-runner-backend", "auto"]),
    ("tp4_ep4_triton_moe_triton", ["--moe-runner-backend", "triton"]),
]''', '''# v3.1: compressed QSA pins page_size=64, and the default mamba radix strategy (no_buffer) requires page_size 1 ->
# use dfranzen's extra_buffer (needs linear_attn_backend triton, the default) or disable the radix cache.
ATTEMPTS = [
    ("tp4_ep4_extrabuf_moe_auto",  ["--mamba-radix-cache-strategy", "extra_buffer", "--mamba-track-interval", "64",
                                    "--moe-runner-backend", "auto"]),
    ("tp4_ep4_noradix_all_triton", ["--disable-radix-cache", "--attention-backend", "triton",
                                    "--moe-runner-backend", "triton"]),
]''')
rep('''        need = ["--tensor-parallel-size", "--ep-size", "--quantization", "--ple-offload-embedding", "--language-model-only",
                "--mamba-backend", "--linear-attn-decode-backend", "--linear-attn-prefill-backend", "--disable-cuda-graph",
                "--max-mamba-cache-size", "--moe-runner-backend"]''',
    '''        need = ["--tensor-parallel-size", "--ep-size", "--quantization", "--ple-offload-embedding",
                "--mamba-backend", "--linear-attn-decode-backend", "--linear-attn-prefill-backend", "--disable-cuda-graph",
                "--max-mamba-cache-size", "--moe-runner-backend", "--mamba-radix-cache-strategy",
                "--mamba-track-interval", "--disable-radix-cache", "--attention-backend"]''')
rep("# FN-SMOKE v3 — Qwen3.8-Flash-Next (Intel W4A16 AutoRound + FP8 PLE) on Kaggle 4×L4 for ARC-AGI-2",
    "# FN-SMOKE v3.1 — Qwen3.8-Flash-Next (Intel W4A16 AutoRound + FP8 PLE) on Kaggle 4×L4 for ARC-AGI-2")
rep('''4. launch, re-checking MemAvailable right before it: TP4 + EP4, Triton GDN, BF16 KV, 8K context, 1 request,
   language-model-only, no CUDA graphs, no DP-attention, no speculative decoding; MoE runner `auto`, then `triton`''',
    '''4. launch, re-checking MemAvailable right before it: TP4 + EP4, Triton GDN, BF16 KV, 8K context, 1 request,
   no CUDA graphs, no DP-attention, no speculative decoding. Attempt 1: mamba radix `extra_buffer` (track interval
   64, as dfranzen) + MoE runner `auto`; attempt 2: radix cache disabled + triton attention + triton MoE''')
rep('''Commit / interactive run: stages 0–5.''',
    '''v3.1 (after the first 4×L4 run): stages 0–3 passed (full SHA256 of all 25 shards OK, 193 s); both launches stopped in
argument resolution because `--language-model-only` is not supported for Qwen4Exp. Removed; radix settings now
follow dfranzen's launcher.

Commit / interactive run: stages 0–5.''')
rep('"experiment": "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4"', '"experiment": "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4-v3.1"')
(HERE / "make_fn_smoke_v3_1.py").write_text(s, encoding="utf-8")
print("wrote", HERE / "make_fn_smoke_v3_1.py")
