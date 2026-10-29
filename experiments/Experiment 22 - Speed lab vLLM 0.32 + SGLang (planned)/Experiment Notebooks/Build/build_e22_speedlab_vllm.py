"""E22 Speed Lab, vLLM 0.32.0 track (user 2026-10-29: target 1000+ tok/s; only vLLM 0.32 and the best SGLang).

Built from the Exp 14 VMS builder (build_vms.py, the Kaggle-run engine matrix: venv install, launcher, /metrics windows
at 8k / 40k / XL, greedy gate vs the FP8 target-only oracle, canary, report, bundle). Only the arm plan changes: every
arm is one lever from the 2026-10-29 speed research (reports/৪xL4 এ ৯০০ টোকেন গতির উপায়.md), measured on the same
prompts:
  FULL CUDA graphs with MTP (TRITON_ATTN + FULL_AND_PIECEWISE / FULL_DECODE_ONLY; FlashInfer falls back to PIECEWISE),
  NCCL channels (NCCL_MIN_CTAS / NCCL_MAX_CTAS 4 and 8, NCCL_NTHREADS 512: all-reduce = ~51 % of a TP4 step here),
  24 streams, max-num-batched-tokens 8192, and the combination; INT4 (W4A16 Marlin; FP8 is W8A16 Marlin on sm_89).
FP8 FULL-graph arm is gated against the oracle (a graph-mode change must not change greedy output).
usage: python build_e22_speedlab_vllm.py -> ../Q27-XHIGH-E22_SPEEDLAB_VLLM032.ipynb"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E14B = Path(r"D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Notebooks/Build")
SRC = (E14B / "build_vms.py").read_text(encoding="utf-8")


def rep(text, a, b, n=1):
    assert text.count(a) == n, (text.count(a), a[:100])
    return text.replace(a, b)


s = SRC
s = rep(s, 'OUT = NBDIR / "Q27-XHIGH-VMS_ENGINE_MATRIX_VLLM.ipynb"',
        'OUT = Path(r"' + str(HERE.parent) + r'") / "Q27-XHIGH-E22_SPEEDLAB_VLLM032.ipynb"')
# no 2 x TP2 slot: nothing is reserved at the end
s = rep(s, '    VMS_RESERVE = _ts(35 * 60)            # the 2 x TP2 slot: no earlier arm may eat it',
        '    VMS_RESERVE = 0.0                     # E22: no 2 x TP2 slot')
i0 = s.index("    PLAN = [\n        # rev 2d: FlashInfer PCIe-IPC")
i1 = s.index("    VMS[\"dropped\"] = {")
NEW_PLAN = r'''    _FULL = {"--compilation-config": json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE"})}
    _FDEC = {"--compilation-config": json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY"})}
    _N4 = {"NCCL_MIN_CTAS": "4", "NCCL_MAX_CTAS": "4", "NCCL_NTHREADS": "512"}
    _N8 = {"NCCL_MIN_CTAS": "8", "NCCL_MAX_CTAS": "8", "NCCL_NTHREADS": "512"}
    PLAN = [
        # E22 (speed research 2026-10-29): one lever per arm, the FP8 graph-mode arm gated against the oracle first
        ("fp8_s16_triton_full", vms_flags(MODEL_PATH, backend="TRITON_ATTN", extra=_FULL), {}),
        ("int4_s20", vms_flags(VMS_INT4, seqs=20), dict(level=20, gate=False)),
        ("int4_s20_triton_full", vms_flags(VMS_INT4, seqs=20, backend="TRITON_ATTN", extra=_FULL), dict(level=20, gate=False)),
        ("int4_s20_triton_fdec", vms_flags(VMS_INT4, seqs=20, backend="TRITON_ATTN", extra=_FDEC), dict(level=20, gate=False)),
        ("int4_s20_nccl4", vms_flags(VMS_INT4, seqs=20), dict(level=20, gate=False, env_extra=_N4)),
        ("int4_s20_nccl8", vms_flags(VMS_INT4, seqs=20), dict(level=20, gate=False, env_extra=_N8)),
        ("int4_s24", vms_flags(VMS_INT4, seqs=24), dict(level=24, gate=False)),
        ("int4_s20_b8192", vms_flags(VMS_INT4, seqs=20, batched=8192), dict(level=20, gate=False)),
        ("int4_s24_triton_full_nccl8", vms_flags(VMS_INT4, seqs=24, backend="TRITON_ATTN", extra=_FULL),
         dict(level=24, gate=False, env_extra=_N8)),
    ]
'''
s = s[:i0] + NEW_PLAN + s[i1:]
# drop the 2 x TP2 arms (KV does not fit; measured 0.57x)
j0 = s.index("    # LAST: 2 x TP2 inside one vLLM server")
j1 = s.index('    for _a in VMS["arms"].values():       # rev 2e: the final classification')
s = s[:j0] + "    VMS_RESERVE = 0.0\n" + s[j1:]
s = rep(s, 'md = """# VMS — vLLM 0.32.0 engine-config matrix (Qwen3.8-27B on 4x L4), engine only',
        'md = """# E22 Speed Lab — vLLM 0.32.0 (Qwen3.8-27B on 4x L4), target 1000+ tok/s\n\n'
        'Arms (after the FP8 reference and the FP8 target-only oracle): fp8_s16_triton_full (FULL CUDA graphs with MTP, '
        'gated), int4_s20 (baseline), int4_s20_triton_full, int4_s20_triton_fdec, int4_s20_nccl4, int4_s20_nccl8, int4_s24, '
        'int4_s20_b8192, int4_s24_triton_full_nccl8 (combination).\n\n'
        '## Base: the Exp 14 VMS matrix')
exec(compile(s, str(E14B / "build_vms.py"), "exec"), {"__file__": str(E14B / "build_vms.py"), "__name__": "__main__"})
