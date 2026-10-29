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
    _LARG = {"--speculative-config": json.dumps({"method": "mtp", "num_speculative_tokens": 3,
                                                 "use_local_argmax_reduction": True})}      # rev 2: dossier V32-ARGMAX
    _PMU = {"--prefix-match-unit": 64}                                                       # rev 2: dossier V32-HYBRID-REUSE
    _PMUS = {"--prefix-match-unit": 64, "--enable-mamba-shared-prefix-checkpoint": None}
    _N2 = {"NCCL_MIN_CTAS": "2", "NCCL_MAX_CTAS": "2"}
    _N4 = {"NCCL_MIN_CTAS": "4", "NCCL_MAX_CTAS": "4", "NCCL_NTHREADS": "512"}
    _N8 = {"NCCL_MIN_CTAS": "8", "NCCL_MAX_CTAS": "8", "NCCL_NTHREADS": "512"}
    _RING = {"NCCL_ALGO": "Ring"}

    def _spec_swap(flags, new_json):
        """the arm's --speculative-config replaced (vms_flags adds the default MTP 3 JSON; a second flag would be ambiguous)"""
        f = list(flags)
        i = f.index("--speculative-config")
        f[i + 1] = new_json
        return f

    # FP8 arms, gated against the FP8 target-only oracle
    PLAN = [
        # E22 (speed research 2026-10-29 + Resources dossiers v5): one lever per arm
        ("fp8_s16_triton_full", vms_flags(MODEL_PATH, backend="TRITON_ATTN", extra=_FULL), {}),
        ("fp8_s16_ssm_bf16", vms_flags(MODEL_PATH, extra={"--mamba-ssm-cache-dtype": "bfloat16"}), {}),
        ("fp8_s16_local_argmax", _spec_swap(vms_flags(MODEL_PATH), _LARG["--speculative-config"]), {}),
    ]
    # INT4 arms, gated against the INT4 target-only oracle (rev 2: dossier P0-05 V32-INT4-ORACLE; run after the FP8 block)
    PLAN_INT4 = [
        ("int4_s20", vms_flags(VMS_INT4, seqs=20), dict(level=20)),
        ("int4_s20_local_argmax", _spec_swap(vms_flags(VMS_INT4, seqs=20), _LARG["--speculative-config"]), dict(level=20)),
        ("int4_s20_triton_full", vms_flags(VMS_INT4, seqs=20, backend="TRITON_ATTN", extra=_FULL), dict(level=20)),
        ("int4_s20_triton_fdec", vms_flags(VMS_INT4, seqs=20, backend="TRITON_ATTN", extra=_FDEC), dict(level=20)),
        ("int4_s20_nccl2", vms_flags(VMS_INT4, seqs=20), dict(level=20, env_extra=_N2)),
        ("int4_s20_nccl4", vms_flags(VMS_INT4, seqs=20), dict(level=20, env_extra=_N4)),
        ("int4_s20_nccl8", vms_flags(VMS_INT4, seqs=20), dict(level=20, env_extra=_N8)),
        ("int4_s20_ring", vms_flags(VMS_INT4, seqs=20), dict(level=20, env_extra=_RING)),
        ("int4_s20_pmu64", vms_flags(VMS_INT4, seqs=20, extra=_PMU), dict(level=20)),
        ("int4_s20_pmu64_sharedckpt", vms_flags(VMS_INT4, seqs=20, extra=_PMUS), dict(level=20)),
        ("int4_s24", vms_flags(VMS_INT4, seqs=24), dict(level=24)),
        ("int4_s20_b8192", vms_flags(VMS_INT4, seqs=20, batched=8192), dict(level=20)),
        ("int4_s24_triton_full_nccl8", vms_flags(VMS_INT4, seqs=24, backend="TRITON_ATTN", extra=_FULL),
         dict(level=24, env_extra=_N8)),
    ]
'''
s = s[:i0] + NEW_PLAN + s[i1:]
# rev 2: after the FP8 plan, the INT4 oracle (same INT4 checkpoint, no speculation, prefix caching off) becomes the gate
# reference of every INT4 arm (gate="ref" fills VMS_REF from its own A/A rows; cleared first so no FP8 row is mixed in)
s = rep(s, '''        vms_arm(_name, _fl, **_kw)
    # LAST: 2 x TP2 inside one vLLM server''', '''        vms_arm(_name, _fl, **_kw)
    if _INT4_OK:
        VMS_REF.clear()
        vms_arm("int4_target_only_noprefix", vms_flags(VMS_INT4, seqs=20, spec=0, prefix=False), gate="ref", level=20)
        VMS["int4_gate_reference"] = "int4_target_only_noprefix (RedHat INT4, no speculation, prefix caching off)"
    for _name, _fl, _kw in PLAN_INT4:
        if not _INT4_OK:
            VMS["arms"][_name] = {"ok": False, "skipped": "INT4 MTP static check failed: " + VMS["int4_mtp_static"].get("why", "")}
            VMS["order"].append(_name)
            continue
        vms_arm(_name, _fl, **_kw)
    # LAST: 2 x TP2 inside one vLLM server''')
# drop the 2 x TP2 arms (KV does not fit; measured 0.57x)
j0 = s.index("    # LAST: 2 x TP2 inside one vLLM server")
j1 = s.index('    for _a in VMS["arms"].values():       # rev 2e: the final classification')
s = s[:j0] + "    VMS_RESERVE = 0.0\n" + s[j1:]
# rev 2: 20 launches (reference, 2 oracles, 17 arms) at ~12-14 min each
s = rep(s, '"time_budget_sec": 260 * 60,', '"time_budget_sec": 420 * 60,')
s = rep(s, 'md = """# VMS — vLLM 0.32.0 engine-config matrix (Qwen3.8-27B on 4x L4), engine only',
        'md = """# E22 Speed Lab rev 2 — vLLM 0.32.0 (Qwen3.8-27B on 4x L4), target 1500+ tok/s\n\n'
        'FP8 block (gated vs the FP8 target-only oracle): fp8_s16_triton_full (FULL CUDA graphs with MTP), fp8_s16_ssm_bf16 '
        '(fused CUDA GDN verify kernel), fp8_s16_local_argmax (use_local_argmax_reduction). INT4 block (gated vs the INT4 '
        'target-only oracle int4_target_only_noprefix): int4_s20 (baseline), int4_s20_local_argmax, int4_s20_triton_full, '
        'int4_s20_triton_fdec, int4_s20_nccl2 / nccl4 / nccl8, int4_s20_ring (NCCL_ALGO=Ring), int4_s20_pmu64 '
        '(--prefix-match-unit 64), int4_s20_pmu64_sharedckpt (+ --enable-mamba-shared-prefix-checkpoint), int4_s24, '
        'int4_s20_b8192, int4_s24_triton_full_nccl8 (combination). Budget 420 min.\n\n'
        '## Base: the Exp 14 VMS matrix')
exec(compile(s, str(E14B / "build_vms.py"), "exec"), {"__file__": str(E14B / "build_vms.py"), "__name__": "__main__"})
