"""E22 Speed Lab, SGLang track (user 2026-10-29: target 1000+ tok/s; only vLLM 0.32 and the best SGLang).

Built from the Exp 14 EMS builder (build_ems.py: the Kaggle-run SGLang engine matrix with the target-only oracle gate,
8k / 40k / XL windows, canary, report, bundle) as a new part SPEED. Base = the best measured SGLang config
(RedHat INT4 + DFlash2 width 6, 20 streams: 597 / 536 / 364 tok/s). Arms, one lever each (speed research 2026-10-29):
  DFlash width 4 at 20 / 24 streams (same tok/s as W6 at 16 with ~25 % shorter steps),
  NCCL channels (NCCL_MIN_CTAS / NCCL_MAX_CTAS 4 and 8, NCCL_NTHREADS 512: all-reduce + all-gather = 51 % of the step),
  chunked prefill 8192, and the combination W4 s24 + NCCL 8.
usage: python build_e22_speedlab_sglang.py -> ../Q27-XHIGH-E22_SPEEDLAB_SGLANG.ipynb"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
E14B = Path(r"D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Notebooks/Build")
SRC = (E14B / "build_ems.py").read_text(encoding="utf-8")


def rep(text, a, b, n=1):
    assert text.count(a) == n, (text.count(a), a[:100])
    return text.replace(a, b)


SPEED_ARMS = ("dflash_w6_s20", "dflash_w4_s20", "dflash_w6_s20_nccl4", "dflash_w6_s20_nccl8", "dflash_w4_s24",
              "dflash_w6_s20_chunk8192", "dflash_w4_s24_nccl8")
s = SRC
s = rep(s, '''                 "MTPX": ("mtp_4_1_5", "mtp_5_1_6", "dflash_w6_t06", "mtp_3_1_4_t06", "mtp_5_1_6_t06")}''',
        '''                 "MTPX": ("mtp_4_1_5", "mtp_5_1_6", "dflash_w6_t06", "mtp_3_1_4_t06", "mtp_5_1_6_t06"),
                 # E22 Speed Lab (2026-10-29): DFlash width / streams / NCCL channels / chunked prefill
                 "SPEED": ''' + repr(SPEED_ARMS) + '''}''')
s = rep(s, '''EMS_MTPX_ONLY = set(EMS_PART_ARMS["MTPX"])''',
        '''EMS_MTPX_ONLY = set(EMS_PART_ARMS["MTPX"]) | (set(EMS_PART_ARMS["SPEED"]) - {"dflash_w6_s20"})   # E22: ALL skips them''')
s = rep(s, '''assert EMS_PART in ("ALL", "SAFE", "RISKY", "MTPX"), EMS_PART''',
        '''assert EMS_PART in ("ALL", "SAFE", "RISKY", "MTPX", "SPEED"), EMS_PART''')
s = rep(s, '''            ("mtp_5_1_6_t06", dict(set_=_mtp(steps=5, draft=6), drop=_SPEC + ("--min-free-slots-delay",), win_temp=0.6)),
        ]''',
        '''            ("mtp_5_1_6_t06", dict(set_=_mtp(steps=5, draft=6), drop=_SPEC + ("--min-free-slots-delay",), win_temp=0.6)),
            # E22 Speed Lab (2026-10-29)
            ("dflash_w4_s20", dict(set_={"--speculative-num-draft-tokens": 4}, streams=20)),
            ("dflash_w6_s20_nccl4", dict(streams=20, env={"NCCL_MIN_CTAS": "4", "NCCL_MAX_CTAS": "4", "NCCL_NTHREADS": "512"})),
            ("dflash_w6_s20_nccl8", dict(streams=20, env={"NCCL_MIN_CTAS": "8", "NCCL_MAX_CTAS": "8", "NCCL_NTHREADS": "512"})),
            ("dflash_w4_s24", dict(set_={"--speculative-num-draft-tokens": 4}, streams=24)),
            ("dflash_w6_s20_chunk8192", dict(set_={"--chunked-prefill-size": 8192}, streams=20)),
            ("dflash_w4_s24_nccl8", dict(set_={"--speculative-num-draft-tokens": 4}, streams=24,
                                         env={"NCCL_MIN_CTAS": "8", "NCCL_MAX_CTAS": "8", "NCCL_NTHREADS": "512"})),
        ]''')
s = rep(s, '''            if EMS_PART != "ALL" and _name not in EMS_PART_ARMS.get(EMS_PART, ()):
                _why = "OTHER_PART: runs in the EMS " + ("SAFE" if EMS_PART == "RISKY" else "RISKY") + " notebook"''',
        '''            if EMS_PART != "ALL" and _name not in EMS_PART_ARMS.get(EMS_PART, ()):
                _why = "OTHER_PART: not in the EMS " + EMS_PART + " plan"''')
s = rep(s, '''    "MTPX": "oracle + reference + MTP 4/1/5 and 5/1/6,''',
        '''    "SPEED": "E22 Speed Lab (target 1000+ tok/s): oracle + reference + DFlash W6 at 20 streams, W4 at 20 / 24, NCCL "
             "channels 4 / 8 (NCCL_MIN_CTAS / NCCL_MAX_CTAS, NCCL_NTHREADS 512), chunked prefill 8192, W4 s24 + NCCL 8; no patch, no TP2.",
    "MTPX": "oracle + reference + MTP 4/1/5 and 5/1/6,''')
s = rep(s, '''for PART in ("SAFE", "RISKY", "MTPX"):''', '''for PART in ("SPEED",):''')
s = rep(s, '''    out = NBDIR / f"Q27-XHIGH-EMS_{PART}_ENGINE_MATRIX_SGLANG.ipynb"''',
        '''    out = Path(r"''' + str(HERE.parent) + '''") / "Q27-XHIGH-E22_SPEEDLAB_SGLANG.ipynb"''')
s = rep(s, '''    md_p = md.replace("# EMS — SGLang engine-config matrix", f"# EMS {PART} — SGLang engine-config matrix", 1) + (''',
        '''    md_p = md.replace("# EMS — SGLang engine-config matrix", "# E22 Speed Lab — SGLang (EMS SPEED part), target 1000+ tok/s", 1) + (''')
exec(compile(s, str(E14B / "build_ems.py"), "exec"), {"__file__": str(E14B / "build_ems.py"), "__name__": "__main__"})
