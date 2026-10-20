"""Build FN-PROBE v4 from the FN-SMOKE v3.3 builder: fix the L4 decode crash (FA4-only kernel in SGLang's Qwen4-Exp sparse
attention) with a PyTorch FA2-varlen stand-in, then measure Flash-Next aggregate decode throughput on 4xL4."""
import hashlib, json, re, runpy, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V33 = HERE.parent / "FN_FP8PLE_build_R1" / "build" / "make_fn_smoke_v3_3.py"
DST = Path(r"D:\ARC AGI 2\Research Base Experiment\FN-PROBE-FLASHNEXT-v4.ipynb")
src = V33.read_text(encoding="utf-8")
SHIM_SRC = (HERE / "q27_fa2_varlen_shim.py").read_text(encoding="utf-8")

def rep(a, b, count=1):
    global src
    n = src.count(a); assert n == count, (n, a[:100]); src = src.replace(a, b)

# ---------------------------------------------------------------- destination + header
rep('DST = Path(__file__).resolve().parent.parent / "notebooks" / "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4.ipynb"',
    'DST = Path(r"' + str(DST) + '")')
rep("C_HEAD = r'''# FN-SMOKE v3.3 — Qwen3.8-Flash-Next (Intel W4A16 AutoRound + FP8 PLE) on Kaggle 4×L4 for ARC-AGI-2\n",
    "C_HEAD = r'''# FN-PROBE v4 — Qwen3.8-Flash-Next on Kaggle 4×L4: fix the decode crash, measure throughput\n\n"
    "**Why:** our Qwen3.8-27B pipeline (Q27-XHIGH v1.4) is limited by tokens: 18/22 queries are cut at 49k thinking tokens\n"
    "and score 0, while the 4 that finish are right or 1 cell off. 4×L4 gives the 27B ~373 tok/s with 16 running. A faster\n"
    "model buys more tokens per query. This probe answers one question: how many tokens/s does Flash-Next give on 4×L4?\n\n"
    "**v4 changes (from the v3.3 log, checked in the Pennyroyal SGLang source):**\n"
    "* v3.3 loaded the model and started the server, then crashed on the first decode in `flash_attn/cute` (FA4, Hopper /\n"
    "  Blackwell only): Qwen4-Exp's full-attention layers use `qwen_sparse_attn_backend.py`, whose decode fallback\n"
    "  (`_forward_paged_attention`) calls `_resolve_flash_attn_varlen_func()` = classic FA2 if importable, else FA4 CuTe.\n"
    "  `--attention-backend triton` does not reach it. Stage `3_fa2_shim` installs a PyTorch stand-in for FA2's\n"
    "  `flash_attn_varlen_func` (`q27_fa2_varlen_shim.py`, tested against a naive reference on CPU and on the GPU in the\n"
    "  run) and makes the resolver return it on GPUs below sm90.\n"
    "* KV room: v3.3 had 15,424 KV tokens at mem-fraction 0.90 (weights 19.06 GB of 22 GB). v4: 0.96, 16K context,\n"
    "  up to 8 running requests (fallback attempt: 0.93, 4 requests).\n"
    "* Stage 5: greedy sanity (the model must count 1..20). Stage 6: aggregate decode throughput at concurrency 1/4/8\n"
    "  (ARC evaluation prompts, thinking on, fixed decode length). Optional stage 7: relaunch WITH CUDA graphs and measure\n"
    "  again. The verdict compares with the 27B's 373 tok/s.\n\n"
    "---\n(v3.3 header kept for reference)\n\n"
    "# FN-SMOKE v3.3 — Qwen3.8-Flash-Next (Intel W4A16 AutoRound + FP8 PLE) on Kaggle 4×L4 for ARC-AGI-2\n")

# ---------------------------------------------------------------- config
rep('LOAD_ONLY = True                    # True: stop after the server loads + one 64-token generation (no ARC test)',
    'LOAD_ONLY = False                   # v4: the throughput probe replaces the ARC test')
rep('SMOKE_BUDGET_SEC = float(os.environ.get("FN_SMOKE_BUDGET", (50 if LOAD_ONLY else 85) * 60))',
    'SMOKE_BUDGET_SEC = float(os.environ.get("FN_SMOKE_BUDGET", 80 * 60))')
rep('"experiment": "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4-v3.3"', '"experiment": "FN-PROBE-FLASHNEXT-v4"')

# ---------------------------------------------------------------- stage 3_fa2_shim (after the pinned-memory probe)
SHIM_STAGE = r'''
    # ---- v4: stage 3_fa2_shim - PyTorch stand-in for FA2 flash_attn_varlen_func in the Qwen4-Exp QSA decode fallback
    if REPORT["stopped_at"] is None:
        try:
            purelib = subprocess.run([PYTHON, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"], text=True,
                                     capture_output=True, env=env, timeout=clamp(120)).stdout.strip()
            (Path(purelib) / "q27_fa2_varlen_shim.py").write_text(SHIM_SRC)
            target = Path(purelib) / "sglang" / "srt" / "layers" / "attention" / "qwen_sparse_attn_backend.py"
            text = target.read_text()
            if SHIM_MARK not in text:
                assert text.count(SHIM_OLD) == 1, "patch anchor not found exactly once"
                target.write_text(text.replace(SHIM_OLD, SHIM_NEW + SHIM_OLD))
                for pyc in target.parent.glob("__pycache__/qwen_sparse_attn_backend*.pyc"):
                    pyc.unlink()
            t1 = subprocess.run([PYTHON, "-c", SHIM_GPU_TEST], text=True, capture_output=True, env=env, timeout=clamp(600))
            t2 = subprocess.run([PYTHON, "-c", "from sglang.srt.layers.attention.qwen_sparse_attn_backend import "
                                 "_resolve_flash_attn_varlen_func as r; print('RESOLVED', r().__module__)"],
                                text=True, capture_output=True, env=env, timeout=clamp(600))
            ok = ("SHIM_GPU_OK" in t1.stdout) and ("RESOLVED q27_fa2_varlen_shim" in t2.stdout)
            stage("3_fa2_shim", ok, target=str(target), gpu_test=(t1.stdout + t1.stderr)[-1500:],
                  resolver=(t2.stdout + t2.stderr)[-1500:])
        except Exception as e:
            stage("3_fa2_shim", False, error=f"{type(e).__name__}: {e}"[:2000])
'''
anchor = '''    except Exception as e:
        stage("3_preflight", False, error=f"{type(e).__name__}: {e}"[:3000])
'''
rep(anchor, anchor + SHIM_STAGE.lstrip("\n"))

SHIM_CONSTS = (
    "\n# ---- v4: FA2 varlen stand-in (see the header)\n"
    "SHIM_SRC = " + repr(SHIM_SRC) + "\n"
    "SHIM_MARK = 'Q27 FN-PROBE v4: Ada'\n"
    "SHIM_OLD = '    try:\\n        from flash_attn import flash_attn_varlen_func\\n\\n        return flash_attn_varlen_func\\n    except ImportError:\\n        pass\\n'\n"
    "SHIM_NEW = ('    try:   # Q27 FN-PROBE v4: Ada (sm89) has no FA2 in this venv and FA4 CuTe does not build for it\\n'\n"
    "            '        import torch as _q27_torch\\n'\n"
    "            '        if _q27_torch.cuda.is_available() and _q27_torch.cuda.get_device_capability()[0] < 9:\\n'\n"
    "            '            from q27_fa2_varlen_shim import flash_attn_varlen_func\\n\\n'\n"
    "            '            return flash_attn_varlen_func\\n'\n"
    "            '    except ImportError:\\n'\n"
    "            '        pass\\n')\n"
    "SHIM_GPU_TEST = " + repr(r'''import torch, math
from q27_fa2_varlen_shim import flash_attn_varlen_func as f
torch.manual_seed(0); dev = "cuda"
lens = [7, 0, 33, 2, 2048]; ck = [0]
for n in lens: ck.append(ck[-1] + n)
b = len(lens); hq, hkv, d = 8, 2, 128
q = torch.randn(b, hq, d, device=dev, dtype=torch.bfloat16)
k = torch.randn(ck[-1], hkv, d, device=dev, dtype=torch.bfloat16); v = torch.randn(ck[-1], hkv, d, device=dev, dtype=torch.bfloat16)
o = f(q, k, v, torch.arange(b + 1, dtype=torch.int32, device=dev), torch.tensor(ck, dtype=torch.int32, device=dev), 1, max(lens),
      softmax_scale=d ** -0.5, causal=True)
err = 0.0
for i, n in enumerate(lens):
    if n == 0:
        assert o[i].abs().max().item() == 0.0; continue
    kk = k[ck[i]:ck[i + 1]].float().repeat_interleave(hq // hkv, 1); vv = v[ck[i]:ck[i + 1]].float().repeat_interleave(hq // hkv, 1)
    s = torch.einsum("hd,khd->hk", q[i].float(), kk) / math.sqrt(d)
    err = max(err, (o[i].float() - torch.einsum("hk,khd->hd", s.softmax(-1), vv)).abs().max().item())
assert err < 3e-2, err
print("SHIM_GPU_OK", round(err, 5), torch.cuda.get_device_name(0))
''') + "\n")
rep("C_CUDA = r'''# ==================================================================== CELL 5 — stage 3: torch + CUDA toolkit in the venv\n",
    "C_CUDA = r'''# ==================================================================== CELL 5 — stage 3: torch + CUDA toolkit in the venv\n"
    + SHIM_CONSTS.replace("'''", "\\'\\'\\'"))

# ---------------------------------------------------------------- launch profile
rep('"--mem-fraction-static", "0.90", "--context-length", "4096", "--page-size", "64",\n'
    '            "--max-running-requests", "1", "--chunked-prefill-size", "1024", "--max-prefill-tokens", "2048",',
    '"--mem-fraction-static", "0.96", "--context-length", "16384", "--page-size", "64",\n'
    '            "--max-running-requests", "8", "--chunked-prefill-size", "2048", "--max-prefill-tokens", "4096",')
a = src.index("ATTEMPTS = ["); b = src.index("]\n", src.index('"--max-mamba-cache-size", "6"]),')) + 2
src = src[:a] + '''ATTEMPTS = [   # v4: radix cache off -> 1 mamba slot per request; 8 requests + margin
    ("noradix_eager_mf096_r8", ["--disable-radix-cache", "--attention-backend", "triton", "--moe-runner-backend", "auto",
                                "--max-mamba-cache-size", "10"]),
    ("noradix_eager_mf093_r4", ["--disable-radix-cache", "--attention-backend", "triton", "--moe-runner-backend", "auto",
                                "--max-mamba-cache-size", "6", "--mem-fraction-static", "0.93", "--max-running-requests", "4"]),
]
''' + src[b:]
rep("            LAUNCHED = name; break", "            LAUNCHED = name; LAUNCHED_EXTRA = extra; break")

# ---------------------------------------------------------------- stage 5-7: sanity + throughput
a = src.index("C_ARC = r'''"); b = src.index("C_REPORT = r'''")
src = src[:a] + (HERE / "c_probe.txt").read_text(encoding="utf-8") + "\n" + src[b:]
rep('_expected = ["0_environment", "1_inputs", "2_install", "3_preflight", "3_pinned_memory", "4_launch",\n'
    '             "5_generate_probe" if LOAD_ONLY else "5_arc_test"]',
    '_expected = ["0_environment", "1_inputs", "2_install", "3_preflight", "3_pinned_memory", "3_fa2_shim", "4_launch",\n'
    '             "5_sanity", "6_throughput_eager"]')
rep('print("VERDICT:", "LOADS AND GENERATES ON 4xL4" if _all_ok else ("RERUN FALLBACK" if RERUN else "NOT SHOWN TO WORK - see stopped_at"))',
    'print("VERDICT:", "LOADS AND GENERATES ON 4xL4" if _all_ok else ("RERUN FALLBACK" if RERUN else "NOT SHOWN TO WORK - see stopped_at"))\n'
    'try:\n'
    '    _e = REPORT["stages"].get("6_throughput_eager", {}).get("best_aggregate_tok_s")\n'
    '    _g = REPORT.get("graphs", {}).get("best_aggregate_tok_s")\n'
    '    _best = max([x for x in (_e, _g) if x] or [0])\n'
    '    print(f"THROUGHPUT: eager best {_e} tok/s | CUDA graphs best {_g} tok/s | 27B baseline {BASELINE_TOK_S} tok/s ->",\n'
    '          "FASTER than the 27B" if _best > BASELINE_TOK_S else "NOT faster than the 27B")\n'
    'except Exception as _x:\n'
    '    print("throughput summary failed:", _x)')
src = src.replace('ZIP = WORK / "FN_SMOKE_BUNDLE.zip"', 'ZIP = WORK / "FN_PROBE_BUNDLE.zip"')
(HERE / "make_fn_probe_v4_generated.py").write_text(src, encoding="utf-8")
runpy.run_path(str(HERE / "make_fn_probe_v4_generated.py"), run_name="__main__")
