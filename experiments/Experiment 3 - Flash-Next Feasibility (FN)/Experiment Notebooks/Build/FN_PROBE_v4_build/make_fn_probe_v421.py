"""Build FN-PROBE v4.2 (v4 + slow-input-disk fixes + review fixes) from the FN-SMOKE v3.3 builder: fix the L4 decode crash (FA4-only kernel in SGLang's Qwen4-Exp sparse
attention) with a PyTorch FA2-varlen stand-in, then measure Flash-Next aggregate decode throughput on 4xL4."""
import hashlib, json, re, runpy, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V33 = HERE.parent / "FN_FP8PLE_build_R1" / "build" / "make_fn_smoke_v3_3.py"
DST = Path(r"D:\ARC AGI 2\Research Base Experiment\FN-PROBE-FLASHNEXT-v4_2_1.ipynb")
src = V33.read_text(encoding="utf-8")
SHIM_SRC = (HERE / "q27_fa2_varlen_shim.py").read_text(encoding="utf-8")

def rep(a, b, count=1):
    global src
    n = src.count(a); assert n == count, (n, a[:100]); src = src.replace(a, b)

# ---------------------------------------------------------------- destination + header
rep('DST = Path(__file__).resolve().parent.parent / "notebooks" / "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4.ipynb"',
    'DST = Path(r"' + str(DST) + '")')
rep("C_HEAD = r'''# FN-SMOKE v3.3 — Qwen3.8-Flash-Next (Intel W4A16 AutoRound + FP8 PLE) on Kaggle 4×L4 for ARC-AGI-2\n",
    "C_HEAD = r'''# FN-PROBE v4.2.1 — Qwen3.8-Flash-Next on Kaggle 4×L4: fix the decode crash, measure throughput\n\n"
    "**v4.2.1 (second external review, each point checked against the code):** the warm-up is untimed and a 256-token\n"
    "calibration run is the baseline of the decode estimate (the warm-up includes first-use JIT); SGLang's own 'gen\n"
    "throughput' log lines are recorded too; stage 6 passes only if the full-concurrency row completed; the preflight\n"
    "requires the prefetch flags and asserts sglang 0.5.19+gd00d88efc8d6; graph capture size = the launched server's\n"
    "max running requests; no valid measurement prints 'NO VALID COMPLETE MEASUREMENT' instead of BELOW; GPU compute\n"
    "processes are recorded before every launch (report only).\n\n"
    "**v4.2 (external review of v4.1, each point checked against the code):** kill_server() now kills the whole\n"
    "process group even when the launcher already exited; no launch starts while port 8001 is still answering; launches\n"
    "keep 15 min for stages 5-6 (two 50-min attempts left no time to measure); throughput levels follow the launched\n"
    "server's --max-running-requests (fallback = 4) and run largest first; one fixed ARC prompt at every level, a warm-up\n"
    "before each level, 2,048 forced decode tokens, and only complete rows count; stage-7 launch limit fixed once\n"
    "(re-clamping every loop stopped at ~half the time); the FA2 shim requires max_seqlen_q == 1 for its decode branch\n"
    "and returns zeros when there are no keys at all; the 27B comparison is labelled a reference, not a like-for-like test;\n"
    "each level also reports a decode-only rate (main run minus the warm-up, same prompts, so prefill cancels).\n"
    "Loading: `--weight-loader-prefetch-checkpoints` (dfranzen's Kaggle setting, in this exact SGLang build) replaces the\n"
    "multi-threaded safetensors loader that had 4 ranks x many threads reading the slow input disk in run 1.\n\n"
    "**v4.1 (run 1 of v4, 2026-10-10):** every stage up to `3_fa2_shim` passed (shim OK on the L4, resolver patched), but\n"
    "both launch attempts timed out at 18 min while still loading weights: the input disk was 2-3x slower than on 2026-10-05\n"
    "(hash 0.29 vs 0.62 GB/s; attempt 1 loaded 5/25 shards in 16 min, attempt 2 18/25; v3.3 loaded all 25 in 6.4 min).\n"
    "Changes: no full SHA256 of the 124.8 GB (header hashes and sizes are still checked; the full read did not keep the\n"
    "page cache warm), a launch attempt may take up to 50 min while serve.log keeps growing (stall limit 15 min), probe\n"
    "budget 80 -> 110 min. Nothing else changed.\n\n"
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
    'SMOKE_BUDGET_SEC = float(os.environ.get("FN_SMOKE_BUDGET", 110 * 60))   # v4.1: 80 -> 110 (slow input disk)')
rep('"experiment": "FN-SMOKE-FLASHNEXT-FP8PLE-4XL4-v3.3"', '"experiment": "FN-PROBE-FLASHNEXT-v4.2.1"')
# ---------------------------------------------------------------- v4.1: slow input disk
rep('SERVER_ATTEMPT_TIMEOUT = 18 * 60    # per launch attempt (also clamped to the deadline)',
    'SERVER_ATTEMPT_TIMEOUT = 50 * 60    # v4.1: 18 -> 50 min per attempt (also clamped to the deadline)\n'
    'STALL_SEC = 15 * 60                 # v4.1: give up earlier only when serve.log has not grown for this long\n'
    'BENCHMARK_RESERVE_SEC = 15 * 60     # v4.2: launches never use the time stages 5-6 need')
rep('FULL_HASH = os.environ.get("FN_SMOKE_FULL_HASH", "1") == "1"',
    'FULL_HASH = os.environ.get("FN_SMOKE_FULL_HASH", "0") == "1"   # v4.1: off (headers + sizes still checked)')

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
# ---------------------------------------------------------------- v4.1: progress-aware wait in stage 4
rep('''        t = time.time(); ready = False
        attempt_limit = clamp(SERVER_ATTEMPT_TIMEOUT)
        while time.time() - t < attempt_limit:
            if SERVER.poll() is not None: break
''', '''        t = time.time(); ready = False; stalled = False
        attempt_limit = clamp(SERVER_ATTEMPT_TIMEOUT)
        _last_size, _last_grow = LOG.stat().st_size, time.time()   # v4.1: progress-aware wait
        while time.time() - t < attempt_limit:
            if SERVER.poll() is not None: break
            _sz = LOG.stat().st_size
            if _sz != _last_size:
                _last_size, _last_grow = _sz, time.time()
            elif time.time() - _last_grow > STALL_SEC:
                stalled = True; break
''')
rep('''        results.append({"attempt": name, "ready": ready, "sec": round(time.time() - t, 1),
                        "exit_code": SERVER.poll(), "log_tail": tail})''',
    '''        results.append({"attempt": name, "ready": ready, "sec": round(time.time() - t, 1), "stalled": stalled,
                        "exit_code": SERVER.poll(), "log_tail": tail})''')
# ---------------------------------------------------------------- v4.2: review fixes (stage 4 + cleanup + verdict)
rep('''def kill_server():
    global SERVER
    if SERVER is not None and SERVER.poll() is None:
        try: os.killpg(SERVER.pid, 15); SERVER.wait(timeout=60)
        except Exception:
            try: os.killpg(SERVER.pid, 9)
            except Exception: pass
    SERVER = None   # only the tracked process group is killed; nothing else on the machine
''', '''import signal

def kill_server():
    """v4.2: kill the launcher's whole process group even if the launcher already exited (its ranks can outlive it);
    only the tracked group is touched, nothing else on the machine. Bounded: <= ~25 s."""
    global SERVER
    if SERVER is None:
        return
    pgid = SERVER.pid                     # start_new_session=True -> the launcher leads its own process group
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except Exception as e:
        print("TERM process-group warning:", repr(e), flush=True)
    t_end = time.time() + 20
    while time.time() < t_end:
        SERVER.poll()                     # reap the leader, or its zombie keeps the group "alive"
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            break
        except Exception:
            break
        time.sleep(0.25)
    else:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass
    try:
        SERVER.wait(timeout=3)
    except Exception:
        pass
    SERVER = None

def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False

def port_free_after_kill(wait_sec=30):
    """v4.2: a server still answering on PORT would make /health == 200 look like the new launch"""
    t_end = time.time() + wait_sec
    while port_open(PORT) and time.time() < t_end:
        time.sleep(0.5)
    return not port_open(PORT)

def launch_budget():
    """v4.2: one attempt may use up to SERVER_ATTEMPT_TIMEOUT, but never the time stages 5-6 need"""
    return max(0.0, min(SERVER_ATTEMPT_TIMEOUT, remaining() - SAFETY_SEC - BENCHMARK_RESERVE_SEC))
''')
rep('''        if remaining() < 10 * 60 + SAFETY_SEC:   # a load needs >= ~10 min; never start one that cannot finish
            results.append({"attempt": name, "skipped": "smoke budget", "remaining_sec": round(remaining())}); break
''', '''        attempt_limit = launch_budget()          # v4.2: keeps BENCHMARK_RESERVE_SEC for stages 5-6
        if attempt_limit < 10 * 60:              # a load needs >= ~10 min; never start one that cannot finish
            results.append({"attempt": name, "skipped": "launch + benchmark budget", "remaining_sec": round(remaining())}); break
        if not port_free_after_kill():           # v4.2: never mistake an old server on PORT for this launch
            results.append({"attempt": name, "skipped": f"port {PORT} still in use"}); break
''')
rep('''        t = time.time(); ready = False; stalled = False
        attempt_limit = clamp(SERVER_ATTEMPT_TIMEOUT)
''', '''        t = time.time(); ready = False; stalled = False
''')
rep('''          "FASTER than the 27B" if _best > BASELINE_TOK_S else "NOT faster than the 27B")''',
    '''          ("ABOVE" if _best > BASELINE_TOK_S else "BELOW") + " the 27B reference (reference only: the 27B number is 16"
          " running requests over long runs, not the same benchmark)")''')
# v4.2 (harness): /health == 200 counts only while OUR launcher is alive (a dead launcher + another listener raced)
rep('''                    if r.status == 200: ready = True; break''',
    '''                    if r.status == 200 and SERVER.poll() is None: ready = True; break''')
rep('''                if r.status == 200: ready = True; break''',
    '''                if r.status == 200 and SERVER.poll() is None: ready = True; break''')
rep('''            "--ple-offload-embedding", "--trust-remote-code",
''', '''            "--ple-offload-embedding", "--trust-remote-code",
            # v4.2: sequential checkpoint prefetch (same flag as dfranzen's Kaggle notebook, PREFETCH_CHECKPOINTS=True,
            # present in this exact build 0.5.19+gd00d88efc8d6). It turns off SGLang's multi-threaded safetensors
            # loader, which in run 1 had 4 ranks reading the same 124.8 GB with many threads each from /kaggle/input.
            "--weight-loader-prefetch-checkpoints",
''')
# ---------------------------------------------------------------- v4.2.1: second review
rep("""                "--mamba-track-interval", "--disable-radix-cache", "--attention-backend"]""",
    """                "--mamba-track-interval", "--disable-radix-cache", "--attention-backend",
                "--weight-loader-prefetch-checkpoints", "--weight-loader-prefetch-num-threads"]   # v4.2.1""")
rep("""                 "print('sglang', sglang.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__);\"""",
    """                 "print('sglang', sglang.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__);"
                 "assert sglang.__version__ == '0.5.19+gd00d88efc8d6', sglang.__version__;"   # v4.2.1: exact build""")
rep("""def launch_budget():""", """def gpu_compute_apps():
    # v4.2.1: GPU compute processes right before a launch (recorded only; unknown processes are never killed)
    out = sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits", timeout=30)
    return [x.strip() for x in out.splitlines() if x.strip()][:20]

def launch_budget():""")
rep("""        args = base_args() + extra
        off = LOG.stat().st_size if LOG.exists() else 0""", """        args = base_args() + extra
        _gpu_before = gpu_compute_apps()           # v4.2.1: leftover CUDA workers would show here (report only)
        off = LOG.stat().st_size if LOG.exists() else 0""")
rep(""""stalled": stalled,
                        "exit_code": SERVER.poll(), "log_tail": tail})""", """"stalled": stalled,
                        "exit_code": SERVER.poll(), "gpu_apps_before": _gpu_before, "log_tail": tail})""")
rep("""    _best = max([x for x in (_e, _g) if x] or [0])
    print(f"THROUGHPUT: eager best {_e} tok/s | CUDA graphs best {_g} tok/s | 27B baseline {BASELINE_TOK_S} tok/s ->",
          ("ABOVE" if _best > BASELINE_TOK_S else "BELOW") + " the 27B reference (reference only: the 27B number is 16"
          " running requests over long runs, not the same benchmark)")""",
"""    _valid = [x for x in (_e, _g) if x and x > 0]      # v4.2.1: 0 / None means "not measured", not "slower"
    if not _valid:
        print("THROUGHPUT: NO VALID COMPLETE MEASUREMENT (see stage 6 rows)")
    else:
        _best = max(_valid)
        print(f"THROUGHPUT: eager best {_e} tok/s | CUDA graphs best {_g} tok/s | 27B baseline {BASELINE_TOK_S} tok/s ->",
              ("ABOVE" if _best > BASELINE_TOK_S else "BELOW") + " the 27B reference (reference only: the 27B number is 16"
              " running requests over long runs, not the same benchmark)")""")
# v4.2.1 final: three separate numbers at full concurrency; the server decode rate is the one compared with the 27B
rep("""    _valid = [x for x in (_e, _g) if x and x > 0]      # v4.2.1: 0 / None means "not measured", not "slower"
    if not _valid:
        print("THROUGHPUT: NO VALID COMPLETE MEASUREMENT (see stage 6 rows)")
    else:
        _best = max(_valid)
        print(f"THROUGHPUT: eager best {_e} tok/s | CUDA graphs best {_g} tok/s | 27B baseline {BASELINE_TOK_S} tok/s ->",
              ("ABOVE" if _best > BASELINE_TOK_S else "BELOW") + " the 27B reference (reference only: the 27B number is 16"
              " running requests over long runs, not the same benchmark)")""",
"""    _rows8 = REPORT["stages"].get("6_throughput_eager", {}).get("rows", [])
    _lv = max([r.get("concurrency", 0) for r in _rows8 if "concurrency" in r] or [0])
    _fr = next((r for r in _rows8 if r.get("concurrency") == _lv and r.get("complete")), None)
    if not _fr:   # v4.2.1: 0 / None means "not measured", not "slower"
        print("THROUGHPUT: NO VALID COMPLETE MEASUREMENT (see stage 6 rows)")
    else:
        _srv, _inc, _wall = _fr.get("server_gen_tok_s_full"), _fr.get("incremental_decode_tok_s_est"), _fr.get("aggregate_tok_s")
        print(f"THROUGHPUT at full concurrency ({_lv} requests): server decode {_srv} tok/s | client incremental decode "
              f"{_inc} tok/s | wall aggregate incl. prefill {_wall} tok/s | CUDA graphs best wall {_g}")
        _primary = _srv or _inc
        if _primary:
            print(f"DECODE vs 27B reference ({BASELINE_TOK_S:.0f} tok/s = 27B decode with 16 running; 27B whole run 248.9):",
                  "ABOVE" if _primary > BASELINE_TOK_S else "BELOW",
                  f"- a reference, not a verdict: {_lv} vs 16 running, 16K context vs the 27B's 49K-token queries")
        else:
            print("DECODE vs 27B reference: no decode-rate measurement (only the wall aggregate, which includes prefill)")""")
rep('(slow input disk)   # whole-probe wall budget', '(slow input disk); whole-probe wall budget')
rep('(headers + sizes still checked)   # full SHA256 of all 25 shards (also warms the page cache)',
    '(headers + sizes still checked); "1" = full SHA256 of all 25 shards')
(HERE / "make_fn_probe_v421_generated.py").write_text(src, encoding="utf-8")
runpy.run_path(str(HERE / "make_fn_probe_v421_generated.py"), run_name="__main__")
