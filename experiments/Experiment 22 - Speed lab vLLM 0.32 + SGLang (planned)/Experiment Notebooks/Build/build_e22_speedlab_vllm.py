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
        # rev 2b (priority plan: FP16 vs BF16 GDN state in the same run): the fp16 control of fp8_s16_ssm_bf16
        ("fp8_s16_ssm_fp16", vms_flags(MODEL_PATH, extra={"--mamba-ssm-cache-dtype": "float16"}), {}),
    ]
    # INT4 arms, gated against the INT4 target-only oracle (rev 2: dossier P0-05 V32-INT4-ORACLE; run after the FP8 block)
    PLAN_INT4 = [
        ("int4_s20", vms_flags(VMS_INT4, seqs=20), dict(level=20)),
        # rev 2b (priority plan: MTP OFF / K1 / K3 at 16 / 20 active): speculation depth vs concurrency, prefix caching on
        ("int4_s16", vms_flags(VMS_INT4, seqs=16), dict(level=16)),
        ("int4_s16_mtp0", vms_flags(VMS_INT4, seqs=16, spec=0), dict(level=16)),
        ("int4_s20_mtp0", vms_flags(VMS_INT4, seqs=20, spec=0), dict(level=20)),
        ("int4_s16_mtp1", vms_flags(VMS_INT4, seqs=16, spec=1), dict(level=16)),
        ("int4_s20_mtp1", vms_flags(VMS_INT4, seqs=20, spec=1), dict(level=20)),
        ("int4_s20_local_argmax", _spec_swap(vms_flags(VMS_INT4, seqs=20), _LARG["--speculative-config"]), dict(level=20)),
        ("int4_s20_triton_full", vms_flags(VMS_INT4, seqs=20, backend="TRITON_ATTN", extra=_FULL), dict(level=20)),
        ("int4_s20_triton_fdec", vms_flags(VMS_INT4, seqs=20, backend="TRITON_ATTN", extra=_FDEC), dict(level=20)),
        ("int4_s20_nccl2", vms_flags(VMS_INT4, seqs=20), dict(level=20, env_extra=_N2)),
        ("int4_s20_nccl4", vms_flags(VMS_INT4, seqs=20), dict(level=20, env_extra=_N4)),
        ("int4_s20_nccl8", vms_flags(VMS_INT4, seqs=20), dict(level=20, env_extra=_N8)),
        ("int4_s20_ring", vms_flags(VMS_INT4, seqs=20), dict(level=20, env_extra=_RING)),
        ("int4_s20_pmu64", vms_flags(VMS_INT4, seqs=20, extra=_PMU), dict(level=20)),
        ("int4_s20_pmu64_sharedckpt", vms_flags(VMS_INT4, seqs=20, extra=_PMUS), dict(level=20)),
        ("int4_s20_pmu16_sharedckpt", vms_flags(VMS_INT4, seqs=20, extra={"--prefix-match-unit": 16,
                                                                          "--enable-mamba-shared-prefix-checkpoint": None}),
         dict(level=20)),
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
        # rev 2b (priority plan step 1): the vLLM step profile. INT4 MTP 3 at 20 streams on the 40k band; torch profiler for
        # ~6 s of steady decode (POST /start_profile, /stop_profile), then per-category CUDA kernel time of rank 0's trace
        # (NCCL all-reduce / all-gather, Marlin GEMM, GDN, attention, sampling, other) per decode step.
        _safe_profile("int4_s20_profile", vms_flags(VMS_INT4, seqs=20), 20)
    for _name, _fl, _kw in PLAN_INT4:
        if not _INT4_OK:
            VMS["arms"][_name] = {"ok": False, "skipped": "INT4 MTP static check failed: " + VMS["int4_mtp_static"].get("why", "")}
            VMS["order"].append(_name)
            continue
        vms_arm(_name, _fl, **_kw)
    # LAST: 2 x TP2 inside one vLLM server''')
# the profile helper, defined before the arm plan runs
s = rep(s, '''def vms_arm(name, flags, **kw):''', '''def _e22_trace_summary(pdir):
    """rank-0 (or the first) torch profiler trace: CUDA kernel time per category, the decode-step count, top kernels"""
    import gzip
    files = sorted(Path(pdir).rglob("*.pt.trace.json*"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {"error": "no trace file under " + str(pdir), "listing": [str(p) for p in Path(pdir).rglob("*")][:20]}
    f = next((p for p in files if "rank0" in p.name or "rank-0" in p.name or "_0." in p.name), files[0])
    opener = gzip.open if f.name.endswith(".gz") else open
    with opener(f, "rt", encoding="utf-8") as fh:
        tr = json.load(fh)
    ev = [e for e in tr.get("traceEvents", []) if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_kernel")]
    cats = {"nccl_allreduce": ("allreduce", "all_reduce"), "nccl_allgather": ("allgather", "all_gather"),
            "nccl_other": ("nccl",), "gemm_marlin": ("marlin",), "gemm_other": ("gemm", "cutlass", "sm89_xmma", "ampere", "cublas"),
            "gdn_linear_attn": ("gdn", "fla", "chunk_", "recurrent", "fused_recurrent", "gated_delta", "causal_conv"),
            "attention": ("flash", "attn", "attention", "paged"), "sampling": ("sampl", "topk", "top_k", "argmax", "softmax")}
    tot, per, names = 0.0, {k: 0.0 for k in list(cats) + ["other"]}, {}
    for e in ev:
        n, d = str(e.get("name", "")).lower(), float(e.get("dur", 0.0))
        tot += d
        names[e.get("name", "")] = names.get(e.get("name", ""), 0.0) + d
        for k, pats in cats.items():
            if any(p in n for p in pats):
                per[k] += d
                break
        else:
            per["other"] += d
    span = (max(e["ts"] + e.get("dur", 0) for e in ev) - min(e["ts"] for e in ev)) if ev else 0.0
    return {"file": f.name, "files": len(files), "kernels": len(ev), "kernel_us_total": round(tot), "span_us": round(span),
            "by_category_us": {k: round(v) for k, v in per.items()},
            "by_category_pct": {k: round(100 * v / tot, 1) for k, v in per.items()} if tot else {},
            "top_kernels_us": dict(sorted(((k, round(v)) for k, v in names.items()), key=lambda x: -x[1])[:40])}


def _safe_profile(name, flags, level):
    """rev 2b: one profiled launch; any failure is recorded on the arm, the server killed, the plan goes on"""
    global VMS_SERVER
    a = {"ok": False, "level_wanted": level}
    VMS["arms"][name] = a
    VMS["order"].append(name)
    try:
        pdir = VMS_DIR / "profile"
        pdir.mkdir(parents=True, exist_ok=True)
        fl = list(flags) + ["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": str(pdir)})]
        ok, rec = vms_launch(name, fl)
        a.update(flags=fl, launch_sec=rec.get("sec"), skipped=rec.get("skipped"), kv_tokens=rec.get("kv_tokens"))
        if not ok:
            return a
        box = {}
        th = threading.Thread(target=lambda: box.update(w=vms_window(_L, level, _ts(150), name + "_long",
                                                                       deadline=HARD_STOP - 600)), daemon=True)
        th.start()
        t0, steady = time.perf_counter(), 0
        while th.is_alive() and time.perf_counter() - t0 < _ts(900):
            time.sleep(_ts(5))
            try:
                p = vms_prom()
            except Exception:
                continue
            run, wait = p.get("vllm:num_requests_running", 0.0), p.get("vllm:num_requests_waiting", 0.0)
            steady = steady + 1 if (run >= level - 1 and wait == 0) else 0
            if steady >= 6:          # ~30 s at the level with no prefill queue
                break
        a["steady_before_profile"] = steady
        for ep in ("/start_profile", "/stop_profile"):
            try:
                _urlreq.urlopen(_urlreq.Request(VMS_URL + ep, data=b"", method="POST"), timeout=600).read()
                a[ep.strip("/")] = "ok"
            except Exception as e:
                a[ep.strip("/")] = repr(e)[:200]
            if ep == "/start_profile":
                time.sleep(_ts(6))
        time.sleep(_ts(20))          # the trace is written by every rank after stop
        th.join(timeout=_ts(600))
        a["window"] = {k: v for k, v in (box.get("w") or {}).items() if k in ("tok_s", "accept", "measured_level", "error")}
        a["trace"] = _e22_trace_summary(pdir)
        a["ok"] = "error" not in a["trace"]
    except Exception as e:
        a["error"] = f"{type(e).__name__}: {e}"[:500]
    finally:
        try:
            if VMS_SERVER is not None:
                VMS_SERVER.kill()
            VMS_SERVER = None
        except Exception:
            pass
        e14_save()
        print("VMS arm", name, json.dumps({k: v for k, v in a.items() if k != "flags"}, default=str)[:1500], flush=True)
    return a


def vms_arm(name, flags, **kw):''')
# drop the 2 x TP2 arms (KV does not fit; measured 0.57x)
j0 = s.index("    # LAST: 2 x TP2 inside one vLLM server")
j1 = s.index('    for _a in VMS["arms"].values():       # rev 2e: the final classification')
s = s[:j0] + "    VMS_RESERVE = 0.0\n" + s[j1:]
# rev 2b: 28 launches (reference, 2 oracles, 24 arms, 1 profile) at ~12-20 min each
s = rep(s, '"time_budget_sec": 260 * 60,', '"time_budget_sec": 600 * 60,')
s = rep(s, 'md = """# VMS — vLLM 0.32.0 engine-config matrix (Qwen3.8-27B on 4x L4), engine only',
        'md = """# E22 Speed Lab rev 2b — vLLM 0.32.0 (Qwen3.8-27B on 4x L4), target 1500+ tok/s\n\n'
        'FP8 block (gated vs the FP8 target-only oracle): fp8_s16_triton_full (FULL CUDA graphs with MTP), fp8_s16_ssm_bf16 '
        '(fused CUDA GDN verify kernel), fp8_s16_local_argmax (use_local_argmax_reduction), fp8_s16_ssm_fp16 (the fp16 control, same run). INT4 block (gated vs the INT4 '
        'target-only oracle int4_target_only_noprefix): int4_s20_profile (torch profiler, per-category kernel time per step), int4_s20 (baseline), int4_s16, MTP off / K1 at 16 and 20 streams (int4_s16_mtp0, int4_s20_mtp0, int4_s16_mtp1, int4_s20_mtp1), int4_s20_local_argmax, int4_s20_triton_full, '
        'int4_s20_triton_fdec, int4_s20_nccl2 / nccl4 / nccl8, int4_s20_ring (NCCL_ALGO=Ring), int4_s20_pmu64 '
        '(--prefix-match-unit 64), int4_s20_pmu64_sharedckpt (+ --enable-mamba-shared-prefix-checkpoint), int4_s20_pmu16_sharedckpt, int4_s24, '
        'int4_s20_b8192, int4_s24_triton_full_nccl8 (combination). Budget 600 min.\n\n'
        '## Base: the Exp 14 VMS matrix')
exec(compile(s, str(E14B / "build_vms.py"), "exec"), {"__file__": str(E14B / "build_vms.py"), "__name__": "__main__"})
