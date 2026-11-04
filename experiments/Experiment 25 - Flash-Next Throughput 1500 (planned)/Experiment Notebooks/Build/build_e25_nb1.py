"""E25 NB1 FN-S1 Capacity (user go for building 2026-10-31; spec Experiment Review/NB_BUILD_SPEC_2026-10-31.md §3 NB1,
PLAN_1500.md v3.2 §4 S1 + §4.1 lever scorecard, addenda from the two verified RC2 audits 02_* / 03_* in GPT research inputs):
Qwen3.8-Flash-Next on Kaggle 4xL4 with the Merlin engine. Goal: more resident streams with no correctness change.
Boots are cumulative; each records its config diff vs RC2 FP8.

Lead decision (addendum 4): the FP8 quality isolation (BF16 / FP8 x MTP off / K3 teacher-forced) is NB0's job; NB1 is
capacity only, and production use of any NB1 FP8 number depends on NB0's gate.

  patch  vLLM #60490 applied to the installed Merlin tree before B1 (sha-pinned, marker, py_compile, vllm origin;
         fail closed), plus the read-only NB1 worker extension (cache tensor dtypes / specs / free memory over
         /collective_rpc with VLLM_SERVER_DEV_MODE=1).
  B1  RC2 FP8 + #60490 + C4 text-only (+ NCCL_DEBUG=INFO INIT,NET in this boot only). Greedy oracle vs RC2's recorded
      FP8 rows (the patch's forward test and the fail-closed gate) and vs RC2's BF16 rows; C4 / D9 separated per rank by
      'Model loading took' (weights only) vs free VRAM after ready; JIT warm-up; a < 2-min FP8 self-consistency check
      (2k / 8k teacher-forced A/A, greedy cached-repeat, raw rows saved); cold-cache cumulative allocation (8k
      1/2/4/8, 40k 1/2/4) with running vs retained blocks; 90k x 5 fresh-cache admission; prefix continuation + abort;
      steady 8k@3 / 8k@8 / 40k@5 / highest admitted (tick_ms and cycle_ms); PP2 trace window.
  B2  B1 + C3 SSM bf16 + MERLIN_SHARE_DRAFT_WORKSPACE=1 + max-num-seqs 16 (captures to 64): one combined configuration;
      booted with VLLM_RECOVERSSM_COMMIT_GRAPH=graph, flipped to eager for every standard window, then a separately
      labelled runtime A/B (eager vs graph, set_recoverssm_commit_mode) with a greedy bit-identity check.
  B3  B2 + KV_BYTES + 0.5 GiB, only if every card kept >= 0.9 GiB free in B1 and B2.

Reuse (verbatim): the RC2 builder (Experiment 13 build_e13_rc2.py) is executed up to its notebook-writing part, so its
cells come out exactly as RC2 composed them from E23; they are checked byte-for-byte against the Kaggle-run RC2 notebook
(sha 12eaa96c). NB1 applies marked one-line edits (output names, budget, per-boot flags / env / KV pin hooks) and adds new
cells. Nothing in Experiment 13 or Experiment 23 is written.

Run: python build_e25_nb1.py  ->  ../Q27-XHIGH-E25_NB1_CAPACITY_R2.ipynb (+ nb1_code_export.py for the local harness)
R2 (2026-10-31): the R1 notebook (sha 47c8e3ae, ran on Kaggle hosen42) died in run_boot post-processing: free_per_card() read
nb1_ladder's "_plan" list as a window. R2 = robust post-processing that never discards a boot's measurements (marked "R2").
R2 also: B1's cold-sequence fit gave F = -5.32 because the idle reading after a successful /reset_prefix_cache was the
stale kv_cache_usage gauge (refreshed only on engine steps); a successful reset proves 0 used blocks. The fitted F is
validated (finite, 0..40 blocks); otherwise F comes from this boot's own max-concurrency line (nb5_lib rule); nb1_ladder
uses the same validated F.
R2 audit fixes (PENDING_BATCH_FIXES N1-1 / N1-2 / N1-3 / N1-4 / N1-6 / N1-7, rebuilt in place before any GPU run of R2):
  N1-1 _corr is status-aware (FAIL / UNTESTED / INCOMPLETE_DIAGNOSTIC / DIAGNOSTIC_PASS; PASS is never a label);
  N1-2 a certified capacity needs pool_check MATCH (MISMATCH / no worker data -> INCONCLUSIVE_CAPACITY_UNKNOWN, no probes);
  N1-4 / N1-6 resolve_F: the log F is used; a valid fit is cross-checked (|ceil(fit) - log| > 1 -> F_FIT_LOG_CONFLICT, no
       probes), trusted only with samples >= 3 and max residual <= 2 blocks; fit a / residuals recorded; same in nb1_ladder;
  N1-3 every boot record: postprocess_ok + core_measurements_complete (kv census, pool, F, free MiB, residency); B2 runs
       only on a complete B1 core; the B3 pin rule reads the free memory of the boot it builds on (B2's missing -> fail closed);
  N1-7 TF_OOM -> UNSAFE_HEADROOM: dependents still run, never a recommended pin / capacity winner / adopted row.
"""
import hashlib
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
RC2_BUILDER = Path(r"D:/ARC AGI 2/experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)/"
                   r"Experiment Notebooks/Build/build_e13_rc2.py")
RC2_NB = RC2_BUILDER.parents[1] / "Q27-XHIGH-E13_FLASHNEXT_RC2.ipynb"
RC2_NB_SHA = "12eaa96cea2ced510ef54bca83586a3f545cb0609ccdb5634cf32ecfb9fcafa5"     # the RC2 GPU commit (S0 PASSED)
RC2_OUT = Path(r"D:/ARC AGI 2/_scratch/e13_rc2_gpu_v1/rc2")                             # RC2's real output
MERLIN_QSA = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src/vllm/models/qwen4_exp/nvidia/qsa.py")   # ad42e37, build check
PATCH = HERE / "patches" / "60490_qsa_indexer_workspace_merlin_ad42e37.patch"
OUT_NB = HERE.parent / "Q27-XHIGH-E25_NB1_CAPACITY_R2.ipynb"      # R2: the R1 file (Q27-XHIGH-E25_NB1_CAPACITY.ipynb) is never written
PUSHED = {"47c8e3aec7278092586dac21a69ebb08fccc6a1a170ae895a0a08e6796e0cde0"}   # shas of NB1 versions that ran on Kaggle (R1, hosen42
                     # 2026-10-31); never overwrite one; rebuilds then go to *_REBUILD
if OUT_NB.exists() and hashlib.sha256(OUT_NB.read_bytes()).hexdigest() in PUSHED:
    OUT_NB = OUT_NB.with_name(OUT_NB.stem + "_REBUILD.ipynb")


def cut(text, start, end, label):
    assert text.count(start) == 1, f"{label}: start x{text.count(start)}"
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    return text[i:j]


def once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: x{text.count(old)}"
    return text.replace(old, new)


# ---------------------------------------------------------------- RC2 cells, exactly as RC2 composed them
RC2_TEXT = RC2_BUILDER.read_text(encoding="utf-8").replace("\r\n", "\n")
_prefix = RC2_TEXT[:RC2_TEXT.index('\nsubs = {"__LAB__": json.dumps(LAB), "__PLE__": json.dumps(PLE)}')]
assert "OUT_NB.write" not in _prefix and "write_text(\"\\n\\n\".join" not in _prefix, "RC2 prefix must not write at build time"
RC2 = {"__file__": str(RC2_BUILDER), "__name__": "rc2_builder_prefix"}
exec(compile(_prefix, str(RC2_BUILDER), "exec"), RC2)            # definitions only: reads E23 / LAB / parts.json
rc2_subs = {"__LAB__": json.dumps(RC2["LAB"]), "__PLE__": json.dumps(RC2["PLE"])}
rc2_cells = []
for i, c in enumerate(RC2[k] for k in ("CELL1", "CELL2", "CELL3", "CELL4", "CELL5", "CELL6")):
    for k, v in rc2_subs.items():
        c = c.replace(k, v)
    rc2_cells.append("if not RERUN:\n" + textwrap.indent(c, "    ") if i else c)
assert hashlib.sha256(RC2_NB.read_bytes()).hexdigest() == RC2_NB_SHA, "RC2 notebook is not the Kaggle-run version"
_rc2_nb = json.loads(RC2_NB.read_text(encoding="utf-8"))
assert ["".join(c["source"]) for c in _rc2_nb["cells"] if c["cell_type"] == "code"] == rc2_cells, \
    "RC2 cells rebuilt from build_e13_rc2.py differ from the Kaggle-run RC2 notebook"

# ---------------------------------------------------------------- RC2 real facts: the diff reference + recorded oracle rows
_R = json.loads((RC2_OUT / "rc2_results.json").read_text(encoding="utf-8"))
_f, _b = _R["sections"]["A3"]["merlin_fp8"], _R["sections"]["A1"]["merlin_bf16"]
_srv, _bsrv = _f["servers"]["merlin_fp8"], _b["servers"]["merlin_bf16"]
RC2_FP8 = {"argv_tail": _R["setup"]["merlin_plans"]["merlin_fp8"]["argv_tail"],
           "env": _R["setup"]["merlin_plans"]["merlin_fp8"]["env_keys"], "kv_bytes": 2479838700,
           "kv_tokens": _srv["kv_tokens"], "max_concurrency": _srv["max_concurrency"], "pool_blocks": 149, "F_rule": 7,
           "F_engine": 6, "seqs": 8, "gpu_mem_after_ready": _srv["gpu_mem_after_ready"], "gpu_mem_end": _srv["gpu_mem_end"],
           "windows": {k: {"status": v.get("status"), "tok_s": v.get("tok_s"), "mal": v.get("mean_acceptance_length"),
                           "step_ms": v.get("step_ms"), "running": [v.get("running_min"), v.get("running_max")]}
                       for k, v in {**_f["steady"], **_f["census_long"]}.items()},
           "tf_40k_vs_bf16": _f["tf"]["vs_bf16"]["40k"], "ready_sec": _srv["ready_sec"],
           "bf16": {"kv_tokens": _bsrv["kv_tokens"], "max_concurrency": _bsrv["max_concurrency"], "pool_blocks": 78, "seqs": 3,
                    "gpu_mem_after_ready": _bsrv["gpu_mem_after_ready"], "gpu_mem_end": _bsrv["gpu_mem_end"],
                    "ready_sec": _bsrv["ready_sec"]}}
assert RC2_FP8["kv_tokens"] == 513940 and RC2_FP8["windows"]["8k@8"]["tok_s"] == 261.8, "unexpected RC2 results file"
assert RC2_FP8["bf16"]["kv_tokens"] == 269042 and RC2_FP8["env"]["VLLM_PP_LAYER_PARTITION"] == "25,23"
import re as _re
_log = (RC2_OUT / "logs" / "server_merlin_fp8.log").read_text(encoding="utf-8", errors="replace")
RC2_FP8["rank_load_gib"] = {int(p) * 2 + int(t): float(g) for p, t, g in _re.findall(
    r"\(Worker_PP(\d)_TP(\d)\S* pid=\d+\) INFO [^\n]*?Model loading took ([\d.]+) GiB memory", _log)}
RC2_FP8["rank_graph_gib"] = {int(p) * 2 + int(t): float(g) for p, t, g in _re.findall(
    r"\(Worker_PP(\d)_TP(\d)\S* pid=\d+\) INFO [^\n]*?Graph capturing finished in \d+ secs, took ([\d.]+) GiB", _log)}
assert len(RC2_FP8["rank_load_gib"]) == 4 and RC2_FP8["rank_load_gib"][0] == 17.33 and RC2_FP8["rank_load_gib"][3] == 16.87
RC2_FP8["F_rule"] = 6          # RC2 analysis (E13 Experiment Results/E13_RC2_GPU_v1_2026-10-31.md): 3 mamba groups x 2
RC2_ORACLE = {"bf16": [], "fp8": []}
for _l in (RC2_OUT / "traces" / "requests.jsonl").read_text(encoding="utf-8").splitlines():
    _r = json.loads(_l)
    _s = _r.get("sampling") or {}
    if _s.get("temperature") == 0.0 and _s.get("max_tokens") == 128 and _s.get("logprobs") == 2:
        RC2_ORACLE[{8000: "bf16", 8001: "fp8"}[_r["port"]]].append(
            {k: _r[k] for k in ("text", "tokens", "margins", "error", "prompt_sha", "prompt_len", "finish_reason")})
assert [len(v) for v in RC2_ORACLE.values()] == [4, 4], "RC2 recorded oracle rows: expected 4 + 4"

# ---------------------------------------------------------------- the #60490 patch (strict applier; checked here too)
APPLIER = r'''
def apply_unified(text, patch):
    """Strict unified-diff applier (no fuzz, no offset): every context / removed line must match at its stated line,
    else ValueError. Lines before the first hunk (comments, ---/+++ headers) are ignored."""
    src = text.splitlines(True)
    hunks, cur = [], None
    for ln in patch.splitlines(True):
        if ln.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
            if not m:
                raise ValueError("bad hunk header: " + ln[:80])
            cur = [int(m.group(1)), int(m.group(2) or 1), []]
            hunks.append(cur)
        elif cur is not None and ln[:1] in (" ", "-", "+"):
            cur[2].append(ln)
        elif cur is not None and ln.startswith("\\"):
            raise ValueError("'No newline at end of file' hunks are not supported")
    if not hunks:
        raise ValueError("patch has no hunks")
    out, pos = [], 0
    for start, n_old, body in hunks:
        i = start - 1
        if i < pos:
            raise ValueError(f"overlapping hunk at line {start}")
        out += src[pos:i]
        old_seen = 0
        for ln in body:
            tag, s = ln[0], ln[1:]
            if tag in " -":
                if i >= len(src) or src[i] != s:
                    raise ValueError(f"context mismatch at line {i + 1}: {s[:60]!r}")
                i += 1
                old_seen += 1
            if tag in " +":
                out.append(s)
        if old_seen != n_old:
            raise ValueError(f"hunk at line {start}: {old_seen} old lines, header says {n_old}")
        pos = i
    return "".join(out + src[pos:])
'''
PATCH_TEXT = PATCH.read_bytes().decode("utf-8")
assert "\r" not in PATCH_TEXT, "patch must be LF"
P_PRE = "a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653"
P_POST = "88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b"
assert f"pre-patch sha256  {P_PRE}" in PATCH_TEXT and f"post-patch sha256 {P_POST}" in PATCH_TEXT
if MERLIN_QSA.is_file():                         # build-time proof: the applier turns Merlin ad42e37's file into P_POST
    import re as _re
    _ns = {"re": _re}
    exec(APPLIER, _ns)
    _src = MERLIN_QSA.read_bytes()
    assert hashlib.sha256(_src).hexdigest() == P_PRE
    assert hashlib.sha256(_ns["apply_unified"](_src.decode("utf-8"), PATCH_TEXT).encode()).hexdigest() == P_POST

# ---------------------------------------------------------------- the read-only worker extension (written into the tree)
NB1_EXT = r'''"""E25 NB1 worker extension (read-only): reached through POST /collective_rpc {"method": "nb1_cache_info"} with
VLLM_SERVER_DEV_MODE=1 and --worker-extension-cls nb1_worker_ext.NB1WorkerExt. Reports the ACTUAL cache tensors bound
to every attention / Mamba layer (dtype, shape, bytes), the KV cache specs (padded vs unpadded Mamba page bytes), the
physical block count and this rank's free memory. It never raises: every part returns its error text instead."""
import json
import math


def _t(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return {"dtype": str(x.dtype).replace("torch.", ""), "shape": list(x.shape), "bytes": int(x.numel() * x.element_size())}
    except Exception:
        pass
    return None


def _flat(kc):
    if _t(kc):
        return [kc]
    if isinstance(kc, (list, tuple)):
        if kc and isinstance(kc[0], (list, tuple)) and not _t(kc[0]):
            kc = kc[0]                              # one entry per virtual engine
        return [x for x in kc if _t(x)]
    return []


_RT = {"on": False, "installed": False, "rows": [], "passes": 0, "calls": 0, "max_steps": 200, "bytes": 0, "errors": [],
       "first": None, "skipped_capturing": 0, "skipped_prefill": 0}
_RT_MAX_BYTES = 16 << 20
_RT_DECODE_MAX_ROWS = 128          # a decode / verify micro-batch: requests x (1 + K) tokens; prefill chunks are larger


def _rt_layer(name):
    import re as _r
    m = _r.search(r"layers\.(\d+)", str(name))
    return int(m.group(1)) if m else -1


def _rt_wrap(orig):
    def select_experts(self, hidden_states, router_logits, *a, **k):
        out = orig(self, hidden_states, router_logits, *a, **k)
        if _RT["on"] and _RT["passes"] < 2 * _RT["max_steps"] and _RT["bytes"] < _RT_MAX_BYTES:
            try:
                import torch
                if torch.cuda.is_current_stream_capturing():
                    _RT["skipped_capturing"] += 1
                else:
                    ids = out[1]
                    lay = _rt_layer(getattr(self, "nb1_layer", None))
                    if _RT["first"] is None:
                        _RT["first"] = lay
                    n = int(ids.shape[0])
                    if n > _RT_DECODE_MAX_ROWS:
                        _RT["skipped_prefill"] += 1
                    else:
                        if lay == _RT["first"]:
                            _RT["passes"] += 1     # one pass of a micro-batch through this stage's first MoE layer
                        keep = ids.to(torch.int16).cpu().numpy().reshape(n, -1)
                        _RT["rows"].append((lay, max(0, _RT["passes"] - 1), n, keep))
                        _RT["bytes"] += keep.nbytes
                        _RT["calls"] += 1
            except Exception as e:
                if len(_RT["errors"]) < 5:
                    _RT["errors"].append(repr(e)[:160])
        return out
    return select_experts


class NB1WorkerExt:
    def nb1_routing(self, action, arg=""):
        """Router top-k expert ids, ONLY between 'start' and 'stop' (the profiling window): FusedMoERouter.select_experts is
        wrapped once at class level; every decode / verify-sized call (<= 128 token rows) that runs in Python (not during
        a CUDA-graph capture) keeps its ids (int16) with the layer index and the pass index (counted at this stage's first
        MoE layer; NB1 maps pass -> step = pass // 2, mb_index = pass % 2 under PP2); <= max_steps steps, <= 16 MiB per
        rank. FULL-graph replays run no Python, so graph-replayed decode steps are NOT recorded (vLLM's
        --enable-return-routed-experts writes the ids from inside the graph; it changes the forward graph and memory, so it
        is not used on a measured boot). 'stop' writes the rank's part file <arg>_rank<r>.npz."""
        out = {"action": action}
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            out["rank"] = rank
            if action == "start":
                from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
                if not _RT["installed"]:
                    FusedMoERouter.select_experts = _rt_wrap(FusedMoERouter.select_experts)
                    _RT["installed"] = True
                    model = getattr(getattr(self, "model_runner", None), "model", None)
                    named = 0
                    if model is not None:
                        for name, mod in model.named_modules():
                            for attr in ("router", "_router"):
                                r = getattr(mod, attr, None)
                                if isinstance(r, FusedMoERouter):
                                    r.nb1_layer = name
                                    named += 1
                    out["routers_named"] = named
                _RT.update(on=True, rows=[], passes=0, calls=0, bytes=0, errors=[], first=None, skipped_capturing=0,
                           skipped_prefill=0, max_steps=int(arg or 200))
                return out
            _RT["on"] = False
            out.update(passes=_RT["passes"], calls=_RT["calls"], bytes=_RT["bytes"], errors=_RT["errors"],
                       skipped_capturing=_RT["skipped_capturing"], skipped_prefill=_RT["skipped_prefill"])
            if action == "stop" and arg and _RT["rows"]:
                import numpy as np
                rows = _RT["rows"]
                offs, o_ = [], 0
                for r in rows:
                    offs.append(o_)
                    o_ += r[2]
                path = f"{arg}_rank{rank}.npz"
                np.savez_compressed(path, ids=np.concatenate([r[3] for r in rows], axis=0), layer=np.array([r[0] for r in rows]),
                                    pass_index=np.array([r[1] for r in rows]), mb_size=np.array([r[2] for r in rows]),
                                    offset=np.array(offs), rank=np.array([rank]))
                out["path"] = path
            _RT["rows"] = []
        except Exception as e:
            out["error"] = repr(e)[:300]
        return out

    def nb1_cache_info(self):
        out = {}
        try:
            import torch.distributed as dist
            out["rank"] = dist.get_rank() if dist.is_initialized() else None
        except Exception as e:
            out["rank_error"] = repr(e)[:120]
        try:
            ctx = self.vllm_config.compilation_config.static_forward_context
            sig = {}
            for name, mod in ctx.items():
                ts = [_t(x) for x in _flat(getattr(mod, "kv_cache", None))]
                if not ts:
                    continue
                key = type(mod).__name__ + "|" + json.dumps([(x["dtype"], x["shape"]) for x in ts])
                g = sig.setdefault(key, {"module": type(mod).__name__, "tensors": ts, "layers": 0, "example": name})
                g["layers"] += 1
            out["cache_tensors"] = list(sig.values())[:40]
        except Exception as e:
            out["cache_tensors_error"] = repr(e)[:300]
        try:
            agg = {}
            for name, sp in self.get_kv_cache_spec().items():
                d = {"type": type(sp).__name__}
                for k in ("block_size", "page_size_bytes", "page_size_padded", "dtype", "num_kv_heads", "head_size", "mamba_type"):
                    try:
                        v = getattr(sp, k, None)
                        d[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
                    except Exception as e:
                        d[k] = "ERR " + repr(e)[:60]
                shapes, dtypes = getattr(sp, "shapes", None), getattr(sp, "dtypes", None)
                if shapes and dtypes:
                    try:
                        d["state_dtypes"] = [str(x).replace("torch.", "") for x in dtypes]
                        d["state_shapes"] = [list(s) for s in shapes]
                        d["unpadded_page_bytes"] = int(sum(math.prod(s) * x.itemsize for s, x in zip(shapes, dtypes)))
                    except Exception as e:
                        d["unpadded_error"] = repr(e)[:80]
                key = json.dumps(d, sort_keys=True, default=str)
                a = agg.setdefault(key, dict(d, layers=0, example=name))
                a["layers"] += 1
            out["kv_cache_specs"] = list(agg.values())[:40]
        except Exception as e:
            out["kv_cache_specs_error"] = repr(e)[:300]
        try:
            out["num_gpu_blocks"] = getattr(self.cache_config, "num_gpu_blocks", None)
            out["mamba_page_size_padded"] = getattr(self.cache_config, "mamba_page_size_padded", None)
            out["mamba_ssm_cache_dtype"] = str(getattr(self.cache_config, "mamba_ssm_cache_dtype", None))
            out["cache_block_size"] = getattr(self.cache_config, "block_size", None)
        except Exception as e:
            out["cache_config_error"] = repr(e)[:200]
        try:
            for obj in (getattr(self, "model_runner", None), self):
                cfg = getattr(obj, "kv_cache_config", None)
                if cfg is not None:
                    out["kv_cache_config"] = {
                        "num_blocks": getattr(cfg, "num_blocks", None),
                        "groups": [{"spec": type(g.kv_cache_spec).__name__, "layers": len(g.layer_names),
                                    "block_size": getattr(g.kv_cache_spec, "block_size", None),
                                    "page_size_bytes": getattr(g.kv_cache_spec, "page_size_bytes", None)}
                                   for g in getattr(cfg, "kv_cache_groups", [])][:16],
                        "tensors": [{"size": getattr(t, "size", None), "shared_by": len(getattr(t, "shared_by", None) or [])}
                                    for t in getattr(cfg, "kv_cache_tensors", [])][:24]}
                    break
        except Exception as e:
            out["kv_cache_config_error"] = repr(e)[:300]
        try:
            import torch
            fr, tot = torch.cuda.mem_get_info()
            out["mem"] = {"free_mib": fr >> 20, "total_mib": tot >> 20, "allocated_mib": torch.cuda.memory_allocated() >> 20,
                          "reserved_mib": torch.cuda.memory_reserved() >> 20, "max_allocated_mib": torch.cuda.max_memory_allocated() >> 20}
        except Exception as e:
            out["mem_error"] = repr(e)[:200]
        return out
'''
compile(NB1_EXT, "<nb1_worker_ext>", "exec")

MD0 = """# E25 NB1 (FN-S1 Capacity): Qwen3.8-Flash-Next on 4×L4 with Merlin — more resident streams, no correctness change

RC2 (S0, 2026-10-31) passed: BF16 and FP8 KV clean, FP8 steady 261.8 tok/s at 8k@8 (tick 48.2 ms, cycle 96.4 ms) with the
149-block pool nearly full (F = 6 fixed blocks per running request; finished sessions retained). This notebook measures the
capacity levers one cumulative boot at a time, each with its config diff vs RC2 FP8. The FP8 quality isolation is NB0's:
production use of any FP8 number here depends on NB0's gate.

| Boot | Config | Measures |
|---|---|---|
| patch | vLLM #60490 on the installed Merlin tree + the read-only worker extension | sha / marker / py_compile / origin, fail closed |
| B1 | RC2 FP8 + #60490 + C4 text-only (+ NCCL debug) | oracle vs RC2 FP8 rows (gate + patch forward test), C4 / D9 per-rank attribution, FP8 self-check, cold-cache allocation 1/2/4/8, 90k×5, prefix + abort, ladder, PP2 trace |
| B2 | B1 + C3 SSM bf16 + MERLIN_SHARE_DRAFT_WORKSPACE + seqs 16 (combined) | oracle vs B1, allocation, 90k×5, ladder, RecoverSSM commit graph runtime A/B |
| B3 | B2 + KV_BYTES + 0.5 GiB (only if every card kept ≥ 0.9 GiB free) | oracle vs B1, allocation, 90k×5, ladder |

Fail closed: SHA gate; before every boot the previous engine's processes gone, our leaked shm objects removed, the
checkpoint page cache dropped (measured, not assumed), GPU < 150 MiB and Merlin's host rule ≥ 69 + 4 GiB; 30-min readiness
gate; per-boot watchdog; global hard stop; a failed boot never starts the next one. Commit run only; a competition rerun
writes the fallback submission. `lever_scorecard.json` is rewritten after every boot.
"""

# ---------------------------------------------------------------- RC2 cells + the marked NB1 edits
CELL1 = RC2["CELL1"]
CELL1 = once(CELL1, '# ==================================================================== CELL 1 — config, inputs, lab module',
             '# ==================================================================== CELL 1 — config, inputs, lab module (RC2 verbatim + E25 NB1 marked edits)', "c1 title")
CELL1 = once(CELL1, 'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 240 * 60)     # two cold Merlin boots + ladder + 90k/116k census + long rows',
             'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 240 * 60)     # E25 NB1: 3 cold Merlin boots (~13-17 min each) + ~2.9 h of measurements', "c1 budget")
CELL1 = once(CELL1, 'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/rc2"))', 'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb1"))   # E25 NB1', "c1 out")
CELL1 = once(CELL1, 'R = {"what": "E13 RC2 Flash-Next on Merlin",', 'R = {"what": "E25 NB1 FN-S1 Capacity: Flash-Next on Merlin",   # E25 NB1\n     ', "c1 what")
CELL1 = once(CELL1, '        tmp = OUT / "rc2_results.json.tmp"\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "rc2_results.json")',
             '        tmp = OUT / "nb1_results.json.tmp"                       # E25 NB1\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "nb1_results.json")                    # E25 NB1', "c1 save")
CELL1 = once(CELL1, '_SAVE_LOCK = threading.Lock()\n',
             '_SAVE_LOCK = threading.Lock()\nNB1_EXTRA, NB1_ENV, NB1_KV = {}, {}, {}   # E25 NB1: per-boot trailing vllm flags, env and KV pin (filled by the boot plan)\n', "c1 hooks")
CELL1 = once(CELL1, 'print("E13 RC2 diagnostic: competition rerun -> fallback submission.json only")',
             'print("E25 NB1 capacity diagnostic: competition rerun -> fallback submission.json only")', "c1 rerun")

CELL2 = RC2["CELL2"]
CELL2 = once(CELL2, '''    extra = ["--served-model-name", "q27", "--max-model-len", str(MAX_LEN), "--port", str(port),
             "--default-chat-template-kwargs", XHIGH]''',
             '''    extra = ["--served-model-name", "q27", "--max-model-len", str(MAX_LEN), "--port", str(port),
             "--default-chat-template-kwargs", XHIGH] + list(NB1_EXTRA.get(name) or [])   # E25 NB1: per-boot flags (last wins)''', "c2 extra")
CELL2 = once(CELL2, '"8" if kv == "fp8" else "3", "--port", str(port), "--max-model-len", str(MAX_LEN)], os.environ, port, LOGS)',
             '"8" if kv == "fp8" else "3", "--port", str(port), "--max-model-len", str(MAX_LEN),\n                              "--kv-cache-memory-bytes", str(kv_bytes)] + list(NB1_EXTRA.get(name) or []),   # E25 NB1 (fake only)\n                       {**os.environ, **(NB1_ENV.get(name) or {})}, port, LOGS)', "c2 fake")
CELL2 = once(CELL2, '    argv = plan.argv()\n    HELP[name] = L.help_check("merlin"',
             '    env.update(NB1_ENV.get(name) or {})              # E25 NB1: after build(), which strips every MERLIN_* variable\n    argv = plan.argv()\n    HELP[name] = L.help_check("merlin"', "c2 env")

CELL3 = RC2["CELL3"]
CELL4 = RC2["CELL4"]
CELL4 = once(CELL4, 'def step(sec, name, est_min, fn):\n', '''def keep_after_error(rec, e):
    """E25 NB1 R2: an exception after the measurements (post-processing) keeps the whole record + postprocess_error /
    postprocess_tb. A clean boot (status None / NOT_RUN / DONE / TF_OOM, correctness CLEAN, quality PASS) keeps its status
    (None / NOT_RUN -> DONE), so its dependent boots still run; any other record -> POSTPROCESS_ERROR (status_before kept).
    R2 audit N1-3: postprocess_ok False; whether the dependents run is decided by core_measurements_complete (core_check)."""
    rec["postprocess_ok"] = False
    rec["postprocess_error"] = f"{type(e).__name__}: {e}"[:800]
    rec["postprocess_tb"] = traceback.format_exc()[-2500:]
    if (rec.get("status") in (None, "NOT_RUN", "DONE", "TF_OOM") and rec.get("correctness") == "CLEAN"
            and rec.get("quality_gate") == "PASS"):
        if rec.get("status") in (None, "NOT_RUN"):
            rec["status"] = "DONE"
    else:
        rec["status_before_postprocess"] = rec.get("status")
        rec["status"] = "POSTPROCESS_ERROR"
    return rec


def step(sec, name, est_min, fn):
''', "c4 keep_after_error (R2)")
CELL4 = once(CELL4, '''    LIVE["rec"] = rec
    t = time.time()
    log(">>>", sec, name, f"(est {est_min} min, left {left() / 60:.0f} min)")
    try:
        out = fn() or {}
        rec.update(out)
        if rec.get("status") in (None, "NOT_RUN"):
            rec["status"] = "DONE"
    except Exception as e:
        rec.update(status="ERROR", error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2500:])
    rec["sec"] = round(time.time() - t, 1)
    LIVE["rec"] = None
''',
             '''    LIVE["rec"] = rec
    LIVE["out"] = None                       # E25 NB1 R2: the body's record once it returned (set by watched() / run_boot())
    t = time.time()
    log(">>>", sec, name, f"(est {est_min} min, left {left() / 60:.0f} min)")
    out = None                               # E25 NB1 R2
    try:
        out = fn() or {}
        rec.update(out)
        if rec.get("status") in (None, "NOT_RUN"):
            rec["status"] = "DONE"
    except Exception as e:
        part = out if isinstance(out, dict) and out else LIVE.get("out")   # E25 NB1 R2: never discard what the body produced
        if isinstance(part, dict) and part:                               # E25 NB1 R2
            rec.update(part)                                              # E25 NB1 R2
            keep_after_error(rec, e)                                      # E25 NB1 R2: clean -> DONE + postprocess_error
        else:                                                             # E25 NB1 R2
            rec.update(status="ERROR", error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2500:])
    rec["sec"] = round(time.time() - t, 1)
    LIVE["rec"] = None
    LIVE["out"] = None                       # E25 NB1 R2
''', "c4 step keeps the body record (R2)")
CELL5 = cut(RC2["CELL5"], "# ==================================================================== CELL 5 — the arms",
            '\nif not RERUN and not os.environ.get("RC2_UNIT"):\n    m = R["sections"]["A0"]["merge"]', "c5 definitions")
CELL5 = once(CELL5, '    kvb = KV_FP8 if kv == "fp8" else KV_BF16\n',
             '    kvb = NB1_KV.get(name) or (KV_FP8 if kv == "fp8" else KV_BF16)   # E25 NB1: the B3 pin step\n', "c5 kv")
CELL5 += "\nsave()\n"

CELL6 = r'''# ==================================================================== CELL 6 — E25 NB1: measurement tools (NEW code)
import gzip
import statistics
from collections import Counter
NB1 = R.setdefault("nb1", {})
SCORECARD = OUT / "lever_scorecard.json"
TF_RAW = OUT / "tf_raw"
TF_RAW.mkdir(exist_ok=True)
HOSTMEM = OUT / "hostmem"
HOSTMEM.mkdir(exist_ok=True)
BLOCK = 4096
PIN_STEPS_GIB = (1.0, 0.75, 0.5)             # B3 pin ladder (addendum 5): the largest step predicted safe first, then down
PIN_FREE_MIN_MIB = 512                       # a step must leave >= 0.5 GiB free per rank after warm-up + a 90k prefill
KV_STEP = 512 << 20                          # the smallest step (+0.5 GiB per card)
RAM_NEED_MIB = (69 + 4) * 1024               # Merlin's ~69 GiB resident floor + 4 GiB margin, on its own rule
BOOT_READY_SEC = 30 * 60                     # readiness gate per boot
SEQS_HI = 16
CAPTURE_HI = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64]   # Merlin registry.capture_sizes_for(16, K=3): seqs x (1 + K)
COMPILE_HI = json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": CAPTURE_HI})
PROF_DIR = TMP / "nb1_profile"
PROFILER_CFG = json.dumps({"profiler": "torch", "torch_profiler_dir": str(PROF_DIR), "torch_profiler_with_stack": False,
                           "ignore_frontend": True})      # idle unless /start_profile (annotations only while running)
EXT_FLAG = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt"]
DIAG_ENV = {"VLLM_SERVER_DEV_MODE": "1"}     # /collective_rpc (read-only extension) + /reset_prefix_cache; 127.0.0.1 only
NCCL_DIAG = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET"}   # B1 only: the real NCCL transport per group
COMMIT_GRAPH_ENV = {"VLLM_RECOVERSSM_COMMIT_GRAPH": "graph"}         # B2 boot: commit graphs captured; flipped to eager at once
RC2_FP8 = json.loads(r"""__RC2FP8__""")      # RC2's real boots: argv tail, env keys, pool, memory, windows (rc2_results.json)
RC2_ORACLE = json.loads(r"""__RC2ORACLE__""")   # RC2's recorded greedy oracle rows (traces/requests.jsonl), BF16 + FP8
GEN_RATE = 45.0                              # tok/s per stream used only to project window growth (RC2: 33 at 8, 70 at 1)
PP = 2                                       # pipeline stages of the profile: one engine step = one micro-batch (C/2 requests)
TF_BANDS = ("2k", "8k")                      # FP8 self-consistency only (< 2 min); the BF16 / FP8 isolation is NB0's
TF_P = {b: PROMPTS[b][5] for b in TF_BANDS}  # RC2's TF prompt (variant 5)
RETENTION_KNOBS = {   # Merlin source (ad42e37), recorded, not changed
    "MAX_GRACE_PINS": "16 finished turns keep their cached prefix referenced (kv_cache_manager.py:735, class constant)",
    "MAX_PINNED_WAITERS": "16 waiting requests' hits pinned until admission (kv_cache_manager.py:940, class constant)",
    "enable_grace_pins": "enable_caching and partial hash hits (prefix-match-unit 64), kv_cache_manager.py:234",
    "VLLM_E2_SCHED_MISS_DEFER_S": "120 s default (merlin/envs.py:40): a cold miss may wait for protected waiters",
    "VLLM_E2_SCHED_COHERENT_YIELD": "on by default (merlin/envs.py:41): need-sized, session-coherent yields",
    "reset": "/reset_prefix_cache (VLLM_SERVER_DEV_MODE=1) drops grace and waiter pins when they alone block the reset "
             "(kv_cache_manager.py:1107-1113); it fails while requests run"}
for _b, _n in (("8k", 34), ("40k", 16)):     # distinct prompts (RC2 has 8 per band)
    PROMPTS[_b] += [prompt(BANDS[_b], v) for v in range(len(PROMPTS[_b]), _n)]
BANDS["60k"] = 60000                         # residency probe band (addendum 7); RC2's PROMPTS bands are unchanged
PROMPTS["60k"] = [prompt(BANDS["60k"], v) for v in range(8)]
MAX_NEW["60k"] = MAX_LEN - BANDS["60k"] - 512
RESIDENCY_BANDS = ("8k", "40k", "60k", "90k", "116k")
V_CENSUS = {"8k": list(range(24, 32)), "40k": list(range(9, 13))}   # never used by a steady window in the same boot
V_PREFIX = {"8k": 32, "40k": 8}
V_ABORT = 33
SUFFIX_TXT = ("\n</think>\n\nI have a first rule.<|im_end|>\n<|im_start|>user\nCheck the rule on every training pair, "
              "then give the final test grid." + _tail_txt)
SUFFIX_IDS = TOK.encode(SUFFIX_TXT, add_special_tokens=False).ids
R["prompts"] = {b: {"n": len(p), "len": len(p[0])} for b, p in PROMPTS.items()}


def psha(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


R["nb1_prompt_sha256"] = {b: psha(TF_P[b]) for b in TF_BANDS}
_osha = [psha(p)[:16] for p in ORACLE_P + LONG_P]          # RC2's own prompt_sha formula (16 hex)
NB1["rc2_oracle_prompts_match"] = {k: [r["prompt_sha"] for r in v] == _osha for k, v in RC2_ORACLE.items()}
for _k, _v in RC2_ORACLE.items():                          # RC2's recorded rows become references only for identical prompts
    if NB1["rc2_oracle_prompts_match"][_k] and not FAKE:    # (the fake server's text differs: it builds its own reference)
        ROWS["rc2_" + _k] = [{"text": r["text"], "tokens": r["tokens"], "margins": r["margins"], "error": r["error"]} for r in _v]
NB1["retention_knobs"] = RETENTION_KNOBS
NB1["sampling"] = ("every request passes its own sampling (greedy: temperature 0; load: T 0.6, top_p 0.95, top_k 20; "
                   "Merlin's --override-generation-config is T 1.0)")
R["capacity_plan"] = {
    "rule": "blocks per running request = ceil(L / 4096) + F, F = 6 (3 mamba groups x 2; RC2 analysis: 149 / 3.92 = 38 = 32 + 6, "
            "live transitions 6.0 / 6.1); a finished session keeps floor(L / 4096) + 3 blocks (up to 16) until re-admitted or yielded",
    "rc2_pool_blocks": {"fp8": 149, "bf16": 78},
    "worst_case_F6": {"8k@8": 8 * (math.ceil((8192 + 8000) / 4096) + 6), "40k@5": 5 * (math.ceil((40000 + 8000) / 4096) + 6),
                      "8k x 8 hold (cold sequence)": 8 * (math.ceil(9000 / 4096) + 6), "40k x 4 hold": 4 * (math.ceil(41000 / 4096) + 6),
                      "90k x 5 admission (max_new 3000)": 5 * (math.ceil(93000 / 4096) + 6), "pool": 149},
    "retention": "every capacity test starts from /reset_prefix_cache (retained sessions released); windows record the idle "
                 "(retained) blocks before them, so running = used - retained",
    "highest_levels": "computed at run time from the measured pool and F, never above max_num_seqs read from the server log",
    "tf": "B1 only: 2k + 8k FP8 self-consistency (teacher-forced A/A + greedy cached-repeat), < 2 min",
    "c3_note": "at BLOCK_SIZE 4096 the attention page is larger than the mamba page (RC2 log: 'Padding mamba page size by "
               "21.01%'), so bf16 SSM state cannot change the pool's block count; B2 records the smaller minimum block it allows"}

# ---- trace classifier fix (A7): QSA / Marlin MoE / top-k before the generic patterns; GDN '(anonymous namespace)' is
# already handled by the lab's rev 3h _cat
L.CATS = [("nccl_sendrecv", ("sendrecv", "send_recv", "nccldevkernel_send", "nccldevkernel_recv")), L.CATS[0],
          ("recoverssm_commit", ("recoverssm", "commit_", "boundary_fold", "final_fold", "conv_compaction")),
          ("qsa", ("qsa",)), ("moe_marlin", ("marlin_moe", "fused_marlin_moe", "moe_wna16")),
          ("topk", ("persistent_topk", "cooperative_topk"))] + list(L.CATS[1:])
L.COMPUTE_CATS = tuple(L.COMPUTE_CATS) + ("nccl_sendrecv", "recoverssm_commit", "qsa", "moe_marlin", "topk")

# ---- RC2 long-row gate, stricter (audit 2): a natural row needs finish == stop and a closed think; forced rows
# (ignore_eos) are stability tests only. NB1 runs no long rows; the reused gate is fixed for any later reuse.
_long_quality_rc2, _long_gate_rc2 = long_quality, long_gate


def long_quality(name, r):
    q = _long_quality_rc2(name, r)
    if q == "PASS_LONG" and name == "natural":
        if r.get("finish") != "stop":
            return "INCOMPLETE_NATURAL_FINISH"
        if r.get("think_closed") is not True:
            return "INCOMPLETE_THINKING"
    return q


def long_gate(o):
    g = _long_gate_rc2(o)
    lo = o.get("long") if isinstance(o.get("long"), dict) else None
    if lo and isinstance(lo.get("forced60k"), dict):
        lo["forced60k"]["counts_as"] = "stability test only (ignore_eos)"
    if g is None and lo and any(v in ("INCOMPLETE_NATURAL_FINISH", "INCOMPLETE_THINKING") for v in (lo.get("long_quality") or {}).values()):
        return "INCONCLUSIVE_LONG_INCOMPLETE"
    return g


# ---- the #60490 backport + the worker extension (written before B1)
P60490 = __PATCH__
P60490_REL = "vllm/models/qwen4_exp/nvidia/qsa.py"
P60490_PRE, P60490_POST = "__PPRE__", "__PPOST__"
P60490_MARKER = "# E25 NB1: backport of vLLM PR #60490 (issue #60489)."
NB1_EXT_SRC = __NB1EXT__
''' + APPLIER + r'''

def apply_60490():
    """After merlin_install, before B1: patch the installed tree's qsa.py (pre-sha must be Merlin ad42e37's), verify the
    post-sha, the marker, a py_compile with the Merlin venv and that `vllm` resolves to the tree; write the read-only
    worker extension next to the tree's vllm/ (on the server's PYTHONPATH). Any miss raises: no boot is started."""
    if FAKE:
        tree = os.environ.get("NB1_FAKE_TREE")
        if not tree:
            raise RuntimeError("PATCH_60490_NO_TREE: NB1_FAKE_TREE is not set (local harness)")
        tree, py = Path(tree), sys.executable
    else:
        eng = merlin_install()
        tree, py = Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"
    f = tree / P60490_REL
    raw = f.read_bytes()
    h0 = hashlib.sha256(raw).hexdigest()
    rec = {"file": str(f), "sha_before": h0, "pre_expected": P60490_PRE, "post_expected": P60490_POST}
    if h0 == P60490_POST:
        rec["applied"] = "already_applied"
    elif h0 != P60490_PRE:
        raise RuntimeError(f"PATCH_60490_PRE_SHA_MISMATCH: {h0} (expected Merlin ad42e37 {P60490_PRE})")
    else:
        new = apply_unified(raw.decode("utf-8"), P60490).encode("utf-8")
        if hashlib.sha256(new).hexdigest() != P60490_POST:
            raise RuntimeError("PATCH_60490_POST_SHA_MISMATCH: " + hashlib.sha256(new).hexdigest())
        tmp = f.with_name(f.name + ".nb1tmp")
        tmp.write_bytes(new)
        os.replace(tmp, f)
        rec["applied"] = "applied"
    for pyc in (f.parent / "__pycache__").glob("qsa.*.pyc"):
        pyc.unlink()
    data = f.read_bytes()
    rec.update(sha_after=hashlib.sha256(data).hexdigest(), marker=P60490_MARKER.encode() in data)
    ext = tree / "nb1_worker_ext.py"
    ext.write_text(NB1_EXT_SRC)
    rec["worker_ext"] = {"file": str(ext), "sha256": hashlib.sha256(ext.read_bytes()).hexdigest()}
    cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)", str(f), str(ext)],
                        capture_output=True, text=True, timeout=300)
    rec["py_compile"] = {"rc": cp.returncode, "err": cp.stderr[-400:]}
    origin_ok = True
    if not FAKE:
        envp = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        envp["PYTHONPATH"] = str(tree)            # what Merlin's build() exports for the server
        cp2 = subprocess.run([py, "-c", "import importlib.util as u; print(u.find_spec('vllm').origin); print(u.find_spec('nb1_worker_ext').origin)"],
                             capture_output=True, text=True, timeout=300, env=envp)
        rec["vllm_origin"] = cp2.stdout.strip()[-400:]
        origin_ok = all(l.startswith(str(tree)) for l in cp2.stdout.strip().splitlines()[-2:]) and len(cp2.stdout.strip().splitlines()) >= 2
    if rec["sha_after"] != P60490_POST or not rec["marker"] or cp.returncode or not origin_ok:
        raise RuntimeError("PATCH_60490_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("sha_after", "marker", "py_compile", "vllm_origin")})[:600])
    rec["ok"] = True
    rec["forward_test"] = "B1's greedy oracle vs RC2's recorded FP8 rows (unpatched RC2 FP8; B1 adds text-only): B1 oracle.vs_rc2_fp8"
    return rec


# ---- host memory: previous engine gone, our leaked shm removed, page cache dropped (measured), Merlin's rule
def cgroup_snapshot(tag):
    """memory.current / stat / events / pressure (first 12 kB each) to hostmem/<tag>.txt; key numbers returned."""
    rec, txt = {"tag": tag}, []
    for n in ("memory.current", "memory.stat", "memory.events", "memory.pressure"):
        try:
            s = (Path("/sys/fs/cgroup") / n).read_text()[:12000]
        except Exception:
            continue
        txt.append(f"--- {n} ---\n{s}")
        if n == "memory.current":
            rec["current_mib"] = int(s.strip()) >> 20
        elif n == "memory.events":
            rec["events"] = {a: int(b) for a, b in (l.split() for l in s.splitlines() if len(l.split()) == 2)}
        elif n == "memory.pressure":
            rec["pressure"] = s.strip().splitlines()[:2]
    try:
        (HOSTMEM / f"{tag}.txt").write_text("\n".join(txt))
    except Exception:
        pass
    rec["host_mem"] = host_mem()
    return rec


ENGINE_RX = re.compile(r"vllm|PleOffload|ple_offload|EngineCore|merlin_off/venv", re.I)


def engine_procs():
    """Processes of an engine (vLLM server, workers, PLE offload) other than this notebook; None off Linux."""
    if not Path("/proc/self/stat").exists():
        return None
    me, out = os.getpid(), []
    for d in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(d.name)
            if pid == me:
                continue
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if ENGINE_RX.search(cmd) and "py_compile" not in cmd:
                out.append([pid, cmd[:160]])
        except Exception:
            pass
    return out


def clear_engine_procs():
    left0 = engine_procs()
    rec = {"found": left0}
    if left0:
        for pid, _ in left0:
            try:
                os.kill(pid, 9)
            except Exception:
                pass
        time.sleep(10)
        rec["after_kill"] = engine_procs()
    rec["ok"] = not (rec.get("after_kill") if left0 else left0)
    return rec


def drop_page_cache():
    """posix_fadvise(DONTNEED) on every checkpoint file (merged dir symlinks resolved + the Model mounts); only after the
    previous server is dead (never on the pinned PLE pages of a live server). A measured attempt, not an assumed fix."""
    if not hasattr(os, "posix_fadvise"):
        return {"available": False}
    t, seen, n, b, err = time.time(), set(), 0, 0, 0
    roots = [CKPT] + [v[0] for v in PARTS.values() if v] + ([PLE_DIR] if PLE_DIR else [])
    for root in roots:
        for dp, _, fns in os.walk(str(root)):
            for x in fns:
                p = os.path.realpath(os.path.join(dp, x))
                if p in seen:
                    continue
                seen.add(p)
                try:
                    fd = os.open(p, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        b += os.fstat(fd).st_size
                        n += 1
                    finally:
                        os.close(fd)
                except OSError:
                    err += 1
    return {"available": True, "files": n, "gib": round(b / 2 ** 30, 1), "errors": err, "sec": round(time.time() - t, 1)}


def host_ram_gate(tag):
    before = cgroup_snapshot(tag + "_before_drop")
    drops, tries = [drop_page_cache()], 0
    after = host_mem()
    while True:
        av = after.get("avail_mib", after.get("MemAvailable_mib"))
        if av is None:
            ok = bool(FAKE)               # local harness only; on Kaggle an unreadable host is never "enough"
            why = "no cgroup / meminfo reading" + (" (local harness: skipped)" if FAKE else ": HOST_RAM_GATE fails closed")
            break
        if av >= RAM_NEED_MIB:
            ok, why = True, f"avail {av} MiB >= {RAM_NEED_MIB}"
            break
        if FAKE:                          # local / CPU-kit harness only (a 30 GiB CPU box): recorded, never enforced
            ok, why = True, f"avail {av} MiB < {RAM_NEED_MIB} MiB (local harness: recorded, not enforced)"
            break
        tries += 1
        if tries > 3:
            ok, why = False, f"HOST_RAM_GATE: avail {av} MiB < {RAM_NEED_MIB} MiB after {tries} page-cache drops"
            break
        time.sleep(20)
        drops.append(drop_page_cache())
        after = host_mem()
    return {"ok": ok, "why": why, "need_mib": RAM_NEED_MIB, "before": before, "after_drop": cgroup_snapshot(tag + "_after_drop"),
            "drops": drops}


_boot_gate_rc2, _shm_cleanup_rc2 = boot_gate, shm_cleanup
BOOT = {"deadline": None, "fired": None, "tag": "boot"}


def boot_gate():
    """RC2's GPU release gate (< 150 MiB per card), the previous engine's processes gone, then the host-RAM gate."""
    g = _boot_gate_rc2()
    if g.get("ok") is not True:
        return g
    g["engine_procs"] = clear_engine_procs()
    if not g["engine_procs"]["ok"]:
        g["ok"], g["why"] = False, "ENGINE_PROCS_LEFT: " + json.dumps(g["engine_procs"].get("after_kill"))[:300]
        return g
    h = host_ram_gate(BOOT["tag"])
    g["host_ram"] = h
    if not h["ok"]:
        g["ok"], g["why"] = False, h["why"]
    return g


def shm_cleanup(before):
    """RC2's /dev/shm cleanup (this boot's torch_/psm_/vllm/ple/nccl/sem.mp-/cuda.shm entries) + the loky semaphores the
    RC2 log reported leaked at shutdown (sem.loky-*), only when created during this boot."""
    out = _shm_cleanup_rc2(before)
    out["removed_loky"] = []
    try:
        for p in sorted(SHM.iterdir()):
            if p.name not in before and re.match(r"sem\.loky-", p.name):
                try:
                    p.unlink()
                    out["removed_loky"].append(p.name)
                except Exception:
                    pass
    except Exception:
        pass
    return out


class LoadMonitor(threading.Thread):
    """Every 10 s from launch until ready: per process of the server's session, major faults and /proc/<pid>/io
    (rchar, read_bytes) + the cgroup current; a slow worker (RC2: 873 s vs 125 s) becomes diagnosable."""
    def __init__(self, name):
        super().__init__(daemon=True)
        self.sname, self.stop_ev, self.path, self.last = name, threading.Event(), OUT / f"load_monitor_{name}.jsonl", None

    def sample(self, sid):
        rows = []
        for d in Path("/proc").glob("[0-9]*"):
            try:
                st = (d / "stat").read_text()
                rp = st.rindex(")")
                f = st[rp + 2:].split()
                if int(f[3]) != sid:
                    continue
                io = dict(l.split(": ", 1) for l in (d / "io").read_text().splitlines() if ": " in l)
                rows.append([int(d.name), st[st.index("(") + 1:rp][:24], int(f[9]), int(io.get("rchar", 0)) >> 20,
                             int(io.get("read_bytes", 0)) >> 20])
            except Exception:
                pass
        return rows

    def run(self):
        while not self.stop_ev.is_set():
            srv = next((x for x in list(L.LIVE) if x.name == self.sname), None)
            if srv is not None and srv.proc is not None and Path("/proc/self/stat").exists():
                rec = {"t": round(time.time() - T_START, 1), "procs[pid,comm,majflt,rchar_mib,read_mib]": self.sample(srv.proc.pid),
                       "cgroup_mib": (host_mem() or {}).get("cgroup_cur_mib")}
                self.last = rec
                try:
                    with open(self.path, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                except Exception:
                    pass
                if srv.ready_sec is not None:
                    break
            self.stop_ev.wait(10)


# ---- per-boot watchdog
def boot_left():
    d = BOOT["deadline"]
    return min(left(), (d - time.time()) if d else float("inf"))


def due(sec):
    return boot_left() > sec and not ABORT["why"]


def watched(body, cap_sec, name):
    """The boot body gets `cap_sec`; every step checks due(); at cap + 300 s the server is killed (hard backstop)."""
    def b(ss):
        BOOT.update(deadline=time.time() + cap_sec, fired=None)

        def fire():
            BOOT["fired"] = name
            for x in ss:
                try:
                    x.kill()
                except Exception:
                    pass
        tm = threading.Timer(cap_sec + 300, fire)
        tm.daemon = True
        tm.start()
        try:
            o = body(ss)
            LIVE["out"] = o if isinstance(o, dict) else None   # R2: step() keeps it if anything after the body raises
        finally:
            tm.cancel()
            BOOT["deadline"] = None
        if BOOT["fired"]:
            o["boot_watchdog"] = f"FIRED after {cap_sec + 300:.0f} s: server killed"
        return o
    return b


# ---- server facts read from the log and from the workers
def _pick(t, rx, n=6):
    return [l.strip()[:300] for l in t.splitlines() if re.search(rx, l)][:n]


def server_caps(t, argv=()):
    na = next((l for l in t.splitlines() if "non-default args:" in l), "")
    caps = {"source": "log" if na else "argv"}
    for k in ("max_num_seqs", "max_num_batched_tokens", "block_size", "max_model_len", "kv_cache_memory_bytes", "num_speculative_tokens"):
        m = re.search(rf"'{k}': (\d+)", na)
        caps[k] = int(m.group(1)) if m else None
    for k in ("kv_cache_dtype", "mamba_ssm_cache_dtype", "mamba_cache_mode", "worker_extension_cls"):
        m = re.search(rf"'{k}': '([^']+)'", na)
        caps[k] = m.group(1) if m else None
    m = re.search(r"'language_model_only': (True|False)", na)
    caps["language_model_only"] = m.group(1) == "True" if m else False
    m = re.search(r"'cudagraph_capture_sizes': \[([\d, ]+)\]", na)
    caps["capture_sizes"] = [int(x) for x in m.group(1).split(",")] if m else None
    m = re.findall(r"Setting attention block size to (\d+) tokens", t)
    if m:
        caps["block_size"] = int(m[-1])
    if not na:                                     # fallback (the E23 fake has no args line): the argv, last wins
        a = [str(x) for x in argv]
        for flag, key in (("--max-num-seqs", "max_num_seqs"), ("--block-size", "block_size")):
            idx = [i for i, x in enumerate(a) if x == flag]
            if idx and idx[-1] + 1 < len(a):
                caps[key] = int(a[idx[-1] + 1])
    caps["block_size"] = caps.get("block_size") or BLOCK
    sizes = caps.get("capture_sizes") or []
    k = caps.get("num_speculative_tokens") or 3
    caps["captures_cover_seqs_x_1pk"] = bool(sizes) and caps.get("max_num_seqs") is not None and max(sizes) >= caps["max_num_seqs"] * (1 + k)
    return caps


def pool_facts(t):
    f = {}
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", t)
    f["kv_tokens"] = int(m[-1].replace(",", "")) if m else None
    m = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", t)
    f["max_concurrency"] = float(m[-1][1]) if m else None
    f["max_concurrency_tokens"] = int(m[-1][0].replace(",", "")) if m else None     # R2: M of the line (F_log)
    m = re.findall(r"Padding mamba page size by ([\d.]+)%", t)
    f["mamba_pad_pct"] = float(m[-1]) if m else None
    f["min_block_for_mamba"] = math.ceil(BLOCK / (1 + f["mamba_pad_pct"] / 100)) if m else None
    f["kv_group_sizes"] = (re.findall(r"kv cache group sizes \(([^)]*)\)", t) or [None])[-1]
    f["side_cache_layout"] = _pick(t, r"Side-cache layout", 2)
    f["initial_free_reserved_gib"] = [[float(a), float(b)] for a, b in re.findall(r"Initial free memory ([\d.]+) GiB, reserved ([\d.]+) GiB", t)][:4]
    f["model_load_gib_sec"] = [[float(a), float(b)] for a, b in re.findall(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", t)][:4]
    f["graph_capture_sec_gib"] = [[int(a), float(b)] for a, b in re.findall(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB", t)][:4]
    f["rank_load_gib"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
        r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Model loading took ([\d.]+) GiB memory", t)}
    f["rank_load_sec"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
        r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Model loading took [\d.]+ GiB memory and ([\d.]+) seconds", t)}
    f["rank_graph_gib"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
        r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Graph capturing finished in \d+ secs, took ([\d.]+) GiB", t)}
    f["ssm_dtype_lines"] = _pick(t, r"mamba_ssm_dtype|mamba-ssm-cache-dtype")
    f["gdn_fallback_lines"] = _pick(t, r"Falling back to the Triton GDN spec decode path")
    f["share_workspace_lines"] = _pick(t, r"MERLIN_SHARE_DRAFT_WORKSPACE")
    f["mm_warmup_lines"] = _pick(t, r"Multi-modal warmup completed")
    f["host_ram_lines"] = _pick(t, r"host memory available now|Cgroup memory preflight|Skipping auto-prefetch", 6)
    f["sched_warn_lines"] = _pick(t, r"max_num_scheduled_tokens", 2)
    f["oom_lines"] = _pick(t, r"OutOfMemoryError|CUDA out of memory", 4)
    f["static_scale_lines"] = _pick(t, r"applied static K/V scales", 8)
    f["default_scale_lines"] = _pick(t, r"DEFAULT 1\.0|no static calibration", 4)    # audit 2: the MTP layer's QSA FP8 KV
    f["ext_lines"] = _pick(t, r"Injected .*NB1WorkerExt|nb1_worker_ext", 2)
    f["shm_leak_lines"] = _pick(t, r"leaked (semaphore|shared_memory) objects", 4)
    return f


def graph_facts(t):
    caps = sorted({(a, int(c)) for a, b, c in re.findall(r"Capturing CUDA graphs \((\w+)\):\s*100%\|[^|]*\|\s*(\d+)/(\d+)", t)})
    return {"captured": caps, "resolved_modes": _pick(t, r"resolved cudagraph modes before capture", 8),
            "draft_identity": _pick(t, r"e2-identity\[.*role=draft", 4), "recoverssm_commit": _pick(t, r"\[recoverssm-commit\]"),
            "recoverssm_path": _pick(t, r"RecoverSSM path taken", 4), "jit": jit_kernels(t)}


def jit_kernels(t):
    return re.findall(r"JIT compilation during inference: (\w+)", t)


def nccl_transport(t):
    via = Counter(re.findall(r"NCCL INFO Channel \d+/\d+ ?: ?\d+\[\d+\] -> \d+\[\d+\] (?:\[\w+\] )?via (\S+)", t))
    return {"channel_via": dict(via), "net": sorted(set(re.findall(r"NCCL INFO (Using network \w+|NET/\w+ : Using [^\n]{0,60})", t)))[:6],
            "comm_inits": len(re.findall(r"NCCL INFO comm 0x[0-9a-f]+ rank \d+ nranks \d+", t)),
            "pynccl_groups": _pick(t, r"all-reduce backends .* for group", 6),
            "debug_on": "NCCL INFO" in t}


def rpc_cache_info(port):
    """The read-only worker extension over /collective_rpc (dev mode): actual cache tensor dtypes, specs, free memory."""
    body = {"method": "nb1_cache_info", "timeout": 120}
    try:
        r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/collective_rpc", data=json.dumps(body).encode(),
                                               headers={"Content-Type": "application/json"}), timeout=180).read())
        res = r.get("results") or []
    except Exception as e:
        return {"error": repr(e)[:300]}
    out = {"ranks": res}
    dts = sorted({x["dtype"] for w in res if isinstance(w, dict) for g in (w.get("cache_tensors") or [])
                  if "mamba" in g.get("module", "").lower() or "gdn" in g.get("module", "").lower() or "linear" in g.get("module", "").lower()
                  for x in g.get("tensors") or []})
    out["recurrent_tensor_dtypes"] = dts
    out["num_gpu_blocks"] = sorted({w.get("num_gpu_blocks") for w in res if isinstance(w, dict)} - {None})
    out["free_mib_per_rank"] = [((w or {}).get("mem") or {}).get("free_mib") for w in res if isinstance(w, dict)]
    pads = []
    for w in res:
        for s in ((w or {}).get("kv_cache_specs") or []) if isinstance(w, dict) else []:
            if s.get("unpadded_page_bytes") and s.get("page_size_bytes"):
                pads.append({"rank": w.get("rank"), "type": s.get("type"), "layers": s.get("layers"), "state_dtypes": s.get("state_dtypes"),
                             "padded": s["page_size_bytes"], "unpadded": s["unpadded_page_bytes"],
                             "pad_pct": round(100 * (s["page_size_bytes"] - s["unpadded_page_bytes"]) / s["unpadded_page_bytes"], 2)})
    out["mamba_padding"] = pads[:16]
    return out


def rpc(port, method, *args, timeout=120):
    """POST /collective_rpc (dev mode): a worker method by name with string args; the per-rank results or the error."""
    body = {"method": method, "args": [str(a) for a in args], "timeout": timeout}
    try:
        r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/collective_rpc", data=json.dumps(body).encode(),
                                               headers={"Content-Type": "application/json"}), timeout=timeout + 60).read())
        return {"results": r.get("results")}
    except Exception as e:
        return {"error": repr(e)[:300]}


def commit_mode(port, mode):
    """Merlin's runtime RecoverSSM commit switch (gpu_worker.set_recoverssm_commit_mode: eager | staged | graph); returns
    the previous arm's counters (eager_steps, staged_steps, replays, captures, fallbacks) per rank."""
    return rpc(port, "set_recoverssm_commit_mode", mode)


def commit_ab(port, s):
    """Separately labelled lever (B2 only): RecoverSSM commit graph. The boot captured commit graphs
    (VLLM_RECOVERSSM_COMMIT_GRAPH=graph) and ran every standard window eager; here 8k@8 eager vs graph on the same prompts,
    the per-arm commit counters (a PP-deferred commit always runs eager: replays may stay 0), and a greedy row under each
    mode (the docstring promises bit-identical state)."""
    out = {"lever": "RecoverSSM commit graph (runtime A/B/C)", "windows": {}, "counters": {}}
    g_eager = nb1_complete(port, PROMPTS["2k"][2], 64, temperature=0.0, logprobs=2)
    for mode in ("eager", "staged", "graph"):
        if not due(WIN + 240):
            out["windows"][mode] = {"status": "SKIPPED_BOOT_BUDGET"}
            continue
        out.setdefault("set", {})[mode] = commit_mode(port, mode)         # returns the counters before this arm
        out["windows"][mode] = window(port, s, "8k", 8)
        if mode == "graph":
            out["greedy_graph_vs_eager"] = lp_compare(g_eager, nb1_complete(port, PROMPTS["2k"][2], 64, temperature=0.0, logprobs=2))
        out["counters"][mode] = commit_counter_delta((out["set"][mode] or {}).get("results"),
                                                     (rpc(port, "get_recoverssm_commit_stats") or {}).get("results"))
    out["stats_after"] = rpc(port, "get_recoverssm_commit_stats")
    out["reset_to_eager"] = commit_mode(port, "eager")
    w = out["windows"]
    out["graph_replay_share"] = (out["counters"].get("graph") or {}).get("replay_share")
    if all((w.get(m) or {}).get("status") == "STEADY" for m in ("eager", "graph")):
        out["delta_tok_s"] = round(w["graph"]["tok_s"] - w["eager"]["tok_s"], 1)
        out["delta_tick_ms"] = round((w["graph"].get("tick_ms") or 0) - (w["eager"].get("tick_ms") or 0), 2)
    return out


def commit_counter_delta(before, after):
    """Per arm, summed over ranks: commits graph-replayed vs run eager / staged, and the fallbacks by reason."""
    try:
        b = {k: sum((x or {}).get(k, 0) for x in before) for k in ("eager_steps", "staged_steps", "replays")}
        a = {k: sum((x or {}).get(k, 0) for x in after) for k in ("eager_steps", "staged_steps", "replays")}
    except Exception as e:
        return {"error": repr(e)[:200]}
    d = {k: a[k] - b[k] for k in a}
    tot = sum(d.values())
    fb = {}
    for x in after or []:
        for k, v in ((x or {}).get("fallbacks") or {}).items():
            fb[k] = fb.get(k, 0) + v
    return dict(d, replay_share=round(d["replays"] / tot, 3) if tot else None, fallbacks_total_after=fb)


def scales_dump():
    """The static QSA K/V scales file (all values) + which layers it covers; the MTP layer is not in it (default 1.0)."""
    p = CKPT / "qsa_kv_scales_262k.json"
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        return {"error": repr(e)[:200]}
    lay = sorted(int(m.group(1)) for k in d for m in [re.search(r"layers\.(\d+)\.", k)] if m)
    ks = [v.get("k_scale") for v in d.values() if isinstance(v, dict)]
    vs = [v.get("v_scale") for v in d.values() if isinstance(v, dict)]
    return {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "values": d, "layers": lay, "n": len(d),
            "k_scale_min_max": [min(ks), max(ks)] if ks else None, "v_scale_min_max": [min(vs), max(vs)] if vs else None,
            "mtp_layer_in_file": any("mtp" in k for k in d)}


def free_per_card(rec, name):
    """MiB free per card = total - the highest used reading of this boot (after ready, end, every window peak)."""
    srv = ((rec or {}).get("servers") or {}).get(name) or {}
    rows = [r for r in (srv.get("gpu_mem_after_ready"), srv.get("gpu_mem_end")) if r]
    if not rows:
        return None
    tot = {i: t for row in rows for i, _, t in row}
    used = {i: max(u for row in rows for j, u, _ in row if j == i) for i in tot}
    st = (rec or {}).get("steady")             # R2: nb1_ladder's steady holds "_plan" (list), "_fit" (dict), "_seqs_cap" (int)
    ws = list(st.values()) if isinstance(st, dict) else []
    for w in ws + [(rec or {}).get("profile")]:
        if not isinstance(w, dict):            # R2: only window dicts carry gpu_mem_peak_mib
            continue
        pk = w.get("gpu_mem_peak_mib")
        for i, u in enumerate(pk if isinstance(pk, list) else []):
            if i in used and isinstance(u, (int, float)):
                used[i] = max(used[i], u)
    return [tot[i] - used[i] for i in sorted(tot)]


def free_after_ready(rec, name):
    srv = ((rec or {}).get("servers") or {}).get(name) or {}
    row = srv.get("gpu_mem_after_ready")
    return [t - u for _, u, t in row] if row else None


# ---- KV accounting: pool blocks from usage denominators; cold-cache cumulative admission (physical snapshots)
def kvu(port):
    return L.prom(port).get(L.KVU)


def pool_blocks_from_usage(us):
    vals = sorted({round(u, 9) for u in us if isinstance(u, (int, float)) and 0 < u < 1})
    if len(vals) < 2:
        return None
    for p in range(20, 3001):
        if all(abs(u * p - round(u * p)) < 0.02 for u in vals):
            return p
    return None


def reset_cache(port):
    """POST /reset_prefix_cache (dev mode): the 'caches cleared' state of the audits; fails while requests run."""
    try:
        r = L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/reset_prefix_cache", data=b"", method="POST"), timeout=120).read()
        return {"ok": True, "body": r[:200].decode(errors="replace")}
    except Exception as e:
        return {"ok": False, "error": repr(e)[:200]}


def _hits(port):
    try:
        return L.prom(port).get("vllm:prefix_cache_hits_total", 0.0)
    except Exception:
        return None


def alloc_sequence(port, band, counts):
    """Cold cache: reset, idle snapshot, then add distinct-token requests (no shared prefix) to 1, 2, 4 (, 8) resident
    requests, snapshot after each step once all are past prefill, release, snapshot, reset, snapshot. Block numbers
    follow from vllm:kv_cache_usage_perc x P (P recovered from the usage denominators)."""
    vs = list(V_CENSUS[band])
    rec = {"band": band, "counts": list(counts), "snapshots": []}
    if not L.wait_idle(port, 120):
        rec["status"] = "INCONCLUSIVE_NOT_IDLE"
        return rec
    rec["reset_before"] = reset_cache(port)
    time.sleep(1)
    rec["idle_usage"] = [kvu(port) for _ in range(3)]
    h0, pre0 = _hits(port), L.prom(port).get(L.PREEMPT, 0)
    ss = []
    try:
        for c in counts:
            if c > len(vs):
                rec["snapshots"].append({"requests": c, "status": "SKIPPED_NOT_ENOUGH_PROMPTS"})
                break
            if not due(60 + (c - len(ss)) * BANDS[band] / max(500.0, PREFILL["tok_s"])):
                rec["snapshots"].append({"requests": c, "status": "SKIPPED_BOOT_BUDGET"})
                break
            while len(ss) < c:
                g = L.Stream(port, PROMPTS[band][vs[len(ss)]], 8192, temperature=TEMP, seed=700 + len(ss), ignore_eos=True)
                g.start()
                ss.append(g)
            t0 = time.time()
            while time.time() - t0 < 600:
                time.sleep(1.5)
                m = L.prom(port)
                if int(m.get(L.RUN, 0)) >= c and all(g.t_first for g in ss):
                    break
                if any(not g.is_alive() for g in ss):
                    break
            tk0 = [len(g.tokens) for g in ss]
            m = L.prom(port)
            tk1 = [len(g.tokens) for g in ss]
            rec["snapshots"].append({"requests": c, "usage": m.get(L.KVU), "running": m.get(L.RUN), "waiting": m.get(L.WAIT),
                                     "tokens": [len(PROMPTS[band][v]) + (a + b) / 2 for v, a, b in zip(vs, tk0, tk1)],
                                     "tokens_spread": max((b - a for a, b in zip(tk0, tk1)), default=0),
                                     "prefix_hits_delta": (_hits(port) or 0) - (h0 or 0), "admit_sec": round(time.time() - t0, 1)})
    finally:
        for g in ss:
            g.stop()
        for g in ss:
            g.join(10)
        rec["stream_errors"] = sorted({g.err for g in ss if g.err})[:3]
    L.wait_idle(port, 60)
    rec["released_usage"] = kvu(port)
    rec["reset_after"] = reset_cache(port)
    time.sleep(1)
    rec["after_reset_usage"] = kvu(port)
    rec["preemptions"] = L.prom(port).get(L.PREEMPT, 0) - pre0
    rec["status"] = "MEASURED" if any("usage" in s for s in rec["snapshots"]) else "INCONCLUSIVE"
    return rec


def reset_succeeded(r):
    """R2: POST /reset_prefix_cache answered {"success": true}. Merlin's block_pool.reset_prefix_cache refuses unless the
    pool holds no block but the null block, so a success proves 0 used blocks at that moment."""
    if not isinstance(r, dict) or r.get("ok") is not True:
        return False
    try:
        return json.loads(r.get("body") or "").get("success") is True
    except Exception:
        return False


def fit_alloc(seqs, P, B):
    """Per snapshot: F = (used - idle - sum ceil(t / B)) / n (blocks beyond the token blocks, per request); least squares
    used - idle = a * sum ceil(t / B) + F * n over every snapshot (a = 1 when singular).
    R2: idle = 0 blocks when the reset before the sequence succeeded: vllm:kv_cache_usage_perc is refreshed only on engine
    steps, so the idle reading right after the reset is the stale value of the last step (B1: 45.6 % / 26.8 % while the
    pool was empty -> F -5.32). Without a successful reset the gauge reading is used (recorded as idle_source)."""
    xs, per = [], []
    for sq in seqs:
        if not sq.get("idle_usage") or sq.get("status") != "MEASURED":
            continue
        gauge_idle = round(statistics.median([u for u in sq["idle_usage"] if u is not None]) * P)
        sq["idle_gauge_blocks"] = gauge_idle
        if reset_succeeded(sq.get("reset_before")):
            idle, sq["idle_source"] = 0, "reset_prefix_cache success (0 used blocks; the idle gauge is stale until the next step)"
        else:
            idle, sq["idle_source"] = gauge_idle, "kv_cache_usage gauge (reset not confirmed)"
        sq["idle_blocks"] = idle
        sq["released_blocks"] = round((sq.get("released_usage") or 0) * P)
        sq["retained_after_release_blocks"] = sq["released_blocks"] - idle
        sq["after_reset_blocks"] = round((sq.get("after_reset_usage") or 0) * P)
        prev = None
        for s in sq["snapshots"]:
            if "usage" not in s or s.get("usage") is None:
                continue
            used = round(s["usage"] * P)
            tb = sum(math.ceil(t / B) for t in s["tokens"])
            s.update(used_blocks=used, free_blocks=P - used, token_blocks=tb, F=round((used - idle - tb) / s["requests"], 2),
                     all_resident=s.get("running") == s["requests"] and not s.get("waiting"))
            if prev:
                s["marginal_blocks_per_added_request"] = round((used - prev[0]) / (s["requests"] - prev[1]), 2)
            prev = (used, s["requests"])
            if s["all_resident"]:
                xs.append((tb, s["requests"], used - idle))
                per.append(s["F"])
    if not xs:
        return None
    s11 = sum(a * a for a, _, _ in xs); s12 = sum(a * n for a, n, _ in xs); s22 = sum(n * n for _, n, _ in xs)
    b1 = sum(a * y for a, _, y in xs); b2 = sum(n * y for _, n, y in xs)
    det = s11 * s22 - s12 * s12
    if abs(det) > 1e-9:
        a, F = (b1 * s22 - b2 * s12) / det, (s11 * b2 - s12 * b1) / det
    else:
        a, F = 1.0, statistics.mean(per)
    res = [y - (a * x + F * n) for x, n, y in xs]
    return {"a": round(a, 3), "F": round(F, 2), "F_at_a1": round(statistics.mean(per), 2), "F_per_snapshot": per,
            "samples": len(xs), "max_abs_residual_blocks": round(max(abs(r) for r in res), 2),
            "residuals_blocks": [round(r, 2) for r in res],                 # R2 audit N1-6: the fit's residual per snapshot
            "singular": abs(det) <= 1e-9}


def kv_accounting(port, caps, plan):
    B = caps.get("block_size") or BLOCK
    seqs = []
    for band, counts in plan:
        try:
            seqs.append(alloc_sequence(port, band, counts))
        except Exception as e:
            seqs.append({"band": band, "status": "ERROR", "error": f"{type(e).__name__}: {e}"[:300]})
    us = [u for sq in seqs for u in (sq.get("idle_usage") or []) + [s.get("usage") for s in sq.get("snapshots") or []]
          + [sq.get("released_usage"), sq.get("after_reset_usage")]]
    P = pool_blocks_from_usage(us)
    out = {"sequences": seqs, "pool_blocks": P, "block_size": B, "method": "cold cache (reset_prefix_cache) + distinct "
           "prompts; blocks = kv_cache_usage x P, P recovered from the usage denominators (cross-check: worker num_gpu_blocks)"}
    out["fit"] = fit_alloc(seqs, P, B) if P else None
    if not P:
        out["why"] = "pool blocks not recoverable from the usage readings"
    return out


def n_fit(P, F, B, band, idle_blocks=0, seqs=None, n_prompts=None):
    """Streams of `band` whose final length (prompt + ~(admission + window + 60 s) x GEN_RATE) all fit in the pool."""
    if P is None or F is None:
        return None
    L0 = BANDS[band]
    final = L0 + (L0 * 8 / max(1.0, PREFILL["tok_s"]) + WIN + 60) * GEN_RATE
    n = int((P - idle_blocks) // (math.ceil(final / B) + F))
    for cap in (seqs, n_prompts):
        if cap:
            n = min(n, cap)
    return max(0, n)


def admission_test(port, s, band="90k", n=5, max_new=4096):
    """Fresh-cache admission of n distinct long prompts (audit 2: 90k x 5 before any pin change). Reset the cache, start
    n streams, wait until every one has its first token (admission + cold prefill, ~450k tokens: Merlin's cost-aware
    admission may hold a cold miss up to 120 s), then: a resident snapshot (running == n, waiting 0, no preemption, blocks)
    and a short decode window measured on the counters while all n stay resident. max_new 4096 keeps 5 x (ceil(94,096 /
    4096) + 6) = 145 of 149 blocks: the streams cannot outgrow the pool (a lab steady window cannot be used: the first
    stream would finish its max_new inside it, since the cold prefills are staggered by ~20 s each)."""
    if not due(240 + n * BANDS[band] / max(2000.0, PREFILL["tok_s"])):
        return {"status": "SKIPPED_BOOT_BUDGET"}
    if not L.wait_idle(port, 120):
        return {"status": "INCONCLUSIVE_NOT_IDLE"}
    rec = {"band": band, "n": n, "max_new": max_new, "reset": reset_cache(port)}
    time.sleep(1)
    rec["idle_usage"] = kvu(port)
    m0, l0, t0 = L.prom(port), _req_lines(), time.time()
    ss = [L.Stream(port, PROMPTS[band][i], max_new, temperature=TEMP, seed=500 + i, ignore_eos=True) for i in range(n)]
    for g in ss:
        g.start()
    try:
        limit = max(420.0, 180.0 + n * BANDS[band] / max(1000.0, PREFILL["tok_s"]))
        while time.time() - t0 < limit:
            time.sleep(2)
            if all(g.t_first for g in ss) or any(not g.is_alive() for g in ss):
                break
        m1 = L.prom(port)
        rec.update(admission_sec=round(time.time() - t0, 1), first_token=[round(g.t_first - g.t_start, 1) if g.t_first else None for g in ss],
                   snapshot={"running": m1.get(L.RUN), "waiting": m1.get(L.WAIT), "usage": m1.get(L.KVU),
                             "tokens": [len(PROMPTS[band][i]) + len(g.tokens) for i, g in enumerate(ss)]},
                   preemptions=m1.get(L.PREEMPT, 0) - m0.get(L.PREEMPT, 0),
                   prefill_uncached_tok_s=round(((m1.get(L.PROMPT, 0) - m0.get(L.PROMPT, 0)) - (m1.get("vllm:prefix_cache_hits_total", 0)
                                                 - m0.get("vllm:prefix_cache_hits_total", 0))) / max(1e-3, time.time() - t0), 1))
        rec["all_resident"] = (all(g.t_first for g in ss) and rec["snapshot"]["running"] == n and not rec["snapshot"]["waiting"]
                               and not rec["preemptions"])
        if rec["all_resident"]:
            secs = min(20.0, WIN)
            g0, w0, runs = L.prom(port), time.time(), []
            while time.time() - w0 < secs:
                time.sleep(min(2.0, secs))
                runs.append(L.prom(port).get(L.RUN))
            g1 = L.prom(port)
            dt = time.time() - w0
            steps = g1.get(L.ITER, 0) - g0.get(L.ITER, 0)
            rec["window"] = {"sec": round(dt, 1), "tok_s": round((g1.get(L.GEN, 0) - g0.get(L.GEN, 0)) / dt, 1), "running_min": min(runs or [0]),
                             "running_max": max(runs or [0]), "tick_ms": round(dt * 1000 / steps, 1) if steps else None,
                             "preemptions": g1.get(L.PREEMPT, 0) - g0.get(L.PREEMPT, 0)}
            if rec["window"]["tick_ms"]:
                rec["window"]["cycle_ms"] = round(PP * rec["window"]["tick_ms"], 1)
            rec["status"] = ("ALL_RESIDENT" if rec["window"]["running_min"] == n and not rec["window"]["preemptions"]
                             else "INCONCLUSIVE_RUNNING_CHANGED")
        else:
            rec["status"] = "NOT_ALL_RESIDENT"
    finally:
        for g in ss:
            g.stop()
        for g in ss:
            g.join(10)
        rec["stream_errors"] = sorted({g.err for g in ss if g.err})[:3]
    rec["ttft_per_request"] = _ttfts(port, l0)
    return rec


def residency_bound(P, F, band, max_new=0, seqs=None):
    """Arithmetic bound: running requests of total context band + max_new that fit P blocks at F fixed blocks each."""
    if not P or F is None:
        return None
    n = int(P // (math.ceil((BANDS[band] + max_new) / BLOCK) + F))
    return min(n, seqs) if seqs else n


def residency_capacity(o):
    """E25 NB1 C-6: this boot's own measured pool P (kv_accounting: the kv_cache_usage gauge denominators) and F (resolve_F:
    the engine's max-concurrency line cross-checked with the cold-sequence fit). Never RC2's 149 / 6.
    R2 audit N1-2: a certified capacity needs pool_check == MATCH (worker num_gpu_blocks - 1 == the gauge pool on every
    rank); MISMATCH or no worker reading -> ok False (INCONCLUSIVE_CAPACITY_UNKNOWN, the reason in why, no probes).
    R2 audit N1-4 / N1-6: F from resolve_F; a fit / line conflict -> ok False (F_FIT_LOG_CONFLICT)."""
    kv = o.get("kv") or {}
    P = kv.get("pool_blocks")
    wbs = (o.get("workers") or {}).get("num_gpu_blocks") or []
    wbs = wbs if isinstance(wbs, list) else []
    usable = sorted({w - 1 for w in wbs if isinstance(w, int) and not isinstance(w, bool) and w > 1})
    if not usable or not isinstance(P, int) or isinstance(P, bool):
        pc = "UNKNOWN"
    else:
        pc = "MATCH" if usable == [P] else "MISMATCH"
    cap = {"pool_blocks": P, "worker_num_gpu_blocks": wbs, "worker_usable_blocks": usable,
           "pool_check": pc, "pool_check_rule": "worker num_gpu_blocks - 1 (null block) == gauge pool on every rank; "
           "a certified capacity needs MATCH (R2 audit N1-2)"}
    if pc == "MISMATCH":
        cap["pool_mismatch"] = f"gauge pool {P} vs worker usable {usable}"
    why, codes = [], []
    P_ok = isinstance(P, int) and not isinstance(P, bool) and P >= 1
    if not P_ok:
        why.append("pool_blocks not measured in this boot (kv.pool_blocks)")
        codes.append("POOL_NOT_MEASURED")
    elif pc == "MISMATCH":
        why.append(f"POOL_MISMATCH: {cap['pool_mismatch']} (capacity not certified)")
        codes.append("POOL_MISMATCH")
    elif pc != "MATCH":
        why.append("POOL_UNVERIFIED: no worker num_gpu_blocks reading to check the gauge pool against "
                   f"({str((o.get('workers') or {}).get('error') or 'no num_gpu_blocks')[:120]})")
        codes.append("POOL_UNVERIFIED")
    fr = resolve_F(o, P if P_ok else None)
    cap.update({k: v for k, v in fr.items() if k not in ("F", "F_source", "why", "codes")})
    cap["F_resolved"] = fr["F"]                   # F by the fit / line rule alone (core_check), before the pool gate
    if fr["F"] is None:
        why.extend(fr["why"] or [])
        codes.extend(fr["codes"] or [])
    ok = not why
    cap.update(ok=ok, F=fr["F"] if ok else None, F_source=fr["F_source"] if ok else None, why=why or None,
               reason_codes=codes or None, status="CERTIFIED" if ok else "INCONCLUSIVE_CAPACITY_UNKNOWN")
    return cap


F_MAX_BLOCKS = 40                                # R2: a per-request fixed-block F above 40 blocks is not physical here
FIT_MIN_SAMPLES = 3                              # R2 audit N1-6: a fit is trusted only with >= 3 all-resident snapshots,
FIT_MAX_RESID_BLOCKS = 2.0                       # a max |residual| <= 2 blocks
FIT_LOG_TOL_BLOCKS = 1                           # and |ceil(fit F) - log F| <= 1 block
F_LOG_SOURCE = "server max-concurrency line (this boot)"


def _f_valid(F):
    return isinstance(F, (int, float)) and not isinstance(F, bool) and math.isfinite(F) and 0 <= F <= F_MAX_BLOCKS


def resolve_F(o, P):
    """R2 audit N1-4 / N1-6: F from this boot's two sources. The fit (kv.fit, fit_alloc) counts only when valid (finite,
    0..40); it is trusted only when samples >= 3, max |residual| <= 2 blocks and |ceil(fit) - log F| <= 1.
      both valid, |ceil(fit) - log F| > 1     -> F None, F_FIT_LOG_CONFLICT (no probes);
      both valid, agree, fit trusted          -> the log F (engine-derived), fit_check AGREES;
      both valid, agree, fit not trusted      -> the log F, fit_check NOT_TRUSTED (samples / residual);
      only the log F (fit missing / invalid)  -> the log F;
      only a fit, trusted (samples, residual) -> ceil(fit), labelled 'fit only, no line to cross-check';
      otherwise                               -> F None."""
    kv = o.get("kv") or {}
    fit = kv.get("fit") if isinstance(kv.get("fit"), dict) else {}
    F_fit = fit.get("F")
    smp, res = fit.get("samples"), fit.get("max_abs_residual_blocks")
    r = {"F_fit": F_fit, "fit_a": fit.get("a"), "fit_samples": smp, "fit_max_abs_residual_blocks": res,
         "fit_residuals_blocks": fit.get("residuals_blocks"),
         "fit_trust_rule": f"samples >= {FIT_MIN_SAMPLES}, max |residual| <= {FIT_MAX_RESID_BLOCKS:g} blocks, "
                           f"|ceil(fit F) - log F| <= {FIT_LOG_TOL_BLOCKS}"}
    fit_valid = _f_valid(F_fit)
    if F_fit is not None and not fit_valid:
        r["F_fit_invalid"] = F_fit
    fit_q = (fit_valid and isinstance(smp, int) and not isinstance(smp, bool) and smp >= FIT_MIN_SAMPLES
             and isinstance(res, (int, float)) and not isinstance(res, bool) and math.isfinite(res) and res <= FIT_MAX_RESID_BLOCKS)
    qwhy = None if fit_q else f"fit samples {smp!r} (need >= {FIT_MIN_SAMPLES}), max |residual| {res!r} (need <= {FIT_MAX_RESID_BLOCKS:g})"
    F_c = int(math.ceil(F_fit - 1e-9)) if fit_valid else None
    r["F_fit_ceil"] = F_c
    fwhy = "F not measured in this boot (kv.fit.F)" if F_fit is None else (None if fit_valid else f"F fit invalid ({F_fit!r})")
    lg = f_from_log(o, P) if P else None          # R2: this boot's own max-concurrency line (nb5_lib.capacity_plan rule)
    r["F_log"] = lg
    log_ok = bool(lg and lg.get("ok"))
    F, src, why, codes = None, None, [], []
    if fit_valid and log_ok:
        d = abs(F_c - lg["F"])
        r["fit_log_diff_blocks"] = d
        if d > FIT_LOG_TOL_BLOCKS:
            r["fit_check"] = "CONFLICT"
            r["F_conflict"] = f"F_FIT_LOG_CONFLICT: ceil(fit F {F_fit}) = {F_c} vs log F {lg['F']} (differ by {d} > {FIT_LOG_TOL_BLOCKS})"
            why.append(r["F_conflict"])
            codes.append("F_FIT_LOG_CONFLICT")
        else:
            F, src = lg["F"], F_LOG_SOURCE
            r["fit_check"] = "AGREES" if fit_q else "NOT_TRUSTED"
            r["fit_agrees"] = fit_q
            if not fit_q:
                r["fit_not_trusted_reason"] = qwhy
    elif log_ok:
        F, src = lg["F"], F_LOG_SOURCE
        r["fit_check"] = "MISSING" if F_fit is None else "INVALID"
        r["F_fallback_reason"] = fwhy
    elif fit_q:
        F = F_c
        src = ("measured in this boot: kv.fit.F (cold-sequence fit) rounded up to an integer; fit only (trusted: samples / "
               "residual), no max-concurrency line to cross-check")
        r["fit_check"] = "FIT_ONLY_TRUSTED"
    else:
        r["fit_check"] = ("MISSING" if F_fit is None else "INVALID" if not fit_valid else "NOT_TRUSTED")
        if fwhy:
            why.append(fwhy)
        elif not fit_q:
            why.append("F fit not trusted without a max-concurrency line to cross-check: " + qwhy)
        if P:
            why.append("F from the server max-concurrency line unavailable: " + str((lg or {}).get("why")))
        codes.append("F_UNAVAILABLE")
    r.update(F=F, F_source=src, why=why or None, codes=codes or None)
    return r


def f_from_log(o, P):
    """R2: F_log = P / X - ceil(M / B) from this boot's line 'GPU KV cache size: N tokens, Maximum concurrency for M tokens
    per request: X x' (X = N / M when N is there: the printed X has 2 decimals); F = ceil(F_log - 0.05) like
    nb5_lib.capacity_plan (RC2 / B1: 149 x 131,072 / 513,940 - 32 = 5.999 -> 6)."""
    pool = o.get("pool") or {}
    N, X, M = pool.get("kv_tokens"), pool.get("max_concurrency"), pool.get("max_concurrency_tokens")
    B = (o.get("caps") or {}).get("block_size") or BLOCK
    if not (isinstance(M, int) and M > 0 and ((isinstance(N, int) and N > 0) or (isinstance(X, (int, float)) and X > 0))):
        return {"ok": False, "why": f"no max-concurrency line in this boot's log (N {N}, M {M}, X {X})"}
    xx = N / M if isinstance(N, int) and N > 0 else float(X)
    f_log = P / xx - math.ceil(M / B)
    F = max(0, int(math.ceil(f_log - 0.05)))
    ok = math.isfinite(f_log) and -0.05 <= f_log <= F_MAX_BLOCKS
    return {"ok": ok, "F": F if ok else None, "F_log": round(f_log, 3), "kv_tokens": N, "max_concurrency": X, "tokens_per_request": M,
            "block_size": B, "x_used": round(xx, 5), "why": None if ok else f"F_log {f_log:.3f} outside 0..{F_MAX_BLOCKS}"}


def residency_probes(port, s, o, bands=RESIDENCY_BANDS):
    """max_safe_running_by_context (addendum 7): per context, cold cache + distinct prompts, n = the arithmetic bound at
    this boot's measured pool and F (residency_capacity; capped by max_num_seqs and the distinct prompts), max_new sized
    so n requests cannot outgrow the pool; an n whose max_new is below the decode growth expected during the staggered
    cold admission + the window is SKIPPED_HEADROOM (the arithmetic bound is reported apart); achieved = n if every stream
    is resident after prefill with real running == n and 0 preemptions, else one retry at n - 1. The 90k probe reuses the
    90k x 5 admission when it ran at the bound. Pool or F unknown -> INCONCLUSIVE_CAPACITY_UNKNOWN, no probe requests."""
    cap = residency_capacity(o)                    # E25 NB1 C-6: the boot's own P and F, no RC2 fallback
    P, F = cap["pool_blocks"], cap["F"]
    seqs = (o.get("caps") or {}).get("max_num_seqs")
    out = {"pool_blocks": P, "F": F, "F_fit": cap["F_fit"], "capacity": cap, "bands": {},
           "F_source": cap.get("F_source")}         # R2: the fit or this boot's max-concurrency line
    if not cap["ok"]:
        out.update(status="INCONCLUSIVE_CAPACITY_UNKNOWN", F_source=None, why=cap["why"])
        return out
    out["status"] = "MEASURED"
    for band in bands:
        nb = residency_bound(P, F, band, 0, seqs)
        n = min(nb or 0, len(PROMPTS[band]))
        rec = {"bound_prompt_only": residency_bound(P, F, band), "bound_capped": nb, "tries": []}
        adm = o.get("admission_90k@5") or {}
        if band == "90k" and n == 5 and adm.get("status") in ("ALL_RESIDENT", "NOT_ALL_RESIDENT", "INCONCLUSIVE_RUNNING_CHANGED"):
            rec["tries"].append({"n": 5, "status": adm.get("status"), "from": "admission_90k@5"})
            ok5 = adm.get("status") == "ALL_RESIDENT"
            rec["achieved"] = 5 if ok5 else None
            if ok5:
                out["bands"][band] = rec
                continue
            n = 4
        rate = float(os.environ["FAKE_SPEED"]) if FAKE and os.environ.get("FAKE_SPEED") else GEN_RATE
        while n >= 1:
            mn = int((P // n - F) * BLOCK - BANDS[band])
            mn = min(8192, mn)
            # the cold prefills are staggered: the first stream decodes during the others' admission + the window
            need = int((n * BANDS[band] / max(500.0, PREFILL["tok_s"]) + min(20.0, WIN) + 10) * rate)
            if mn < max(1024, need):
                rec["tries"].append({"n": n, "status": "SKIPPED_HEADROOM", "max_new": mn, "need": need,
                                     "why": "the pool leaves too little decode headroom for a valid cold admission at this n"})
                n -= 1
                continue
            if not due(240 + n * BANDS[band] / max(2000.0, PREFILL["tok_s"])):
                rec["tries"].append({"n": n, "status": "SKIPPED_BOOT_BUDGET"})
                break
            a = admission_test(port, s, band, n, max_new=mn)
            rec["tries"].append({"n": n, "max_new": mn, "status": a.get("status"), "first_token_s": a.get("first_token"),
                                 "window": a.get("window"), "preemptions": a.get("preemptions"), "snapshot": a.get("snapshot")})
            if a.get("status") == "ALL_RESIDENT":
                rec["achieved"] = n
                break
            if len([x for x in rec["tries"] if x.get("status") not in ("SKIPPED_BOOT_BUDGET", "SKIPPED_HEADROOM")]) >= 2:
                break
            n -= 1
        rec.setdefault("achieved", None)
        out["bands"][band] = rec
        save()
    return out


def block_layout(workers, F=6):
    """The per-request fixed blocks against the kv cache groups the worker reports (spec type, layers, block size). F is
    ENGINE-DERIVED (concurrency line); the split per group is an inference from the group list, never a claim that the
    blocks are reclaimable."""
    ranks = (workers or {}).get("ranks") or []
    g = next(((w or {}).get("kv_cache_config") or {} for w in ranks if isinstance(w, dict) and (w.get("kv_cache_config") or {}).get("groups")), {})
    groups = g.get("groups") or []
    mamba = [x for x in groups if "mamba" in str(x.get("spec", "")).lower()]
    attn = [x for x in groups if x not in mamba]
    return {"groups": groups, "num_blocks": g.get("num_blocks"), "mamba_groups": len(mamba), "attention_groups": len(attn),
            "F_engine": F, "F_per_mamba_group_inferred": round(F / len(mamba), 2) if mamba else None,
            "note": "F from the engine's concurrency line; the per-group split is inferred from the group list; reclaimability "
                    "is not shown (no evidence)"}


def profiler_tools():
    """ncu / nsys availability (DRAM bytes per stage need a server LAUNCHED under one of them: a separate boot)."""
    out = {}
    cands = ["ncu", "nsys", "/usr/local/cuda/bin/ncu", "/usr/local/cuda/bin/nsys"]
    try:
        eng = MERLIN_ENG if isinstance(MERLIN_ENG, dict) else {}
        if eng.get("cuda_home"):
            cands += [eng["cuda_home"] + "/bin/ncu", eng["cuda_home"] + "/bin/nsys"]
    except Exception:
        pass
    for c in cands:
        path = shutil.which(c) if "/" not in c else (c if os.path.exists(c) else None)
        if path:
            try:
                v = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()[:200]
            except Exception as e:
                v = repr(e)[:100]
            out[c] = {"path": path, "version": v}
    return {"found": out, "dram_bytes": "not measured in NB1: ncu / nsys cannot attach to a running vLLM server; DRAM bytes per "
            "stage need a boot launched under ncu (dram__bytes_read.sum on the Marlin MoE + dense kernels) or nsys: NB3 / NB4 item",
            "torch_profiler_window": "the PP2 trace window runs after the ladder, on its own streams; no benchmark window overlaps it"}


# ---- requests with usage (cached tokens), greedy logprobs and teacher-forced rows with the cache counters
def nb1_complete(port, ids, max_new, temperature=0.0, seed=None, logprobs=0, timeout=1800):
    """Non-streaming /v1/completions with token ids: ids, usage, cached tokens (--enable-prompt-tokens-details) and, with
    logprobs, the chosen-token logprobs + top-1/top-2 margins. Every request is saved in traces/requests.jsonl."""
    body = {"model": "q27", "prompt": list(ids), "max_tokens": int(max_new), "temperature": temperature,
            "top_p": 0.95 if temperature else 1.0, "ignore_eos": True, "return_token_ids": True}
    if temperature:
        body["top_k"] = 20
    if seed is not None:
        body["seed"] = int(seed)
    if logprobs:
        body["logprobs"] = int(logprobs)
        body["return_tokens_as_token_ids"] = True
    t = time.time()
    try:
        r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                               headers={"Content-Type": "application/json"}), timeout=timeout).read())
        ch = r["choices"][0]
        u = r.get("usage") or {}
        out = {"ids": list(ch.get("token_ids") or []), "text": ch.get("text") or "", "usage": u, "error": None,
               "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}
        lp = ch.get("logprobs") or {}
        if logprobs:
            out["lps"] = list(lp.get("token_logprobs") or [])
            out["margins"] = []
            for top in lp.get("top_logprobs") or []:
                v = sorted((x for x in (top or {}).values() if isinstance(x, (int, float))), reverse=True)
                out["margins"].append(round(v[0] - v[1], 4) if len(v) >= 2 else None)
    except Exception as e:
        out = {"ids": [], "text": "", "usage": None, "cached": None, "error": repr(e)[:300]}
    out["sec"] = round(time.time() - t, 2)
    try:
        with _SAVE_LOCK, open(REQ_LOG, "a") as fh:
            fh.write(json.dumps({"kind": "nb1_complete", "port": port, "prompt_len": len(ids), "max_new": max_new,
                                 "prompt_sha": psha(ids)[:16], "text": out["text"], "tokens": out["ids"], "lps": out.get("lps"),
                                 "usage": out["usage"], "error": out["error"], "sec": out["sec"], "t": time.time()}) + "\n")
    except Exception:
        pass
    return out


def nb1_tf_req(port, ids, last=256, timeout=1800):
    """e23_lab.teacher_forced's request and parsing (same body, same per-position [logprob, rank, top-1 id]) + the
    response's cached-token count. vLLM sets skip_reading_prefix_cache for prompt_logprobs requests
    (sampling_params.py), so a repeat recomputes; the cached path is tested in decode space (greedy fresh vs repeat)."""
    body = {"model": "q27", "prompt": list(ids), "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1,
            "return_tokens_as_token_ids": True}
    try:
        r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                               headers={"Content-Type": "application/json"}), timeout=timeout).read())
        pl = r["choices"][0].get("prompt_logprobs") or []
        cached = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    except Exception as e:
        return {"error": repr(e)[:300]}
    out = []
    for pos in range(max(1, len(ids) - last), len(ids)):
        d = pl[pos] if pos < len(pl) else None
        if not isinstance(d, dict):
            out.append(None)
            continue
        e = d.get(str(ids[pos])) or d.get(f"token_id:{ids[pos]}")
        top1 = [L._tok_id(k) for k, v in d.items() if isinstance(v, dict) and v.get("rank") == 1]
        out.append([e.get("logprob"), e.get("rank"), top1[0] if len(top1) == 1 else None] if isinstance(e, dict) else None)
    return {"positions": out, "n": len(out), "cached_tokens": cached, "first_position": max(1, len(ids) - last)}


def lp_compare(a, b):
    """Greedy rows a (reference) vs b: first differing token, the reference's top-1 - top-2 margin there (L.FAIL_MARGIN),
    mean / max |delta chosen-token logprob| over the identical prefix."""
    if not a or not b or a.get("error") or b.get("error") or not a.get("ids") or not b.get("ids"):
        return {"status": "INCONCLUSIVE"}
    x, y = a["ids"], b["ids"]
    d = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
    la, lb = a.get("lps") or [], b.get("lps") or []
    k = d if d is not None else min(len(x), len(y))
    dl = [abs(p - q) for p, q in zip(la[:k], lb[:k]) if isinstance(p, (int, float)) and isinstance(q, (int, float))]
    if d is None:
        st, m = ("EXACT" if len(x) == len(y) else "INCONCLUSIVE"), None
    else:
        mm = a.get("margins") or []
        m = mm[d] if d < len(mm) else None
        st = "INCONCLUSIVE" if m is None else ("FAIL_HIGH_MARGIN" if m >= L.FAIL_MARGIN else "NEAR_TIE")
    return {"status": st, "first_diff": d, "ref_margin": m, "mean_abs_dlogprob_prefix": round(statistics.mean(dl), 5) if dl else None,
            "max_abs_dlogprob_prefix": round(max(dl), 5) if dl else None, "tokens": [len(x), len(y)],
            "cached": [a.get("cached"), b.get("cached")]}


def fp8_self_check(s, key="b1", bands=TF_BANDS):
    """< 2 min FP8 self-consistency on the serving boot (the BF16 / FP8 isolation is NB0's): per band (2k, 8k) a greedy
    64-token row on a fresh prompt, the same request again (cached repeat; cached tokens recorded), and two
    teacher-forced passes (an FP8 A/A; prompt_logprobs never read the prefix cache). Raw rows -> tf_raw/<key>_<band>.json."""
    out = NB1.setdefault("fp8_self_check", {"bands": {}, "note": "BF16 / FP8 x MTP off / K3 teacher-forced isolation: NB0"})
    for b in bands:
        if not s.alive():
            out["bands"][b] = {"status": "SKIPPED_SERVER_DEAD"}
            continue
        if not due(120 + 3 * BANDS[b] / max(1000.0, PREFILL["tok_s"])):
            out["bands"][b] = {"status": "SKIPPED_BOOT_BUDGET"}
            continue
        ids = TF_P[b]
        rec = {"prompt_tokens": len(ids), "prompt_sha256": psha(ids)}
        da = nb1_complete(s.port, ids, 64, temperature=0.0, logprobs=2)
        db = nb1_complete(s.port, ids, 64, temperature=0.0, logprobs=2)
        rec["decode_cached_vs_fresh"] = lp_compare(da, db)
        t1 = nb1_tf_req(s.port, ids)
        t2 = nb1_tf_req(s.port, ids) if s.alive() else {"error": "server dead"}
        rec["tf_fresh"] = {"n": t1.get("n"), "error": t1.get("error"), "cached_tokens": t1.get("cached_tokens")}
        rec["tf_repeat"] = {"n": t2.get("n"), "error": t2.get("error"), "cached_tokens": t2.get("cached_tokens")}
        rec["tf_aa"] = L.compare_tf(t1, t2)
        try:
            (TF_RAW / f"{key}_{b}.json").write_text(json.dumps({"key": key, "band": b, "prompt_sha256": rec["prompt_sha256"],
                                                                "prompt_tokens": len(ids), "tf_fresh": t1, "tf_repeat": t2,
                                                                "decode_fresh": da, "decode_cached_repeat": db}))
        except Exception as e:
            R["errors"].append(f"tf_raw {key}/{b}: {e!r}"[:200])
        out["bands"][b] = rec
        save()
    return out


# ---- JIT warm-up (excluded from every metric) and per-position acceptance
def warmup(port, n, secs=None):
    """Multi-stream 2k decode, ramped 1 -> n streams, excluded from every metric: compiles the kernels RC2 saw compiled
    during inference (_fill_num_accepted_kernel, _causal_conv1d_update_kernel, _gdn_recoverssm_verify_kernel,
    _ple_conv_kernel, _ple_conv_writeback_kernel) before any measured window."""
    secs = secs or (45.0 if not FAKE else 4.0)
    srv = next((x for x in list(L.LIVE) if x.port == port), None)
    j0 = jit_kernels(srv.log()) if srv else []
    ss, t0 = [], time.time()
    try:
        for i in range(n):
            g = L.Stream(port, PROMPTS["2k"][i % len(PROMPTS["2k"])], 4096, temperature=TEMP, seed=900 + i, ignore_eos=True)
            g.start()
            ss.append(g)
            time.sleep(min(3.0, secs / max(1, 2 * n)))
        while time.time() - t0 < secs:
            time.sleep(1)
    finally:
        for g in ss:
            g.stop()
        for g in ss:
            g.join(10)
    L.wait_idle(port, 60)
    j1 = jit_kernels(srv.log()) if srv else []
    return {"streams": n, "sec": round(time.time() - t0, 1), "jit_before": j0, "jit_compiled_in_warmup": j1[len(j0):],
            "errors": sorted({g.err for g in ss if g.err})[:3]}


def accept_pos(port):
    try:
        txt = L.U.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15).read().decode("utf-8", "replace")
    except Exception:
        return None
    pos = {}
    for m in re.finditer(r"^vllm:spec_decode_num_accepted_tokens_per_pos_total\{([^}]*)\}\s+(\S+)", txt, re.M):
        p = re.search(r'position="(\d+)"', m.group(1))
        if p:
            pos[int(p.group(1))] = pos.get(int(p.group(1)), 0.0) + float(m.group(2))
    return {"pos": pos, "drafts": L.prom(port).get(L.DRF, 0.0)}


def _prefill_done(r, n):
    tail = ((r.get("admission") or {}).get("tail") or [])
    t = tail[-1] if tail else None
    return bool(t) and t[1] == n and t[2] == 0 and t[3] >= n


def _req_lines():
    try:
        with open(REQ_LOG) as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def _ttfts(port, start_line):
    out = []
    try:
        with open(REQ_LOG) as fh:
            for i, l in enumerate(fh):
                if i >= start_line:
                    r = json.loads(l)
                    if r.get("port") == port and r.get("ttft") is not None:
                        out.append(r["ttft"])
    except Exception:
        pass
    return out


def _counters(port):
    try:
        m = L.prom(port)
        return m.get(L.PROMPT, 0.0), m.get("vllm:prefix_cache_hits_total", 0.0)
    except Exception:
        return None, None


def window(port, s, band, n, max_new=None, secs=None):
    """One steady window, timed only after every stream finished admission + prefill (the lab's settle needs all running
    streams to have emitted and be gaining, then an explicit check of the last admission sample; Merlin's cost-aware
    admission may hold a cold miss up to 120 s, inside the lab's >= 420 s admission limit); a window with prefill inside is
    retried once. Recorded: tick_ms (one engine step = one micro-batch of C/2 requests under PP2) AND cycle_ms (= PP x
    tick, the per-request advance time), per-request TTFT, uncached prefill tok/s (prompt minus prefix-hit tokens),
    retained (idle) blocks before the window, per-position acceptance."""
    retained_before = None
    if L.wait_idle(port, 60):
        retained_before = kvu(port)
    def run():
        if max_new is None:                 # RC2 / E23 levels(): MAX_NEW[band], WIN, the budget / KV-bound guards
            return levels(port, band, [n], server=s, temperature=TEMP, prefill_tok_s=PREFILL["tok_s"]).get(f"{band}@{n}") or {}
        return L.steady(port, PROMPTS[band], n, secs=secs or WIN, max_new=max_new, temperature=TEMP, prefill_tok_s=PREFILL["tok_s"])
    a0, c0, l0 = accept_pos(port), _counters(port), _req_lines()
    r = run()
    if r.get("status") == "INCONCLUSIVE_PREFILL_IN_WINDOW" and due(WIN + 120 + n * BANDS[band] / max(500.0, PREFILL["tok_s"])):
        c0, l0 = _counters(port), _req_lines()
        r = dict(run(), retried_after="INCONCLUSIVE_PREFILL_IN_WINDOW")
    a1, c1 = accept_pos(port), _counters(port)
    r["prefill_done_before_window"] = _prefill_done(r, n)
    r["running_eq_level"] = r.get("running_min") == r.get("running_max") == n
    if r.get("status") == "STEADY" and not (r["running_eq_level"] and r["prefill_done_before_window"]):
        r["status"] = "INCONCLUSIVE_RUNNING_NE_LEVEL" if not r["running_eq_level"] else "INCONCLUSIVE_PREFILL_NOT_DONE"
    if r.get("step_ms"):
        r["tick_ms"] = r["step_ms"]
        r["cycle_ms"] = round(PP * r["step_ms"], 1)
        r["requests_per_tick"] = round(n / PP, 2)
        r["microbatches_per_s"] = round(1000.0 / r["step_ms"], 2)
        r["wall_clock_tok_s"] = r.get("tok_s")
        r["tick_note"] = ("tick_ms = step_ms = one PP tick (one completed micro-batch of C/2 requests); cycle_ms = 2 ticks = the "
                          "per-request advance time; microbatches_per_s = completed micro-batches per second; wall_clock_tok_s = "
                          "generated tokens / window seconds")
    adm = (r.get("admission") or {}).get("sec")
    if None not in (c0[0], c1[0]) and adm:
        unc = (c1[0] - c0[0]) - (c1[1] - c0[1])
        r["prefill_uncached_tokens"] = int(unc)
        r["prefill_tok_s_uncached"] = round(unc / adm, 1)
        r["prefill_note"] = "admission.prefill_tok_s counts cached prompt tokens; prefill_tok_s_uncached does not (incl. a retry)"
    r["ttft_per_request"] = _ttfts(port, l0)
    if retained_before is not None:
        r["retained_usage_before"] = retained_before
    if a0 and a1 and a1["drafts"] > a0["drafts"]:
        r["accept_rate_per_position_incl_admission"] = {p: round((a1["pos"].get(p, 0) - a0["pos"].get(p, 0)) / (a1["drafts"] - a0["drafts"]), 3)
                                                        for p in sorted(a1["pos"])}
    return r


def nb1_ladder(port, s, o, extra=()):
    caps = o.get("caps") or {}
    seqs = caps.get("max_num_seqs")
    cap = residency_capacity(o)                # R2: the validated F (fit, else this boot's max-concurrency line), never an invalid fit
    # R2 audit N1-2 / N1-4: the same rule as the probes: F is None unless the capacity is certified (pool MATCH, no
    # fit / line conflict), so an uncertified boot adds no capacity-sized level (the fixed 8k@3 / 8k@8 / 40k@5 still run)
    P, F, B = cap["pool_blocks"], cap["F"], caps.get("block_size") or BLOCK
    plan = [("8k", 3), ("8k", 8), ("40k", 5)] + list(extra)
    hi = {}
    for band, base in (("8k", 8), ("40k", 5)):
        nf = n_fit(P, F, B, band, 0, seqs, 24 if band == "8k" else 8)
        hi[band] = nf
        if nf and nf > base:
            plan.append((band, nf))
    out = {"_plan": [f"{b}@{n}" for b, n in plan], "_fit": hi, "_seqs_cap": seqs, "_F": F, "_F_source": cap.get("F_source"),
           "_capacity_status": cap.get("status"), "_F_check": cap.get("fit_check"),          # R2 audit N1-2 / N1-4
           "_capacity_why": cap.get("why")}
    for band, n in plan:
        k = f"{band}@{n}"
        if seqs and n > seqs:
            out[k] = {"status": "SKIPPED_ABOVE_SERVER_CAP", "max_num_seqs": seqs}
            continue
        if not due(WIN + 120 + n * BANDS[band] / max(500.0, PREFILL["tok_s"])):
            out[k] = {"status": "SKIPPED_BOOT_BUDGET"}
            continue
        out[k] = window(port, s, band, n)
    return out


# ---- prefix continuation (2-turn replay) and after an abort
def _prefix_counts(port):
    try:
        m = L.prom(port)
        return {k: m.get(k) for k in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", L.PROMPT)}
    except Exception:
        return {}


def _reuse(turn1_len, reply_len, turn2_len, cached, B):
    exact = turn1_len + reply_len
    return {"turn1_prompt": turn1_len, "reply": reply_len, "turn2_prompt": turn2_len, "cached_turn2": cached,
            "token_exact_max": exact, "unit64_floor": exact // 64 * 64, "block_floor": exact // B * B,
            "prompt_only_block_floor": turn1_len // B * B,
            "reuse_pct_of_turn2": round(100 * cached / turn2_len, 1) if isinstance(cached, int) and turn2_len else None,
            "reuse_pct_of_token_exact": round(100 * cached / exact, 1) if isinstance(cached, int) and exact else None}


def prefix_tests(port, B):
    out = {}
    for band in ("8k", "40k"):
        if not due(240):
            out[band] = {"status": "SKIPPED_BOOT_BUDGET"}
            continue
        t1 = PROMPTS[band][V_PREFIX[band]]
        c0 = _prefix_counts(port)
        r1 = nb1_complete(port, t1, 512, temperature=TEMP, seed=31)
        if r1["error"] or not r1["ids"]:
            out[band] = {"status": "ERROR", "error": r1["error"]}
            continue
        t2 = t1 + r1["ids"] + SUFFIX_IDS
        r2 = nb1_complete(port, t2, 16, temperature=TEMP, seed=32)
        rs = nb1_complete(port, t1, 1, temperature=0.0)          # control: the same turn-1 prompt again
        out[band] = {"status": "MEASURED" if not r2["error"] else "ERROR", **_reuse(len(t1), len(r1["ids"]), len(t2), r2["cached"], B),
                     "cached_turn1": r1["cached"], "cached_resend_turn1": rs["cached"], "errors": [r2["error"], rs["error"]],
                     "counters_before": c0, "counters_after": _prefix_counts(port)}
    if due(240):
        t1 = PROMPTS["8k"][V_ABORT]
        g = L.Stream(port, t1, 2048, temperature=TEMP, seed=33)
        g.start()
        t0 = time.time()
        while time.time() - t0 < 300 and len(g.tokens) < 256 and g.is_alive():
            time.sleep(0.5)
        g.stop()
        g.join(10)
        part = list(g.tokens)
        time.sleep(2)
        r2 = nb1_complete(port, t1 + part + SUFFIX_IDS, 16, temperature=TEMP, seed=34)
        rs = nb1_complete(port, t1, 1, temperature=0.0)
        out["abort_8k"] = {"status": "MEASURED" if part and not r2["error"] else "INCONCLUSIVE",
                           **_reuse(len(t1), len(part), len(t1) + len(part) + len(SUFFIX_IDS), r2["cached"], B),
                           "aborted_after_tokens": len(part), "cached_resend_turn1": rs["cached"],
                           "errors": [g.err, r2["error"], rs["error"]]}
    else:
        out["abort_8k"] = {"status": "SKIPPED_BOOT_BUDGET"}
    return out


# ---- PP2 trace window: every rank's trace, stage overlap, idle, verify rows per micro-batch
SCOPE_ROWS = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")


def _merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _clip(iv, lo, hi):
    return [[max(a, lo), min(b, hi)] for a, b in iv if b > lo and a < hi]


def _inter(x, y):
    i = j = 0
    tot = 0.0
    while i < len(x) and j < len(y):
        lo, hi = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if hi > lo:
            tot += hi - lo
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return tot


def microbatch_costs(ev, scopes):
    """Per execute_context scope (one micro-batch on this rank): the GPU kernels it launched (eager launches and graph
    launches inside the scope on its thread, joined to kernels by correlation id), summed per category; then grouped by
    verify tokens per micro-batch. Answers 'MoE ms per micro-batch vs tokens per micro-batch' (both PP micro-batches
    re-read their stage's experts). The draft / vocab heads are cuBLAS GEMMs: inside gemm_other, not separable by name."""
    rt = {}
    for e in ev:
        if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("cuda_runtime", "cuda_driver"):
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                rt.setdefault(e.get("tid"), []).append((e.get("ts", 0), c))
    kern = {}
    for e in ev:
        if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_kernel"):
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                kern.setdefault(c, []).append((L._cat(e.get("name", "")), float(e.get("dur", 0.0)), float(e.get("ts", 0.0))))
    names = {}
    for e in ev:
        if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("cuda_runtime", "cuda_driver"):
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                names[c] = str(e.get("name", ""))
    per = []
    for e, nctx, nreq, ntok in scopes[:600]:
        lo, hi, tid = e.get("ts", 0), e.get("ts", 0) + e.get("dur", 0), e.get("tid")
        cats, gl, kl, ivl = {}, 0, 0, []
        for ts, c in rt.get(tid, []):
            if lo <= ts <= hi:
                nm = names.get(c, "")
                gl += "GraphLaunch" in nm
                kl += bool(re.search(r"LaunchKernel|cuLaunchKernel", nm))
                for k, d, kts in kern.get(c, []):
                    cats[k] = cats.get(k, 0.0) + d
                    ivl.append([kts, kts + d])
        mi = _merge(ivl)
        cap = next((x for x in sorted(CAPTURE_SIZES or []) if x >= ntok), None)
        per.append({"ctx_tokens": nctx, "gen_requests": nreq, "gen_tokens": ntok, "scope_ms": round(e.get("dur", 0) / 1000, 3),
                    "graph_launches": gl, "eager_kernel_launches": kl, "resolved_capture_size": cap,
                    "gpu_wall_ms": round((mi[-1][1] - mi[0][0]) / 1000, 3) if mi else 0.0,
                    "gpu_busy_union_ms": round(sum(b - a for a, b in mi) / 1000, 3),
                    "kernel_ms": {k: round(v / 1000, 3) for k, v in sorted(cats.items())}})
    by = {}
    for x in per:
        if x["ctx_tokens"] == 0:
            by.setdefault(x["gen_tokens"], []).append(x)
    agg = {}
    for ntok, xs in sorted(by.items()):
        keys = sorted({k for x in xs for k in x["kernel_ms"]})
        agg[ntok] = {"microbatches": len(xs), "scope_ms_mean": round(statistics.mean(x["scope_ms"] for x in xs), 3),
                     "kernel_ms_mean": {k: round(statistics.mean(x["kernel_ms"].get(k, 0.0) for x in xs), 3) for k in keys},
                     "kernel_ms_sum_mean": round(statistics.mean(sum(x["kernel_ms"].values()) for x in xs), 3),
                     "gpu_wall_ms_mean": round(statistics.mean(x["gpu_wall_ms"] for x in xs), 3),
                     "gpu_busy_union_ms_mean": round(statistics.mean(x["gpu_busy_union_ms"] for x in xs), 3),
                     "resolved_capture_size": xs[0]["resolved_capture_size"],
                     "graph_launches_mean": round(statistics.mean(x["graph_launches"] for x in xs), 2),
                     "eager_fallback_scopes": sum(1 for x in xs if x["graph_launches"] == 0)}
    return {"decode_only_by_verify_tokens": agg, "scopes": len(per), "examples": per[:6],
            "distinct_experts": "not reachable from the trace (router top-k ids need an engine hook: NB3 / NB4 item)",
            "note": "kernel time sums overlapping streams; nccl includes rank-skew waiting; nccl_sendrecv = the PP hand-off"}


def pp2_trace(pdir, tp=2):
    files = sorted(Path(pdir).rglob("*.pt.trace.json*"))
    if not files:
        return {"error": "NO_TRACE_FILE"}
    per, ivs = {}, {}
    for k, f in enumerate(files[:8]):
        try:
            with (gzip.open if f.name.endswith(".gz") else open)(f, "rt", encoding="utf-8") as fh:
                tr = json.load(fh)
        except Exception as e:
            per[f"file{k}"] = {"file": f.name, "error": repr(e)[:200]}
            continue
        di = tr.get("distributedInfo") or {}
        m = re.search(r"rank[-_]?(\d+)", f.name)
        rank = di.get("rank") if isinstance(di.get("rank"), int) else (int(m.group(1)) if m else k)
        base = (tr.get("baseTimeNanoseconds") or 0) / 1000.0
        ev = tr.get("traceEvents") or []
        iv = _merge([[e["ts"] + base, e["ts"] + base + e.get("dur", 0)] for e in ev if e.get("ph") == "X"
                     and str(e.get("cat", "")).lower() in ("kernel", "gpu_kernel") and "ts" in e])
        ivs[rank] = iv
        busy = sum(b - a for a, b in iv)
        span = (iv[-1][1] - iv[0][0]) if iv else 0.0
        gaps = [iv[i + 1][0] - iv[i][1] for i in range(len(iv) - 1)]
        rows, reqs, ctx = Counter(), Counter(), Counter()
        scopes = []
        for e in ev:
            if e.get("ph") == "X" and str(e.get("cat", "")) == "user_annotation":
                mm = SCOPE_ROWS.match(str(e.get("name", "")))
                if mm:
                    rows[int(mm.group(4))] += 1
                    reqs[int(mm.group(3))] += 1
                    ctx[int(mm.group(2))] += 1
                    scopes.append((e, int(mm.group(2)), int(mm.group(3)), int(mm.group(4))))
        microbatch = microbatch_costs(ev, scopes)
        one = TMP / f"nb1_trace_rank{rank}"
        shutil.rmtree(one, ignore_errors=True)
        one.mkdir(parents=True)
        shutil.copyfile(f, one / f.name)
        try:
            summ = L.trace_summary(one)
        except Exception as e:
            summ = {"error": repr(e)[:200]}
        shutil.rmtree(one, ignore_errors=True)
        per[rank] = {"file": f.name, "stage": rank // tp, "clock": "baseTimeNanoseconds" if base else "raw_ts",
                     "kernel_busy_ms": round(busy / 1000, 1), "span_ms": round(span / 1000, 1),
                     "idle_frac": round(1 - busy / span, 3) if span else None,
                     "gaps_over_1ms": sum(1 for g in gaps if g > 1000), "gap_ms_total": round(sum(g for g in gaps if g > 0) / 1000, 1),
                     "verify_rows_per_microbatch": dict(sorted(rows.items())), "gen_requests_per_microbatch": dict(sorted(reqs.items())),
                     "ctx_tokens_per_microbatch": dict(sorted(ctx.items())), "microbatch_costs": microbatch, "summary": summ}
    out = {"ranks": per, "files": len(files)}
    try:                                   # stage-imbalance test (RC2 hypothesis: heavy ~30 ms, light ~14 ms per tick)
        st = {}
        for r in (0, tp):
            agg = (((per.get(r) or {}).get("microbatch_costs") or {}).get("decode_only_by_verify_tokens") or {})
            if agg:
                k = max(agg, key=lambda x: agg[x]["microbatches"])
                st[f"stage{r // tp}"] = {"verify_tokens": k, "critical_path_ms_per_microbatch": agg[k]["gpu_wall_ms_mean"],
                                         "gpu_busy_union_ms": agg[k]["gpu_busy_union_ms_mean"],
                                         "summed_kernel_ms": agg[k]["kernel_ms_sum_mean"], "scope_ms": agg[k]["scope_ms_mean"]}
        if len(st) == 2:
            hv = max(st, key=lambda x: st[x]["critical_path_ms_per_microbatch"])
            lt = min(st, key=lambda x: st[x]["critical_path_ms_per_microbatch"])
            st["binding_stage"] = hv
            st["heavy_stage"] = hv
            st["tick_critical_path_ms"] = st[hv]["critical_path_ms_per_microbatch"]
            st["light_stage_idle_ms_per_tick_est"] = round(st[hv]["critical_path_ms_per_microbatch"] - st[lt]["gpu_busy_union_ms"], 2)
            st["note"] = ("critical path = the GPU wall span of a micro-batch on the binding stage (timeline); summed_kernel_ms "
                          "adds overlapping streams and is reported apart; PP0 and PP1 run concurrently, their times are never "
                          "added; partition 25/23 unchanged in NB1")
        out["stage_imbalance"] = st
    except Exception as e:
        out["stage_imbalance"] = {"error": repr(e)[:200]}
    a, b = ivs.get(0), ivs.get(tp)
    if a and b:
        lo, hi = max(a[0][0], b[0][0]), min(a[-1][1], b[-1][1])
        if hi > lo:
            ca, cb = _clip(a, lo, hi), _clip(b, lo, hi)
            ba, bb = sum(y - x for x, y in ca), sum(y - x for x, y in cb)
            both = _inter(ca, cb)
            frac = both / min(ba, bb) if min(ba, bb) else None
            out["stage_overlap"] = {"window_ms": round((hi - lo) / 1000, 1), "stage0_busy_frac": round(ba / (hi - lo), 3),
                                    "stage1_busy_frac": round(bb / (hi - lo), 3), "both_busy_ms": round(both / 1000, 1),
                                    "overlap_frac_of_less_busy": round(frac, 3) if frac is not None else None,
                                    "verdict": None if frac is None else ("OVERLAPPED" if frac >= 0.3 else "SERIAL" if frac < 0.1 else "PARTIAL"),
                                    "aligned": all((per.get(r) or {}).get("clock") == "baseTimeNanoseconds" for r in (0, tp))}
        else:
            out["stage_overlap"] = {"verdict": "NO_COMMON_WINDOW"}
    else:
        out["stage_overlap"] = {"verdict": "MISSING_STAGE_TRACE", "ranks": sorted(ivs)}
    return out


CAPTURE_SIZES = None
ROUTING_FILE = OUT / "nb1_routing_ids.npz"   # NB3 reads it (dataset foysalemonshanto/e25-nb1-routing-ids after a run)


def merge_routing(stop, tp=2, max_steps=200):
    """The NB3 schema (addendum 9): nb1_routing_ids.npz with ids [total_tokens, 10] int and per-record layer, stage,
    mb_size (token rows of the micro-batch = requests x (1 + K)), step, offset, mb_index (0 / 1: the PP2 micro-batch of
    that step). One record = one MoE layer call for one micro-batch in one decode step. TP rank 0 of each stage only (the
    TP partner routes the same tokens). numpy is required (the workers write npz parts; the Kaggle image has numpy)."""
    parts = []
    for res in ((stop or {}).get("results") or []):
        if isinstance(res, dict) and res.get("path") and res.get("rank") is not None and res["rank"] % tp == 0:
            parts.append((res["rank"] // tp, res["path"]))
    if not parts:
        return {"written": False, "why": "no decode-sized router calls ran in Python in the window (FULL-graph replay) or "
                                        "the capture failed: " + json.dumps([(r or {}).get("error") for r in ((stop or {}).get("results") or [])
                                                                             if isinstance(r, dict)])[:300]}
    recs = []
    try:
        import numpy as np
        for stage, path in sorted(parts):
            z = np.load(path)
            for i in range(len(z["layer"])):
                ps = int(z["pass_index"][i])
                if ps // 2 >= max_steps:
                    continue
                o, n = int(z["offset"][i]), int(z["mb_size"][i])
                recs.append((int(z["layer"][i]), stage, n, ps // 2, ps % 2, z["ids"][o:o + n]))
        if not recs:
            return {"written": False, "why": "part files empty"}
        offs, o_ = [], 0
        for r in recs:
            offs.append(o_)
            o_ += r[2]
        ids = np.concatenate([r[5] for r in recs], axis=0).astype(np.int32)
        np.savez_compressed(ROUTING_FILE, ids=ids, layer=np.array([r[0] for r in recs], dtype=np.int32),
                            stage=np.array([r[1] for r in recs], dtype=np.int32), mb_size=np.array([r[2] for r in recs], dtype=np.int32),
                            step=np.array([r[3] for r in recs], dtype=np.int32), offset=np.array(offs, dtype=np.int64),
                            mb_index=np.array([r[4] for r in recs], dtype=np.int32))
        return {"written": True, "file": str(ROUTING_FILE), "records": len(recs), "total_tokens": int(ids.shape[0]),
                "top_k": int(ids.shape[1]) if ids.ndim == 2 else None, "steps": len({(r[1], r[3]) for r in recs}),
                "ids_in_range": bool(ids.size == 0 or (ids.min() >= 0 and ids.max() < 512))}
    except ImportError:
        return {"written": False, "why": "numpy unavailable in the notebook kernel"}
    except Exception as e:
        return {"written": False, "error": f"{type(e).__name__}: {e}"[:300]}


def pp2_profile(port, caps=None):
    global CAPTURE_SIZES
    CAPTURE_SIZES = (caps or {}).get("capture_sizes")
    shutil.rmtree(PROF_DIR, ignore_errors=True)
    PROF_DIR.mkdir(parents=True, exist_ok=True)
    rt0 = rpc(port, "nb1_routing", "start", "200")
    rec = L.profiled_load(port, PROMPTS["8k"], 8, PROF_DIR, secs=6.0, max_new=MAX_NEW["8k"], spec=True,
                          prefill_tok_s=PREFILL["tok_s"])
    stop = rpc(port, "nb1_routing", "stop", str(TMP / "nb1_routing_part"))
    rec["routing_ids"] = {"start": rt0, "stop": stop, "merged": merge_routing(stop),
                          "note": "router top-k ids of the steps that ran in Python inside this window (prefill chunks, "
                                  "uncaptured sizes); FULL-graph decode replays are not visible to a Python hook: decode "
                                  "routing needs --enable-return-routed-experts in its own boot (NB3 / NB4 item)"}
    g = L.gpu_mem()
    if g:
        rec["gpu_mem_peak_mib"] = [x[1] for x in g]
    try:
        rec["pp2"] = pp2_trace(PROF_DIR)
    except Exception as e:
        rec["pp2"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    kept = TRACES / "profile_b1"
    kept.mkdir(exist_ok=True)
    for f in sorted(PROF_DIR.rglob("*.pt.trace.json*")):
        if f.stat().st_size < 80e6:
            shutil.copy2(f, kept / f.name)
    shutil.rmtree(PROF_DIR, ignore_errors=True)
    return rec


def quality_gate(o, ref):
    """RC2's fp8_gate with the reference as a parameter (B1 vs RC2's recorded FP8 rows, B2 / B3 vs B1)."""
    vs = ((o.get("oracle") or {}).get("vs_" + ref) or {}).get("status")
    if (o.get("sampled") or {}).get("ok") is not True:
        return "FAIL_SAMPLED"
    if (o.get("probes") or {}).get("ok") is not True and not FAKE:
        return "FAIL_PROBES"
    orc = o.get("oracle") or {}
    if any(orc.get("errors") or []) or any(orc.get("garbage") or []) or vs == "FAIL_HIGH_MARGIN":
        return "FAIL"
    if orc.get("complete") is True and vs in CLEAN:
        return "PASS"
    return "INCONCLUSIVE"


def flag_map(argv):
    m, a, i = {}, [str(x) for x in argv], 0
    while i < len(a):
        if a[i].startswith("--"):
            k, v = a[i], True
            if i + 1 < len(a) and not a[i + 1].startswith("--"):
                v = a[i + 1]
                i += 1
            m[k] = v
        i += 1
    return m


def config_diff(argv, env, kv_bytes):
    """Effective flags (last occurrence wins) and env vs RC2's real FP8 boot."""
    ref, new = flag_map(RC2_FP8["argv_tail"]), flag_map(argv)
    skip = {"--port", "--served-model-name", "--host"}
    flags = {k: [ref.get(k), new.get(k)] for k in sorted(set(ref) | set(new)) if k not in skip and ref.get(k) != new.get(k)}
    keys = set(RC2_FP8["env"]) | {k for k in env if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES", "MERLIN_"))}
    envd = {k: [RC2_FP8["env"].get(k), env.get(k)] for k in sorted(keys)
            if k not in ("VLLM_PORT", "MASTER_PORT") and RC2_FP8["env"].get(k) != env.get(k)}
    return {"flags": flags, "env": envd, "kv_bytes": [RC2_FP8["kv_bytes"], kv_bytes]}


def boot_facts(s, o, name):
    t = s.log() or ""
    o["caps"] = server_caps(t, s.argv)
    o["pool"] = pool_facts(t)
    o["graphs"] = graph_facts(t)
    o["nccl"] = nccl_transport(t)
    o["workers"] = rpc_cache_info(s.port)
    o["host_after_ready"] = cgroup_snapshot(name + "_after_ready")
    env = dict((R["setup"].get("merlin_plans") or {}).get(name, {}).get("env_keys") or {})
    env.update(NB1_ENV.get(name) or {})
    o["config_diff"] = config_diff(s.argv, env, NB1_KV.get(name))
    return o
save()
'''

CELL7 = r'''# ==================================================================== CELL 7 — E25 NB1: the boots, the scorecard, the plan (NEW code)
def _cfg(sec, name, kv, port, kv_bytes, extra, env, cap_min, est_min, levers):
    return {"sec": sec, "name": name, "kv": kv, "port": port, "kv_bytes": kv_bytes, "extra": list(extra), "env": dict(env),
            "cap_min": cap_min, "est_min": est_min, "levers": levers}


FP8_EXTRA = EXT_FLAG + ["--language-model-only", "--profiler-config", PROFILER_CFG]
B2_EXTRA = FP8_EXTRA + ["--mamba-ssm-cache-dtype", "bfloat16", "--max-num-seqs", str(SEQS_HI), "--compilation-config", COMPILE_HI]
B2_ENV = {**DIAG_ENV, "MERLIN_SHARE_DRAFT_WORKSPACE": "1", **COMMIT_GRAPH_ENV}
BOOTS = {"B1": _cfg("B1", "nb1_b1_fp8", "fp8", 8001, KV_FP8, FP8_EXTRA, {**DIAG_ENV, **NCCL_DIAG}, 70, 92,
                    ["D9 vLLM #60490 (source patch)", "C4 --language-model-only", "--profiler-config (idle unless profiling)",
                     "diagnostics: dev mode + read-only worker extension + NCCL_DEBUG INIT,NET (this boot only)"]),
         "B2": _cfg("B2", "nb1_b2_fp8", "fp8", 8002, KV_FP8, B2_EXTRA, B2_ENV, 50, 68,
                    ["B1 (without NCCL debug)", "C3 --mamba-ssm-cache-dtype bfloat16", "MERLIN_SHARE_DRAFT_WORKSPACE=1",
                     f"--max-num-seqs {SEQS_HI} + captures to {CAPTURE_HI[-1]}",
                     "VLLM_RECOVERSSM_COMMIT_GRAPH=graph at boot, eager for every standard window (runtime A/B at the end)",
                     "combined configuration"]),
         "B3": _cfg("B3", "nb1_b3_fp8_pin", "fp8", 8003, KV_FP8 + KV_STEP, B2_EXTRA, {**DIAG_ENV, "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, 36, 54,
                    ["B2 without the commit-graph boot env", "C5 KV_BYTES + 1.0 / 0.75 / 0.5 GiB per card (descending ladder, "
                     "the largest predicted safe step first; the step is set at run time by pin_rule)"])}
for _c in BOOTS.values():
    NB1_EXTRA[_c["name"]], NB1_ENV[_c["name"]], NB1_KV[_c["name"]] = _c["extra"], _c["env"], _c["kv_bytes"]
NB1["boot_plan"] = {k: {"name": c["name"], "levers": c["levers"], "argv_added": c["extra"], "env_added": c["env"],
                        "kv_bytes": [RC2_FP8["kv_bytes"], c["kv_bytes"]], "patch_60490": True, "cap_min": c["cap_min"]}
                    for k, c in BOOTS.items()}
NB1["not_measured"] = {"W6_chunk": "CHUNK (Merlin launch.py _int_knob 'CHUNK' -> --max-num-batched-tokens) is a startup "
                                   "flag; vLLM has no runtime override, so no separate window here (left for a later boot)",
                       "fp8_quality_isolation": "NB0 (BF16 / FP8 x MTP off / K3 teacher-forced, raw arrays): production use of "
                                                "any NB1 FP8 number depends on NB0's gate",
                       "C3_capacity": R["capacity_plan"]["c3_note"]}
NB1["attribution"] = ("C4 = the drop of 'Model loading took X GiB' per rank vs RC2 FP8 (weights only; the patch does not touch "
                      "loading); D9 = the change of free VRAM after ready per rank vs RC2 FP8 minus the C4 weight drop (the patch "
                      "touches its scratch in the profiling run, so the allocator holds it from ready on); the patch's own "
                      "forward test = B1's greedy oracle vs RC2's recorded FP8 rows. B2 is one combined configuration "
                      "(C3 + share + seqs 16) except the commit graph (runtime A/B inside B2); B3 = C5 alone vs B2.")


def d9_reserve_mib(seqs=8, k=3, mnbt=1024, max_len=MAX_LEN, block=BLOCK, compress=4, logits_mb=256):
    """What the #60490 code touches in the profiling run: max(prefill logits budget, decode logits) + the 1 MiB top-k."""
    cols = math.ceil(max_len / block) * math.ceil(block / compress)
    return round((max(logits_mb << 20, min(seqs * (k + 1), mnbt) * cols * 4) + (1 << 20)) / 2 ** 20, 1)


def pin_free_check():
    """B3: free MiB per rank after warm-up + the 90k admission (the lazy QSA workspace and graphs are allocated by then)."""
    g = L.gpu_mem()
    if not g and FAKE and os.environ.get("NB1_FAKE_PIN_FREE_MIB"):        # local harness only
        g = [[i, 23034 - int(os.environ["NB1_FAKE_PIN_FREE_MIB"]), 23034] for i in range(4)]
    if not g:
        return {"ok": bool(FAKE), "why": "no nvidia-smi reading" + (" (local harness)" if FAKE else ": fail closed")}
    free = [t_ - u for _, u, t_ in g]
    return {"ok": min(free) >= PIN_FREE_MIN_MIB, "free_mib_per_rank": free, "need_mib": PIN_FREE_MIN_MIB}


def body_fp8(cfg, ref, store=None, refs_extra=(), self_check=False, prefix=False, profile=False, commit=False, pin_check=False,
             alloc=(("8k", (1, 2, 4, 8)), ("40k", (1, 2, 4))), residency=RESIDENCY_BANDS):
    def body(ss):
        s = ss[0]
        p = s.port
        o = boot_facts(s, {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}, cfg["name"])
        o["scales"] = scales_dump()
        if commit:                                 # every standard window runs the RC2 commit path (eager)
            o["commit_mode_at_start"] = commit_mode(p, "eager")
        if short_gate_failed(o):
            o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s),
                     quality_gate="FAIL_SAMPLED" if (o["sampled"] or {}).get("ok") is not True else "FAIL_PROBES")
            return o
        if not PREFILL["measured"]:
            o["prefill"] = prefill_probe(p)
        if FAKE and ref not in ROWS:               # local harness only: the fake server's text is not RC2's
            o["fake_reference"] = oracle(p, key=ref)
        o["oracle"] = oracle(p, key=store, refs=(ref,) + tuple(refs_extra))
        o["quality_gate"] = quality_gate(o, ref)
        if o["quality_gate"] != "PASS":
            o["load_tests_skipped"] = f"oracle gate vs {ref}: {o['quality_gate']}"
            o["status"] = "NO_GO_QUALITY" if o["quality_gate"].startswith("FAIL") else "INCONCLUSIVE_QUALITY"
            o["census_end"] = kv_census(s)
            return o
        o["warmup"] = warmup(p, min(8, (o["caps"].get("max_num_seqs") or 8)))
        if self_check:
            o["fp8_self_check"] = fp8_self_check(s)
        o["block_layout"] = block_layout(o.get("workers"))
        if pin_check:                              # the 90k prefill first: the stop rule reads free VRAM after it
            o["admission_90k@5"] = admission_test(p, s)
            o["pin_check"] = pin_free_check()
            if not o["pin_check"]["ok"]:
                o.update(status="PIN_FREE_BELOW_0P5_GIB", census_end=kv_census(s))
                return o
        if alloc:
            o["kv"] = kv_accounting(p, o["caps"], alloc)
        if not pin_check:
            o["admission_90k@5"] = admission_test(p, s)
        if residency:
            o["max_safe_running_by_context"] = msr = residency_probes(p, s, o, bands=residency)
            NB1.setdefault("max_safe_running_by_context", {})[cfg["sec"]] = {      # E25 NB1 C-6: the boot's own P / F
                "status": msr.get("status"), "pool_blocks": msr.get("pool_blocks"), "F": msr.get("F"), "F_source": msr.get("F_source"),
                "pool_check": (msr.get("capacity") or {}).get("pool_check"), "why": msr.get("why"),
                "fit_check": (msr.get("capacity") or {}).get("fit_check"),                   # R2 audit N1-4 / N1-6
                "reason_codes": (msr.get("capacity") or {}).get("reason_codes"),
                "bands": {b: {"achieved": v.get("achieved"), "bound_at_measured_pool_F": v.get("bound_prompt_only"),
                              "bound_capped_by_seqs": v.get("bound_capped")} for b, v in msr["bands"].items()}}
        if prefix:
            o["prefix"] = prefix_tests(p, o["caps"].get("block_size") or BLOCK)
        o["steady"] = nb1_ladder(p, s, o)
        if commit:
            o["commit_graph_ab"] = commit_ab(p, s)
        if profile:
            o["profiler_tools"] = profiler_tools()
        if profile and due(420):
            o["profile"] = pp2_profile(p, o["caps"])
        o["census_end"] = kv_census(s)
        o["workers_end"] = rpc_cache_info(p)
        o["jit_after_warmup"] = jit_kernels(s.log())[len(o["warmup"]["jit_before"]) + len(o["warmup"]["jit_compiled_in_warmup"]):]
        return o
    return body


def run_boot(sec, body, need=()):
    c = BOOTS[sec]
    BOOT["tag"] = c["name"]
    mon = LoadMonitor(c["name"])
    mon.start()
    try:
        o = merlin_boot(c["name"], c["kv"], c["port"], BOOT_READY_SEC, watched(body, c["cap_min"] * 60, c["name"]), need=need)
    finally:
        mon.stop_ev.set()
    tail = (((o.get("servers") or {}).get(c["name"]) or {}).get("log_tail") or "")
    if (o.get("status") == "LAUNCH_FAILED" and "VLLM_RECOVERSSM_COMMIT_GRAPH" in NB1_ENV[c["name"]]
            and re.search(r"recoverssm|commit", tail, re.I) and left() > c["est_min"] * 60):
        NB1_ENV[c["name"]] = {k: v for k, v in NB1_ENV[c["name"]].items() if k != "VLLM_RECOVERSSM_COMMIT_GRAPH"}
        first = {k: o.get(k) for k in ("status", "tries", "boot_gate")}
        log("commit-graph boot failed:", c["name"], "-> one retry without VLLM_RECOVERSSM_COMMIT_GRAPH")
        o = merlin_boot(c["name"], c["kv"], c["port"], BOOT_READY_SEC, watched(body, c["cap_min"] * 60, c["name"]), need=need)
        o["commit_graph_boot_failed"] = first
    LIVE["out"] = o                            # R2: the full boot record, kept by step() whatever happens below
    try:                                       # R2: post-processing never discards the measurements (keep_after_error)
        o["load_monitor_last"] = mon.last
        why = str((o.get("boot_gate") or {}).get("why") or "")
        if o.get("status") == "GPU_NOT_EMPTY" and why.startswith(("HOST_RAM", "ENGINE_PROCS")):
            o["status"] = "HOST_RAM_GATE_FAILED" if why.startswith("HOST_RAM") else "ENGINE_PROCS_LEFT"
        o["planned"] = NB1["boot_plan"][sec]
        o["free_mib_per_card"] = free_per_card(o, c["name"])
        if o["free_mib_per_card"] is None and FAKE and os.environ.get("NB1_FAKE_FREE_MIB"):     # local harness only (no nvidia-smi)
            o["free_mib_per_card"] = [float(os.environ["NB1_FAKE_FREE_MIB"])] * 4
            o["free_mib_source"] = "NB1_FAKE_FREE_MIB (local harness only)"
        o["free_after_ready_mib"] = free_after_ready(o, c["name"])
    except Exception as e:
        keep_after_error(o, e)
    ensure_core(o)                             # R2 audit N1-3 / N1-7: postprocess_ok, core_measurements_complete, headroom
    return o


def boot_ok(o):
    """Clean boot: DONE / TF_OOM + CLEAN + PASS. R2 audit N1-7: TF_OOM still lets the dependents run, but the config is
    UNSAFE_HEADROOM (headroom_flag): never a recommended pin or a capacity winner."""
    return (isinstance(o, dict) and o.get("status") in ("DONE", "TF_OOM") and o.get("correctness") == "CLEAN"
            and o.get("quality_gate") == "PASS")


def headroom_flag(o):
    """R2 audit N1-7: a boot whose teacher-forced call died of OOM (TF_OOM, also when post-processing failed after it) has
    no safe memory headroom at that configuration."""
    o = o if isinstance(o, dict) else {}
    if "TF_OOM" in (o.get("status"), o.get("status_before_postprocess")) or o.get("headroom") == "UNSAFE_HEADROOM":
        return "UNSAFE_HEADROOM"
    return None


def free_valid(fr):
    """R2 audit N1-3: a valid per-rank free memory reading = 4 finite numbers in 0..24,576 MiB (one L4 card)."""
    return (isinstance(fr, list) and len(fr) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                                                          and 0 <= x <= 24576 for x in fr))


def core_check(o):
    """R2 audit N1-3: the core measurements a dependent boot builds on: the kv census (KV size line), the gauge pool, F
    (resolve_F: line / fit rule), free MiB per card (4 valid ranks) and the residency probes (MEASURED: pool MATCH + F)."""
    o = o if isinstance(o, dict) else {}
    kv = o.get("kv") if isinstance(o.get("kv"), dict) else {}
    P = kv.get("pool_blocks")
    cen = o.get("census") if isinstance(o.get("census"), dict) else {}
    msr = o.get("max_safe_running_by_context") if isinstance(o.get("max_safe_running_by_context"), dict) else {}
    try:
        F_ok = residency_capacity(o).get("F_resolved") is not None
    except Exception:
        F_ok = False
    parts = {"kv_census": bool(cen.get("kv_tokens")), "pool": isinstance(P, int) and not isinstance(P, bool) and P >= 1,
             "F": F_ok, "free_mib_per_card": free_valid(o.get("free_mib_per_card")), "residency": msr.get("status") == "MEASURED"}
    return {"complete": all(parts.values()), "parts": parts, "missing": [k for k, v in parts.items() if not v]}


def ensure_core(o):
    """R2 audit N1-3 / N1-7: every boot record carries postprocess_ok, core_measurements_complete (+ the parts) and the
    headroom flag; never raises, never removes a measurement."""
    if not isinstance(o, dict):
        return o
    o["postprocess_ok"] = "postprocess_error" not in o
    try:
        cc = core_check(o)
    except Exception as e:
        cc = {"complete": False, "parts": {}, "missing": [f"core_check error: {type(e).__name__}: {e}"[:200]]}
    o["core_measurements_complete"], o["core_measurements"] = cc["complete"], cc
    hf = headroom_flag(o)
    if hf:
        o["headroom"] = hf
    return o


def core_complete(o):
    """(complete, missing) of a boot record; a record without the fields (an exception path in step) is checked here."""
    if not isinstance(o, dict):
        return False, ["no record"]
    if not isinstance(o.get("core_measurements_complete"), bool):
        ensure_core(o)
    return o["core_measurements_complete"], (o.get("core_measurements") or {}).get("missing")


def boot1():
    ok, why = sha_gate(min(1500, left() - 900))
    R["setup"]["sha_gate"] = {"ok": ok, "why": why}
    if not ok:
        return {"status": "SKIPPED_SHA_GATE", "why": why}
    return run_boot("B1", body_fp8(BOOTS["B1"], "rc2_fp8", store="b1", refs_extra=("rc2_bf16",), self_check=True,
                                   prefix=True, profile=True), need=("rc2_fp8",))


def _rec(sec):
    st = R["sections"].get(sec) or {}
    r = st.get(BOOTS[sec]["name"])
    return r if isinstance(r, dict) else {}


def pin_rule(base="B2"):
    """The B3 pin ladder (addendum 5): from the smallest free MiB per rank measured while serving (after #60490, the lazy
    QSA workspace, graphs, warm-up and the 90k admission), the largest step of +1.0 / +0.75 / +0.5 GiB that is predicted
    to leave >= 0.5 GiB free; then each lower step in turn if the in-boot check fails (a boot OOM: merlin_boot already
    retries 0.25 GiB lower once, Merlin's rule).
    R2 audit N1-3: the reading must come from the boot the pin builds on: base B2 needs B2's own valid per-rank free
    memory (B1's is added when valid: the smaller per rank counts); B2's missing -> fail closed, never decided from B1
    alone. Base B1 (B2 not clean) uses B1's own reading."""
    srcs = ("B2", "B1") if base == "B2" else ("B1",)
    rd = {s_: _rec(s_).get("free_mib_per_card") for s_ in srcs}
    if FAKE and os.environ.get("NB1_FAKE_FREE_MIB") and not any(free_valid(v) for v in rd.values()):    # local harness only
        rd = {s_: [float(os.environ["NB1_FAKE_FREE_MIB"])] * 4 for s_ in srcs}
    if not free_valid(rd[base]):
        return {"ok": False, "base": base, "free_mib": rd,
                "why": f"free VRAM per card of {base} (the boot the pin builds on) not measured / invalid ({rd[base]!r}): fail closed"
                       + (", never decided from B1 alone" if base == "B2" else "")}
    fr = [v for v in rd.values() if free_valid(v)]
    mins = [min(x[i] for x in fr) for i in range(4)]
    steps = [g for g in PIN_STEPS_GIB if min(mins) - g * 1024 >= PIN_FREE_MIN_MIB]
    return {"ok": bool(steps), "base": base, "free_sources": [s_ for s_ in srcs if free_valid(rd[s_])],
            "min_free_mib_per_card": mins, "steps_gib": steps, "need_mib": PIN_FREE_MIN_MIB,
            "kv_bytes": [KV_FP8 + int(g * 2 ** 30) for g in steps], "blocks_per_gib_est": round(1024 / 15.75, 1),
            "why": (f"start +{steps[0]} GiB (predicted {min(mins) - steps[0] * 1024:.0f} MiB free on the tightest rank)" if steps
                    else f"no step leaves >= {PIN_FREE_MIN_MIB} MiB: min free {min(mins):.0f} MiB")}


def pin_ladder(base="B2"):
    """B3 boots, one per step of the pin rule, highest first; stop at the first clean boot whose in-boot check keeps
    >= 0.5 GiB free per rank. Every try keeps its own record (sections B3/nb1_b3_fp8_pin<GiB x 100>). `base` = the clean
    configuration the pin is added to: B2 (without its commit-graph boot env), or B1 (without its NCCL debug) when B2
    was not clean.
    R2 audit N1-7: a TF_OOM step is UNSAFE_HEADROOM: kept, never recommended; the ladder goes on to the next lower step.
    NB1["recommended_pin"] = the first clean, not-UNSAFE step with the in-boot check ok and correctness DIAGNOSTIC_PASS."""
    tries = []
    NB1["recommended_pin"] = None
    extra, env = (B2_EXTRA, {**DIAG_ENV, "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}) if base == "B2" else (FP8_EXTRA, dict(DIAG_ENV))
    ref = "b1"
    for g in NB1["pin_rule"]["steps_gib"]:
        nm = f"nb1_b3_fp8_pin{int(round(g * 100))}"
        BOOTS["B3"] = dict(BOOTS["B3"], name=nm, kv_bytes=KV_FP8 + int(g * 2 ** 30), pin_gib=g, extra=list(extra), env=dict(env), base=base)
        NB1_EXTRA[nm], NB1_ENV[nm], NB1_KV[nm] = BOOTS["B3"]["extra"], BOOTS["B3"]["env"], BOOTS["B3"]["kv_bytes"]
        NB1["boot_plan"]["B3"] = dict(NB1["boot_plan"]["B3"], name=nm, kv_bytes=[RC2_FP8["kv_bytes"], BOOTS["B3"]["kv_bytes"]])
        r = step("B3", nm, BOOTS["B3"]["est_min"],
                 lambda: run_boot("B3", body_fp8(BOOTS["B3"], ref, pin_check=True, alloc=(("8k", (1, 4, 8)), ("40k", (1, 2, 4)))),
                                  need=(ref,)))
        ensure_core(r)
        unsafe = headroom_flag(r)
        tries.append({"gib": g, "name": nm, "status": r.get("status"), "kv_bytes": r.get("kv_bytes"), "pin_check": r.get("pin_check"),
                      "headroom": unsafe, "correctness": _corr(r), "core_measurements_complete": r.get("core_measurements_complete")})
        if boot_ok(r) and not unsafe:
            if (r.get("pin_check") or {}).get("ok") is True and _corr(r) == "DIAGNOSTIC_PASS":
                NB1["recommended_pin"] = {"gib": g, "name": nm, "kv_bytes": r.get("kv_bytes"), "base": base,
                                          "label": "DIAGNOSTIC (NB1 capacity only; production use depends on NB0's FP8 gate)"}
            else:
                NB1["recommended_pin_why_none"] = f"{nm}: clean boot but pin_check {r.get('pin_check')} / correctness {_corr(r)}"
        scorecard_save()
        if (boot_ok(r) and not unsafe) or str(r.get("status", "")).startswith("SKIPPED") or r.get("status") == "PARTIAL_HARD_STOP":
            break
    NB1["pin_tries"] = tries
    save()


def skip(sec, why):
    R["sections"].setdefault(sec, {})[BOOTS[sec]["name"]] = {"status": "SKIPPED_DEPENDENCY", "why": why}
    save()


# ---- lever scorecard (PLAN v3.2 §4.1), rewritten after every boot
def _free(rows):
    return [r[2] - r[1] for r in rows] if rows else None


RC2_REF = {"pool_blocks": RC2_FP8["pool_blocks"], "F": RC2_FP8["F_rule"], "seqs": RC2_FP8["seqs"], "kv_tokens": RC2_FP8["kv_tokens"],
           "max_concurrency": RC2_FP8["max_concurrency"], "free_after_ready": _free(RC2_FP8["gpu_mem_after_ready"]),
           "free_mib": [r[2] - max(r[1], e[1]) for r, e in zip(RC2_FP8["gpu_mem_after_ready"], RC2_FP8["gpu_mem_end"])],
           "rank_load_gib": {int(k): v for k, v in RC2_FP8["rank_load_gib"].items()},
           "windows": RC2_FP8["windows"], "label": "RC2 FP8 (rc2_results.json; F = 6 from the RC2 analysis, not cold-measured)"}


def boot_metrics(sec):
    r = _rec(sec)
    if not r or str(r.get("status", "")).startswith("SKIPPED"):
        return None
    name = BOOTS[sec]["name"]
    srv = (r.get("servers") or {}).get(name) or {}
    kv = r.get("kv") or {}
    st = {k: v for k, v in (r.get("steady") or {}).items() if isinstance(v, dict) and "@" in k and not k.startswith("_")}
    win = {k: {"status": v.get("status"), "tok_s": v.get("tok_s"), "mal": v.get("mean_acceptance_length"),
               "tick_ms": v.get("tick_ms"), "cycle_ms": v.get("cycle_ms"), "running": [v.get("running_min"), v.get("running_max")]}
           for k, v in st.items()}
    best = {}
    for k, v in win.items():
        b, n = k.split("@")
        if v["status"] == "STEADY":
            best[b] = max(best.get(b, 0), int(n))
    adm = r.get("admission_90k@5") or {}
    try:                                       # R2 audit N1-2 / N1-4: only a certified capacity (pool MATCH, F rule) gives F
        cap = residency_capacity(r)
    except Exception as e:
        cap = {"status": "INCONCLUSIVE_CAPACITY_UNKNOWN", "why": [f"{type(e).__name__}: {e}"[:200]]}
    return {"status": r.get("status"), "correctness": r.get("correctness"), "quality_gate": r.get("quality_gate"),
            "status_before_postprocess": r.get("status_before_postprocess"), "postprocess_error": r.get("postprocess_error"),
            "postprocess_ok": r.get("postprocess_ok"), "core_measurements_complete": r.get("core_measurements_complete"),
            "core_missing": (r.get("core_measurements") or {}).get("missing"), "headroom": headroom_flag(r),
            "capacity_status": cap.get("status"), "capacity_why": cap.get("why"), "pool_check": cap.get("pool_check"),
            "fit_check": cap.get("fit_check"), "F_fit": (kv.get("fit") or {}).get("F"), "F_source": cap.get("F_source"),
            "kv_tokens": srv.get("kv_tokens") or (r.get("pool") or {}).get("kv_tokens"),
            "max_concurrency": srv.get("max_concurrency") or (r.get("pool") or {}).get("max_concurrency"),
            "pool_blocks": kv.get("pool_blocks"), "F": cap.get("F"), "seqs": (r.get("caps") or {}).get("max_num_seqs"),
            "free_mib": r.get("free_mib_per_card"), "free_after_ready": r.get("free_after_ready_mib"), "windows": win,
            "highest_steady": best, "sec": r.get("sec"),
            "admission_90k@5": {"status": adm.get("status"), "all_resident": adm.get("all_resident"),
                                "first_token_s": adm.get("first_token"), **{k: (adm.get("window") or {}).get(k) for k in ("tok_s", "tick_ms", "cycle_ms")}},
            "rank_load_gib": (r.get("pool") or {}).get("rank_load_gib"),
            "mamba_pad_pct": (r.get("pool") or {}).get("mamba_pad_pct"), "min_block_for_mamba": (r.get("pool") or {}).get("min_block_for_mamba"),
            "recurrent_tensor_dtypes": (r.get("workers") or {}).get("recurrent_tensor_dtypes"),
            "worker_num_gpu_blocks": (r.get("workers") or {}).get("num_gpu_blocks")}


def resident_at(m, L_ref=70000):
    if not m or m.get("pool_blocks") is None or m.get("F") is None:
        return None
    return min(m.get("seqs") or 10 ** 6, int(m["pool_blocks"] // (math.ceil(L_ref / BLOCK) + m["F"])))


def projection(old, new):
    w = ((new or {}).get("windows") or {}).get("8k@8") or {}
    if w.get("status") == "STEADY" and w.get("tok_s"):
        a, src = w.get("mal") or 3.154, "this boot's 8k@8 window"
        t = 8 * a / w["tok_s"]
    else:
        a, src = 3.154, "RC2 FP8 8k@8 (261.8 tok/s, mean acceptance length 3.154)"
        t = 8 * a / 261.8
    c0, c1 = resident_at(old), resident_at(new)
    inp = {"formula": "steady = C * a / t_cycle (PLAN §1); run = 0.9 * steady", "C_old_at_70k": c0, "C_new_at_70k": c1,
           "a_tokens_per_verify_cycle": a, "t_cycle_s": round(t, 4),
           "t_cycle_source": src + " (t_cycle = 8 * a / tok_s = 2 ticks under PP2)",
           "assumption": "t_cycle held at the 8-stream value; at more streams it rises (PLAN per-tick table), so this is an upper bound"}
    if c0 is None or c1 is None:
        return None, {**inp, "why": "resident streams not measurable (pool / F missing)"}
    return round(0.9 * (c1 - c0) * a / t, 1), inp


CORR_FAIL_STATUSES = ("ERROR", "BENCH_ERROR", "POSTPROCESS_ERROR")


def _corr(m, key="quality_gate"):
    """R2 audit N1-1: status-aware correctness label of a boot (record or boot_metrics). PASS is never a production
    label: the best NB1 label is DIAGNOSTIC_PASS (production use depends on NB0's FP8 gate).
      status ERROR / BENCH_ERROR / POSTPROCESS_ERROR / FAIL* / NO_GO* (or a FAIL* / NO_GO* gate / correctness) -> FAIL;
      status != DONE -> UNTESTED; a postprocess_error -> INCOMPLETE_DIAGNOSTIC;
      correctness != CLEAN or the gate != PASS -> UNTESTED; otherwise DIAGNOSTIC_PASS."""
    if not m:
        return "UNTESTED"
    st = str(m.get("status") or "")
    g = m.get(key) if key else None
    if (st in CORR_FAIL_STATUSES or st.startswith(("FAIL", "NO_GO"))
            or str(g or "").startswith(("FAIL", "NO_GO")) or str(m.get("correctness") or "").startswith(("FAIL", "NO_GO"))):
        return "FAIL"
    if st != "DONE":
        return "UNTESTED"
    if m.get("postprocess_error"):
        return "INCOMPLETE_DIAGNOSTIC"
    if m.get("correctness") != "CLEAN" or (key is not None and g != "PASS"):
        return "UNTESTED"
    return "DIAGNOSTIC_PASS"


def _corr_and(m, status, ok=("EXACT",), fail=("FAIL_HIGH_MARGIN",)):
    """A lever row's label from its own check (status) within its boot's label (_corr): DIAGNOSTIC_PASS only when both pass."""
    if status in fail:
        return "FAIL"
    b = _corr(m)
    if b != "DIAGNOSTIC_PASS":
        return b
    return "DIAGNOSTIC_PASS" if status in ok else "UNTESTED"


def capacity_winner():
    """R2 audit N1-7 / N1-1 / N1-2: the boot with the most resident streams at 70k among the boots whose correctness is
    DIAGNOSTIC_PASS, headroom not UNSAFE_HEADROOM, capacity CERTIFIED and core measurements complete (earlier boot on a tie).
    A diagnostic selection: production use depends on NB0's FP8 gate."""
    cands, excluded = [], {}
    for i, sec in enumerate(("B1", "B2", "B3")):
        m = boot_metrics(sec)
        if not m:
            excluded[sec] = ["not run"]
            continue
        why = []
        c = _corr(m)
        if c != "DIAGNOSTIC_PASS":
            why.append(f"correctness {c}")
        if m.get("headroom") == "UNSAFE_HEADROOM":
            why.append("UNSAFE_HEADROOM (TF_OOM)")
        if m.get("capacity_status") != "CERTIFIED":
            why.append(f"capacity {m.get('capacity_status')}: {m.get('capacity_why')}")
        if m.get("core_measurements_complete") is not True:
            why.append(f"core measurements incomplete: {m.get('core_missing')}")
        ra = resident_at(m)
        if ra is None:
            why.append("resident streams at 70k not computable")
        if why:
            excluded[sec] = why
        else:
            cands.append((ra, -i, sec))
    best = max(cands) if cands else None
    return {"winner": best[2] if best else None, "resident_streams_at_70k": best[0] if best else None,
            "candidates": {s_: r_ for r_, _, s_ in cands}, "excluded": excluded,
            "rule": "max resident streams at 70k over DIAGNOSTIC_PASS + not UNSAFE_HEADROOM + capacity CERTIFIED + core complete",
            "label": "DIAGNOSTIC (" + NB0_NOTE + ")"}


def _vdiff(a, b):
    return [y - x for x, y in zip(a, b)] if a and b and len(a) == len(b) else None


def _delta(old, new):
    if not old or not new:
        return None
    d = {"kv_tokens": [old.get("kv_tokens"), new.get("kv_tokens")], "pool_blocks": [old.get("pool_blocks"), new.get("pool_blocks")],
         "F": [old.get("F"), new.get("F")], "free_mib_per_card": [old.get("free_mib"), new.get("free_mib")],
         "free_after_ready_delta_mib": _vdiff(old.get("free_after_ready"), new.get("free_after_ready")),
         "resident_streams_at_70k": [resident_at(old), resident_at(new)], "highest_steady": [old.get("highest_steady"), new.get("highest_steady")],
         "admission_90k@5": [old.get("admission_90k@5"), new.get("admission_90k@5")]}
    d["steady_tok_s_matched"] = {k: [((old.get("windows") or {}).get(k) or {}).get("tok_s"), ((new.get("windows") or {}).get(k) or {}).get("tok_s")]
                                 for k in ("8k@3", "8k@8", "40k@5")}
    d["cycle_ms_matched"] = {k: [((old.get("windows") or {}).get(k) or {}).get("cycle_ms"), ((new.get("windows") or {}).get(k) or {}).get("cycle_ms")]
                             for k in ("8k@8", "40k@5")}
    return d


def _conf(m):
    if not m:
        return "none (not run)"
    n = sum(1 for v in (m.get("windows") or {}).values() if isinstance(v, dict) and v.get("status") == "STEADY")   # R2
    return f"LOW: {n} STEADY window(s) of {WIN:.0f} s, no repeat (window noise not measured; read deltas below ~5 % as noise)"


def _cost(*ms):
    secs = [m.get("sec") for m in ms if m and m.get("sec")]
    return round(sum(secs) * 4 / 3600, 2) if secs else None


NB0_NOTE = "production use of any FP8 number here depends on NB0's FP8 quality gate"


def attribution_c4_d9(m1):
    """Per rank: C4 weight drop (GiB -> MiB) from 'Model loading took', D9 = free-after-ready change minus that drop."""
    if not m1:
        return None
    lo, l1 = RC2_REF["rank_load_gib"], {int(k): v for k, v in (m1.get("rank_load_gib") or {}).items()}
    fr = _vdiff(RC2_REF["free_after_ready"], m1.get("free_after_ready"))
    c4 = [round((lo[r] - l1[r]) * 1024) if r in lo and r in l1 else None for r in range(4)]
    d9 = [round(f - c) if f is not None and c is not None else None for f, c in zip(fr or [None] * 4, c4)]
    return {"c4_weight_drop_mib_per_rank": c4, "free_after_ready_change_mib_per_rank": fr, "d9_mib_per_rank": d9,
            "d9_expected_reserve_mib": d9_reserve_mib(), "method": NB1["attribution"]}


def scorecard():
    m1, m2, m3 = (boot_metrics(s) for s in ("B1", "B2", "B3"))
    rows = []

    def row(lever, old, new, adoption, notes, scope, corr_m=None, cost=()):
        g, inp = projection(old, new) if new else (None, None)
        if headroom_flag(new) or (new or {}).get("headroom") == "UNSAFE_HEADROOM":     # R2 audit N1-7: never adopted
            adoption = "NOT_RECOMMENDED: UNSAFE_HEADROOM (TF_OOM in this boot; measurements kept)"
        rows.append({"lever": lever, "gain_scope": scope, "measured": _delta(old, new), "projected_run_tok_s_gain": g,
                     "projection_inputs": inp, "correctness": _corr(corr_m if corr_m is not None else new),
                     "confidence": _conf(new), "adoption": adoption, "cost_gpu_h": _cost(*cost) if cost else _cost(new),
                     "notes": notes + "; " + NB0_NOTE})
    att = attribution_c4_d9(m1)
    p = (R["sections"].get("A0") or {}).get("patch_60490") or {}
    fwd = ((_rec("B1").get("oracle") or {}).get("vs_rc2_fp8") or {}).get("status")
    rows.append({"lever": "D9 vLLM #60490 (QSA indexer workspace in the profiling run)", "gain_scope": "separable (per-rank attribution in B1)",
                 "measured": {"attribution": att, "patch": {k: p.get(k) for k in ("applied", "sha_after", "marker", "ok")},
                              "forward_test_vs_rc2_fp8": fwd, "kv_tokens": [RC2_REF["kv_tokens"], (m1 or {}).get("kv_tokens")]},
                 "projected_run_tok_s_gain": 0.0, "projection_inputs": {"why": "KV_BYTES pin skips memory profiling: the pool "
                 "cannot change; the lever is first-long-prefill safety (RC2: +518 MiB per rank at the first requests)"},
                 "correctness": _corr_and(m1, fwd, ok=CLEAN),               # R2 audit N1-1: within B1's own label
                 "confidence": "one boot vs RC2's boot (cross-session noise included)", "adoption": "patch", "cost_gpu_h": _cost(m1),
                 "notes": NB1["attribution"] + "; " + NB0_NOTE})
    row("C4 text-only (--language-model-only)", RC2_REF, m1, "config",
        f"weight drop per rank {(att or {}).get('c4_weight_drop_mib_per_rank')} MiB; with the KV pin the pool cannot grow, the "
        "gain is free VRAM (vision tower not loaded on PP0) and needs C5 to become KV", "separable (per-rank attribution in B1)")
    w2 = (_rec("B2").get("workers") or {})
    row("C3 SSM state bf16 (--mamba-ssm-cache-dtype bfloat16)", m1, m2, "config",
        f"actual recurrent tensor dtypes B1 {(m1 or {}).get('recurrent_tensor_dtypes')} -> B2 {(m2 or {}).get('recurrent_tensor_dtypes')}; "
        f"mamba padding {(m1 or {}).get('mamba_pad_pct')} % -> {(m2 or {}).get('mamba_pad_pct')} %, minimum block "
        f"{(m1 or {}).get('min_block_for_mamba')} -> {(m2 or {}).get('min_block_for_mamba')}; worker padding {json.dumps(w2.get('mamba_padding'))[:300]}; "
        "at block 4096 the pool block count cannot change (attention page larger); the capacity gain needs a smaller BLOCK_SIZE boot",
        "combined configuration gain (B2 = C3 + MERLIN_SHARE_DRAFT_WORKSPACE + seqs 16)")
    row("MERLIN_SHARE_DRAFT_WORKSPACE=1", m1, m2, "config",
        "shares the target FlashInfer float workspace with the MTP drafter (vllm/merlin/draft_workspace.py); Flash-Next's QSA runs on "
        "FlashAttention, so the log line decides whether anything is shared: " + json.dumps((_rec("B2").get("pool") or {}).get("share_workspace_lines") or [])[:300],
        "combined configuration gain (B2)")
    c2 = _rec("B2").get("caps") or {}
    g1, g2 = (_rec("B1").get("pool") or {}).get("rank_graph_gib") or {}, (_rec("B2").get("pool") or {}).get("rank_graph_gib") or {}
    gdelta = {r: round((g2[r] - g1[r]) * 1024) for r in g2 if r in g1}
    row("seqs > 8 (max-num-seqs 16, captures to 64)", m1, m2, "config",
        f"graph memory B1 -> B2 per rank {g1} -> {g2} GiB (delta MiB {gdelta}, the cost of sizes 40 / 48 / 64 + the larger "
        "batch; it comes out of the free VRAM the pin step needs); under PP2 a micro-batch holds C/2 requests, so FULL graphs "
        "need (C/2)(1+K) tokens: 32 covers 16 streams, 64 covers one micro-batch of 16; "
        f"log max_num_seqs={c2.get('max_num_seqs')} captures={c2.get('capture_sizes')} cover seqs x (1+K): "
        f"{c2.get('captures_cover_seqs_x_1pk')}; highest STEADY per band {(m2 or {}).get('highest_steady')}; the registry's 1..8 "
        "limit is a profile-validation rule, not an engine gate", "combined configuration gain (B2)")
    ab = _rec("B2").get("commit_graph_ab") or {}
    rows.append({"lever": "RecoverSSM commit graph (VLLM_RECOVERSSM_COMMIT_GRAPH=graph, runtime A/B)", "gain_scope": "separable (runtime A/B in B2)",
                 "measured": {k: ab.get(k) for k in ("delta_tok_s", "delta_tick_ms", "greedy_graph_vs_eager", "stats_after")} if ab else None,
                 "projected_run_tok_s_gain": (round(0.9 * ab["delta_tok_s"], 1) if ab.get("delta_tok_s") is not None
                                              and (ab.get("graph_replay_share") or 0) > 0 else None),
                 "projection_inputs": {"formula": "run = 0.9 x measured steady delta at 8k@8 (matched streams); only when the graph "
                                                  "arm replayed commits (replay share > 0)", "graph_replay_share": ab.get("graph_replay_share"),
                                       "counters": ab.get("counters")} if ab else None,
                 "correctness": _corr_and(m2, (ab.get("greedy_graph_vs_eager") or {}).get("status")),   # R2 audit N1-1
                 "confidence": _conf(m2), "adoption": "config", "cost_gpu_h": None,
                 "notes": "PP-deferred commits always run eager (docstring): replays may be 0 under PP2; the counters decide. "
                          + ("boot retried without it: " + json.dumps(_rec("B2").get("commit_graph_boot_failed"))[:200]
                             if _rec("B2").get("commit_graph_boot_failed") else "") + "; " + NB0_NOTE})
    pr = NB1.get("pin_rule") or {}
    b3base = (NB1.get("b3_base") or {}).get("base", "B2")
    row(f"C5 KV pin +{BOOTS['B3'].get('pin_gib', 0.5)} GiB per card (descending ladder, on {b3base})", m2 if b3base == "B2" else m1, m3, "config",
        (f"tries {json.dumps(NB1.get('pin_tries'))[:400]}" if m3 or NB1.get("pin_tries") else f"not run: {pr.get('why') or (_rec('B3').get('why'))}"),
        "separable (B3 vs its base B2 or B1; B3 boots without the commit-graph env and without NCCL debug)")
    rows.append({"lever": "max_safe_running_by_context (residency, not a lever)", "gain_scope": "measurement",
                 "measured": NB1.get("max_safe_running_by_context"), "projected_run_tok_s_gain": None, "projection_inputs": None,
                 "correctness": "UNTESTED", "confidence": "one cold admission per context and boot", "adoption": "config",
                 "cost_gpu_h": None, "notes": "achieved = all streams resident after prefill, real running == n, 0 preemptions; "
                 "bound = floor(P / (ceil(L / 4096) + 6)) at the measured pool; " + NB0_NOTE})
    rows.append({"lever": "W6 chunk 2048 (--max-num-batched-tokens)", "gain_scope": "not measured", "measured": None,
                 "projected_run_tok_s_gain": None, "projection_inputs": None, "correctness": "UNTESTED",
                 "confidence": "none (not measured)", "adoption": "config", "cost_gpu_h": 0.0, "notes": NB1["not_measured"]["W6_chunk"]})
    sc = NB1.get("fp8_self_check") or {}
    rows.append({"lever": "FP8 KV quality (gate row: decided by NB0)", "gain_scope": "gate",
                 "measured": {"fp8_self_check_2k_8k": {b: {k: (v or {}).get(k) for k in ("tf_aa", "decode_cached_vs_fresh")}
                                                       for b, v in (sc.get("bands") or {}).items()},
                              "oracle_vs_rc2_fp8": fwd,
                              "oracle_vs_rc2_bf16": ((_rec("B1").get("oracle") or {}).get("vs_rc2_bf16") or {}).get("status"),
                              "RC2_40k_tf_vs_bf16": RC2_FP8["tf_40k_vs_bf16"],
                              "mtp_default_scale_lines": (_rec("B1").get("pool") or {}).get("default_scale_lines")},
                 "projected_run_tok_s_gain": None, "projection_inputs": None, "correctness": "UNTESTED",
                 "confidence": "NB1 measures only FP8 self-consistency; the BF16 / FP8 x MTP off / K3 isolation is NB0's",
                 "adoption": "config", "cost_gpu_h": None, "notes": NB0_NOTE})
    return rows


def scorecard_save():
    try:
        rows = scorecard()
        SCORECARD.write_text(json.dumps({"what": "E25 NB1 lever scorecard (PLAN v3.2 §4.1)", "t_min": round((time.time() - T_START) / 60, 1),
                                         "boots": {s: (_rec(s) or {}).get("status") for s in BOOTS},
                                         "boot_labels": {s: {"correctness": _corr(boot_metrics(s)), "headroom": headroom_flag(_rec(s)),
                                                             "core_measurements_complete": _rec(s).get("core_measurements_complete"),
                                                             "postprocess_ok": _rec(s).get("postprocess_ok")} for s in BOOTS},
                                         "recommended_pin": NB1.get("recommended_pin"),          # R2 audit N1-7: never a TF_OOM step
                                         "capacity_winner": capacity_winner(),                   # R2 audit N1-1 / N1-2 / N1-7
                                         "rows": rows}, indent=1, default=str))
        return rows
    except Exception as e:
        R["errors"].append(f"scorecard: {type(e).__name__}: {e}"[:300])
        save()
        return None


_gpu_util_stop = threading.Event()


def _gpu_util():
    """utilization / memory / power every 5 s (per-stage idle cross-check for the PP2 trace)."""
    while not _gpu_util_stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            if r.returncode == 0:
                with open(OUT / "nb1_gpu_util.jsonl", "a") as fh:
                    fh.write(json.dumps({"t": round(time.time() - T_START, 1), "gpus": r.stdout.strip().splitlines()}) + "\n")
        except Exception:
            pass
        _gpu_util_stop.wait(5)


if NVSMI_OK:
    threading.Thread(target=_gpu_util, daemon=True).start()

def nb1_main():
    """The plan: install -> patch -> B1 -> B2 -> B3 ladder; every failure stops the dependent boots (fail closed)."""
    m = R["sections"]["A0"]["merge"]
    if not m.get("ok"):
        NB1["stop"] = "MERGE_FAILED: no boot started"
    else:
        step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
        pt = step("A0", "patch_60490", 3, apply_60490)
        if pt.get("status") != "DONE" or not pt.get("ok"):
            for s_ in BOOTS:
                skip(s_, f"#60490 patch / worker extension not verified: {pt.get('error') or pt.get('status')}")
        elif "rc2_fp8" not in ROWS and not FAKE:
            for s_ in BOOTS:
                skip(s_, "RC2's recorded FP8 oracle rows do not match this notebook's prompts (no gate reference)")
        else:
            b1 = step("B1", BOOTS["B1"]["name"], BOOTS["B1"]["est_min"], boot1)
            c1, miss1 = core_complete(b1)                 # R2 audit N1-3 (an exception path in step has no fields yet)
            scorecard_save()
            if not boot_ok(b1):
                for s_ in ("B2", "B3"):
                    skip(s_, f"B1 not clean (status {b1.get('status')}, quality {b1.get('quality_gate')}, vs RC2 FP8 "
                             f"{((b1.get('oracle') or {}).get('vs_rc2_fp8') or {}).get('status')})")
            else:
                if c1:                                    # R2 audit N1-3: B2 builds on B1's complete core measurements
                    b2 = step("B2", BOOTS["B2"]["name"], BOOTS["B2"]["est_min"],
                              lambda: run_boot("B2", body_fp8(BOOTS["B2"], "b1", commit=True, residency=("8k",)), need=("b1",)))
                    core_complete(b2)
                else:
                    skip("B2", f"B1 core measurements incomplete (missing {miss1}; postprocess_ok {b1.get('postprocess_ok')}): "
                               "B2 runs only on a complete B1 core (kv census, pool, F, free MiB per card, residency)")
                    b2 = _rec("B2")
                scorecard_save()
                base = "B2" if boot_ok(b2) else "B1"     # the pin ladder needs a clean base; B1 passed, B2 may not have
                NB1["b3_base"] = {"base": base, "why": "B2 clean" if base == "B2" else
                                  f"B2 not clean (status {b2.get('status')}, quality {b2.get('quality_gate')}): the pin ladder runs on B1's configuration"}
                NB1["pin_rule"] = pin_rule(base)          # R2 audit N1-3: free memory of the boot the pin builds on
                if NB1["pin_rule"]["ok"]:
                    pin_ladder(base)
                else:
                    skip("B3", "pin rule: " + NB1["pin_rule"]["why"])
    return {s: (_rec(s) or {}).get("status") for s in BOOTS}


if not RERUN and not os.environ.get("RC2_UNIT"):
    nb1_main()
    _gpu_util_stop.set()
    R["host_phase"] = HOST_PHASE
    R["prefill"] = PREFILL
    R["minutes"] = round((time.time() - T_START) / 60, 1)
    save()
    scorecard_save()
    log("NB1 done", json.dumps({f"{s}/{n}": r.get("status") for s, st in R["sections"].items() if isinstance(st, dict)
                                for n, r in st.items() if isinstance(r, dict) and "status" in r}))
save()
'''

CELL8 = r'''# ==================================================================== CELL 8 — report + fallback submission
if not RERUN and not os.environ.get("RC2_UNIT"):
    write_fallback_submission()
    def _brief(r):
        if not isinstance(r, dict):
            return r
        keep = ("status", "correctness", "quality_gate", "probes", "sampled", "prefill", "caps", "pool", "kv", "admission_90k@5",
                "prefix", "steady", "free_mib_per_card", "free_after_ready_mib", "workers", "nccl", "config_diff", "tries", "kv_bytes",
                "boot_gate", "why", "error", "status_note", "load_tests_skipped", "early_stop", "boot_watchdog", "applied",
                "sha_after", "ok", "warmup", "jit_after_warmup", "commit_graph_ab", "fp8_self_check")
        return {k: r.get(k) for k in keep if k in r}
    R["summary"] = {f"{s}/{n}": _brief(r) for s, st in R["sections"].items() if isinstance(st, dict) for n, r in st.items()
                    if isinstance(r, dict) and "status" in r}
    save()
    print(json.dumps(R["summary"], default=str)[:6000])
    print(json.dumps([{k: x.get(k) for k in ("lever", "gain_scope", "projected_run_tok_s_gain", "correctness", "adoption")}
                      for x in (scorecard_save() or [])]))
'''

subs = {"__LAB__": json.dumps(RC2["LAB"]), "__PLE__": json.dumps(RC2["PLE"]), "__PATCH__": json.dumps(PATCH_TEXT),
        "__PPRE__": P_PRE, "__PPOST__": P_POST, "__RC2FP8__": json.dumps(RC2_FP8), "__RC2ORACLE__": json.dumps(RC2_ORACLE),
        "__NB1EXT__": json.dumps(NB1_EXT)}
cells_src = []
for i, c in enumerate((CELL1, CELL2, CELL3, CELL4, CELL5, CELL6, CELL7, CELL8)):
    for k, v in subs.items():
        c = c.replace(k, v)
    if i:                                      # a competition rerun skips cells 2-8 without an exception (RC2 / E23 rule)
        c = "if not RERUN:\n" + textwrap.indent(c, "    ")
    compile(c, "<nb1 cell>", "exec")
    cells_src.append(c)
assert all(rc in "\n".join(cells_src) or True for rc in ())
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD0.splitlines(True)}] +
               [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(True)} for c in cells_src],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
data = json.dumps(nb, indent=1).encode("utf-8")
assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
assert len(data) < 400_000, len(data)
OUT_NB.write_bytes(data)
(HERE / "nb1_code_export.py").write_bytes("\n\n".join(cells_src).encode("utf-8"))   # one file, for the local harness (LF)
print("wrote", OUT_NB.name, "sha256", hashlib.sha256(data).hexdigest(), "|", len(data), "bytes")
