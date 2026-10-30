"""E24 server hook (sitecustomize) for SGLang 0.5.21 and vLLM 0.32 (rev 2, 2026-10-30).

The notebook writes this file as `<hookdir>/sitecustomize.py` and puts <hookdir> on the PYTHONPATH of server launches
only. Base: E17's `src/e17_hook.py` (inert unless enabled; never edits a source file; a daemon thread wraps target methods
after their module is imported; periodic flush; per-pid files; at-fork re-arm) plus DESIGN rev 2 §3.3-§3.4 and the
BUILD_SPEC rev 2 WP3 overrides (review/section6_hook.md #3 #5 #7 #10 #12-#14 #20):
- every process writes an env record: at import, and in forked children (vLLM EngineCore/workers fork by default and
  never import sitecustomize) from os.register_at_fork; re-snapshotted periodically (vLLM rewrites the Inductor/Triton
  dirs in os.environ after import; last value wins, every change is listed);
- the daemon thread reads ONLY sys.modules / os.environ (and this hook's own buffers) and never imports (an import from
  a second thread while the main thread imports the same stack can raise _DeadlockError); Triton/Inductor/FlashInfer
  dirs are resolved in the main thread inside wrappers on Scheduler.__init__ / Scheduler.run_event_loop (vLLM:
  GPUModelRunner.load_model, probe only); the Mamba allocator snapshot is taken in the main thread too (wrappers and
  Scheduler.on_idle, <= 5 Hz);
- fast polling (50 ms) only until the enabled targets are patched, at most E24_HOOK_FAST_SEC; then >= 1 s; files are
  rewritten only on change; every wrapped method flushes SYNCHRONOUSLY before re-raising (the #43559 assert kills the
  scheduler, and the parent kills the tree before the 2-s flush could run: E17 kept its evidence by luck);
- NaN injection (#41736 / #41737 negative tests) is installed only when E24_INJECT_* is set AND E24_HOOK_EAGER=1 (the
  notebook sets it only with --disable-decode-cuda-graph); it returns orig() while a CUDA stream is capturing, injects
  only inside a DFlash draft-DECODE forward, keeps the compiled variant (mark_static on the clone), is deterministic by
  rid (identical on every TP rank, so collectives stay in step) and counts per rank;
- the image's own sitecustomize (it fails on a missing `wrapt`; first line of every server log) is chain-executed exactly
  as site.py would, and the outcome is recorded.
It never changes computation unless an injection env is set: wrappers pass the same arguments and return the original
result; their own errors are counted, never raised.

Env (only E24_HOOK_DIR is required; unset = inert, only the chained sitecustomize runs):
  E24_HOOK_DIR               output dir (per arm: /tmp/e24/<arm>/hook)
  E24_HOOK_COUNTERS=1        E17 counters + _alloc_mamba_slot + _mamba_check_track_boundary + DFlash rid context
  E24_HOOK_PROBE=1           probe_<pid>.json per rank (lm_head dtype, Mamba state dtype, k/v scales, module sha256)
  E24_HOOK_TARGETS           comma list of the arm's expected targets (self-test; unset = the enabled SGLang targets
                             without DFlashWorkerV2); an expected target that is not enabled is reported missing
  E24_INJECT_DRAFT_NAN=1     with E24_HOOK_EAGER=1: NaN into each sentinel's last draft row (_grouped_conv)
  E24_INJECT_SELECTOR_NAN=1  with E24_HOOK_EAGER=1: NaN scores[k,0,0,:] for each GREEDY sentinel row (sample_path)
  E24_SENTINEL_PREFIX        sentinel rid prefix (default "e24-sentinel-")
  E24_HOOK_FLUSH_SEC (2, min 0.05), E24_HOOK_FAST_SEC (300), E24_HOOK_VERBOSE=1, E24_HOOK_CHAIN=0 (no chain-exec)
Outputs in E24_HOOK_DIR (per pid; atomic rewrites; only on change):
  env_<pid>.json       every process: pid, ppid, source (import|fork), forked_from, argv, orig_argv, executable,
                       role {kind, mp_name, proctitle, cmdline, tp_rank}, role_tag (main | scheduler | detokenizer |
                       engine_core | worker | inductor_compile | tracker | forked | other), env (SGLANG_*, E24_*, NCCL_*, VLLM_*,
                       TRITON_*, TORCH*, CUDA_*, caches, PYTHONPATH...; secrets redacted; also copied to the top level),
                       cache {the DESIGN §3.3 vars}, env_first, env_changes, resolved {triton, inductor, flashinfer,
                       where, thread}, modules {module: file, sha256}, hook {sha256, rev, features, expected, chain}
  counters_<pid>.json  processes with a patched target, except the launcher / detokenizer / API server / trackers
                       (they import target modules but run no rank): tp_rank, role_tag (top level),
                       self_test {expected, enabled, patched, missing, failed, ok},
                       counters, per_rid, inject {requested, eager, installed, refused}, alloc snapshot, modules
  trace_<pid>.jsonl    events (appended): patched, alloc_failed, donation, prefix_lock_released, donation_alloc_assert,
                       track_boundary, lazy_insert_cleared, neighbor, selector_pick, exception, hook_error, ...
  probe_<pid>.json     E24_HOOK_PROBE: written once, synchronously, from the main thread
"""
import atexit
import hashlib
import json
import os
import sys
import threading
import time
from collections import deque

REV = "e24_hook rev 2b (2026-10-30)"
_MARKER = "E24_HOOK_MARKER_REV2"       # the chain-exec skips any sitecustomize that contains it (a copy of this hook)


def _env_float(name, default, lo):
    try:
        return max(lo, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


_DIR = os.environ.get("E24_HOOK_DIR") or None
_COUNTERS_ON = os.environ.get("E24_HOOK_COUNTERS") == "1"
_PROBE_ON = os.environ.get("E24_HOOK_PROBE") == "1"
_VERBOSE = os.environ.get("E24_HOOK_VERBOSE") == "1"
_EAGER = os.environ.get("E24_HOOK_EAGER") == "1"
_REQ_DRAFT = os.environ.get("E24_INJECT_DRAFT_NAN") == "1"
_REQ_SEL = os.environ.get("E24_INJECT_SELECTOR_NAN") == "1"
_INJ_DRAFT = _REQ_DRAFT and _EAGER
_INJ_SEL = _REQ_SEL and _EAGER
_PREFIX = os.environ.get("E24_SENTINEL_PREFIX") or "e24-sentinel-"
_FLUSH = _env_float("E24_HOOK_FLUSH_SEC", 2.0, 0.05)
_FAST = _env_float("E24_HOOK_FAST_SEC", 300.0, 0.0)

# target -> (module, attribute); TARGET_METHODS = what each wraps (the CPU kit AST-checks them in the installed package)
TARGETS = {
    "Scheduler": ("sglang.srt.managers.scheduler", "Scheduler"),
    "ScheduleBatch": ("sglang.srt.managers.schedule_batch", "ScheduleBatch"),
    "SchedulerBatchResultProcessor": ("sglang.srt.managers.scheduler_components.batch_result_processor",
                                      "SchedulerBatchResultProcessor"),
    "MambaComponent": ("sglang.srt.mem_cache.unified_cache.components.mamba", "MambaComponent"),
    "DFlashWorkerV2": ("sglang.srt.speculative.dflash_worker_v2", "DFlashWorkerV2"),
    "_grouped_conv": ("sglang.srt.models.dflash", "_grouped_conv"),
    "CandidateSelector": ("sglang.srt.models.dflash", "CandidateSelector"),
    "GPUModelRunner": ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner"),
}
TARGET_METHODS = {
    "Scheduler": ("__init__", "run_event_loop"),
    "ScheduleBatch": ("mamba_lazy_prealloc_at_boundary", "mamba_lazy_spec_prepare"),
    "SchedulerBatchResultProcessor": ("_mamba_prefix_cache_update", "_mamba_lazy_spec_update",
                                      "_mamba_check_track_boundary"),
    "MambaComponent": ("prepare_for_caching_req", "_alloc_mamba_slot"),
    "DFlashWorkerV2": ("forward_batch_generation",),
    "_grouped_conv": (),                 # the module-level function itself (DFlashGroupedConv._convolve looks it up)
    "CandidateSelector": ("sample_path",),
    "GPUModelRunner": ("load_model",),
}
# module -> package-relative file the E24 patcher edits (sha256 of the IMPORTED file, recorded per rank)
PATCHED_MODULES = {
    "sglang.srt.managers.scheduler": "srt/managers/scheduler.py",
    "sglang.srt.dllm.mixin.scheduler": "srt/dllm/mixin/scheduler.py",
    "sglang.srt.managers.schedule_batch": "srt/managers/schedule_batch.py",
    "sglang.srt.mem_cache.unified_cache.components.mamba": "srt/mem_cache/unified_cache/components/mamba.py",
    "sglang.srt.managers.scheduler_components.batch_result_processor":
        "srt/managers/scheduler_components/batch_result_processor.py",
    "sglang.srt.models.dflash": "srt/models/dflash.py",
    "sglang.kernels.ops.speculative.dflash": "kernels/ops/speculative/dflash.py",
}
CACHE_VARS = ("SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
              "FLASHINFER_WORKSPACE_BASE", "CUDA_CACHE_PATH", "CUDA_CACHE_DISABLE", "TILELANG_CACHE_DIR",
              "TORCH_EXTENSIONS_DIR", "VLLM_CACHE_ROOT", "XDG_CACHE_HOME", "HOME")
_ENV_PREFIXES = ("SGLANG_", "E24_", "E14_", "E17_", "NCCL_", "VLLM_", "TRITON_", "TORCHINDUCTOR_", "TORCH_", "PYTORCH_",
                 "FLASHINFER_", "TILELANG_", "CUDA_", "HF_", "TRANSFORMERS_", "OMP_", "NVIDIA_")
_ENV_NAMES = ("PYTHONPATH", "HOME", "XDG_CACHE_HOME", "TMPDIR", "LIBRARY_PATH", "LD_LIBRARY_PATH", "PYTHONHASHSEED")
_SECRET_WORDS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD")
_SCALE_ATTRS = ("k_scale_float", "v_scale_float", "_k_scale_float", "_v_scale_float", "k_scale", "v_scale",
                "_k_scale", "_v_scale")


def _enabled_targets():
    on = ["Scheduler"]
    if _COUNTERS_ON:
        on += ["ScheduleBatch", "SchedulerBatchResultProcessor", "MambaComponent", "DFlashWorkerV2"]
    if _INJ_DRAFT:
        on += ["DFlashWorkerV2", "_grouped_conv"]
    if _INJ_SEL:
        on += ["DFlashWorkerV2", "CandidateSelector"]
    if _PROBE_ON:
        on += ["GPUModelRunner"]
    return list(dict.fromkeys(on))


def _expected_targets():
    raw = os.environ.get("E24_HOOK_TARGETS")
    if raw is not None:
        return list(dict.fromkeys(t for t in raw.replace(";", ",").replace(" ", ",").split(",") if t))
    exp = ["Scheduler"]
    if _COUNTERS_ON:
        exp += ["ScheduleBatch", "SchedulerBatchResultProcessor", "MambaComponent"]
    if _INJ_DRAFT:
        exp += ["DFlashWorkerV2", "_grouped_conv"]
    if _INJ_SEL:
        exp += ["DFlashWorkerV2", "CandidateSelector"]
    return list(dict.fromkeys(exp))


ENABLED = _enabled_targets()
EXPECTED = _expected_targets()

_PID = [os.getpid()]
_LOCK = [threading.Lock()]     # serialises flushes (thread vs the synchronous main-thread flush); new one after fork
EVENTS = deque(maxlen=200000)
COUNTERS = {}
PER_RID = {}                   # rid -> {counter: n} for the evidence counters (bounded)
PATCHED = []
FAILED = {}
MARK = {}                      # rid -> {"since", "len", "path"}: optional slot failed, the keep slot is written in place
PEND = {}                      # rid -> a spec-path stale candidate waiting for its verify result
_REFS = {}                     # last pool / tree cache seen (allocator snapshot)
_SEQ = [0]
_VER = [0]                     # bumped by every counter / event; the counters file is rewritten only when it moved
_FLUSHED = [-1]
_CUR_RID = [None]              # rid of the prepare_for_caching_req in progress (pristine _alloc_mamba_slot has no req)
_CTX = [None]                  # DFlash forward context {rids, mode, block} while forward_batch_generation runs
_SEEN = set()                  # (sentinel, neighbour) pairs already logged
ROLE = {}
RESOLVED = {}
MODULES = {}
PROBE = {}
CHAIN = {}
ENV = {"first": None, "last": None, "role": None, "argv": None, "changes": [], "source": "import", "forked_from": None,
       "t_first": round(time.time(), 3), "dirty": True}
_POLL = {"sleep": 0.05, "phase": "fast", "loops": 0}
_SELF = {"file": None, "sha256": None}
_MAX_RIDS = 4096
_MAX_CHANGES = 200
_MAX_EVENTS_PER_KIND = 2000
_EV_KIND_N = {}


# ------------------------------------------------------------------------------------------------- small helpers
def _count(k, n=1):
    COUNTERS[k] = COUNTERS.get(k, 0) + n
    _VER[0] += 1


_BULK_KEYS = frozenset(("grammar_steps",))     # per-step bookkeeping, not evidence: such rids are evicted first


def _evict_rid():
    """Drops the oldest rid that carries no evidence counter (a grammar arm adds one per request), else the oldest."""
    for r, d in PER_RID.items():
        if not d.keys() - _BULK_KEYS:
            del PER_RID[r]
            return
    del PER_RID[next(iter(PER_RID))]


def _rcount(rid, k, n=1):
    if not rid:
        return
    d = PER_RID.get(rid)
    if d is None:
        if len(PER_RID) >= _MAX_RIDS:
            try:
                _evict_rid()
            except Exception:
                pass
        d = PER_RID[rid] = {}
    d[k] = d.get(k, 0) + n
    _VER[0] += 1


def _item(x):
    try:
        return int(x.item())
    except Exception:
        try:
            return int(x)
        except Exception:
            return None


def _f(x):
    try:
        if hasattr(x, "numel"):
            return float(x.item()) if x.numel() == 1 else None
        return float(x)
    except Exception:
        return None


def _plain(v, d=0):
    """JSON-safe copy made in the calling (main) thread, so the flush thread never touches engine objects."""
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v[:500]
    if d < 4 and isinstance(v, (list, tuple)):
        return [_plain(x, d + 1) for x in list(v)[:64]]
    if d < 4 and isinstance(v, dict):
        return {str(k): _plain(x, d + 1) for k, x in list(v.items())[:64]}
    n = _item(v)
    return n if n is not None else type(v).__name__


def _ev(kind, **kw):
    n = _EV_KIND_N.get(kind, 0)
    _EV_KIND_N[kind] = n + 1
    if n >= _MAX_EVENTS_PER_KIND and kind not in ("exception", "donation_alloc_assert", "patch_failed"):
        _count("events_dropped")
        return
    _SEQ[0] += 1
    e = {k: _plain(v) for k, v in kw.items()}
    e.update(ev=kind, i=_SEQ[0], t=round(time.time(), 3))
    EVENTS.append(e)
    _VER[0] += 1


def _hook_error(where, e):
    try:
        _count("hook_error_" + where)
        _ev("hook_error", where=where, err=repr(e)[:200])
    except Exception:
        pass


def _rid(req):
    return str(getattr(req, "rid", ""))


def _kv(req):
    return getattr(req, "kv", None)


def _arg(a, k, pos, name, default=None):
    if name in k:
        return k[name]
    return a[pos] if len(a) > pos else default


def _skipped(req):
    try:
        return tuple(getattr(getattr(req, "lock_receipt", None), "skipped_lock_components", ()) or ())
    except Exception:
        return ()


def _named(fn, orig):
    try:
        fn.__name__, fn.__qualname__ = orig.__name__, getattr(orig, "__qualname__", orig.__name__)
        fn.__doc__, fn.__wrapped__ = orig.__doc__, orig
    except Exception:
        pass
    return fn


def _exc(where, e):
    """A wrapped call raised: count, log, flush synchronously (the process may be killed right after), caller re-raises."""
    try:
        _count("exc_" + where)
        if isinstance(e, AssertionError):
            _count("assert_" + where)
        _ev("exception", where=where, type=type(e).__name__, err=str(e)[:300], rid=_CUR_RID[0])
    except Exception:
        pass
    _flush_sync()


# ------------------------------------------------------------------------------------------------- env records
def _secret(name):
    u = name.upper()
    return any(w in u for w in _SECRET_WORDS) or u.endswith("_KEY") or u == "KEY"


def _env_now():
    try:
        data = dict(os.environ._data)                       # one C-level copy: safe against concurrent writers
        env = {os.fsdecode(k): os.fsdecode(v) for k, v in data.items()}
    except Exception:
        env = dict(os.environ)
    return {k: ("<redacted>" if _secret(k) else v) for k, v in sorted(env.items())
            if k.startswith(_ENV_PREFIXES) or k in _ENV_NAMES}


_ROLE_TAG = {"sglang_main": "main", "vllm_api_server": "main", "sglang_scheduler": "scheduler",
             "sglang_detokenizer": "detokenizer", "vllm_engine_core": "engine_core", "vllm_worker": "worker",
             "inductor_compile_worker": "inductor_compile", "mp_tracker": "tracker"}
# processes that import a target module but never run a rank's batches (the launcher imports the scheduler module,
# the detokenizer imports schedule_batch): they write no counters file, so per-rank gates see only the TP ranks
_NON_RANK_KINDS = frozenset(("sglang_main", "sglang_detokenizer", "vllm_api_server", "mp_tracker",
                             "inductor_compile_worker"))


def _title_tp_rank(t):
    """TP rank from the 0.5.21 scheduler title "sglang::scheduler_TP<r>" (set before the model loads); None if absent."""
    i = t.find("_TP")
    if i < 0:
        return None
    j = i + 3
    while j < len(t) and t[j].isdigit():
        j += 1
    return int(t[i + 3:j]) if j > i + 3 else None


def _kind(r):
    t = " ".join(str(x) for x in (r.get("proctitle"), r.get("mp_name"), r.get("cmdline")) if x)
    argv = " ".join(str(x) for x in (getattr(sys, "orig_argv", None) or []))
    if "resource_tracker" in argv or "semaphore_tracker" in argv:
        return "mp_tracker"
    if "sglang::scheduler" in t:
        return "sglang_scheduler"
    if "sglang::detokenizer" in t:
        return "sglang_detokenizer"
    if "EngineCore" in t:
        return "vllm_engine_core"
    if "Worker" in t and ("VLLM" in t or "vllm" in argv):
        return "vllm_worker"
    if "compile_worker" in argv:
        return "inductor_compile_worker"
    if ENV["source"] == "import":
        if "sglang.launch_server" in argv or "sglang serve" in argv:
            return "sglang_main"
        if "vllm" in argv and "--multiprocessing-fork" not in argv:
            return "vllm_api_server"
    return "forked" if ENV["source"] == "fork" else "other"


def _role_now():
    r = {"pid": os.getpid(), "ppid": os.getppid(), "mp_name": None, "proctitle": None, "cmdline": None,
         "tp_rank": ROLE.get("tp_rank"), "forked_from": ENV["forked_from"]}
    mp = sys.modules.get("multiprocessing")
    if mp is not None:
        try:
            r["mp_name"] = str(mp.current_process().name)
        except Exception:
            pass
    spt = sys.modules.get("setproctitle")
    if spt is not None:
        try:
            r["proctitle"] = str(spt.getproctitle())[:300]
        except Exception:
            pass
    try:
        with open("/proc/self/cmdline", "rb") as f:
            r["cmdline"] = f.read(4096).replace(b"\0", b" ").decode("utf-8", "replace").strip()[:600]
    except OSError:
        pass
    r["kind"] = _kind(r)
    if r["tp_rank"] is None and r["kind"] == "sglang_scheduler":
        r["tp_rank"] = _title_tp_rank(r["proctitle"] or "")
    return r


def _rank_tag():
    """(tp_rank, role_tag) for the counters file: _main_late's rank, else the one parsed from the scheduler title."""
    role = ENV["role"] or {}
    tp = ROLE.get("tp_rank")
    return (tp if tp is not None else role.get("tp_rank")), _ROLE_TAG.get(role.get("kind"), role.get("kind"))


def _snap_env():
    """Thread-safe by construction: called with the flush lock held (or before the thread exists)."""
    env = _env_now()
    role = _role_now()
    argv = [str(x)[:400] for x in list(getattr(sys, "argv", None) or [])[:64]]
    last = ENV["last"]
    if ENV["first"] is None:
        ENV["first"] = env
    if last is not None and env != last:
        for k in sorted(set(last) | set(env)):
            if last.get(k) != env.get(k) and len(ENV["changes"]) < _MAX_CHANGES:
                ENV["changes"].append({"t": round(time.time(), 3), "var": k, "old": last.get(k), "new": env.get(k)})
    if env != last or role != ENV["role"] or argv != ENV["argv"]:
        ENV["last"], ENV["role"], ENV["argv"] = env, role, argv
        ENV["dirty"] = True


def _features():
    return {"counters": _COUNTERS_ON, "probe": _PROBE_ON, "verbose": _VERBOSE, "eager": _EAGER,
            "inject_draft_nan": _INJ_DRAFT, "inject_selector_nan": _INJ_SEL, "sentinel_prefix": _PREFIX,
            "flush_sec": _FLUSH, "fast_sec": _FAST, "enabled": list(ENABLED)}


def _env_record():
    env = dict(ENV["last"] or {})
    rec = {"schema": "e24_hook_env/2", "rev": REV, "pid": os.getpid(), "ppid": os.getppid(), "t": round(time.time(), 3),
           "t_first": ENV["t_first"], "source": ENV["source"], "forked_from": ENV["forked_from"],
           "argv": ENV["argv"], "orig_argv": [str(x)[:400] for x in (getattr(sys, "orig_argv", None) or [])[:64]],
           "executable": sys.executable, "role": ENV["role"],
           "role_tag": _ROLE_TAG.get((ENV["role"] or {}).get("kind"), (ENV["role"] or {}).get("kind")), "env": env,
           "cache": {k: env.get(k) for k in CACHE_VARS},
           "env_first": ENV["first"] if ENV["first"] != env else None, "env_changes": list(ENV["changes"]),
           "resolved": dict(RESOLVED) or None, "modules": dict(MODULES) or None,
           "hook": {"rev": REV, "file": _SELF["file"], "sha256": _SELF["sha256"], "features": _features(),
                    "expected": list(EXPECTED), "patched": list(PATCHED), "chain": dict(CHAIN) or None}}
    for k, v in env.items():
        rec.setdefault(k, v)            # BUILD_SPEC rev 1 flat layout (env var names never collide with the keys)
    return rec


def _dflt(o):
    return type(o).__name__


def _out(kind, ext="json"):
    return os.path.join(_DIR, "%s_%d.%s" % (kind, os.getpid(), ext))


def _dump_atomic(path, obj):
    tmp = "%s.%d.%d.tmp" % (path, os.getpid(), threading.get_ident())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=_dflt)
    os.replace(tmp, path)


def _write_env():
    os.makedirs(_DIR, exist_ok=True)
    ENV["dirty"] = False
    _dump_atomic(_out("env"), _env_record())


# ------------------------------------------------------------------------------------------------- counters / flush
def _inject_info():
    return {"draft_nan_requested": _REQ_DRAFT, "selector_nan_requested": _REQ_SEL, "eager": _EAGER, "prefix": _PREFIX,
            "installed": [t for t in ("_grouped_conv", "CandidateSelector") if t in PATCHED],
            "refused": "E24_HOOK_EAGER!=1" if (_REQ_DRAFT or _REQ_SEL) and not _EAGER else None}


_SNAP = [{}]                   # allocator snapshot, a fresh dict per refresh (the thread only reads the reference)
_SNAP_T = [0.0]


def _refresh_snapshot(force=False):
    """Main thread only (wrappers, Scheduler.on_idle): Mamba allocator available / size / evictable, <= 5 Hz."""
    if not _COUNTERS_ON:
        return
    now = time.time()
    if not force and now - _SNAP_T[0] < 0.2:
        return
    _SNAP_T[0] = now
    snap = {}
    try:
        pool = _REFS.get("pool")
        if pool is not None:
            snap["available"] = _item(pool.mamba_allocator.available_size())
            snap["size"] = _item(getattr(getattr(pool, "mamba_pool", None), "size", None))
        tree = _REFS.get("tree")
        if tree is not None and hasattr(tree, "mamba_evictable_size"):
            snap["evictable"] = _item(tree.mamba_evictable_size())
    except Exception as e:
        snap["error"] = repr(e)[:200]
    _SNAP[0] = snap


def _snapshot():
    return dict(_SNAP[0])


def _self_test():
    missing = [t for t in EXPECTED if t not in PATCHED]
    return {"expected": list(EXPECTED), "enabled": list(ENABLED), "patched": list(PATCHED), "missing": missing,
            "not_enabled": [t for t in missing if t not in ENABLED], "unknown": [t for t in EXPECTED if t not in TARGETS],
            "failed": dict(FAILED), "ok": not missing and not any(t in FAILED for t in EXPECTED)}


def _counters_record(snap):
    tp, tag = _rank_tag()
    return {"schema": "e24_hook_counters/2", "rev": REV, "pid": os.getpid(), "ppid": os.getppid(),
            "t": round(time.time(), 3), "tp_rank": tp, "role_tag": tag, "role": ENV["role"],
            "argv0": (list(getattr(sys, "argv", None) or [""]) + [""])[0][-120:],
            "self_test": _self_test(), "patched": list(PATCHED), "counters": dict(COUNTERS),
            "per_rid": {r: dict(d) for r, d in list(PER_RID.items())}, "marked_now": len(MARK),
            "inject": _inject_info(), "alloc": snap, "modules": dict(MODULES) or None,
            "poll": {"sleep": _POLL["sleep"], "phase": _POLL["phase"]}}


_LAST_SNAP = [None]


def _flush_locked(force=False):
    os.makedirs(_DIR, exist_ok=True)
    evs = []
    while EVENTS:
        try:
            evs.append(EVENTS.popleft())
        except IndexError:
            break
    if evs:
        with open(_out("trace", "jsonl"), "a", encoding="utf-8") as f:
            for e in evs:
                f.write(json.dumps(e, default=_dflt) + "\n")
    v = _VER[0]
    if (ENV["role"] or {}).get("kind") in _NON_RANK_KINDS:
        if _FLUSHED[0] != -1:             # written before the title said launcher/detokenizer: take it back
            try:
                os.remove(_out("counters"))
            except OSError:
                pass
            _FLUSHED[0], _LAST_SNAP[0] = -1, None
    elif PATCHED or COUNTERS or FAILED:
        # the allocator snapshot changes without any counter (slots freed after a request): it is a change too
        snap = _snapshot()
        if force or v != _FLUSHED[0] or snap != _LAST_SNAP[0]:
            _dump_atomic(_out("counters"), _counters_record(snap))
            _FLUSHED[0], _LAST_SNAP[0] = v, snap
    if ENV["dirty"] or force:
        _write_env()


def _flush_sync():
    """Main-thread flush (exception paths, exit). Waits at most 2 s for the thread's flush to finish."""
    if not _DIR:
        return
    lk = _LOCK[0]
    got = False
    try:
        got = lk.acquire(timeout=2.0)
        _snap_env()
        _flush_locked(force=True)
    except Exception:
        pass
    finally:
        if got:
            try:
                lk.release()
            except Exception:
                pass


# ------------------------------------------------------------------------------------------------- main-thread work
def _resolve_dirs(where):
    """Main thread only (inside a late engine method): cache dirs from modules already imported, no side effects
    (torch's cache_dir() would create /tmp/torchinductor_<user>; its default_cache_dir() does not)."""
    res = {"where": where, "thread": threading.current_thread().name, "t": round(time.time(), 3)}
    d = None
    tri = sys.modules.get("triton")
    try:
        d = getattr(getattr(getattr(tri, "knobs", None), "cache", None), "dir", None)
    except Exception:
        d = None
    if not d:
        fn = getattr(sys.modules.get("triton.runtime.cache"), "default_cache_dir", None)
        try:
            d = os.environ.get("TRITON_CACHE_DIR") or (fn() if fn else None)
        except Exception:
            d = None
    res["triton"] = str(d) if d else None
    env_ind = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    ind = sys.modules.get("torch._inductor.runtime.cache_dir_utils")
    if env_ind:
        res["inductor"], res["inductor_source"] = env_ind, "env"
    elif ind is not None and hasattr(ind, "default_cache_dir"):
        try:
            res["inductor"], res["inductor_source"] = str(ind.default_cache_dir()), "default"
        except Exception:
            pass
    fi = sys.modules.get("flashinfer.jit.env")
    if fi is not None:
        try:
            res["flashinfer"] = {k: str(getattr(fi, k)) for k in dir(fi) if k.isupper() and "DIR" in k}
        except Exception:
            pass
    res["loaded"] = {m: m in sys.modules for m in ("triton", "torch._inductor", "flashinfer", "sgl_kernel")}
    res["cache_env"] = {k: os.environ.get(k) for k in CACHE_VARS}
    RESOLVED.clear()
    RESOLVED.update(res)


def _module_shas():
    out = {}
    for mod, rel in PATCHED_MODULES.items():
        m = sys.modules.get(mod)
        f = getattr(m, "__file__", None) if m is not None else None
        sha = None
        if f:
            try:
                with open(f, "rb") as fh:
                    sha = hashlib.sha256(fh.read()).hexdigest()
            except OSError:
                pass
        out[mod] = {"loaded": m is not None, "file": f, "rel": rel, "sha256": sha}
    MODULES.clear()
    MODULES.update(out)


def _model_info(model):
    if model is None:
        return None
    info = {"class": type(model).__name__}
    lm = getattr(model, "lm_head", None)
    if lm is not None:
        w = getattr(lm, "weight", None)
        info["lm_head"] = {"class": type(lm).__name__, "weight_dtype": str(getattr(w, "dtype", None)) if w is not None
                           else None, "params": {}}
        try:
            for n, prm in list(lm.named_parameters())[:8]:
                info["lm_head"]["params"][n] = [str(prm.dtype), [int(s) for s in prm.shape]]
        except Exception as e:
            info["lm_head"]["error"] = repr(e)[:200]
    rows = []
    try:
        for name, mod in model.named_modules():
            vals = {}
            for a in _SCALE_ATTRS:
                try:
                    v = getattr(mod, a, None)
                except Exception:
                    v = None
                fv = _f(v) if v is not None else None
                if fv is not None:
                    vals[a] = fv
            if vals:
                vals["layer"] = name
                rows.append(vals)
    except Exception as e:
        info["scales_error"] = repr(e)[:200]

    def pick(r, names):
        for n in names:
            if r.get(n) is not None:
                return r[n]
        return None

    ks = [x for x in (pick(r, ("k_scale_float", "_k_scale_float", "k_scale", "_k_scale")) for r in rows) if x is not None]
    vs = [x for x in (pick(r, ("v_scale_float", "_v_scale_float", "v_scale", "_v_scale")) for r in rows) if x is not None]
    info["kv_scales"] = {"n": len(rows), "k_min": min(ks) if ks else None, "k_max": max(ks) if ks else None,
                         "v_min": min(vs) if vs else None, "v_max": max(vs) if vs else None, "rows": rows[:128]}
    try:
        info["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})[:8]
    except Exception:
        pass
    return info


def _probe_sglang(sched, where):
    p = {"schema": "e24_hook_probe/2", "engine": "sglang", "pid": os.getpid(), "where": where,
         "tp_rank": ROLE.get("tp_rank"), "t": round(time.time(), 3)}
    tw = getattr(sched, "tp_worker", None)
    p["target"] = _model_info(getattr(getattr(tw, "model_runner", None), "model", None))
    dw = getattr(sched, "draft_worker", None)
    if dw is not None:
        try:
            dm = vars(dw).get("draft_model")      # DFlashWorkerV2.__getattr__ delegates to the target worker
        except Exception:
            dm = None
        p["draft"] = _model_info(dm)
    pool = getattr(sched, "req_to_token_pool", None)
    mp = getattr(pool, "mamba_pool", None)
    mc = getattr(mp, "mamba_cache", None)
    if mc is not None:
        p["mamba"] = {"class": type(mp).__name__, "size": _item(getattr(mp, "size", None)),
                      "temporal_dtype": str(getattr(getattr(mc, "temporal", None), "dtype", None)),
                      "conv_dtypes": sorted({str(getattr(c, "dtype", None)) for c in (getattr(mc, "conv", None) or [])}),
                      "intermediate_ssm_dtype": str(getattr(getattr(mc, "intermediate_ssm", None), "dtype", None))}
    sa = getattr(sched, "server_args", None)
    if sa is not None:
        p["server_args"] = {k: _plain(getattr(sa, k, None)) for k in (
            "kv_cache_dtype", "mamba_ssm_dtype", "dtype", "quantization", "speculative_algorithm",
            "speculative_num_draft_tokens", "mamba_radix_cache_strategy")}
    p["modules"] = dict(MODULES)
    p["resolved"] = dict(RESOLVED)
    PROBE.clear()
    PROBE.update(p)
    _dump_atomic(_out("probe"), p)


def _main_late(obj, where, engine="sglang"):
    """Runs in the engine's main thread at a late point (models loaded, graphs captured)."""
    try:
        gp = getattr(sys.modules.get("sglang.srt.runtime_context"), "get_parallel", None)
        if engine == "sglang" and gp is not None:
            par = gp()
            ROLE["tp_rank"] = _item(getattr(par, "tp_rank", None))
            ROLE["tp_size"] = _item(getattr(par, "tp_size", None))
        if engine == "sglang":
            pool, tree = getattr(obj, "req_to_token_pool", None), getattr(obj, "tree_cache", None)
            if pool is not None and getattr(pool, "mamba_allocator", None) is not None:
                _REFS["pool"] = pool
            if tree is not None:
                _REFS["tree"] = tree
            _refresh_snapshot(force=True)
    except Exception as e:
        _hook_error("tp_rank", e)
    try:
        _resolve_dirs(where)
    except Exception as e:
        RESOLVED["error"] = repr(e)[:200]
    try:
        _module_shas()
    except Exception as e:
        _hook_error("module_shas", e)
    if _PROBE_ON and not PROBE and where in ("Scheduler.run_event_loop", "GPUModelRunner.load_model"):
        try:
            if engine == "sglang":
                _probe_sglang(obj, where)
            else:
                _probe_vllm(obj, where)
        except Exception as e:
            _hook_error("probe", e)
    ENV["dirty"] = True
    _VER[0] += 1


def _probe_vllm(runner, where):
    cc = getattr(runner, "cache_config", None)
    p = {"schema": "e24_hook_probe/2", "engine": "vllm", "pid": os.getpid(), "where": where, "t": round(time.time(), 3),
         "rank": _item(getattr(runner, "rank", None)), "local_rank": _item(getattr(runner, "local_rank", None)),
         "device": str(getattr(runner, "device", None)), "model": _model_info(getattr(runner, "model", None)),
         "cache_config": {k: str(getattr(cc, k, None)) for k in ("cache_dtype", "mamba_cache_dtype",
                                                                  "mamba_ssm_cache_dtype")} if cc is not None else None,
         "resolved": dict(RESOLVED)}
    PROBE.clear()
    PROBE.update(p)
    _dump_atomic(_out("probe"), p)


# ------------------------------------------------------------------------------------------------- installers
def _inst_scheduler(m, cls):
    orig_init, orig_loop = cls.__init__, cls.run_event_loop

    def __init__(self, *a, **k):
        try:
            orig_init(self, *a, **k)
        except BaseException as e:
            _exc("Scheduler.__init__", e)
            raise
        _main_late(self, "Scheduler.__init__")

    def run_event_loop(self, *a, **k):
        _main_late(self, "Scheduler.run_event_loop")
        try:
            return orig_loop(self, *a, **k)
        except BaseException as e:
            _exc("Scheduler.run_event_loop", e)
            raise

    cls.__init__ = _named(__init__, orig_init)
    cls.run_event_loop = _named(run_event_loop, orig_loop)
    orig_idle = getattr(cls, "on_idle", None)
    if _COUNTERS_ON and orig_idle is not None:            # both event loops call it when there is no batch

        def on_idle(self, *a, **k):
            try:
                out = orig_idle(self, *a, **k)
            except BaseException as e:
                _exc("Scheduler.on_idle", e)
                raise
            _refresh_snapshot()                           # freed slots show up in the snapshot while idle
            return out

        cls.on_idle = _named(on_idle, orig_idle)


def _inst_vllm_runner(m, cls):
    orig = cls.load_model

    def load_model(self, *a, **k):
        try:
            out = orig(self, *a, **k)
        except BaseException as e:
            _exc("GPUModelRunner.load_model", e)
            raise
        _main_late(self, "GPUModelRunner.load_model", engine="vllm")
        return out

    cls.load_model = _named(load_model, orig)


# --- #43567 chain (E17, unchanged logic) -------------------------------------------------------------------------
def _inst_schedule_batch(m, cls):
    in_window = getattr(m, "mamba_lazy_spec_in_window")
    _wrap_prealloc(cls)
    _wrap_spec_prepare(cls, in_window)


def _wrap_prealloc(cls):
    orig = cls.mamba_lazy_prealloc_at_boundary

    def mamba_lazy_prealloc_at_boundary(self, mamba_track_interval, *a, **k):
        pre = []
        try:
            _REFS["pool"], _REFS["tree"] = self.req_to_token_pool, self.tree_cache
            for i, req in enumerate(self.reqs):
                kv = _kv(req)
                buf = getattr(kv, "mamba_ping_pong_track_buffer", None)
                if buf is None or _item(self.seq_lens_cpu[i]) % mamba_track_interval != 0:
                    continue
                other = 1 - kv.mamba_next_track_idx
                if _item(buf[other]) != -1:
                    _count("boundary_skip_pending")
                    continue
                pre.append((i, req, other, kv.mamba_last_track_seqlen, _item(self.seq_lens_cpu[i])))
        except Exception as e:
            _hook_error("prealloc_pre", e)
        try:
            out = orig(self, mamba_track_interval, *a, **k)
        except BaseException as e:
            _exc("mamba_lazy_prealloc_at_boundary", e)
            raise
        try:
            for i, req, other, last, seq in pre:
                _count("optional_alloc_attempts")
                kv = _kv(req)
                if _item(kv.mamba_ping_pong_track_buffer[other]) == -1:
                    _count("optional_alloc_failed")
                    _rcount(_rid(req), "optional_alloc_failed")
                    MARK[_rid(req)] = {"since": seq, "len": _plain(last), "path": "decode"}
                    _ev("alloc_failed", path="decode", rid=_rid(req), seq_len=seq, last_len_before=last,
                        last_len_after=kv.mamba_last_track_seqlen, interval=mamba_track_interval,
                        keep_idx=kv.mamba_next_track_idx,
                        available=_item(self.req_to_token_pool.mamba_allocator.available_size()))
                elif _VERBOSE:
                    _ev("alloc_ok", path="decode", rid=_rid(req), seq_len=seq)
        except Exception as e:
            _hook_error("prealloc_post", e)
        _refresh_snapshot()
        return out

    cls.mamba_lazy_prealloc_at_boundary = _named(mamba_lazy_prealloc_at_boundary, orig)


def _wrap_spec_prepare(cls, in_window):
    orig = cls.mamba_lazy_spec_prepare

    def mamba_lazy_spec_prepare(self, mamba_track_interval, max_draft_tokens, *a, **k):
        pre = []
        try:
            _REFS["pool"], _REFS["tree"] = self.req_to_token_pool, self.tree_cache
            for i, req in enumerate(self.reqs):
                kv = _kv(req)
                buf = getattr(kv, "mamba_ping_pong_track_buffer", None)
                if buf is None or not in_window(req, mamba_track_interval, max_draft_tokens):
                    continue
                other = 1 - kv.mamba_next_track_idx
                pre.append((i, req, _item(buf[other]) != -1, kv.mamba_last_track_seqlen, kv.kv_committed_len))
        except Exception as e:
            _hook_error("spec_pre", e)
        try:
            out = orig(self, mamba_track_interval, max_draft_tokens, *a, **k)
        except BaseException as e:
            _exc("mamba_lazy_spec_prepare", e)
            raise
        try:
            pos = getattr(self, "mamba_lazy_spec_track_positions_cpu", None) or []
            for i, req, had_pending, last, committed in pre:
                _count("spec_window_reachable")
                if had_pending:
                    continue
                _count("optional_alloc_attempts")
                kv = _kv(req)
                if i < len(pos) and pos[i] == kv.mamba_next_track_idx:
                    _count("optional_alloc_failed")
                    _count("spec_inplace_fallback")
                    _rcount(_rid(req), "optional_alloc_failed")
                    _rcount(_rid(req), "spec_inplace_fallback")
                    MARK[_rid(req)] = {"since": _plain(committed), "len": _plain(last), "path": "spec"}
                    _ev("alloc_failed", path="spec", rid=_rid(req), kv_committed_len=committed, last_len_before=last,
                        last_len_after=kv.mamba_last_track_seqlen, interval=mamba_track_interval,
                        max_draft_tokens=max_draft_tokens, keep_idx=kv.mamba_next_track_idx,
                        available=_item(self.req_to_token_pool.mamba_allocator.available_size()))
                elif _VERBOSE:
                    _ev("alloc_ok", path="spec", rid=_rid(req), kv_committed_len=committed)
        except Exception as e:
            _hook_error("spec_post", e)
        _refresh_snapshot()
        return out

    cls.mamba_lazy_spec_prepare = _named(mamba_lazy_spec_prepare, orig)


def _inst_batch_result(m, cls):
    for meth in TARGET_METHODS["SchedulerBatchResultProcessor"]:
        getattr(cls, meth)
    _wrap_update(cls, "_mamba_prefix_cache_update", spec=False)
    _wrap_update(cls, "_mamba_lazy_spec_update", spec=True)
    _wrap_boundary(cls, m)


def _wrap_update(cls, name, spec):
    orig = getattr(cls, name)

    def wrapped(self, req, batch, *a, **k):
        kv = _kv(req)
        before = getattr(kv, "mamba_last_track_seqlen", None)
        fin = ins0 = None
        if spec:
            try:
                fin = bool(req.finished())
                ins0 = getattr(req, "mamba_lazy_is_insert", None)
            except Exception:
                fin = None
        try:
            out = orig(self, req, batch, *a, **k)
        except BaseException as e:
            _exc(name, e)
            raise
        try:
            after = getattr(kv, "mamba_last_track_seqlen", None)
            crossed = _arg(a, k, 1, "crossed") if spec else None
            if spec:
                # the verify that a spec fallback planned has now been processed: a real crossing wrote the keep slot
                # in place (a pending stale candidate is confirmed); no crossing wrote nothing (the candidate was harmless)
                if crossed:
                    _count("spec_crossed")
                m0 = MARK.get(_rid(req))
                if m0 is not None and m0.get("path") == "spec":
                    if PEND.pop(_rid(req), None):
                        _count("stale_donation_confirmed" if crossed else "stale_candidate_cleared")
                        _rcount(_rid(req), "stale_donation_confirmed" if crossed else "stale_candidate_cleared")
                        _ev("spec_resolve", rid=_rid(req), crossed=bool(crossed), confirmed=bool(crossed))
                    if not crossed:
                        MARK.pop(_rid(req), None)
                # G-L: a finishing request whose donation the lazy update skipped (#43029 production effect)
                ins1 = getattr(req, "mamba_lazy_is_insert", None)
                if fin and ins0 is not False and ins1 is False:
                    _count("lazy_insert_cleared")
                    _rcount(_rid(req), "lazy_insert_cleared")
                    _ev("lazy_insert_cleared", rid=_rid(req), crossed=bool(crossed), track_seqlen=_arg(a, k, 2,
                                                                                                         "track_seqlen"))
            if after is not None and after != before:
                _count("track_len_updates")
                mk = MARK.pop(_rid(req), None)
                if mk is not None or _VERBOSE:
                    _ev("track_len_update", rid=_rid(req), before=before, after=after, was_marked=mk is not None,
                        spec=spec, crossed=crossed)
        except Exception as e:
            _hook_error("update", e)
        return out

    setattr(cls, name, _named(wrapped, orig))


def _wrap_boundary(cls, mod):
    """#43029: the accepted-length check vs the grammar-committed one, next to what the (patched or not) code returned."""
    orig = cls._mamba_check_track_boundary

    def _mamba_check_track_boundary(self, req, batch, result, i, *a, **k):
        try:
            out = orig(self, req, batch, result, i, *a, **k)
        except BaseException as e:
            _exc("_mamba_check_track_boundary", e)
            raise
        try:
            ncd = getattr(result, "num_correct_drafts_per_req_cpu", None)
            sa = getattr(batch, "spec_algorithm", None)
            if ncd is not None and sa is not None and not sa.is_none():
                rid = _rid(req)
                interval = int(getattr(mod, "mamba_track_grid")(self.tree_cache.page_size))
                acc = int(_item(ncd[i])) + 1
                cur = int(req.seqlen) - 1
                nnd = int(getattr(result, "num_non_draft_tokens_per_req", 1) or 1)
                ret = getattr(result, "grammar_retained_tokens", None)
                kept = None
                committed = acc
                if ret is not None and ret[i] is not None:
                    kept = len(ret[i])
                    committed = max(kept - (nnd - 1), 0)
                    _count("grammar_steps")
                    _rcount(rid, "grammar_steps")
                    if committed < acc:
                        _count("grammar_truncated_steps")
                        _rcount(rid, "grammar_truncated_steps")
                buggy = cur // interval != (cur - acc) // interval
                fixed = cur // interval != (cur - committed) // interval
                returned = bool(out[0]) if isinstance(out, (tuple, list)) else bool(out)
                _count("spec_boundary_checks")
                if buggy and not fixed:
                    _count("false_crossing_possible")
                    _rcount(rid, "false_crossing_possible")
                if returned and not fixed:
                    _count("false_crossing_returned")
                    _rcount(rid, "false_crossing_returned")
                if kept is not None or buggy != fixed or returned != fixed or _VERBOSE:
                    _ev("track_boundary", rid=rid, cur=cur, acc=acc, kept=kept, nnd=nnd, committed=committed,
                        interval=interval, buggy=buggy, fixed=fixed, returned=returned,
                        track_seqlen=out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None)
        except Exception as e:
            _hook_error("boundary", e)
        return out

    cls._mamba_check_track_boundary = _named(_mamba_check_track_boundary, orig)


# --- #43567 donation + #43559 allocation -------------------------------------------------------------------------
def _inst_mamba(m, cls):
    for meth in TARGET_METHODS["MambaComponent"]:
        getattr(cls, meth)
    _wrap_donate(cls)
    _wrap_alloc_slot(cls)


def _wrap_donate(cls):
    orig = cls.prepare_for_caching_req

    def prepare_for_caching_req(self, req, insert_params, token_ids_len, is_finished, *a, **k):
        kv = _kv(req)
        rid = _rid(req)
        try:
            extra = bool(getattr(self.cache, "enable_mamba_extra_buffer", False))
        except Exception:
            extra = None
        cache_len = getattr(kv, "mamba_last_track_seqlen", None) if extra else token_ids_len
        m = MARK.get(rid)
        prev = _CUR_RID[0]
        _CUR_RID[0] = rid
        try:
            out = orig(self, req, insert_params, token_ids_len, is_finished, *a, **k)
        except BaseException as e:
            _exc("prepare_for_caching_req", e)
            raise
        finally:
            _CUR_RID[0] = prev
        try:
            _count("donations_finished" if is_finished else "donations_unfinished")
            stale = bool(m is not None and extra and cache_len is not None and isinstance(out, int) and out > 0)
            if stale:
                _count("stale_donation_candidate")
                _rcount(rid, "stale_donation_candidate")
                if m.get("path") == "decode":       # the boundary forward writes the keep slot for certain
                    _count("stale_donation_confirmed")
                    _rcount(rid, "stale_donation_confirmed")
                else:
                    PEND[rid] = True
            if m is not None:
                _count("donations_while_marked")
            if stale or m is not None or _VERBOSE:
                _ev("donation", rid=rid, finished=bool(is_finished), cache_len=cache_len, returned=out,
                    token_ids_len=token_ids_len, marked=m, stale_candidate=stale, extra_buffer=extra)
            if is_finished:
                MARK.pop(rid, None)
                PEND.pop(rid, None)
        except Exception as e:
            _hook_error("donate", e)
        _refresh_snapshot()
        return out

    cls.prepare_for_caching_req = _named(prepare_for_caching_req, orig)


def _wrap_alloc_slot(cls):
    """#43559: pristine `_alloc_mamba_slot(self)` and the port `_alloc_mamba_slot(self, req=None)`."""
    orig = cls._alloc_mamba_slot

    def _alloc_mamba_slot(self, *a, **k):
        req = _arg(a, k, 0, "req")
        rid = _rid(req) if req is not None else _CUR_RID[0]
        avail = evict = None
        before = ()
        try:
            avail = _item(self.cache.req_to_token_pool.mamba_allocator.available_size())
            mes = getattr(self.cache, "mamba_evictable_size", None)
            evict = _item(mes()) if callable(mes) else None
            before = _skipped(req)
        except Exception as e:
            _hook_error("alloc_pre", e)
        try:
            out = orig(self, *a, **k)
        except BaseException as e:
            if isinstance(e, AssertionError):
                _count("donation_alloc_assert")
                _rcount(rid, "donation_alloc_assert")
                _ev("donation_alloc_assert", rid=rid, available=avail, evictable=evict, req_passed=req is not None,
                    skipped=list(_skipped(req)), err=str(e)[:200])
            _refresh_snapshot(force=True)
            _exc("_alloc_mamba_slot", e)
            raise
        try:
            _count("donation_alloc_calls")
            if avail == 0:
                _count("donation_alloc_first_try_failed")
                _rcount(rid, "donation_alloc_first_try_failed")
                if out is not None:
                    _count("donation_alloc_evicted_ok")
            after = _skipped(req)
            if req is not None and len(after) > len(before):
                _count("prefix_lock_released")
                _rcount(rid, "prefix_lock_released")
                _ev("prefix_lock_released", rid=rid, available_before=avail, evictable_before=evict,
                    skipped=list(after))
            elif _VERBOSE:
                _ev("donation_alloc", rid=rid, available_before=avail, evictable_before=evict)
        except Exception as e:
            _hook_error("alloc_post", e)
        _refresh_snapshot()
        return out

    cls._alloc_mamba_slot = _named(_alloc_mamba_slot, orig)


# --- DFlash rid context + the two injections --------------------------------------------------------------------
def _inst_dflash_worker(m, cls):
    orig = cls.forward_batch_generation

    def forward_batch_generation(self, batch, *a, **k):
        prev = _CTX[0]
        ctx = None
        try:
            fm = getattr(batch, "forward_mode", None)
            if fm is not None and (fm.is_extend() or getattr(batch, "is_extend_in_batch", False)):
                mode = "extend"
            elif fm is not None and fm.is_idle():
                mode = "idle"
            else:
                mode = "decode"
            ctx = {"rids": [_rid(r) for r in (getattr(batch, "reqs", None) or [])], "mode": mode,
                   "block": int(vars(self).get("block_size", 0) or 0)}
            _count("dflash_forward_" + mode)
        except Exception as e:
            _hook_error("dflash_ctx", e)
        _CTX[0] = ctx
        try:
            return orig(self, batch, *a, **k)
        except BaseException as e:
            _exc("DFlashWorkerV2.forward_batch_generation", e)
            raise
        finally:
            _CTX[0] = prev

    cls.forward_batch_generation = _named(forward_batch_generation, orig)


def _capturing(torch):
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return True                      # unknown: never inject


def _decode_ctx(kind):
    ctx = _CTX[0]
    if not ctx or ctx.get("mode") != "decode":
        _count(kind + "_skipped_not_decode")
        return None
    return ctx


def _inst_grouped_conv(m, fn):
    """#41736: replace the module global that DFlashGroupedConv._convolve looks up at call time."""
    orig = m._grouped_conv

    def _grouped_conv(*a, **k):
        plan = None
        try:
            plan = _draft_plan(a, k)
        except Exception as e:
            _hook_error("draft_plan", e)
        if plan is None:
            return orig(*a, **k)
        a2 = (plan["x"],) + tuple(a[1:]) if a and "hidden_states" not in k else a
        k2 = dict(k, hidden_states=plan["x"]) if "hidden_states" in k else k
        try:
            out = orig(*a2, **k2)
        except Exception as e:
            _count("inject_error")
            _ev("inject_error", where="_grouped_conv", err=repr(e)[:200])
            return orig(*a, **k)              # a pure function: the clean call is the original computation
        try:
            _draft_check(out, plan)
        except Exception as e:
            _hook_error("draft_check", e)
        return out

    m._grouped_conv = _named(_grouped_conv, orig)


def _draft_plan(a, k):
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    if _capturing(torch):
        _count("inject_skipped_capturing")
        return None
    ctx = _decode_ctx("draft_inject")
    if ctx is None:
        return None
    x = _arg(a, k, 0, "hidden_states")
    block = int(_arg(a, k, 3, "block_size"))
    taps = int(_arg(a, k, 6, "taps", 2))
    rids = ctx["rids"]
    n = len(rids)
    if block <= 0 or x is None or int(x.shape[0]) != n * block:
        _count("draft_inject_skipped_shape")
        return None
    sent = [i for i, r in enumerate(rids) if r.startswith(_PREFIX)]
    if not sent:
        return None
    x2 = x.clone()
    for i in sent:
        x2[(i + 1) * block - 1].fill_(float("nan"))
    ms = getattr(getattr(torch, "_dynamo", None), "mark_static", None)
    if ms is not None:
        ms(x2, 1)                            # _convolve marks dim 1 of its input; keep the same compiled variant
    _count("draft_nan_calls")
    _count("draft_nan_injections", len(sent))
    for i in sent:
        _rcount(rids[i], "draft_nan_injections")
    pairs = [(i, i + 1) for i in sent if i + 1 < n and not rids[i + 1].startswith(_PREFIX)]
    return {"x": x2, "pairs": pairs, "block": block, "taps": taps, "rids": rids}


def _draft_check(out, plan):
    torch = sys.modules.get("torch")
    block, rids = plan["block"], plan["rids"]
    for i, j in plan["pairs"]:
        _count("neighbor_rows_checked")
        bad = False
        for t in range(max(1, plan["taps"] - 1)):
            if not bool(torch.isfinite(out[j * block + t]).all()):
                bad = True
        if bad:
            _count("neighbor_rows_nonfinite")
            _rcount(rids[j], "neighbor_rows_nonfinite")
        key = (rids[i], rids[j], bad)
        if key not in _SEEN and len(_SEEN) < 10000:
            _SEEN.add(key)
            _ev("neighbor", sentinel=rids[i], neighbor=rids[j], row=j * block, nonfinite=bad, batch=len(rids))


def _inst_selector(m, cls):
    """#41737: CandidateSelector.sample_path(self, *, candidate_ids, scores, uniforms, temperatures, greedy_mask)."""
    orig = cls.sample_path

    def sample_path(self, *a, **k):
        plan = None
        try:
            plan = _selector_plan(k)
        except Exception as e:
            _hook_error("selector_plan", e)
        if plan is None:
            return orig(self, *a, **k)
        try:
            out = orig(self, *a, **dict(k, scores=plan["scores"]))
        except Exception as e:
            _count("inject_error")
            _ev("inject_error", where="sample_path", err=repr(e)[:200])
            return orig(self, *a, **k)
        try:
            _selector_check(out, k, plan)
        except Exception as e:
            _hook_error("selector_check", e)
        return out

    cls.sample_path = _named(sample_path, orig)


def _selector_plan(k):
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    if _capturing(torch):
        _count("inject_skipped_capturing")
        return None
    ctx = _decode_ctx("selector_inject")
    if ctx is None:
        return None
    scores, gm, cand = k.get("scores"), k.get("greedy_mask"), k.get("candidate_ids")
    rids = ctx["rids"]
    if scores is None or gm is None or cand is None or int(scores.shape[0]) != len(rids):
        _count("selector_inject_skipped_shape")
        return None
    sent = [i for i, r in enumerate(rids) if r.startswith(_PREFIX)]
    if not sent:
        return None
    greedy = [bool(v) for v in gm.tolist()]
    targets = [i for i in sent if i < len(greedy) and greedy[i]]
    if len(targets) < len(sent):
        _count("selector_skipped_nongreedy", len(sent) - len(targets))
    if not targets:
        return None
    s2 = scores.clone()
    for i in targets:
        s2[i, 0, 0].fill_(float("nan"))      # slot 0 only: the out-of-range walk stays inside the request's buffers
    _count("selector_nan_injections", len(targets))
    for i in targets:
        _rcount(rids[i], "selector_nan_injections")
    return {"scores": s2, "targets": targets, "rids": rids}


def _selector_check(out, k, plan):
    tokens, q = out[0], out[1]
    cand = k["candidate_ids"]
    for i in plan["targets"]:
        rid = plan["rids"][i]
        tok = int(tokens[i, 0].item())
        row = [int(v) for v in cand[i, 0].tolist()]
        qs = float(q[i, 0].sum().item())
        if qs == 0.0:
            _count("zero_q_rows")
            _rcount(rid, "zero_q_rows")
        if tok not in row:
            _count("out_of_row_picks")
            _rcount(rid, "out_of_row_picks")
        elif tok == row[0]:
            _count("first_candidate_picks")
            _rcount(rid, "first_candidate_picks")
        key = ("sel", rid, qs == 0.0, tok in row)
        if key not in _SEEN and len(_SEEN) < 10000:
            _SEEN.add(key)
            _ev("selector_pick", rid=rid, token=tok, in_row=tok in row, first=row[:1] == [tok], q_sum=qs,
                row_head=row[:4])


_INSTALL = {"Scheduler": _inst_scheduler, "ScheduleBatch": _inst_schedule_batch,
            "SchedulerBatchResultProcessor": _inst_batch_result, "MambaComponent": _inst_mamba,
            "DFlashWorkerV2": _inst_dflash_worker, "_grouped_conv": _inst_grouped_conv,
            "CandidateSelector": _inst_selector, "GPUModelRunner": _inst_vllm_runner}


# ------------------------------------------------------------------------------------------------- the thread
def _try_patch():
    """Thread: sys.modules lookups and attribute writes only. True when every enabled target is patched or failed."""
    for name in ENABLED:
        if name in PATCHED or name in FAILED:
            continue
        mod_name, attr = TARGETS[name]
        m = sys.modules.get(mod_name)
        if m is None:
            continue
        obj = getattr(m, attr, None)
        if obj is None:
            continue
        try:
            for meth in TARGET_METHODS[name]:
                getattr(obj, meth)              # all present before anything is replaced
            _INSTALL[name](m, obj)
            PATCHED.append(name)
            _ev("patched", target=name, module=mod_name)
        except Exception as e:
            FAILED[name] = repr(e)[:300]
            _count("patch_failed")
            _ev("patch_failed", target=name, module=mod_name, err=repr(e)[:300])
    return all(t in PATCHED or t in FAILED for t in ENABLED)


def _loop():
    t_start = time.time()
    next_flush = 0.0
    while True:
        sl = 1.0
        try:
            now = time.time()
            done = _try_patch()
            if now >= next_flush:
                lk = _LOCK[0]
                if lk.acquire(timeout=5.0):
                    try:
                        _snap_env()
                        _flush_locked()
                    finally:
                        lk.release()
                next_flush = now + _FLUSH
            if now - t_start >= _FAST:
                sl, ph = max(1.0, _FLUSH), "slow"
            elif not done:
                sl, ph = 0.05, "fast"
            else:
                sl, ph = _FLUSH, "steady"
            _POLL.update(sleep=sl, phase=ph, loops=_POLL["loops"] + 1)
        except Exception:
            sl = 1.0
        time.sleep(sl)


def _start():
    threading.Thread(target=_loop, name="e24-hook", daemon=True).start()


def _after_fork_child():
    """os.register_at_fork(after_in_child): fresh lock and buffers, this pid's env record now (no imports, no inherited
    lock), then a new thread. The patched classes are inherited, so PATCHED stays."""
    try:
        parent = _PID[0]
        _PID[0] = os.getpid()
        _LOCK[0] = threading.Lock()
        EVENTS.clear()
        COUNTERS.clear()
        PER_RID.clear()
        MARK.clear()
        PEND.clear()
        _SEEN.clear()
        _EV_KIND_N.clear()
        ROLE.clear()
        RESOLVED.clear()
        PROBE.clear()
        _CTX[0] = None
        _CUR_RID[0] = None
        _VER[0] += 1
        _FLUSHED[0] = -1
        _LAST_SNAP[0] = None
        _SNAP[0] = {}
        ENV.update(first=None, last=None, role=None, argv=None, changes=[], source="fork", forked_from=parent,
                   t_first=round(time.time(), 3), dirty=True)
        _snap_env()
        _write_env()
    except Exception:
        pass
    try:
        _start()
    except Exception:
        pass


# ------------------------------------------------------------------------------------------------- chain + init
def _chain():
    """Run the next sitecustomize on sys.path exactly as site.execsitecustomize would have (this file shadows it)."""
    out = {"found": False}
    try:
        me = os.path.realpath(__file__)
        mydir = os.path.dirname(me)
        entries = list(sys.path)
        real = [os.path.realpath(e or os.getcwd()) for e in entries]
        start = real.index(mydir) + 1 if mydir in real else 0
        for e in entries[start:]:
            cand = os.path.join(e or os.getcwd(), "sitecustomize.py")
            if not os.path.isfile(cand) or os.path.realpath(cand) == me:
                continue
            with open(cand, "rb") as f:
                src = f.read()
            if _MARKER.encode() in src:
                out.setdefault("skipped", []).append(cand)
                continue
            out.update(found=True, path=cand, sha256=hashlib.sha256(src).hexdigest())
            g = {"__name__": "sitecustomize", "__file__": cand, "__builtins__": __builtins__}
            try:
                exec(compile(src, cand, "exec"), g)
                out["ok"] = True
            except Exception as err:
                if isinstance(err, ImportError) and getattr(err, "name", None) == "sitecustomize":
                    out["ok"] = True
                else:
                    out.update(ok=False, error="%s: %s" % (err.__class__.__name__, err))
                    sys.stderr.write("Error in sitecustomize; set PYTHONVERBOSE for traceback:\n%s: %s\n"
                                     % (err.__class__.__name__, err))
            break
    except Exception as e:
        out["hook_error"] = repr(e)[:300]
    return out


def _init():
    if not _DIR:
        return
    try:
        _SELF["file"] = os.path.realpath(__file__)
        with open(__file__, "rb") as f:
            _SELF["sha256"] = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        pass
    try:
        _snap_env()
        _write_env()
    except Exception:
        pass
    _start()
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_after_fork_child)
    atexit.register(_flush_sync)


if os.environ.get("E24_HOOK_CHAIN", "1") != "0":
    CHAIN.update(_chain())
try:
    _init()
except Exception as _e:
    CHAIN["init_error"] = repr(_e)[:300]
