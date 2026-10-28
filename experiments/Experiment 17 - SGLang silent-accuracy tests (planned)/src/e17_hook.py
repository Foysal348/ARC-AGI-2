"""E17 instrumentation for SGLang 0.5.21, the #43567 stale lazy-checkpoint chain (PLAN_E17 rev 2 §2).

Installed as `sitecustomize.py` in a directory put on PYTHONPATH of the server launch only. It is inert unless
E17_TRACE_DIR is set. It never edits a source file: a daemon thread waits until the target modules are imported, then
wraps four methods in place. Each wrapper infers the outcome from the request state before and after the original
call. The unpatched arm therefore runs byte-identical SGLang code.

Wrapped (0.5.21 line numbers):
- ScheduleBatch.mamba_lazy_prealloc_at_boundary (schedule_batch.py:3464-3493). Eligible = seq_len % interval == 0
  and no pending slot. Afterwards the pending slot is still -1 -> optional_alloc_failed; the request is then marked
  `inplace_pending` (the forward writes the new state into the keep slot).
- ScheduleBatch.mamba_lazy_spec_prepare (:3495-3533). In window and no pending before; the planned position equals the
  keep index afterwards -> spec_inplace_fallback (same mark).
- SchedulerBatchResultProcessor._mamba_prefix_cache_update / _mamba_lazy_spec_update (batch_result_processor.py:
  1381-1490). A new non-None mamba_last_track_seqlen clears the mark (the checkpoint is consistent again).
- MambaComponent.prepare_for_caching_req (unified_cache/components/mamba.py:539). Logs every donation.
  stale_donation_candidate = marked and the length is not None and the returned length is > 0.
- ModelRunner.load_model (TP rank 0): logs the KV-cache scales of every attention layer once.

Output, per process: trace_<pid>.jsonl (events, appended) and counters_<pid>.json (counters, the list of patched
targets and an allocator snapshot), both flushed every E17_FLUSH_SEC (default 2 s) by a daemon thread. Nothing is
written on the scheduler's hot path, and there is no CUDA sync beyond the .item() reads the original code does for the
same requests.
"""
import json
import os
import sys
import threading
import time
from collections import deque

_DIR = os.environ.get("E17_TRACE_DIR")
_VERBOSE = os.environ.get("E17_TRACE_VERBOSE") == "1"
_FLUSH = float(os.environ.get("E17_FLUSH_SEC", "2"))
_PID = os.getpid()
_LOCK = threading.Lock()
EVENTS = deque(maxlen=200000)
COUNTERS = {}
PATCHED = []
MARK = {}             # rid -> {"since": seq_len, "len": mamba_last_track_seqlen at the failure, "path": ...}
PEND = {}             # rid -> a spec-path stale candidate waiting for its verify result (crossed or not)
_REFS = {}            # the last pool / tree cache seen (allocator snapshot)
_SEQ = [0]
TARGETS = {
    "sglang.srt.managers.schedule_batch": "ScheduleBatch",
    "sglang.srt.managers.scheduler_components.batch_result_processor": "SchedulerBatchResultProcessor",
    "sglang.srt.mem_cache.unified_cache.components.mamba": "MambaComponent",
    "sglang.srt.model_executor.model_runner": "ModelRunner",
}


def _count(k, n=1):
    COUNTERS[k] = COUNTERS.get(k, 0) + n


def _ev(kind, **kw):
    _SEQ[0] += 1
    kw.update(ev=kind, i=_SEQ[0], t=round(time.time(), 3))
    EVENTS.append(kw)


def _item(x):
    try:
        return int(x.item())
    except Exception:
        try:
            return int(x)
        except Exception:
            return None


def _rid(req):
    return str(getattr(req, "rid", ""))


def _kv(req):
    return getattr(req, "kv", None)


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
            _count("hook_error_prealloc_pre")
            _ev("hook_error", where="prealloc_pre", err=repr(e)[:200])
        out = orig(self, mamba_track_interval, *a, **k)
        try:
            for i, req, other, last, seq in pre:
                _count("optional_alloc_attempts")
                kv = _kv(req)
                if _item(kv.mamba_ping_pong_track_buffer[other]) == -1:
                    _count("optional_alloc_failed")
                    MARK[_rid(req)] = {"since": seq, "len": last, "path": "decode"}
                    _ev("alloc_failed", path="decode", rid=_rid(req), seq_len=seq, last_len_before=last,
                        last_len_after=kv.mamba_last_track_seqlen, interval=mamba_track_interval,
                        keep_idx=kv.mamba_next_track_idx,
                        available=_item(self.req_to_token_pool.mamba_allocator.available_size()))
                elif _VERBOSE:
                    _ev("alloc_ok", path="decode", rid=_rid(req), seq_len=seq)
        except Exception as e:
            _count("hook_error_prealloc_post")
            _ev("hook_error", where="prealloc_post", err=repr(e)[:200])
        return out

    cls.mamba_lazy_prealloc_at_boundary = mamba_lazy_prealloc_at_boundary


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
            _count("hook_error_spec_pre")
            _ev("hook_error", where="spec_pre", err=repr(e)[:200])
        out = orig(self, mamba_track_interval, max_draft_tokens, *a, **k)
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
                    MARK[_rid(req)] = {"since": committed, "len": last, "path": "spec"}
                    _ev("alloc_failed", path="spec", rid=_rid(req), kv_committed_len=committed, last_len_before=last,
                        last_len_after=kv.mamba_last_track_seqlen, interval=mamba_track_interval,
                        max_draft_tokens=max_draft_tokens, keep_idx=kv.mamba_next_track_idx,
                        available=_item(self.req_to_token_pool.mamba_allocator.available_size()))
                elif _VERBOSE:
                    _ev("alloc_ok", path="spec", rid=_rid(req), kv_committed_len=committed)
        except Exception as e:
            _count("hook_error_spec_post")
            _ev("hook_error", where="spec_post", err=repr(e)[:200])
        return out

    cls.mamba_lazy_spec_prepare = mamba_lazy_spec_prepare


def _wrap_update(cls, name, spec):
    orig = getattr(cls, name)

    def wrapped(self, req, batch, *a, **k):
        kv = _kv(req)
        before = getattr(kv, "mamba_last_track_seqlen", None)
        out = orig(self, req, batch, *a, **k)
        try:
            after = getattr(kv, "mamba_last_track_seqlen", None)
            if spec:
                # the verify that a spec fallback planned has now been processed: a real crossing wrote the keep slot in
                # place (a pending stale candidate is confirmed); no crossing wrote nothing (the candidate was harmless)
                crossed = a[1] if len(a) > 1 else k.get("crossed")
                if crossed:
                    _count("spec_crossed")
                m0 = MARK.get(_rid(req))
                if m0 is not None and m0.get("path") == "spec":
                    if PEND.pop(_rid(req), None):
                        _count("stale_donation_confirmed" if crossed else "stale_candidate_cleared")
                        _ev("spec_resolve", rid=_rid(req), crossed=bool(crossed), confirmed=bool(crossed))
                    if not crossed:
                        MARK.pop(_rid(req), None)
            if after is not None and after != before:
                _count("track_len_updates")
                m = MARK.pop(_rid(req), None)
                if m is not None or _VERBOSE:
                    _ev("track_len_update", rid=_rid(req), before=before, after=after, was_marked=m is not None,
                        spec=spec, crossed=(a[1] if spec and len(a) > 1 else None))
        except Exception as e:
            _count("hook_error_update")
            _ev("hook_error", where=name, err=repr(e)[:200])
        return out

    wrapped.__name__ = name
    setattr(cls, name, wrapped)


def _wrap_donate(cls):
    orig = cls.prepare_for_caching_req

    def prepare_for_caching_req(self, req, insert_params, token_ids_len, is_finished, *a, **k):
        kv = _kv(req)
        try:
            extra = bool(getattr(self.cache, "enable_mamba_extra_buffer", False))
        except Exception:
            extra = None
        cache_len = getattr(kv, "mamba_last_track_seqlen", None) if extra else token_ids_len
        m = MARK.get(_rid(req))
        out = orig(self, req, insert_params, token_ids_len, is_finished, *a, **k)
        try:
            _count("donations_finished" if is_finished else "donations_unfinished")
            stale = bool(m is not None and extra and cache_len is not None and isinstance(out, int) and out > 0)
            if stale:
                _count("stale_donation_candidate")
                if m.get("path") == "decode":       # the boundary forward writes the keep slot for certain
                    _count("stale_donation_confirmed")
                else:
                    PEND[_rid(req)] = True
            if m is not None:
                _count("donations_while_marked")
            if stale or m is not None or _VERBOSE:
                _ev("donation", rid=_rid(req), finished=bool(is_finished), cache_len=cache_len, returned=out,
                    token_ids_len=token_ids_len, marked=m, stale_candidate=stale, extra_buffer=extra)
            if is_finished:
                MARK.pop(_rid(req), None)
                PEND.pop(_rid(req), None)
        except Exception as e:
            _count("hook_error_donate")
            _ev("hook_error", where="donate", err=repr(e)[:200])
        return out

    cls.prepare_for_caching_req = prepare_for_caching_req


def _wrap_load(cls):
    orig = cls.load_model

    def load_model(self, *a, **k):
        out = orig(self, *a, **k)
        try:
            if getattr(self, "tp_rank", 0) == 0 and not getattr(self, "is_draft_worker", False):
                rows = []
                for name, mod in self.model.named_modules():
                    ks, vs = getattr(mod, "k_scale_float", None), getattr(mod, "v_scale_float", None)
                    if ks is None and vs is None and not hasattr(mod, "k_scale"):
                        continue
                    rows.append({"layer": name, "k_scale_float": ks, "v_scale_float": vs,
                                 "k_scale": _f(getattr(mod, "k_scale", None)), "v_scale": _f(getattr(mod, "v_scale", None))})
                _ev("kv_scales", n=len(rows), rows=rows[:128])
        except Exception as e:
            _ev("hook_error", where="load_model", err=repr(e)[:200])
        return out

    cls.load_model = load_model


def _f(x):
    try:
        return float(x.item() if hasattr(x, "item") and getattr(x, "numel", lambda: 1)() == 1 else x)
    except Exception:
        return None if x is None else str(type(x).__name__)


def _try_patch():
    left = [m for m in TARGETS if TARGETS[m] + "@" + m not in PATCHED]
    for mod in left:
        m = sys.modules.get(mod)
        if m is None or not hasattr(m, TARGETS[mod]):
            continue
        cls = getattr(m, TARGETS[mod])
        try:
            if mod.endswith("schedule_batch"):
                _wrap_prealloc(cls)
                _wrap_spec_prepare(cls, getattr(m, "mamba_lazy_spec_in_window"))
            elif mod.endswith("batch_result_processor"):
                _wrap_update(cls, "_mamba_prefix_cache_update", spec=False)
                _wrap_update(cls, "_mamba_lazy_spec_update", spec=True)
            elif mod.endswith("components.mamba"):
                _wrap_donate(cls)
            else:
                _wrap_load(cls)
            PATCHED.append(TARGETS[mod] + "@" + mod)
            _ev("patched", target=TARGETS[mod], module=mod)
        except Exception as e:
            PATCHED.append(TARGETS[mod] + "@" + mod)
            _ev("patch_failed", target=TARGETS[mod], module=mod, err=repr(e)[:300])
            _count("patch_failed")
    return len(PATCHED) == len(TARGETS)


def _snapshot():
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
    return snap


def _flush():
    os.makedirs(_DIR, exist_ok=True)
    evs = []
    while EVENTS:
        evs.append(EVENTS.popleft())
    if evs:
        with open(os.path.join(_DIR, f"trace_{os.getpid()}.jsonl"), "a", encoding="utf-8") as f:
            for e in evs:
                f.write(json.dumps(e, default=str) + "\n")
    if PATCHED or COUNTERS:
        p = os.path.join(_DIR, f"counters_{os.getpid()}.json")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "argv0": (sys.argv or [""])[0][-120:], "patched": PATCHED, "counters": COUNTERS,
                       "marked_now": len(MARK), "alloc": _snapshot(), "t": round(time.time(), 3)}, f, default=str)
        os.replace(tmp, p)


def _loop():
    t_end = time.time() + 3600
    done = False
    while True:
        try:
            if not done:
                done = _try_patch() or time.time() > t_end
            _flush()
        except Exception:
            pass
        time.sleep(_FLUSH if done else 0.05)


def _start():
    threading.Thread(target=_loop, name="e17-hook", daemon=True).start()


def _after_fork():
    # a forked child inherits the patched classes and the buffers but not the thread; its files carry its own pid
    EVENTS.clear()
    COUNTERS.clear()
    MARK.clear()
    PEND.clear()
    _start()


if _DIR:
    _start()
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_after_fork)
