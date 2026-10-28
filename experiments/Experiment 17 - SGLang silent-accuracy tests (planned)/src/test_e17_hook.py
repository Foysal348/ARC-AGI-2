"""CPU tests of e17_hook.py against mock SGLang 0.5.21 classes.

The method bodies below are reduced copies of the 0.5.21 logic the hook wraps (schedule_batch.py:3464-3533,
batch_result_processor.py:1381-1490, mamba.py:539-575), with plain ints in place of tensors. The #43567 backport (the
two `else: ... = None` hunks) is switched by PATCHED_43567. Run: python -m pytest -q test_e17_hook.py
"""
import importlib
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


class T(int):
    def item(self):
        return int(self)


class Alloc:
    def __init__(self, free):
        self.free = free

    def available_size(self):
        return self.free

    def alloc(self, n):
        if self.free < n:
            return None
        self.free -= n
        return [T(100 + self.free)]


class Pool:
    def __init__(self, free):
        self.mamba_allocator = Alloc(free)
        self.mamba_pool = types.SimpleNamespace(size=4)

    def set_mamba_ping_pong_slot(self, req, idx, slot):
        req.kv.mamba_ping_pong_track_buffer[idx] = T(slot)


class Tree:
    def mamba_evictable_size(self):
        return 2


class KV:
    def __init__(self):
        self.mamba_ping_pong_track_buffer = [T(7), T(-1)]
        self.mamba_next_track_idx = 0
        self.mamba_last_track_seqlen = None
        self.kv_committed_len = 0


class Req:
    def __init__(self, rid):
        self.rid, self.kv = rid, KV()


PATCHED_43567 = [False]
FORCE_FAIL = [False]


def in_window(req, interval, max_draft):
    s = req.kv.kv_committed_len
    return s // interval != (s + 2 * max_draft) // interval


class ScheduleBatch:
    def __init__(self, reqs, seq_lens, free):
        self.reqs, self.seq_lens_cpu = reqs, [T(x) for x in seq_lens]
        self.req_to_token_pool, self.tree_cache = Pool(free), Tree()

    def mamba_lazy_prealloc_at_boundary(self, mamba_track_interval):
        pool = self.req_to_token_pool
        for i, req in enumerate(self.reqs):
            buf = req.kv.mamba_ping_pong_track_buffer
            if self.seq_lens_cpu[i].item() % mamba_track_interval != 0:
                continue
            other_idx = 1 - req.kv.mamba_next_track_idx
            if buf[other_idx].item() != -1:
                continue
            new_slot = None if FORCE_FAIL[0] else pool.mamba_allocator.alloc(1)
            if new_slot is not None:
                pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])
                req.kv.mamba_next_track_idx = other_idx
            elif PATCHED_43567[0]:
                req.kv.mamba_last_track_seqlen = None

    def mamba_lazy_spec_prepare(self, mamba_track_interval, max_draft_tokens):
        pool = self.req_to_token_pool
        pos = []
        for req in self.reqs:
            buf = req.kv.mamba_ping_pong_track_buffer
            if not in_window(req, mamba_track_interval, max_draft_tokens):
                pos.append(req.kv.mamba_next_track_idx)
                continue
            other_idx = 1 - req.kv.mamba_next_track_idx
            has_pending = buf[other_idx].item() != -1
            if not has_pending:
                new_slot = None if FORCE_FAIL[0] else pool.mamba_allocator.alloc(1)
                if new_slot is not None:
                    pool.set_mamba_ping_pong_slot(req, other_idx, new_slot[0])
                    has_pending = True
                elif PATCHED_43567[0]:
                    req.kv.mamba_last_track_seqlen = None
            pos.append(other_idx if has_pending else req.kv.mamba_next_track_idx)
        self.mamba_lazy_spec_track_positions_cpu = pos


class SchedulerBatchResultProcessor:
    def _mamba_prefix_cache_update(self, req, batch, result, i, known_boundary=False):
        req.kv.mamba_last_track_seqlen = result

    def _mamba_lazy_spec_update(self, req, batch, i, crossed, track_seqlen):
        if crossed:
            req.kv.mamba_last_track_seqlen = track_seqlen


class MambaComponent:
    def __init__(self):
        self.cache = types.SimpleNamespace(enable_mamba_extra_buffer=True)

    def prepare_for_caching_req(self, req, insert_params, token_ids_len, is_finished):
        cache_len = req.kv.mamba_last_track_seqlen
        if is_finished:
            return cache_len or 0
        return 0 if cache_len is None else cache_len


class ModelRunner:
    tp_rank = 0

    def load_model(self):
        layer = types.SimpleNamespace(k_scale_float=0.5, v_scale_float=0.25)
        self.model = types.SimpleNamespace(named_modules=lambda: [("layers.3.attn", layer), ("layers.0.gdn", object())])


_ORIG = {c: dict(vars(c)) for c in (ScheduleBatch, SchedulerBatchResultProcessor, MambaComponent, ModelRunner)}


def _restore():
    for c, d in _ORIG.items():
        for k, v in d.items():
            if callable(v) and not k.startswith("__"):
                setattr(c, k, v)


def _load(tmp, verbose="1"):
    _restore()
    os.environ["E17_TRACE_DIR"], os.environ["E17_FLUSH_SEC"], os.environ["E17_TRACE_VERBOSE"] = tmp, "0.05", verbose
    for name, cls in (("sglang.srt.managers.schedule_batch", ScheduleBatch),
                      ("sglang.srt.managers.scheduler_components.batch_result_processor", SchedulerBatchResultProcessor),
                      ("sglang.srt.mem_cache.unified_cache.components.mamba", MambaComponent),
                      ("sglang.srt.model_executor.model_runner", ModelRunner)):
        m = types.ModuleType(name)
        setattr(m, cls.__name__, cls)
        if name.endswith("schedule_batch"):
            m.mamba_lazy_spec_in_window = in_window
        sys.modules[name] = m
    sys.path.insert(0, str(HERE))
    sys.modules.pop("e17_hook", None)
    h = importlib.import_module("e17_hook")
    end = time.time() + 5
    while len(h.PATCHED) < 4 and time.time() < end:
        time.sleep(0.02)
    assert len(h.PATCHED) == 4, h.PATCHED
    return h


def _read(tmp):
    time.sleep(0.3)
    cs = [json.loads(p.read_text()) for p in Path(tmp).glob("counters_*.json")]
    evs = [json.loads(l) for p in Path(tmp).glob("trace_*.jsonl") for l in p.read_text().splitlines()]
    return cs[0], evs


def _bug_order(h):
    """#43567 upstream order: prefill sets the 192 checkpoint; the first decode crosses 256 with no free slot; the
    overlapped prefill-result processing donates; then the decode's result processing commits 256."""
    req = Req("r1")
    req.kv.mamba_last_track_seqlen = 192
    b = ScheduleBatch([req], [256], free=0)
    b.mamba_lazy_prealloc_at_boundary(128)
    don = MambaComponent().prepare_for_caching_req(req, None, 255, False)
    SchedulerBatchResultProcessor()._mamba_prefix_cache_update(req, b, 256, 0)
    return don


def test_unpatched_chain_is_confirmed():
    PATCHED_43567[0], FORCE_FAIL[0] = False, False
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        h = _load(tmp)
        assert _bug_order(h) == 192
        c, evs = _read(tmp)
        k = c["counters"]
        assert k["optional_alloc_attempts"] == 1 and k["optional_alloc_failed"] == 1
        assert k["stale_donation_candidate"] == 1 and k["stale_donation_confirmed"] == 1
        assert any(e["ev"] == "track_len_update" and e["was_marked"] for e in evs)
        assert c["alloc"] == {"available": 0, "size": 4, "evictable": 2}
        assert len(c["patched"]) == 4


def test_patched_has_no_candidate():
    PATCHED_43567[0], FORCE_FAIL[0] = True, False
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        h = _load(tmp)
        assert _bug_order(h) == 0
        k = _read(tmp)[0]["counters"]
        assert k["optional_alloc_failed"] == 1 and k.get("stale_donation_candidate", 0) == 0
    PATCHED_43567[0] = False


def test_success_is_not_a_failure():
    PATCHED_43567[0], FORCE_FAIL[0] = False, False
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        h = _load(tmp)
        req = Req("r2")
        req.kv.mamba_last_track_seqlen = 192
        b = ScheduleBatch([req], [256], free=3)
        b.mamba_lazy_prealloc_at_boundary(128)
        MambaComponent().prepare_for_caching_req(req, None, 255, False)
        k = _read(tmp)[0]["counters"]
        assert k["optional_alloc_attempts"] == 1 and k.get("optional_alloc_failed", 0) == 0
        assert k.get("stale_donation_candidate", 0) == 0


def test_forced_fail_counted_without_alloc_call():
    PATCHED_43567[0], FORCE_FAIL[0] = False, True
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        h = _load(tmp)
        _bug_order(h)
        k = _read(tmp)[0]["counters"]
        assert k["optional_alloc_failed"] == 1 and k["stale_donation_confirmed"] == 1
    FORCE_FAIL[0] = False


def test_spec_candidate_confirmed_or_cleared():
    PATCHED_43567[0], FORCE_FAIL[0] = False, False
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        h = _load(tmp)
        rp = SchedulerBatchResultProcessor()
        for rid, crossed in (("s1", True), ("s2", False)):
            req = Req(rid)
            req.kv.mamba_last_track_seqlen, req.kv.kv_committed_len = 192, 250
            b = ScheduleBatch([req], [250], free=0)
            b.mamba_lazy_spec_prepare(64, 6)
            assert b.mamba_lazy_spec_track_positions_cpu == [0]
            assert MambaComponent().prepare_for_caching_req(req, None, 249, False) == 192
            rp._mamba_lazy_spec_update(req, b, 0, crossed, 256)
        k = _read(tmp)[0]["counters"]
        assert k["spec_inplace_fallback"] == 2 and k["stale_donation_candidate"] == 2
        assert k["stale_donation_confirmed"] == 1 and k["stale_candidate_cleared"] == 1
        assert not h.MARK and not h.PEND


def test_spec_out_of_window_not_counted():
    PATCHED_43567[0], FORCE_FAIL[0] = False, False
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        h = _load(tmp)
        req = Req("s3")
        req.kv.kv_committed_len = 200
        b = ScheduleBatch([req], [200], free=0)
        b.mamba_lazy_spec_prepare(64, 6)
        k = _read(tmp)[0]["counters"]
        assert k.get("spec_window_reachable", 0) == 0 and k.get("optional_alloc_failed", 0) == 0


def test_kv_scales_logged():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        h = _load(tmp)
        r = ModelRunner()
        r.load_model()
        evs = _read(tmp)[1]
        ks = [e for e in evs if e["ev"] == "kv_scales"]
        assert ks and ks[0]["n"] == 1 and ks[0]["rows"][0]["k_scale_float"] == 0.5


def test_inert_without_trace_dir():
    os.environ.pop("E17_TRACE_DIR", None)
    sys.modules.pop("e17_hook", None)
    h = importlib.import_module("e17_hook")
    time.sleep(0.2)
    assert h._DIR is None and not h.PATCHED
