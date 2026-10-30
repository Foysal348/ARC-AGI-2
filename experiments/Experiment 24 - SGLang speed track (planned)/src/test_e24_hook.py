"""CPU tests of e24_hook.py (rev 2, 2026-10-30), E17 style, every case in its own subprocess.

Each scenario runs `python3.12 scen.py` with PYTHONPATH=<tmp>/hook (the hook copied there as sitecustomize.py, exactly
as the notebook installs it) and a temp E24_HOOK_DIR, so the real site -> sitecustomize startup path runs. The scenario
imports THIS module for the fakes: the wrapped SGLang methods are the REAL 0.5.21 statements cut with ast from the
pristine source (E24_SGLANG_SRC, else the scratch copy of the wheel) or from a copy patched by e24_sglang_patches
(E24T_VARIANT=patched); torch / triton / the DFlash conv and selector are small fakes with the 0.5.21 semantics.
Run: python3.12 -m unittest -v test_e24_hook.py   (stdlib only; ~40 s)
"""
import ast
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import types
import typing
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SCR = Path("/private/tmp/claude-501/-Users-md-hamidhosen-Documents-Kaggle-ARC-AGI-2/e21dd5ac-4964-44ae-8fe0-b7d1156db571/"
           "scratchpad/e24")
SRC = Path(os.environ.get("E24_SGLANG_SRC") or SCR / "downloads/sglang_source_0521/src/sglang")
HAVE_SRC = (SRC / "srt/managers/schedule_batch.py").is_file()
NAN = float("nan")

SB = "srt/managers/schedule_batch.py"
BRP = "srt/managers/scheduler_components/batch_result_processor.py"
MAMBA = "srt/mem_cache/unified_cache/components/mamba.py"
BPC = "srt/mem_cache/base_prefix_cache.py"
M_SB = "sglang.srt.managers.schedule_batch"
M_BRP = "sglang.srt.managers.scheduler_components.batch_result_processor"
M_MAMBA = "sglang.srt.mem_cache.unified_cache.components.mamba"
M_SCHED = "sglang.srt.managers.scheduler"
M_DFLASH = "sglang.srt.models.dflash"
M_WORKER = "sglang.srt.speculative.dflash_worker_v2"


# ======================================================================================================================
# Fakes (imported by the scenario subprocesses)
class TI(int):
    """int with the tensor methods the 0.5.21 code calls on CPU index values."""

    def item(self):
        return int(self)

    def unsqueeze(self, d):
        return self

    def clone(self):
        return self

    def view(self, *a):
        return self


class Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v

    def __bool__(self):
        return bool(self.v)

    def __int__(self):
        return int(self.v)

    def __float__(self):
        return float(self.v)


def _deep(d):
    return [_deep(x) for x in d] if isinstance(d, list) else d


def _leaves(d):
    if isinstance(d, list):
        for x in d:
            yield from _leaves(x)
    else:
        yield d


def _fill(d, v):
    for i, x in enumerate(d):
        if isinstance(x, list):
            _fill(x, v)
        else:
            d[i] = v


class FT:
    """Minimal tensor over nested lists; indexing returns views that share the lists (fill_ writes through)."""

    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        s, d = [], self.data
        while isinstance(d, list):
            s.append(len(d))
            d = d[0] if d else None
        return tuple(s)

    def __getitem__(self, idx):
        d = self.data
        for i in (idx if isinstance(idx, tuple) else (idx,)):
            d = d[i]
        return FT(d) if isinstance(d, list) else Scalar(d)

    def clone(self):
        return FT(_deep(self.data))

    def fill_(self, v):
        _fill(self.data, v)
        return self

    def tolist(self):
        return _deep(self.data)

    def sum(self):
        return Scalar(sum(_leaves(self.data)))

    def all(self):
        return Scalar(all(_leaves(self.data)))


def _map(d, fn):
    return [_map(x, fn) for x in d] if isinstance(d, list) else fn(d)


STATE = {"capturing": False, "mark_static": [], "conv_inputs": [], "cache_dir_calls": 0, "alloc_threads": []}


def fake_torch():
    t = types.ModuleType("torch")
    t.cuda = SimpleNamespace(is_current_stream_capturing=lambda: STATE["capturing"])
    t._dynamo = SimpleNamespace(mark_static=lambda x, d: STATE["mark_static"].append((id(x), d)))
    t.isfinite = lambda x: FT(_map(x.data, math.isfinite)) if isinstance(x, FT) else Scalar(math.isfinite(x.v))
    return t


def _cut_def(text, node):
    seg = textwrap.dedent(ast.get_source_segment(text, node))
    return "".join("@%s\n" % d.id for d in node.decorator_list if isinstance(d, ast.Name)) + seg


def _read(pkg, rel):
    with open(os.path.join(pkg, rel), encoding="utf-8") as f:
        return f.read()


def cut_methods(pkg, rel, cls, names):
    text, out = _read(pkg, rel), {}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in names:
                    out[item.name] = _cut_def(text, item)
    missing = [n for n in names if n not in out]
    assert not missing, (rel, cls, missing)
    return [out[n] for n in names]


def cut_top(pkg, rel, name, deco=""):
    text = _read(pkg, rel)
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return deco + ast.get_source_segment(text, node)
    raise KeyError((rel, name))


def build_class(mod, name, methods):
    body = "\n".join(textwrap.indent(s, "    ") for s in methods) if methods else "    pass"
    exec(compile("from __future__ import annotations\nclass %s:\n%s\n" % (name, body), "<e24t:%s>" % name, "exec"),
         mod.__dict__)
    return mod.__dict__[name]


class _Flag:
    def __init__(self, name):
        self.name = name

    def get(self):
        return os.environ.get(self.name, "0") in ("1", "true", "True")


class Alloc:
    def __init__(self, free):
        self.free_n = free

    def available_size(self):
        return self.free_n

    def alloc(self, n):
        if self.free_n < n:
            return None
        self.free_n -= n
        return [TI(100 + self.free_n)]

    def free(self, x):
        self.free_n += 1


class Pool:
    def __init__(self, free):
        self.mamba_allocator = Alloc(free)
        self.mamba_pool = SimpleNamespace(size=4)

    def set_mamba_ping_pong_slot(self, req, idx, slot):
        req.kv.mamba_ping_pong_track_buffer[idx] = TI(slot)

    def get_mamba_ping_pong_keep_idx(self, req):
        return req.kv.mamba_next_track_idx

    def donate_mamba_ping_pong_slot(self, req, new_slot):
        k = req.kv.mamba_next_track_idx
        old = req.kv.mamba_ping_pong_track_buffer[k]
        req.kv.mamba_ping_pong_track_buffer[k] = TI(new_slot[0])
        return old


class Cache:
    """UnifiedRadixCache stand-in: dec_lock_ref of a Mamba prefix lock makes that prefix evictable."""

    def __init__(self, pool, evictable=0):
        self.req_to_token_pool, self.evictable = pool, evictable
        self.enable_mamba_extra_buffer = True
        self.tree_components = (0, 2)
        self.dec_calls = []

    def evict_for_alloc(self, params):
        n = min(params.mamba_num, self.evictable)
        self.evictable -= n
        self.req_to_token_pool.mamba_allocator.free_n += n

    def mamba_evictable_size(self):
        return self.evictable

    def dec_lock_ref(self, node, params, skip_swa=False):
        self.dec_calls.append((node, tuple(params.skipped_lock_components)))
        if 2 not in params.skipped_lock_components:
            self.evictable += 1


class KV:
    def __init__(self):
        self.mamba_ping_pong_track_buffer = [TI(7), TI(-1)]
        self.mamba_next_track_idx = 0
        self.mamba_last_track_seqlen = None
        self.mamba_last_track_idx = None
        self.kv_committed_len = 0


class Req:
    def __init__(self, rid):
        self.rid, self.kv = rid, KV()
        self.finished_reason = None
        self.mamba_lazy_is_insert = True
        self.last_node, self.swa_prefix_lock_released, self.lock_receipt = None, False, None
        self.seqlen, self.decode_batch_idx = 0, 0

    def finished(self):
        return self.finished_reason is not None


class SA:
    def __init__(self, none):
        self.none = none

    def is_none(self):
        return self.none


class FM:
    def __init__(self, mode):
        self.mode = mode

    def is_extend(self):
        return self.mode == "extend"

    def is_idle(self):
        return self.mode == "idle"


def _sb_init(self, reqs, seq_lens, pool, tree=None):
    self.reqs, self.seq_lens_cpu = reqs, [TI(x) for x in seq_lens]
    self.req_to_token_pool, self.tree_cache = pool, tree or SimpleNamespace(page_size=1)


INTERVAL = [int(os.environ.get("E24T_INTERVAL", "128"))]


# --- DFlash fakes with the 0.5.21 semantics (variant: 41736 / 41737 applied or not) -----------------------------
def _grouped_conv_fake(patched):
    def _grouped_conv(hidden_states, delta, base, block_size, num_groups, group_size, taps):
        STATE["conv_inputs"].append(id(hidden_states))
        rows = hidden_states.data
        out = []
        for r in range(len(rows)):
            pos = r % block_size
            o = [base[0] * v for v in rows[r]]
            for tap in range(1, taps):
                sh = rows[r - tap] if r - tap >= 0 else [0.0] * len(rows[r])
                if patched:                                        # torch.where(position >= tap, shifted, 0)
                    sh = sh if pos >= tap else [0.0] * len(sh)
                    o = [a + base[tap] * s for a, s in zip(o, sh)]
                else:                                              # shifted * (position >= tap): NaN * 0 = NaN
                    mask = 1.0 if pos >= tap else 0.0
                    o = [a + base[tap] * s * mask for a, s in zip(o, sh)]
            out.append(o)
        return FT(out)
    return _grouped_conv


class DFlashGroupedConv:
    def __init__(self, block_size, taps=2):
        self.block_size, self.taps = block_size, taps

    def _convolve(self, hidden_states, delta, side):
        sys.modules["torch"]._dynamo.mark_static(hidden_states, 1)
        mod = sys.modules[M_DFLASH]
        return mod._grouped_conv(hidden_states, delta, [1.0, 0.5], self.block_size, 1, 4, self.taps)


def _selector_fake(patched):
    def sample_path(self, *, candidate_ids, scores, uniforms, temperatures, greedy_mask):
        C, S = candidate_ids.data, scores.data
        bs, slots, topk = len(C), len(C[0]), len(C[0][0])
        flat_c = list(_leaves(C))
        rows2d = [row for b in S for s in b for row in s]          # [bs*slots*topk][topk]
        gm = list(_leaves(greedy_mask.data))
        toks, qs = [], []
        for b in range(bs):
            prev, trow, qrow = 0, [], []
            for slot in range(slots):
                base = (b * slots + slot) * topk
                idx = base + prev
                sc = rows2d[idx] if idx < len(rows2d) else [NAN] * topk
                if gm[b]:
                    nan = [x != x for x in sc]
                    if patched:                                    # first NaN if any, else the first max
                        index = nan.index(True) if any(nan) else sc.index(max(sc))
                    else:                                          # tl.max is NaN on an all-NaN row: no hit
                        fin = [x for x in sc if x == x]
                        index = topk if not fin else min(i for i, x in enumerate(sc) if x == max(fin))
                    probs = [1.0 if i == index else 0.0 for i in range(topk)]
                else:
                    fin = [(x, i) for i, x in enumerate(sc) if x == x]
                    index = min(max(fin)[1], topk - 1) if fin else topk - 1
                    probs = [1.0 / topk] * topk
                ci = base + index
                trow.append(flat_c[ci] if ci < len(flat_c) else -1)
                qrow.append(probs)
                prev = index
            toks.append(trow)
            qs.append(qrow)
        return FT(toks), FT(qs)
    return sample_path


class CandidateSelector:
    pass


class DFlashWorkerV2:
    def __init__(self, block_size=6):
        self.block_size = block_size
        self.selector = CandidateSelector()
        self.conv = DFlashGroupedConv(block_size)

    def forward_batch_generation(self, batch, on_publish=None):
        bs = getattr(batch, "rows_bs", None) or len(batch.reqs)
        x = FT([[float(r * 10 + c) for c in range(4)] for r in range(bs * self.block_size)])
        out = self.conv._convolve(x, None, 0)
        res = {"conv": out.tolist(), "x_id": id(x)}
        if batch.forward_mode.is_extend():
            return res
        topk, slots = 4, self.block_size - 1
        cand = FT([[[1000 * b + 10 * s + j for j in range(topk)] for s in range(slots)] for b in range(bs)])
        scores = FT([[[[float((j * 7 + s + b + p) % 5) for j in range(topk)] for p in range(topk)]
                      for s in range(slots)] for b in range(bs)])
        gm = FT([bool(batch.greedy[b]) if b < len(batch.greedy) else True for b in range(bs)])
        tokens, q = self.selector.sample_path(candidate_ids=cand, scores=scores, uniforms=None, temperatures=None,
                                              greedy_mask=gm)
        res.update(tokens=tokens.tolist(), q=q.tolist())
        return res


def batch(rids, mode="decode", greedy=None, rows_bs=None):
    return SimpleNamespace(reqs=[SimpleNamespace(rid=r) for r in rids], forward_mode=FM(mode), is_extend_in_batch=False,
                           greedy=greedy if greedy is not None else [True] * len(rids), rows_bs=rows_bs)


# --- Scheduler / probe fakes ------------------------------------------------------------------------------------
class FakeModel:
    def __init__(self):
        w = SimpleNamespace(dtype="torch.bfloat16", shape=(248320, 5120))
        self.lm_head = SimpleNamespace(weight=w, named_parameters=lambda: [("weight", w)])

    def named_modules(self):
        return [("model.layers.3.self_attn.attn", SimpleNamespace(k_scale_float=0.028, v_scale_float=0.254)),
                ("model.layers.7.self_attn.attn", SimpleNamespace(k_scale_float=0.051, v_scale_float=0.021)),
                ("model.layers.0.linear_attn", SimpleNamespace())]

    def parameters(self):
        return [SimpleNamespace(dtype="torch.bfloat16")]


class RecAlloc(Alloc):
    """Records which thread reads the allocator (the hook must only do it from the main thread)."""

    def available_size(self):
        STATE["alloc_threads"].append(threading.current_thread().name)
        return self.free_n


class Scheduler:
    def __init__(self, server_args=None, port_args=None):
        if os.environ.get("E24T_INIT_FAIL") == "1":
            raise RuntimeError("scheduler init failed")
        self.server_args = SimpleNamespace(kv_cache_dtype="fp8_e4m3", mamba_ssm_dtype="float16")
        self.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(model=FakeModel()))
        self.draft_worker = None
        cache = SimpleNamespace(temporal=SimpleNamespace(dtype="torch.float16"),
                                conv=[SimpleNamespace(dtype="torch.bfloat16")], intermediate_ssm=None)
        self.req_to_token_pool = SimpleNamespace(mamba_pool=SimpleNamespace(size=68, mamba_cache=cache),
                                                 mamba_allocator=RecAlloc(5))
        self.tree_cache = SimpleNamespace(mamba_evictable_size=lambda: 2)

    def run_event_loop(self):
        return "loop-done"

    def on_idle(self):
        return None


def install_fakes(variant=None, only=None, torch=True, drop_methods=()):
    """Build every fake module completely, then publish them in sys.modules at once (the hook thread may patch any
    module the moment it appears, as it would after a real import)."""
    variant = variant or os.environ.get("E24T_VARIANT", "pristine")
    pkg = os.environ["E24T_PATCHED"] if variant == "patched" else os.environ["E24T_SRC"]
    mods = {}

    def mk(name, rel=None):
        m = types.ModuleType(name)
        if rel:
            m.__file__ = os.path.join(pkg, rel)
        mods[name] = m
        return m

    exec_ctx = SimpleNamespace(mamba=SimpleNamespace(enable_mamba_extra_buffer_lazy=True))
    get_exec = lambda: exec_ctx                                     # noqa: E731
    sb = mk(M_SB, SB)
    sb.__dict__.update(envs=SimpleNamespace(SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=_Flag("SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL")),
                       List=typing.List, Optional=typing.Optional)
    exec(compile("from __future__ import annotations\n" + cut_top(pkg, SB, "mamba_lazy_spec_in_window"), "<e24t>",
                 "exec"), sb.__dict__)
    build_class(sb, "ScheduleBatch", cut_methods(pkg, SB, "ScheduleBatch", ["mamba_lazy_prealloc_at_boundary",
                                                                             "mamba_lazy_spec_prepare"]))
    sb.ScheduleBatch.__init__ = _sb_init
    brp = mk(M_BRP, BRP)
    brp.__dict__.update(get_exec=get_exec, mamba_track_grid=lambda page: INTERVAL[0],
                        max_speculative_num_draft_tokens=lambda: 6, mamba_lazy_spec_in_window=sb.mamba_lazy_spec_in_window)
    build_class(brp, "SchedulerBatchResultProcessor", cut_methods(pkg, BRP, "SchedulerBatchResultProcessor", [
        "_mamba_prefix_cache_update", "_mamba_lazy_spec_update", "_mamba_assert_committed_len_lookahead",
        "_mamba_check_track_boundary", "mamba_lazy_post_decode_at_boundary"]))
    mm = mk(M_MAMBA, MAMBA)
    mm.__dict__.update(dataclasses=dataclasses, get_exec=get_exec, replace=dataclasses.replace, Optional=typing.Optional)
    # the real dataclasses resolve their string annotations through sys.modules[cls.__module__]: give them their own
    # module, published now (it is not a hook target)
    bpc = types.ModuleType("sglang.srt.mem_cache.base_prefix_cache")
    bpc.__dict__.update(dataclasses=dataclasses, Optional=typing.Optional)
    sys.modules[bpc.__name__] = bpc
    for c in ("EvictParams", "DecLockRefParams"):
        exec(compile("from __future__ import annotations\n" + cut_top(pkg, BPC, c, "@dataclasses.dataclass\n"),
                     "<e24t>", "exec"), bpc.__dict__)
        mm.__dict__[c] = bpc.__dict__[c]
    names = [n for n in ("int8_ckpt_pool", "prepare_for_caching_req", "_alloc_mamba_slot") if n not in drop_methods]
    build_class(mm, "MambaComponent", cut_methods(pkg, MAMBA, "MambaComponent", names))
    sch = mk(M_SCHED, "srt/managers/scheduler.py")
    sch.Scheduler = Scheduler
    rc = mk("sglang.srt.runtime_context")
    rc.get_parallel = lambda: SimpleNamespace(tp_rank=int(os.environ.get("E24T_TP_RANK", "0")), tp_size=4)
    mk("sglang.srt.dllm.mixin.scheduler", "srt/dllm/mixin/scheduler.py")
    mk("sglang.kernels.ops.speculative.dflash", "kernels/ops/speculative/dflash.py")
    df = mk(M_DFLASH, "srt/models/dflash.py")
    df._grouped_conv = _grouped_conv_fake(variant == "patched")
    sel = type("CandidateSelector", (), {"sample_path": _selector_fake(variant == "patched")})
    df.CandidateSelector = sel
    df.DFlashGroupedConv = DFlashGroupedConv
    wk = mk(M_WORKER)
    wk.DFlashWorkerV2 = DFlashWorkerV2
    if torch:
        mods["torch"] = fake_torch()
    tri = mk("triton")
    tri.knobs = SimpleNamespace(cache=SimpleNamespace(dir=os.environ.get("E24T_TRITON_DIR", "/fake/triton")))
    ind = mk("torch._inductor.runtime.cache_dir_utils")
    ind.default_cache_dir = lambda: "/tmp/torchinductor_fakeuser"

    def cache_dir():                                                # creates dirs in torch: the hook must not call it
        STATE["cache_dir_calls"] += 1
        return "/never"
    ind.cache_dir = cache_dir
    if os.environ.get("E24T_TITLE"):
        spt = mk("setproctitle")
        spt.getproctitle = lambda: os.environ["E24T_TITLE"]
    if only is not None:
        keep = set(only) | {"torch", "triton", "torch._inductor.runtime.cache_dir_utils", "setproctitle",
                            "sglang.srt.runtime_context"}
        mods = {k: v for k, v in mods.items() if k in keep}
    sys.modules.update(mods)
    return SimpleNamespace(sb=sb, brp=brp, mm=mm, df=df, wk=wk, sch=sch, sel=sel, orig_gc=df._grouped_conv,
                           orig_sp=sel.__dict__["sample_path"], orig={k: dict(vars(getattr(m, n))) for k, m, n in (
                               ("sb", sb, "ScheduleBatch"), ("brp", brp, "SchedulerBatchResultProcessor"),
                               ("mm", mm, "MambaComponent"))})


def hook():
    return sys.modules["sitecustomize"]


def wait_patched(names=None, timeout=10.0):
    """Until every name (default: the hook's enabled targets; none when inert) is patched or failed."""
    h = hook()
    if names is None:
        names = list(h.ENABLED) if h._DIR else []
    end = time.time() + timeout
    while time.time() < end and not all(n in h.PATCHED or n in h.FAILED for n in names):
        time.sleep(0.01)
    return list(h.PATCHED)


def make_worker(F):
    w = F.wk.DFlashWorkerV2()
    w.selector = F.df.CandidateSelector()
    return w


def child_sleep(sec):
    time.sleep(sec)


def done(result):
    with open(os.environ["E24T_RESULT"], "w") as f:
        json.dump(result, f)


# ======================================================================================================================
# Tests
SCEN_HEAD = "import os, sys, json, time, threading\nsys.path.insert(0, os.environ['E24T_HERE'])\nimport test_e24_hook as T\n"


@unittest.skipUnless(HAVE_SRC, "no SGLang 0.5.21 source (set E24_SGLANG_SRC)")
class HookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="e24h_")
        cls.hookdir = os.path.join(cls.tmp, "hook")
        os.makedirs(cls.hookdir)
        shutil.copy2(HERE / "e24_hook.py", os.path.join(cls.hookdir, "sitecustomize.py"))
        with open(HERE / "e24_hook.py", "rb") as f:
            cls.hook_sha = hashlib.sha256(f.read()).hexdigest()
        spec = importlib.util.spec_from_file_location("e24_sglang_patches_t", HERE / "e24_sglang_patches.py")
        P = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(P)
        cls.P = P
        cls.patched = os.path.join(cls.tmp, "patched", "sglang")
        for rel in list(P.REFERENCE_FILE_SHA256) + ["_version.py", BPC]:
            dst = os.path.join(cls.patched, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(SRC / rel, dst)
        r = P.apply(cls.patched, P.E24_NAMES)
        assert r["ok"], r
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_scen(self, body, env=None, hook_dir=True, pythonpath=None, timeout=90, expect_rc=0):
        HookTests.n += 1
        d = os.path.join(self.tmp, "s%03d" % HookTests.n)
        out = os.path.join(d, "out")
        os.makedirs(out)
        scen = os.path.join(d, "scen.py")
        with open(scen, "w") as f:
            f.write(SCEN_HEAD + textwrap.dedent(body))
        e = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": d, "PYTHONPATH": pythonpath or self.hookdir,
             "E24T_HERE": str(HERE), "E24T_SRC": str(SRC), "E24T_PATCHED": self.patched,
             "E24T_RESULT": os.path.join(d, "result.json"), "E24_HOOK_FLUSH_SEC": "0.05", "E24_HOOK_CHAIN": "0",
             "PYTHONDONTWRITEBYTECODE": "1"}
        if hook_dir:
            e["E24_HOOK_DIR"] = out
        e.update(env or {})
        cp = subprocess.run([sys.executable, scen], env=e, cwd=d, capture_output=True, text=True, timeout=timeout)
        self.assertEqual(cp.returncode, expect_rc, "rc=%s\nstdout:\n%s\nstderr:\n%s" % (cp.returncode, cp.stdout[-3000:],
                                                                                       cp.stderr[-3000:]))
        res = {}
        if os.path.exists(e["E24T_RESULT"]):
            with open(e["E24T_RESULT"]) as f:
                res = json.load(f)
        return SimpleNamespace(cp=cp, out=out, res=res, dir=d)

    def outputs(self, out):
        o = {"env": {}, "counters": {}, "trace": {}, "probe": {}, "files": sorted(os.listdir(out))}
        for fn in o["files"]:
            p = os.path.join(out, fn)
            kind, _, rest = fn.partition("_")
            if fn.endswith(".tmp"):
                continue
            pid = int(rest.split(".")[0])
            with open(p) as f:
                if fn.endswith(".jsonl"):
                    o["trace"][pid] = [json.loads(x) for x in f.read().splitlines() if x.strip()]
                else:
                    o[kind][pid] = json.load(f)
        return o

    # ----------------------------------------------------------------------------------------------- static
    def test_static_no_import_inside_functions(self):
        src = (HERE / "e24_hook.py").read_text(encoding="utf-8")
        compile(src, "e24_hook.py", "exec")
        tree = ast.parse(src)
        inner = [n.lineno for f in ast.walk(tree) if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
                 for n in ast.walk(f) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertEqual(inner, [], "imports inside functions (the hook thread must never import)")
        tops = {a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        tops |= {n.module for n in tree.body if isinstance(n, ast.ImportFrom)}
        self.assertLessEqual(tops, {"atexit", "hashlib", "json", "os", "sys", "threading", "time", "collections"})
        self.assertIn("E24_HOOK_MARKER_REV2", src)

    def test_target_specs_match_the_0521_source(self):
        """Every target module/class/method exists in the 0.5.21 source (what the kit AST-checks on the install)."""
        text = (HERE / "e24_hook.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        ns = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") in ("TARGETS", "TARGET_METHODS"):
                ns[node.targets[0].id] = ast.literal_eval(node.value)
        self.assertEqual(set(ns["TARGETS"]), set(ns["TARGET_METHODS"]))
        for name, (mod, attr) in ns["TARGETS"].items():
            if mod.startswith("vllm"):
                continue
            rel = mod.replace("sglang.", "", 1).replace(".", "/") + ".py"
            t = (SRC / rel).read_text(encoding="utf-8")
            top = {n.name: n for n in ast.parse(t).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
            self.assertIn(attr, top, (name, rel))
            if isinstance(top[attr], ast.ClassDef):
                meths = {n.name for n in top[attr].body if isinstance(n, ast.FunctionDef)}
                for m in ns["TARGET_METHODS"][name]:
                    self.assertIn(m, meths, (name, m))

    # ----------------------------------------------------------------------------------------------- inert / env
    def test_inert_without_hook_dir(self):
        r = self.run_scen("""
            F = T.install_fakes()
            time.sleep(0.3)
            h = T.hook()
            T.done({"dir": h._DIR, "threads": [t.name for t in threading.enumerate()], "patched": h.PATCHED,
                    "gc_same": F.df._grouped_conv is F.orig_gc,
                    "same": all(getattr(F.sb.ScheduleBatch, k) is v for k, v in F.orig["sb"].items() if callable(v))})
        """, hook_dir=False)
        self.assertIsNone(r.res["dir"])
        self.assertNotIn("e24-hook", r.res["threads"])
        self.assertEqual(r.res["patched"], [])
        self.assertTrue(r.res["gc_same"] and r.res["same"])
        self.assertEqual(os.listdir(r.out), [])

    def test_env_record_roles_changes_redaction(self):
        r = self.run_scen("""
            import multiprocessing
            multiprocessing.current_process().name = "EngineCore_0"
            T.install_fakes(only=[])
            time.sleep(0.3)
            os.environ["TRITON_CACHE_DIR"] = "/x/new"
            time.sleep(0.4)
            T.done({"pid": os.getpid()})
        """, env={"TRITON_CACHE_DIR": "/x/old", "SGLANG_LOGPROB_CHUNK_SIZE": "128", "NCCL_MIN_CTAS": "4",
                  "E24_INJECT_DRAFT_NAN": "0", "HF_TOKEN": "hf_secret", "SGLANG_FAKE_API_KEY": "k",
                  "VLLM_CACHE_ROOT": "/x/vllm", "E24T_TITLE": "VLLM::EngineCore", "UNRELATED_VAR": "1"})
        o = self.outputs(r.out)
        rec = o["env"][r.res["pid"]]
        self.assertEqual(rec["source"], "import")
        self.assertEqual(rec["pid"], r.res["pid"])
        self.assertEqual(rec["env"]["SGLANG_LOGPROB_CHUNK_SIZE"], "128")
        self.assertEqual(rec["env"]["NCCL_MIN_CTAS"], "4")
        self.assertEqual(rec["env"]["E24_INJECT_DRAFT_NAN"], "0")
        self.assertEqual(rec["env"]["PYTHONPATH"], self.hookdir)
        self.assertEqual(rec["env"]["HF_TOKEN"], "<redacted>")
        self.assertEqual(rec["env"]["SGLANG_FAKE_API_KEY"], "<redacted>")
        self.assertNotIn("UNRELATED_VAR", rec["env"])
        self.assertEqual(rec["env"]["TRITON_CACHE_DIR"], "/x/new")             # last value wins
        self.assertEqual(rec["TRITON_CACHE_DIR"], "/x/new")                    # flat copy
        self.assertEqual(rec["cache"]["VLLM_CACHE_ROOT"], "/x/vllm")
        ch = [c for c in rec["env_changes"] if c["var"] == "TRITON_CACHE_DIR"]
        self.assertEqual((ch[0]["old"], ch[0]["new"]), ("/x/old", "/x/new"))
        self.assertEqual(rec["role"]["mp_name"], "EngineCore_0")
        self.assertEqual(rec["role"]["proctitle"], "VLLM::EngineCore")
        self.assertEqual(rec["role"]["kind"], "vllm_engine_core")
        self.assertEqual(rec["role_tag"], "engine_core")
        self.assertIn("scen.py", " ".join(rec["orig_argv"]))
        self.assertEqual(rec["hook"]["sha256"], self.hook_sha)
        self.assertEqual(rec["hook"]["expected"], ["Scheduler"])

    def test_counters_only_from_rank_processes(self):
        """The 0.5.21 launcher imports the scheduler module and the detokenizer imports schedule_batch, so both patch
        targets; neither may leave a counters file (per-rank gates count files). A scheduler's file carries its TP rank,
        parsed from its title before _main_late runs."""
        body = """
            if os.environ.get("E24T_ARGV"):
                sys.orig_argv = json.loads(os.environ["E24T_ARGV"])
            F = T.install_fakes()
            T.wait_patched()
            time.sleep(0.3)
            seen = sorted(os.listdir(os.environ["E24_HOOK_DIR"]))
            late = os.environ.get("E24T_LATE_TITLE")
            if late:
                os.environ["E24T_TITLE"] = late
                time.sleep(0.4)
            T.done({"pid": os.getpid(), "seen_before": seen})
        """
        env = {"E24_HOOK_COUNTERS": "1"}
        r = self.run_scen(body, env=dict(env, E24T_TITLE="sglang::scheduler_TP2"))
        c = self.outputs(r.out)["counters"][r.res["pid"]]
        self.assertEqual((c["tp_rank"], c["role_tag"]), (2, "scheduler"))
        self.assertTrue(c["self_test"]["ok"], c["self_test"])
        r = self.run_scen(body, env=dict(env, E24T_TITLE="sglang::detokenizer"))
        o = self.outputs(r.out)
        self.assertEqual(o["counters"], {})
        self.assertIn(r.res["pid"], o["env"])
        r = self.run_scen(body, env=dict(env, E24T_TITLE="python",
                                         E24T_ARGV=json.dumps([sys.executable, "-m", "sglang.launch_server"])))
        o = self.outputs(r.out)
        self.assertEqual(o["counters"], {})
        self.assertEqual(o["env"][r.res["pid"]]["role_tag"], "main")
        # a title set after the first counters write (spawned detokenizer): the file is taken back
        r = self.run_scen(body, env=dict(env, E24T_TITLE="python", E24T_LATE_TITLE="sglang::detokenizer"))
        self.assertIn("counters_%d.json" % r.res["pid"], r.res["seen_before"])
        self.assertEqual(self.outputs(r.out)["counters"], {})

    def test_per_rid_eviction_keeps_evidence(self):
        r = self.run_scen("""
            h = T.hook()
            h._MAX_RIDS = 3
            h._rcount("r1", "zero_q_rows")
            h._rcount("g1", "grammar_steps")
            h._rcount("r2", "optional_alloc_failed")
            h._rcount("g2", "grammar_steps")          # full: evicts g1 (bookkeeping only), not r1
            a = sorted(h.PER_RID)
            h._rcount("r3", "prefix_lock_released")  # full: evicts g2
            b = sorted(h.PER_RID)
            h._rcount("r4", "zero_q_rows")            # full of evidence: evicts the oldest (r1)
            T.done({"a": a, "b": b, "c": sorted(h.PER_RID)})
        """, hook_dir=False)
        self.assertEqual(r.res["a"], ["g2", "r1", "r2"])
        self.assertEqual(r.res["b"], ["r1", "r2", "r3"])
        self.assertEqual(r.res["c"], ["r2", "r3", "r4"])

    def test_forked_and_spawned_children_write_records(self):
        r = self.run_scen("""
            import multiprocessing
            def main():
                pid = os.fork()
                if pid == 0:
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/x/inductor_from_child"
                    time.sleep(0.8)
                    os._exit(0)
                p1 = multiprocessing.get_context("fork").Process(target=T.child_sleep, args=(0.8,), name="EngineCore_0")
                p1.start()
                p2 = multiprocessing.get_context("spawn").Process(target=T.child_sleep, args=(0.8,), name="Worker_TP1")
                p2.start()
                os.waitpid(pid, 0)
                p1.join()
                p2.join()
                T.done({"parent": os.getpid(), "fork": pid, "mp_fork": p1.pid, "spawn": p2.pid})
            if __name__ == "__main__":
                main()
        """, env={"TORCHINDUCTOR_CACHE_DIR": "/x/inductor_parent"})
        o = self.outputs(r.out)
        res = r.res
        for k in ("parent", "fork", "mp_fork", "spawn"):
            self.assertIn(res[k], o["env"], (k, sorted(o["env"])))
        fk = o["env"][res["fork"]]
        self.assertEqual((fk["source"], fk["forked_from"], fk["ppid"]), ("fork", res["parent"], res["parent"]))
        self.assertEqual(fk["env"]["TORCHINDUCTOR_CACHE_DIR"], "/x/inductor_from_child")     # re-snapshot in the child
        self.assertTrue(any(c["var"] == "TORCHINDUCTOR_CACHE_DIR" for c in fk["env_changes"]))
        mpf = o["env"][res["mp_fork"]]
        self.assertEqual(mpf["source"], "fork")
        self.assertEqual(mpf["role"]["mp_name"], "EngineCore_0")
        self.assertEqual(mpf["role"]["kind"], "vllm_engine_core")
        sp = o["env"][res["spawn"]]
        self.assertEqual(sp["source"], "import")
        self.assertEqual(sp["role"]["mp_name"], "Worker_TP1")
        self.assertEqual(o["env"][res["parent"]]["env"]["TORCHINDUCTOR_CACHE_DIR"], "/x/inductor_parent")

    # ----------------------------------------------------------------------------------------------- counters
    def _bug_order(self, variant, extra_env=None):
        return self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["ScheduleBatch", "SchedulerBatchResultProcessor", "MambaComponent"])
            req = T.Req("seed-r1")
            req.kv.mamba_last_track_seqlen = 192
            pool = T.Pool(free=0)
            b = F.sb.ScheduleBatch([req], [256], pool)
            b.mamba_lazy_prealloc_at_boundary(128)
            comp = F.mm.MambaComponent()
            comp.cache, comp.component_type = T.Cache(pool, evictable=1), 2
            don = comp.prepare_for_caching_req(req, T.SimpleNamespace(mamba_value=None), 255, False)
            rp = F.brp.SchedulerBatchResultProcessor()
            rp.tree_cache = T.SimpleNamespace(page_size=1)
            req.kv.kv_committed_len, req.decode_batch_idx = 256, 0
            bt = T.SimpleNamespace(spec_algorithm=T.SA(True), mamba_decode_batch_idx_cpu=[0],
                                   mamba_track_buffer_indices=[0], req_to_token_pool=pool)
            rp._mamba_prefix_cache_update(req, bt, None, 0)
            T.done({"pid": os.getpid(), "donated": don, "last": req.kv.mamba_last_track_seqlen})
        """, env=dict({"E24_HOOK_COUNTERS": "1", "E24T_VARIANT": variant}, **(extra_env or {})))

    def test_43567_unpatched_chain_confirmed(self):
        r = self._bug_order("pristine")
        self.assertEqual(r.res["donated"], 192)
        o = self.outputs(r.out)
        c = o["counters"][r.res["pid"]]
        k = c["counters"]
        self.assertEqual((k["optional_alloc_attempts"], k["optional_alloc_failed"]), (1, 1))
        self.assertEqual((k["stale_donation_candidate"], k["stale_donation_confirmed"]), (1, 1))
        self.assertEqual(c["per_rid"]["seed-r1"]["optional_alloc_failed"], 1)
        self.assertEqual(c["per_rid"]["seed-r1"]["stale_donation_confirmed"], 1)
        self.assertEqual((k["donation_alloc_calls"], k["donation_alloc_first_try_failed"]), (1, 1))
        evs = o["trace"][r.res["pid"]]
        self.assertTrue(any(e["ev"] == "track_len_update" and e["was_marked"] for e in evs))
        self.assertTrue(any(e["ev"] == "alloc_failed" and e["rid"] == "seed-r1" and e["last_len_before"] == 192
                            for e in evs))
        self.assertTrue(c["self_test"]["ok"], c["self_test"])
        self.assertEqual(c["alloc"], {"available": 0, "size": 4})

    def test_43567_patched_no_candidate(self):
        r = self._bug_order("patched")
        self.assertEqual(r.res["donated"], 0)
        k = self.outputs(r.out)["counters"][r.res["pid"]]["counters"]
        self.assertEqual(k["optional_alloc_failed"], 1)                     # the trigger is reached
        self.assertEqual(k.get("stale_donation_candidate", 0), 0)

    def test_forced_fail_env_counted(self):
        r = self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["ScheduleBatch"])
            req = T.Req("seed-ff")
            req.kv.mamba_last_track_seqlen = 192
            b = F.sb.ScheduleBatch([req], [256], T.Pool(free=3))
            b.mamba_lazy_prealloc_at_boundary(128)
            T.done({"pid": os.getpid()})
        """, env={"E24_HOOK_COUNTERS": "1", "SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL": "1"})
        o = self.outputs(r.out)
        k = o["counters"][r.res["pid"]]["counters"]
        self.assertEqual(k["optional_alloc_failed"], 1)
        self.assertEqual(o["env"][r.res["pid"]]["env"]["SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL"], "1")

    def test_spec_candidate_confirmed_or_cleared(self):
        r = self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["ScheduleBatch", "SchedulerBatchResultProcessor", "MambaComponent"])
            rp = F.brp.SchedulerBatchResultProcessor()
            rp.tree_cache = T.SimpleNamespace(page_size=1)
            comp = F.mm.MambaComponent()
            pos = []
            for rid, crossed in (("s1", True), ("s2", False)):
                req = T.Req(rid)
                req.kv.mamba_last_track_seqlen, req.kv.kv_committed_len = 192, 250
                pool = T.Pool(free=0)
                comp.cache, comp.component_type = T.Cache(pool, evictable=1), 2
                b = F.sb.ScheduleBatch([req], [250], pool)
                b.mamba_lazy_spec_prepare(64, 6)
                pos.append(b.mamba_lazy_spec_track_positions_cpu)
                comp.prepare_for_caching_req(req, T.SimpleNamespace(), 249, False)
                rp._mamba_lazy_spec_update(req, b, 0, crossed, 256)
            h = T.hook()
            T.done({"pid": os.getpid(), "pos": pos, "mark": len(h.MARK), "pend": len(h.PEND)})
        """, env={"E24_HOOK_COUNTERS": "1"})
        self.assertEqual(r.res["pos"], [[0], [0]])
        k = self.outputs(r.out)["counters"][r.res["pid"]]["counters"]
        self.assertEqual((k["spec_inplace_fallback"], k["stale_donation_candidate"]), (2, 2))
        self.assertEqual((k["stale_donation_confirmed"], k["stale_candidate_cleared"]), (1, 1))
        self.assertEqual((r.res["mark"], r.res["pend"]), (0, 0))

    def test_43559_prefix_lock_released(self):
        r = self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["MambaComponent"])
            pool = T.Pool(free=0)
            comp = F.mm.MambaComponent()
            comp.cache, comp.component_type = T.Cache(pool, evictable=0), 2
            req = T.Req("seed-np")
            req.last_node, req.lock_receipt = 7, F.mm.DecLockRefParams(node_id=7, skipped_lock_components=())
            slot = comp._alloc_mamba_slot(req)
            T.done({"pid": os.getpid(), "slot": list(slot), "skipped": list(req.lock_receipt.skipped_lock_components),
                    "dec": comp.cache.dec_calls})
        """, env={"E24_HOOK_COUNTERS": "1", "E24T_VARIANT": "patched"})
        self.assertEqual(r.res["skipped"], [2])
        o = self.outputs(r.out)
        c = o["counters"][r.res["pid"]]
        k = c["counters"]
        self.assertEqual((k["prefix_lock_released"], k["donation_alloc_first_try_failed"], k["donation_alloc_evicted_ok"]),
                         (1, 1, 1))
        self.assertEqual(c["per_rid"]["seed-np"]["prefix_lock_released"], 1)
        ev = [e for e in o["trace"][r.res["pid"]] if e["ev"] == "prefix_lock_released"]
        self.assertEqual((ev[0]["rid"], ev[0]["available_before"], ev[0]["evictable_before"]), ("seed-np", 0, 0))

    def test_assert_path_flushes_synchronously(self):
        """The #43559 crash: _alloc_mamba_slot asserts inside prepare_for_caching_req; the process dies at once
        (os._exit, no atexit, periodic flush 30 s away): the evidence must already be on disk."""
        r = self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["MambaComponent", "ScheduleBatch"])
            pool = T.Pool(free=0)
            comp = F.mm.MambaComponent()
            comp.cache, comp.component_type = T.Cache(pool, evictable=0), 2
            req = T.Req("seed-crash")
            req.kv.mamba_last_track_seqlen = 128
            try:
                comp.prepare_for_caching_req(req, T.SimpleNamespace(), 255, False)
            except AssertionError as e:
                with open(os.environ["E24T_RESULT"], "w") as f:
                    json.dump({"pid": os.getpid(), "err": str(e)}, f)
                os._exit(3)
        """, env={"E24_HOOK_COUNTERS": "1", "E24_HOOK_FLUSH_SEC": "30"}, expect_rc=3)
        self.assertEqual(r.res["err"], "Can not alloc mamba cache")
        o = self.outputs(r.out)
        pid = r.res["pid"]
        evs = o["trace"][pid]
        a = [e for e in evs if e["ev"] == "donation_alloc_assert"]
        self.assertEqual(len(a), 1, evs)
        self.assertEqual((a[0]["rid"], a[0]["available"], a[0]["evictable"], a[0]["req_passed"]),
                         ("seed-crash", 0, 0, False))
        self.assertTrue(any(e["ev"] == "exception" and e["where"] == "prepare_for_caching_req" for e in evs))
        c = o["counters"][pid]
        self.assertEqual(c["counters"]["donation_alloc_assert"], 1)
        self.assertEqual(c["per_rid"]["seed-crash"]["donation_alloc_assert"], 1)

    def test_43029_boundary_counters(self):
        body = """
            F = T.install_fakes()
            T.wait_patched(["SchedulerBatchResultProcessor"])
            rp = F.brp.SchedulerBatchResultProcessor()
            rp.tree_cache = T.SimpleNamespace(page_size=1)
            bt = T.SimpleNamespace(spec_algorithm=T.SA(False))
            out = []
            for seqlen, nc, ret in ((260, 5, [7, 7]), (260, 5, None), (258, 5, [7, 7, 7])):
                req = T.Req("g-%d-%s" % (seqlen, ret is not None))
                req.seqlen = seqlen
                res = T.SimpleNamespace(num_correct_drafts_per_req_cpu=[nc], num_non_draft_tokens_per_req=1,
                                        grammar_retained_tokens=None if ret is None else [ret])
                out.append(list(rp._mamba_check_track_boundary(req, bt, res, 0)))
            T.done({"pid": os.getpid(), "out": out})
        """
        env = {"E24_HOOK_COUNTERS": "1", "E24T_INTERVAL": "256"}
        r = self.run_scen(body, env=dict(env, E24T_VARIANT="pristine"))
        self.assertEqual(r.res["out"], [[True, 256], [True, 256], [True, 256]])
        c = self.outputs(r.out)["counters"][r.res["pid"]]
        k = c["counters"]
        self.assertEqual((k["grammar_steps"], k["grammar_truncated_steps"]), (2, 2))
        self.assertEqual((k["false_crossing_possible"], k["false_crossing_returned"]), (1, 1))
        self.assertEqual(c["per_rid"]["g-260-True"]["false_crossing_returned"], 1)
        r = self.run_scen(body, env=dict(env, E24T_VARIANT="patched"))
        self.assertEqual(r.res["out"], [[False, 0], [True, 256], [True, 256]])
        k = self.outputs(r.out)["counters"][r.res["pid"]]["counters"]
        self.assertEqual((k["grammar_truncated_steps"], k["false_crossing_possible"]), (2, 1))
        self.assertEqual(k.get("false_crossing_returned", 0), 0)

    def test_lazy_insert_cleared_on_finish(self):
        r = self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["SchedulerBatchResultProcessor"])
            rp = F.brp.SchedulerBatchResultProcessor()
            rp.tree_cache = T.SimpleNamespace(page_size=1)
            req = T.Req("gl-1")
            req.finished_reason = "stop"
            b = T.SimpleNamespace(mamba_lazy_spec_track_positions_cpu=[0])
            rp._mamba_lazy_spec_update(req, b, 0, True, 256)
            T.done({"pid": os.getpid(), "insert": req.mamba_lazy_is_insert})
        """, env={"E24_HOOK_COUNTERS": "1", "E24T_INTERVAL": "64"})
        self.assertFalse(r.res["insert"])
        c = self.outputs(r.out)["counters"][r.res["pid"]]
        self.assertEqual(c["counters"]["lazy_insert_cleared"], 1)
        self.assertEqual(c["per_rid"]["gl-1"]["lazy_insert_cleared"], 1)

    # ----------------------------------------------------------------------------------------------- injection
    FWD = """
        F = T.install_fakes()
        T.wait_patched()
        w = T.make_worker(F)
        outs = {}
        outs["mixed"] = w.forward_batch_generation(T.batch(["e24-sentinel-0", "n1", "n2"], greedy=[True, True, False]))
        outs["plain"] = w.forward_batch_generation(T.batch(["a", "b"]))
        for o in outs.values():
            o.pop("x_id", None)
        h = T.hook()
        T.done({"pid": os.getpid(), "outs": outs, "gc_same": F.df._grouped_conv is F.orig_gc,
                "sp_same": F.df.CandidateSelector.__dict__["sample_path"] is F.orig_sp, "patched": h.PATCHED,
                "inject": h._inject_info()})
    """

    def test_injection_gating_off_is_identical(self):
        base = self.run_scen(self.FWD, hook_dir=False)
        self.assertTrue(base.res["gc_same"] and base.res["sp_same"])
        for env in ({"E24_HOOK_COUNTERS": "1"}, {"E24_INJECT_DRAFT_NAN": "1", "E24_INJECT_SELECTOR_NAN": "1"},
                    {"E24_INJECT_DRAFT_NAN": "1", "E24_HOOK_EAGER": "0", "E24_HOOK_COUNTERS": "1"}):
            with self.subTest(env=env):
                r = self.run_scen(self.FWD, env=env)
                self.assertEqual(r.res["outs"], base.res["outs"])
                self.assertTrue(r.res["gc_same"] and r.res["sp_same"], r.res)
                self.assertNotIn("_grouped_conv", r.res["patched"])
                if "E24_INJECT_DRAFT_NAN" in env:
                    self.assertEqual(r.res["inject"]["refused"], "E24_HOOK_EAGER!=1")
                    c = self.outputs(r.out)["counters"][r.res["pid"]]
                    self.assertEqual(c["inject"]["refused"], "E24_HOOK_EAGER!=1")
        # installed but no sentinel in the batch: still identical
        r = self.run_scen(self.FWD.replace('"e24-sentinel-0"', '"s0"'),
                          env={"E24_INJECT_DRAFT_NAN": "1", "E24_INJECT_SELECTOR_NAN": "1", "E24_HOOK_EAGER": "1"})
        self.assertFalse(r.res["gc_same"] or r.res["sp_same"])
        base2 = self.run_scen(self.FWD.replace('"e24-sentinel-0"', '"s0"'), hook_dir=False)
        self.assertEqual(r.res["outs"], base2.res["outs"])

    def _draft(self, variant, body_extra=""):
        head = textwrap.dedent("""
            F = T.install_fakes()
            T.wait_patched(["DFlashWorkerV2", "_grouped_conv"])
            w = T.make_worker(F)
            o = w.forward_batch_generation(T.batch(["e24-sentinel-0", "n1", "n2"]))
            row6 = o["conv"][6]
            ms = T.STATE["mark_static"][-1]
            used = T.STATE["conv_inputs"][-1]
            res = {"pid": os.getpid(), "row6_finite": all(v == v for v in row6), "x_id": o["x_id"],
                   "mark_static_id": ms[0], "mark_static_dim": ms[1], "used": used}
        """)
        return self.run_scen(head + textwrap.dedent(body_extra) + "T.done(res)\n",
                             env={"E24_INJECT_DRAFT_NAN": "1", "E24_HOOK_EAGER": "1", "E24T_VARIANT": variant})

    def test_draft_nan_unpatched_leaks_patched_clean(self):
        r = self._draft("pristine", """
            o2 = w.forward_batch_generation(T.batch(["n0", "n1", "e24-sentinel-9"]))         # sentinel last: no neighbour
            o3 = w.forward_batch_generation(T.batch(["e24-sentinel-0", "n1", "n2"], mode="extend"))
            res["extend_used_orig"] = T.STATE["conv_inputs"][-1] == o3["x_id"]
            T.STATE["capturing"] = True
            o4 = w.forward_batch_generation(T.batch(["e24-sentinel-0", "n1", "n2"]))
            res["capture_used_orig"] = T.STATE["conv_inputs"][-1] == o4["x_id"]
            T.STATE["capturing"] = False
            o5 = w.forward_batch_generation(T.batch(["e24-sentinel-0", "n1", "n2"], rows_bs=4))
            res["shape_used_orig"] = T.STATE["conv_inputs"][-1] == o5["x_id"]
        """)
        self.assertFalse(r.res["row6_finite"])                               # the #41736 leak into request 1
        self.assertNotEqual(r.res["used"], r.res["x_id"])                    # injected into a clone ...
        self.assertEqual((r.res["mark_static_id"], r.res["mark_static_dim"]), (r.res["used"], 1))   # ... marked static
        self.assertTrue(r.res["extend_used_orig"] and r.res["capture_used_orig"] and r.res["shape_used_orig"])
        o = self.outputs(r.out)
        c = o["counters"][r.res["pid"]]
        k = c["counters"]
        self.assertEqual((k["draft_nan_calls"], k["draft_nan_injections"]), (2, 2))
        self.assertEqual((k["neighbor_rows_checked"], k["neighbor_rows_nonfinite"]), (1, 1))
        self.assertEqual(k["inject_skipped_capturing"], 1)
        self.assertEqual(k["draft_inject_skipped_not_decode"], 1)
        self.assertEqual(k["draft_inject_skipped_shape"], 1)
        self.assertEqual(c["per_rid"]["n1"]["neighbor_rows_nonfinite"], 1)
        self.assertEqual(c["per_rid"]["e24-sentinel-0"]["draft_nan_injections"], 1)
        nb = [e for e in o["trace"][r.res["pid"]] if e["ev"] == "neighbor"]
        self.assertEqual([(e["sentinel"], e["neighbor"], e["nonfinite"]) for e in nb], [("e24-sentinel-0", "n1", True)])
        self.assertEqual(c["inject"]["installed"], ["_grouped_conv"])

    def test_draft_nan_patched_no_leak(self):
        r = self._draft("patched")
        self.assertTrue(r.res["row6_finite"])
        o = self.outputs(r.out)
        k = o["counters"][r.res["pid"]]["counters"]
        self.assertEqual((k["draft_nan_injections"], k["neighbor_rows_checked"]), (1, 1))
        self.assertEqual(k.get("neighbor_rows_nonfinite", 0), 0)
        nb = [e for e in o["trace"][r.res["pid"]] if e["ev"] == "neighbor"]
        self.assertEqual([(e["neighbor"], e["nonfinite"]) for e in nb], [("n1", False)])

    def _selector(self, variant):
        return self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["DFlashWorkerV2", "CandidateSelector"])
            w = T.make_worker(F)
            o = w.forward_batch_generation(T.batch(["e24-sentinel-0", "n1", "e24-sentinel-2"], greedy=[True, True, False]))
            T.done({"pid": os.getpid(), "tok00": o["tokens"][0][0], "q00": o["q"][0][0], "tok10": o["tokens"][1][0]})
        """, env={"E24_INJECT_SELECTOR_NAN": "1", "E24_HOOK_EAGER": "1", "E24T_VARIANT": variant})

    def test_selector_nan_unpatched_walks_out_patched_first_nan(self):
        r = self._selector("pristine")
        self.assertEqual(r.res["tok00"], 10)                                 # candidate_ids[0, 1, 0]: the next slot's
        self.assertEqual(r.res["q00"], [0.0, 0.0, 0.0, 0.0])
        c = self.outputs(r.out)["counters"][r.res["pid"]]
        k = c["counters"]
        self.assertEqual((k["selector_nan_injections"], k["zero_q_rows"], k["out_of_row_picks"]), (1, 1, 1))
        self.assertEqual(k["selector_skipped_nongreedy"], 1)                 # sampled sentinel never poisoned
        self.assertEqual(c["per_rid"]["e24-sentinel-0"]["out_of_row_picks"], 1)
        p = self._selector("patched")
        self.assertEqual(p.res["tok00"], 0)                                  # candidate_ids[0, 0, 0]
        self.assertEqual(p.res["q00"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(p.res["tok10"], r.res["tok10"])                     # the neighbour is untouched
        k = self.outputs(p.out)["counters"][p.res["pid"]]["counters"]
        self.assertEqual((k["selector_nan_injections"], k["first_candidate_picks"]), (1, 1))
        self.assertEqual((k.get("zero_q_rows", 0), k.get("out_of_row_picks", 0)), (0, 0))

    # ----------------------------------------------------------------------------------------------- chain-exec
    def test_chain_exec_image_sitecustomize(self):
        img = os.path.join(self.tmp, "image_fail")
        os.makedirs(img, exist_ok=True)
        with open(os.path.join(img, "sitecustomize.py"), "w") as f:
            f.write("import os\nos.environ['E24T_CHAINED'] = 'yes'\nimport wrapt_e24_missing_module\n")
        body = """
            T.done({"pid": os.getpid(), "chained": os.environ.get("E24T_CHAINED"), "chain": T.hook().CHAIN})
        """
        r = self.run_scen(body, env={"E24_HOOK_CHAIN": "1"}, pythonpath=self.hookdir + os.pathsep + img)
        self.assertEqual(r.res["chained"], "yes")
        self.assertIn("Error in sitecustomize; set PYTHONVERBOSE for traceback:\nModuleNotFoundError: No module named "
                      "'wrapt_e24_missing_module'", r.cp.stderr)
        rec = self.outputs(r.out)["env"][r.res["pid"]]
        ch = rec["hook"]["chain"]
        self.assertEqual((ch["found"], ch["ok"], ch["path"]), (True, False, os.path.join(img, "sitecustomize.py")))
        self.assertIn("wrapt_e24_missing_module", ch["error"])
        # inert (no E24_HOOK_DIR): the image's sitecustomize still runs, exactly as without the hook
        r = self.run_scen(body, env={"E24_HOOK_CHAIN": "1"}, hook_dir=False, pythonpath=self.hookdir + os.pathsep + img)
        self.assertEqual(r.res["chained"], "yes")
        self.assertEqual(os.listdir(r.out), [])
        # a copy of this hook later on the path is skipped; the next real one runs
        cp_dir, ok_dir = os.path.join(self.tmp, "image_copy"), os.path.join(self.tmp, "image_ok")
        os.makedirs(cp_dir, exist_ok=True)
        os.makedirs(ok_dir, exist_ok=True)
        shutil.copy2(os.path.join(self.hookdir, "sitecustomize.py"), os.path.join(cp_dir, "sitecustomize.py"))
        with open(os.path.join(ok_dir, "sitecustomize.py"), "w") as f:
            f.write("import os\nos.environ['E24T_CHAINED'] = 'ok'\n")
        r = self.run_scen(body, env={"E24_HOOK_CHAIN": "1"},
                          pythonpath=os.pathsep.join([self.hookdir, cp_dir, ok_dir]))
        self.assertEqual(r.res["chained"], "ok")
        self.assertEqual(r.res["chain"]["skipped"], [os.path.join(cp_dir, "sitecustomize.py")])
        self.assertTrue(r.res["chain"]["ok"])
        self.assertEqual(r.cp.stderr.count("Error in sitecustomize"), 0)
        # E24_HOOK_CHAIN=0: not run
        r = self.run_scen(body, env={"E24_HOOK_CHAIN": "0"}, pythonpath=self.hookdir + os.pathsep + img)
        self.assertIsNone(r.res["chained"])

    # ----------------------------------------------------------------------------------------------- probe / dirs
    def test_probe_resolution_main_thread_no_imports_from_thread(self):
        r = self.run_scen("""
            BAD = []
            def audit(ev, args):
                if ev == "import" and threading.current_thread().name == "e24-hook":
                    BAD.append(str(args[0]))
            sys.addaudithook(audit)
            F = T.install_fakes()
            T.wait_patched(["Scheduler"])
            s = F.sch.Scheduler()
            loop = s.run_event_loop()
            time.sleep(0.5)                                          # several flush cycles of the thread
            h = T.hook()
            T.done({"pid": os.getpid(), "loop": loop, "bad": BAD, "cache_dir_calls": T.STATE["cache_dir_calls"],
                    "resolved": h.RESOLVED})
        """, env={"E24_HOOK_PROBE": "1", "E24T_TP_RANK": "2", "E24T_TRITON_DIR": "/arm/triton",
                  "E24_HOOK_TARGETS": "Scheduler"})
        self.assertEqual(r.res["loop"], "loop-done")
        self.assertEqual(r.res["bad"], [])
        self.assertEqual(r.res["cache_dir_calls"], 0)
        o = self.outputs(r.out)
        pid = r.res["pid"]
        pr = o["probe"][pid]
        self.assertEqual((pr["engine"], pr["tp_rank"], pr["where"]), ("sglang", 2, "Scheduler.run_event_loop"))
        self.assertEqual(pr["target"]["lm_head"]["weight_dtype"], "torch.bfloat16")
        ks = pr["target"]["kv_scales"]
        self.assertEqual((ks["n"], ks["k_min"], ks["k_max"], ks["v_min"], ks["v_max"]), (2, 0.028, 0.051, 0.021, 0.254))
        self.assertEqual(pr["mamba"]["temporal_dtype"], "torch.float16")
        self.assertEqual(pr["mamba"]["conv_dtypes"], ["torch.bfloat16"])
        mods = pr["modules"]
        self.assertEqual(mods[M_SB]["sha256"], self.P.REFERENCE_FILE_SHA256[SB])
        self.assertEqual(mods["sglang.srt.models.dflash"]["sha256"],
                         self.P.REFERENCE_FILE_SHA256["srt/models/dflash.py"])
        rec = o["env"][pid]
        res = rec["resolved"]
        self.assertEqual((res["thread"], res["where"]), ("MainThread", "Scheduler.run_event_loop"))
        self.assertEqual(res["triton"], "/arm/triton")
        self.assertEqual((res["inductor"], res["inductor_source"]), ("/tmp/torchinductor_fakeuser", "default"))
        self.assertEqual(rec["role"]["tp_rank"], 2)
        self.assertEqual(rec["modules"][M_SB]["sha256"], self.P.REFERENCE_FILE_SHA256[SB])
        c = o["counters"][pid]
        self.assertTrue(c["self_test"]["ok"])
        # the patched copy is what a patched rank imports: its sha is the reference patched sha
        r2 = self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["Scheduler"])
            F.sch.Scheduler().run_event_loop()
            T.done({"pid": os.getpid()})
        """, env={"E24_HOOK_PROBE": "1", "E24T_VARIANT": "patched", "TORCHINDUCTOR_CACHE_DIR": "/arm/inductor"})
        o2 = self.outputs(r2.out)
        rec2 = o2["env"][r2.res["pid"]]
        self.assertEqual(rec2["modules"][M_SB]["sha256"], self.P.REFERENCE_PATCHED_SHA256[SB])
        self.assertEqual((rec2["resolved"]["inductor"], rec2["resolved"]["inductor_source"]), ("/arm/inductor", "env"))

    def test_allocator_snapshot_main_thread_on_idle(self):
        r = self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["Scheduler"])
            s = F.sch.Scheduler()
            s.req_to_token_pool.mamba_allocator.free_n = 3          # slots freed after a request, no counter moves
            time.sleep(0.3)
            s.on_idle()
            time.sleep(0.5)
            T.done({"pid": os.getpid(), "threads": sorted(set(T.STATE["alloc_threads"])),
                    "wrapped": hasattr(F.sch.Scheduler.on_idle, "__wrapped__")})
        """, env={"E24_HOOK_COUNTERS": "1", "E24_HOOK_TARGETS": "Scheduler"})
        self.assertTrue(r.res["wrapped"])
        self.assertEqual(r.res["threads"], ["MainThread"])
        c = self.outputs(r.out)["counters"][r.res["pid"]]
        self.assertEqual(c["alloc"], {"available": 3, "size": 68, "evictable": 2})
        # without counters on_idle is not wrapped and no snapshot is taken
        r = self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["Scheduler"])
            s = F.sch.Scheduler()
            s.on_idle()
            T.done({"threads": T.STATE["alloc_threads"], "wrapped": hasattr(F.sch.Scheduler.on_idle, "__wrapped__")})
        """)
        self.assertEqual(r.res, {"threads": [], "wrapped": False})

    def test_scheduler_init_failure_flushes_and_propagates(self):
        r = self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["Scheduler"])
            try:
                F.sch.Scheduler()
            except RuntimeError as e:
                with open(os.environ["E24T_RESULT"], "w") as f:
                    json.dump({"pid": os.getpid(), "err": str(e)}, f)
                os._exit(4)
        """, env={"E24T_INIT_FAIL": "1", "E24_HOOK_FLUSH_SEC": "30"}, expect_rc=4)
        self.assertEqual(r.res["err"], "scheduler init failed")
        evs = self.outputs(r.out)["trace"][r.res["pid"]]
        self.assertTrue(any(e["ev"] == "exception" and e["where"] == "Scheduler.__init__" for e in evs))

    # ----------------------------------------------------------------------------------------------- self-test/poll
    def test_targets_selftest_slow_polling_flush_on_change(self):
        r = self.run_scen("""
            F = T.install_fakes(only=[T.M_SCHED, T.M_SB])
            T.wait_patched(["Scheduler", "ScheduleBatch"])
            h = T.hook()
            time.sleep(0.8)                                           # past E24_HOOK_FAST_SEC
            phase, sleep = h._POLL["phase"], h._POLL["sleep"]
            p = os.path.join(os.environ["E24_HOOK_DIR"], "counters_%d.json" % os.getpid())
            t1 = json.load(open(p))["t"]
            time.sleep(2.5)                                           # >= 2 slow flushes, nothing changed
            t2 = json.load(open(p))["t"]
            h._count("e24t_marker")
            time.sleep(1.6)
            t3 = json.load(open(p))["t"]
            T.done({"pid": os.getpid(), "phase": phase, "sleep": sleep, "t1": t1, "t2": t2, "t3": t3})
        """, env={"E24_HOOK_COUNTERS": "1", "E24_HOOK_TARGETS": "Scheduler,ScheduleBatch,DFlashWorkerV2,_grouped_conv",
                  "E24_HOOK_FAST_SEC": "0.4"})
        self.assertEqual(r.res["phase"], "slow")
        self.assertGreaterEqual(r.res["sleep"], 1.0)
        self.assertEqual(r.res["t1"], r.res["t2"])                          # flush only on change
        self.assertGreater(r.res["t3"], r.res["t2"])
        st = self.outputs(r.out)["counters"][r.res["pid"]]["self_test"]
        self.assertFalse(st["ok"])
        self.assertEqual(st["missing"], ["DFlashWorkerV2", "_grouped_conv"])
        self.assertEqual(st["not_enabled"], ["_grouped_conv"])

    # ----------------------------------------------------------------------------------------------- robustness
    def test_robustness(self):
        r = self.run_scen("""
            F = T.install_fakes(drop_methods=("_alloc_mamba_slot",))
            T.wait_patched(["ScheduleBatch", "MambaComponent", "SchedulerBatchResultProcessor"])
            h = T.hook()
            res = {"pid": os.getpid(), "failed": sorted(h.FAILED),
                   "mamba_untouched": F.mm.MambaComponent.__dict__["prepare_for_caching_req"] is
                   F.orig["mm"]["prepare_for_caching_req"]}
            # a broken engine object: the wrapper's own error is counted, the ORIGINAL exception propagates unchanged
            class Broken:
                req_to_token_pool = None
            def call(fn):
                try:
                    fn(Broken(), 128)
                    return None
                except Exception as e:
                    return [type(e).__name__, str(e)]
            res["wrapped_exc"] = call(F.sb.ScheduleBatch.mamba_lazy_prealloc_at_boundary)
            res["orig_exc"] = call(F.sb.ScheduleBatch.mamba_lazy_prealloc_at_boundary.__wrapped__)
            # garbage results from the engine never break the boundary wrapper
            rp = F.brp.SchedulerBatchResultProcessor()
            rp.tree_cache = T.SimpleNamespace(page_size=1)
            req = T.Req("x")
            req.seqlen = 260
            bad = T.SimpleNamespace(num_correct_drafts_per_req_cpu=["?"], num_non_draft_tokens_per_req=1,
                                    grammar_retained_tokens=None)
            try:
                res["boundary"] = list(rp._mamba_check_track_boundary(req, T.SimpleNamespace(spec_algorithm=T.SA(False)),
                                                                      bad, 0))
            except Exception as e:
                res["boundary"] = [type(e).__name__]
            odd = T.SimpleNamespace(num_correct_drafts_per_req_cpu=[5], num_non_draft_tokens_per_req=1,
                                    grammar_retained_tokens=[object()])     # the original ignores it, the hook cannot len() it
            res["boundary2"] = list(rp._mamba_check_track_boundary(req, T.SimpleNamespace(spec_algorithm=T.SA(False)),
                                                                   odd, 0))
            res["flush"], res["fast"] = h._FLUSH, h._FAST
            h._flush_sync()
            T.done(res)
        """, env={"E24_HOOK_COUNTERS": "1", "E24_HOOK_FLUSH_SEC": "abc", "E24_HOOK_FAST_SEC": "-5"})
        self.assertEqual(r.res["failed"], ["MambaComponent"])
        self.assertTrue(r.res["mamba_untouched"])
        self.assertEqual(r.res["wrapped_exc"], r.res["orig_exc"])
        self.assertEqual(r.res["wrapped_exc"][0], "AttributeError")
        self.assertEqual(r.res["boundary"], ["TypeError"])                  # the original raises on '?' - 1; unchanged
        self.assertEqual(r.res["boundary2"], [True, 256])                   # the original result, hook error counted
        self.assertEqual((r.res["flush"], r.res["fast"]), (2.0, 0.0))
        k = self.outputs(r.out)["counters"][r.res["pid"]]["counters"]
        self.assertGreaterEqual(k["hook_error_prealloc_pre"], 1)
        self.assertGreaterEqual(k["hook_error_boundary"], 1)
        self.assertEqual(k["patch_failed"], 1)
        # an unusable E24_HOOK_DIR (a regular file) never breaks the process
        blocker = os.path.join(self.tmp, "not_a_dir")
        with open(blocker, "w") as f:
            f.write("x")
        r = self.run_scen("""
            F = T.install_fakes()
            T.wait_patched(["ScheduleBatch"])
            req = T.Req("y")
            req.kv.mamba_last_track_seqlen = 192
            b = F.sb.ScheduleBatch([req], [256], T.Pool(free=0))
            b.mamba_lazy_prealloc_at_boundary(128)
            T.hook()._flush_sync()
            time.sleep(0.2)
            T.done({"ok": True, "failed": T.hook().COUNTERS.get("optional_alloc_failed")})
        """, env={"E24_HOOK_COUNTERS": "1", "E24_HOOK_DIR": blocker})
        self.assertEqual(r.res, {"ok": True, "failed": 1})


if __name__ == "__main__":
    unittest.main()
