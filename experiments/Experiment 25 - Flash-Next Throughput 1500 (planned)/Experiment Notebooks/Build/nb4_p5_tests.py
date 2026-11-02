"""Quick CPU tests for nb4_p5_engine.py (NB4 arm P): the strict patch on a copy of the real Merlin ad42e37
cuda_communicator.py (applies, refuses on any modified file, reverts), the patched dispatcher / init block / log line
executed with stubs, the backend (threshold, engaged counters, self-test, abort, config) on a 2-rank CPU simulator of the
P5 ops, the read-back parser and labels, and that the gate / worker-extension sources compile.

    python nb4_p5_tests.py            (needs torch; CPU only; ~seconds)
NOT covered here (GPU only): the real p5_host_ar.so, CUDA graph capture, both TP pairs at once, the engine boot."""
import ast
import json
import os
import shutil
import sys
import tempfile
import textwrap
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nb4_p5_engine as E  # noqa: E402

MERLIN_SRC = Path(os.environ.get("NB4_MERLIN_SRC", r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src"))
REAL_CC = MERLIN_SRC / E.CC_REL


def load_backend():
    m = types.ModuleType("nb4_p5_backend")
    m.__file__ = "<nb4_p5_backend>"
    exec(compile(E.BACKEND_SRC, "nb4_p5_backend.py", "exec"), m.__dict__)
    sys.modules["nb4_p5_backend"] = m
    return m


B = load_backend()


# ---------------------------------------------------------------- 2-rank CPU simulator of torch.ops.p5_host_ar
class SimWorld:
    def __init__(self, world=2, corrupt=False):
        self.world = world
        self.bar = threading.Barrier(world, timeout=30)
        self.slots, self.names = {}, {}
        self.calls = [0] * world
        self.abort = 0
        self.corrupt = corrupt
        self.bcast = [None]

    def broadcast(self, rank):
        def f(obj, group):
            if rank == 0:
                self.bcast[0] = obj[0]
            self.bar.wait()
            obj[0] = self.bcast[0]
            self.bar.wait()
        return f


class SimOps:
    def __init__(self, w, rank):
        self.w, self.rank = w, rank

    def info(self):
        return "p5_host_ar v1 two-shot host-staged all-reduce; worlds 2/4/8; sim"

    def init(self, rank, world, name, max_bytes, device, timeout_ms):
        assert rank == self.rank and world == self.w.world and name.startswith("/p5har_nb4_")
        self.w.names[rank] = name
        self.w.bar.wait()
        assert len(set(self.w.names.values())) == 1, "ranks got different shm names"
        return 0

    def status(self, h):
        return [self.w.abort] + [0] * self.w.world + [48]

    def epoch(self, h):
        return self.w.calls[self.rank]

    def all_reduce(self, h, inp, out, mode, blocks):
        assert inp.is_contiguous() and out.is_contiguous() and inp.data_ptr() % 16 == 0 and mode == 0 and blocks == 0
        assert inp.numel() == out.numel() and inp.dtype == out.dtype
        idx = self.w.calls[self.rank]
        self.w.slots[(idx, self.rank)] = inp.clone()
        self.w.bar.wait()
        ref = B.ordered_ref([self.w.slots[(idx, r)] for r in range(self.w.world)], inp.dtype)
        if self.w.corrupt:
            ref = ref.view(torch.int16 if ref.element_size() == 2 else torch.int32).add(1).view(ref.dtype)
        out.copy_(ref)
        self.w.bar.wait()
        self.w.calls[self.rank] += 1


ENV = {"NB4_P5_SO": "/x/p5_host_ar.so", "NB4_P5_SO_SHA256": "ab" * 32, "NB4_P5_LOG_S": "0"}


def run2(fn):
    res, errs = [None, None], [None, None]

    def t(r):
        try:
            res[r] = fn(r)
        except Exception as e:  # noqa: BLE001
            errs[r] = e
    th = [threading.Thread(target=t, args=(r,)) for r in range(2)]
    [x.start() for x in th]
    [x.join(60) for x in th]
    return res, errs


def make_pair(w=None, env=None, capflag=None):
    w = w or SimWorld()
    capflag = capflag if capflag is not None else {"on": False}

    def mk(r):
        return B.P5HostAllReduce(group=None, device=torch.device("cpu"), unique_name="tp:0", rank=r, world_size=2,
                                 ops=SimOps(w, r), environ=dict(ENV, **(env or {})), broadcast=w.broadcast(r),
                                 capturing=lambda: capflag["on"])
    res, errs = run2(mk)
    return w, res, errs, capflag


# ---------------------------------------------------------------- patch tests
@unittest.skipUnless(REAL_CC.is_file(), f"real Merlin file not found: {REAL_CC}")
class PatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="nb4p5_"))
        dst = self.tmp / E.CC_REL
        dst.parent.mkdir(parents=True)
        shutil.copy2(REAL_CC, dst)
        (self.tmp / "vllm" / "__init__.py").write_text("")
        self.f = dst

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_real_file_is_pre(self):
        self.assertEqual(E.sha256_bytes(REAL_CC.read_bytes()), E.CC_PRE)
        self.assertNotIn(b"\r", REAL_CC.read_bytes())

    def test_patch_text(self):
        self.assertNotIn("\r", E.PATCH)
        self.assertIn(f"pre-patch sha256  {E.CC_PRE}", E.PATCH)
        self.assertIn(f"post-patch sha256 {E.CC_POST}", E.PATCH)
        self.assertIn("+        " + E.MARKER, E.PATCH)

    def test_apply_verify_reapply_revert(self):
        rec = E.apply_p5_patch(self.tmp, check_origin=True)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["applied"], "applied")
        self.assertEqual(rec["sha_after"], E.CC_POST)
        self.assertTrue(rec["marker"])
        self.assertEqual(E.verify_p5_patch(self.tmp)["state"], "applied")
        self.assertEqual(E.apply_p5_patch(self.tmp, check_origin=False)["applied"], "already_applied")
        r = E.revert_p5_patch(self.tmp)
        self.assertEqual(r["reverted"], "reverted")
        self.assertEqual(E.sha256_bytes(self.f.read_bytes()), E.CC_PRE)
        self.assertFalse((self.tmp / E.BACKEND_REL).exists())
        self.assertEqual(E.verify_p5_patch(self.tmp)["state"], "not_applied")
        self.assertEqual(E.revert_p5_patch(self.tmp)["reverted"], "not_applied")

    def test_refuses_modified(self):
        raw = self.f.read_bytes()
        for label, bad in (("one char", raw.replace(b"use_fi_ar = (", b"use_fi_ar  = (", 1)),
                           ("crlf", raw.replace(b"\n", b"\r\n")),
                           ("trailing newline", raw + b"\n")):
            self.f.write_bytes(bad)
            with self.assertRaises(RuntimeError, msg=label) as cm:
                E.apply_p5_patch(self.tmp, check_origin=False)
            self.assertIn("PATCH_P5_PRE_SHA_MISMATCH", str(cm.exception))
            self.assertEqual(self.f.read_bytes(), bad, "a refused patch must not touch the file")
            self.assertFalse((self.tmp / E.BACKEND_REL).exists())
            with self.assertRaises(RuntimeError):
                E.revert_p5_patch(self.tmp)
            self.assertEqual(E.verify_p5_patch(self.tmp)["state"], "foreign")

    def test_strict_applier_rejects_shifted_context(self):
        txt = REAL_CC.read_text(encoding="utf-8")
        with self.assertRaises(ValueError):
            E.apply_unified("# extra line\n" + txt, E.PATCH)

    def _patched(self):
        return E.apply_unified(REAL_CC.read_text(encoding="utf-8"), E.PATCH)

    def _funcs(self, text):
        tree = ast.parse(text)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CudaCommunicator")
        return {n.name: textwrap.dedent(ast.get_source_segment(text, n)) for n in cls.body
                if isinstance(n, ast.FunctionDef)}

    def _ns(self):
        logs = []
        ns = {"torch": torch, "envs": SimpleNamespace(VLLM_BATCH_INVARIANT=False),
              "current_platform": SimpleNamespace(is_rocm=lambda: False),
              "should_nccl_symm_mem_allreduce": lambda ws, t: False,
              "logger": SimpleNamespace(info_once=lambda msg, *a, **k: logs.append(msg % a)),
              "NCCL_SYMM_MEM_ALL_REDUCE_CONFIG": {"min_world_size": 4, "custom_ar_preferred_ranges": {},
                                                  "always_use_above_world_size": 8},
              "is_symmetric_memory_enabled": lambda: False}
        return ns, logs

    def _fake_self(self, p5=None, with_attr=True):
        calls = []

        class PyNccl:
            world_size, disabled = 2, False

            def all_reduce(self, x):
                calls.append(("pynccl", x.numel()))
                return x * 2
        d = dict(fi_ar_comm=None, pynccl_comm=PyNccl(), qr_comm=None, fi_pcie_ipc_ar_comm=None, use_aiter_allreduce=False,
                 aiter_ar_comm=None, ca_comm=None, symm_mem_comm=None, device_group=None, unique_name="tp:0", world_size=2)
        if with_attr:
            d["nb4_p5_comm"] = p5
        return SimpleNamespace(**d), calls

    def test_patched_dispatch_order_and_fallthrough(self):
        f = self._funcs(self._patched())
        ns, _ = self._ns()
        exec(f["all_reduce"], ns)
        ar = ns["all_reduce"]
        hits = []

        class P5:
            def maybe_all_reduce(self, x):
                hits.append(x.numel())
                return x * 3 if x.numel() * x.element_size() <= 327680 else None
        s, calls = self._fake_self(P5())
        small = torch.ones(16 * 2560, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(ar(s, small), small * 3))
        self.assertEqual(calls, [])
        big = torch.ones(65 * 2560, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(ar(s, big), big * 2))
        self.assertEqual(calls, [("pynccl", big.numel())])
        self.assertEqual(hits, [small.numel(), big.numel()])
        for p5, attr in ((None, True), (None, False)):  # NB4_P5 unset / an object built without the attribute
            s, calls = self._fake_self(p5, with_attr=attr)
            self.assertTrue(torch.equal(ar(s, small), small * 2))
            self.assertEqual(calls, [("pynccl", small.numel())])

    def test_patched_log_line(self):
        pre, post = self._funcs(REAL_CC.read_text(encoding="utf-8")), self._funcs(self._patched())
        ns0, logs0 = self._ns()
        exec(pre["_log_all_reduce_backend_selection"], ns0)
        ns1, logs1 = self._ns()
        exec(post["_log_all_reduce_backend_selection"], ns1)
        s0, _ = self._fake_self(None, with_attr=False)
        ns0["_log_all_reduce_backend_selection"](s0)
        s1, _ = self._fake_self(None)
        ns1["_log_all_reduce_backend_selection"](s1)
        self.assertEqual(logs0, logs1, "NB4_P5 unset: the read-back line must be byte-identical to Merlin's")
        s2, _ = self._fake_self(object())
        ns1["_log_all_reduce_backend_selection"](s2)
        self.assertIn("Using ['P5_HOST', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0' out of "
                      "potential backends: ['P5_HOST', 'FLASHINFER_PCIE_IPC'", logs1[-1])
        info = E.parse_log("(Worker_PP0_TP0 pid=1) INFO 11-01 [cuda_communicator.py:372] " + logs1[-1])
        self.assertTrue(info["p5_first_in_dispatch"])

    def test_patched_init_block_gating(self):
        text = self._patched()
        i = text.index("        " + E.MARKER)
        j = text.index("        if self.world_size > 1:\n            self._log_all_reduce_backend_selection()", i)
        self.assertLess(text.index("self.qr_comm = QuickAllReduce("), i)
        block = textwrap.dedent(text[i:j])
        made = []
        fake_mod = types.ModuleType("nb4_p5_backend")

        class Rec:
            def __init__(self, **kw):
                made.append(kw)
        fake_mod.P5HostAllReduce = Rec
        saved = sys.modules.get("nb4_p5_backend")
        sys.modules["nb4_p5_backend"] = fake_mod
        old_env = os.environ.get("NB4_P5")
        try:
            cases = [("1", "tp:0", 2, True), ("1", "tp:0", 1, False), ("1", "ep:0", 2, False), ("1", "pp:0", 2, False),
                     ("0", "tp:0", 2, False), (None, "tp:0", 2, False)]
            for env, name, ws, want in cases:
                if env is None:
                    os.environ.pop("NB4_P5", None)
                else:
                    os.environ["NB4_P5"] = env
                made.clear()
                s = SimpleNamespace(world_size=ws, cpu_group="CPUG", device=torch.device("cpu"), rank=1)
                exec(block, {"self": s, "unique_name": name})
                self.assertEqual(s.nb4_p5_comm is not None, want, (env, name, ws))
                if want:
                    self.assertEqual(made, [dict(group="CPUG", device=torch.device("cpu"), unique_name=name, rank=1,
                                                 world_size=2)])
        finally:
            sys.modules.pop("nb4_p5_backend")
            if saved is not None:
                sys.modules["nb4_p5_backend"] = saved
            if old_env is None:
                os.environ.pop("NB4_P5", None)
            else:
                os.environ["NB4_P5"] = old_env


# ---------------------------------------------------------------- backend tests (CPU simulator)
class BackendTests(unittest.TestCase):
    def test_decide_threshold(self):
        d, bf = B.decide, torch.bfloat16
        self.assertEqual(d(327680, bf, True, 1, 327680), "p5")
        self.assertEqual(d(327682, bf, True, 1, 327680), "size_gt_max")
        self.assertEqual(d(5120, bf, True, 1, 327680), "p5")
        self.assertEqual(d(2, bf, True, 1, 327680), "p5")
        self.assertEqual(d(0, bf, True, 1, 327680), "empty")
        self.assertEqual(d(1024, bf, True, 2048, 327680), "size_lt_min")
        self.assertEqual(d(1024, torch.int8, True, 1, 327680), "dtype")
        self.assertEqual(d(1024, torch.int32, True, 1, 327680), "dtype")
        self.assertEqual(d(1024, torch.float16, True, 1, 327680), "p5")
        self.assertEqual(d(1024, torch.float32, True, 1, 327680), "p5")
        self.assertEqual(d(1024, bf, False, 1, 327680), "not_cuda")

    def test_env_config(self):
        c = B.env_config(ENV)
        self.assertEqual((c["max_bytes"], c["min_bytes"], c["timeout_ms"], c["selftest"]), (327680, 1, 300000, True))
        self.assertEqual(B.env_config(dict(ENV, NB4_P5_MAX_BYTES="655360"))["max_bytes"], 655360)
        for bad in ({"NB4_P5_MAX_BYTES": "655362"}, {"NB4_P5_MAX_BYTES": "abc"}, {"NB4_P5_TIMEOUT_MS": "10"},
                    {"NB4_P5_SO": ""}, {"NB4_P5_SO_SHA256": "xyz"}, {"NB4_P5_MIN_BYTES": "400000"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                B.env_config(dict(ENV, **bad))
        self.assertEqual(E.server_env("/a.so", "AB" * 32)["NB4_P5_SO_SHA256"], "ab" * 32)
        with self.assertRaises(ValueError):
            E.server_env("/a.so", "ab" * 32, max_bytes=E.HARD_MAX_BYTES + 2)
        self.assertEqual(B.DEFAULT_MAX_BYTES, E.DEFAULT_MAX_BYTES)
        self.assertEqual(B.HARD_MAX_BYTES, E.HARD_MAX_BYTES)

    def test_init_selftest_and_counters(self):
        w, comms, errs, cap = make_pair()
        self.assertEqual(errs, [None, None], errs)
        for c in comms:
            self.assertTrue(c.selftest["ok"])
            self.assertEqual(c.selftest_calls, 8)          # 4 sizes (5120, 81920, 102400, max) x bf16 + fp32
            self.assertEqual(c.counts, {"eager": {"p5": 0, "fallback": 0}, "capture": {"p5": 0, "fallback": 0}})
        x = [torch.randn(16, 2560).to(torch.bfloat16) * (r + 1) for r in range(2)]
        big = [torch.zeros(65 * 2560, dtype=torch.bfloat16)] * 2
        i8 = [torch.zeros(4096, dtype=torch.int8)] * 2

        def step(r):
            c = comms[r]
            o = c.maybe_all_reduce(x[r])
            o2 = c.maybe_all_reduce(x[r].t())             # non-contiguous view: copied, same decision on both ranks
            return o, o2, c.maybe_all_reduce(big[r]), c.maybe_all_reduce(i8[r])
        res, errs = run2(step)
        self.assertEqual(errs, [None, None], errs)
        ref = B.ordered_ref(x, torch.bfloat16)
        for o, o2, ob, oi in res:
            self.assertEqual(B.bits_diff(o, ref), 0)
            self.assertEqual(tuple(o2.shape), (2560, 16))
            self.assertEqual(B.bits_diff(o2.contiguous(), B.ordered_ref([t.t().contiguous() for t in x], torch.bfloat16)), 0)
            self.assertIsNone(ob)
            self.assertIsNone(oi)
        cap["on"] = True                                    # one capture of 3 calls: counted once per captured call

        def capstep(r):
            return [comms[r].maybe_all_reduce(x[r]) for _ in range(3)] + [comms[r].maybe_all_reduce(big[r])]
        res, errs = run2(capstep)
        self.assertEqual(errs, [None, None], errs)
        cap["on"] = False
        for r, c in enumerate(comms):
            st = c.stats(read_device=True)
            self.assertEqual(st["counts"]["eager"], {"p5": 2, "fallback": 2})
            self.assertEqual(st["counts"]["capture"], {"p5": 3, "fallback": 1})
            self.assertEqual(st["fallback_reasons"], {"size_gt_max": 2, "dtype": 1})
            self.assertEqual(st["capture_by_size"], {"81920": [3, 0], str(65 * 5120): [0, 1]})
            self.assertEqual(st["eager_by_size"]["81920"], [2, 0])
            self.assertEqual(st["p5_executed"], 5)           # device epoch minus the 8 self-test calls
            self.assertEqual(st["device_epoch"], 13)
            self.assertFalse(st["aborted"])
        sj = json.loads(B.stats_json(read_device=False))
        self.assertGreaterEqual(len(sj["comms"]), 2)
        # the worker extension sees the same registry
        ext = {}
        exec(E.WORKER_EXT_SRC, ext)
        self.assertIn("comms", json.loads(ext["NB4P5WorkerExt"]().nb4_p5_stats("0")))
        # abort: sticky; next eager call raises, a capturing call does not
        w.abort = 1
        c0 = comms[0]
        self.assertEqual(c0.poll_status()[0], 1)
        self.assertTrue(c0.aborted)
        with self.assertRaises(RuntimeError) as cm:
            c0.maybe_all_reduce(x[0])
        self.assertIn("NB4_P5_ABORTED", str(cm.exception))
        self.assertIsNone(c0.maybe_all_reduce(big[0]))      # fallback path is unaffected

    def test_selftest_failure_fails_closed(self):
        _, comms, errs, _ = make_pair(w=SimWorld(corrupt=True))
        for e in errs:
            self.assertIsInstance(e, RuntimeError)
            self.assertIn("NB4_P5_INIT_FAILED", str(e))
            self.assertIn("NB4_P5_SELFTEST_FAILED", str(e))

    def test_selftest_skip_and_max_bytes(self):
        _, comms, errs, _ = make_pair(env={"NB4_P5_SELFTEST": "0", "NB4_P5_MAX_BYTES": "655360"})
        self.assertEqual(errs, [None, None], errs)
        self.assertEqual(comms[0].selftest, {"skipped": True})
        self.assertEqual(comms[0].max_bytes, 655360)

    def test_world_and_lib_fail_closed(self):
        with self.assertRaises(RuntimeError) as cm:
            B.P5HostAllReduce(group=None, device=torch.device("cpu"), unique_name="tp:0", rank=0, world_size=4,
                              ops=SimOps(SimWorld(4), 0), environ=ENV, broadcast=lambda o, g: None)
        self.assertIn("world_size 4 != 2", str(cm.exception))
        with tempfile.TemporaryDirectory() as td:
            so = Path(td) / "p5_host_ar.so"
            so.write_bytes(b"not a library")
            env = dict(ENV, NB4_P5_SO=str(so), NB4_P5_SO_SHA256="0" * 64)
            with self.assertRaises(RuntimeError) as cm:     # real-ops path: hashed BEFORE torch.ops.load_library
                B.P5HostAllReduce(group=None, device=torch.device("cpu"), unique_name="tp:0", rank=0, world_size=2,
                                  environ=env, broadcast=lambda o, g: None)
            self.assertIn("sha256", str(cm.exception))
            self.assertIsNone(B._LIB["path"])


# ---------------------------------------------------------------- read-back / labels / sources
class ReadbackTests(unittest.TestCase):
    def _log(self, n_ok=4, fail=False, abort=False, p5_first=True):
        lines = []
        for r in range(n_ok):
            ev = {"event": "init", "ok": True, "group": "tp:0", "rank": r % 2, "pid": 100 + r,
                  "selftest": {"ok": True, "cases": [], "calls": 8}}
            lines.append(f"(Worker_PP{r // 2}_TP{r % 2} pid={100 + r}) NB4_P5 " + json.dumps(ev))
        if fail:
            lines.append("(Worker pid=9) NB4_P5 " + json.dumps({"event": "init_failed", "error": "boom"}))
        if abort:
            lines.append("(Worker pid=9) NB4_P5 " + json.dumps({"event": "abort", "status": [1, 0, 16777216, 48]}))
        be = "'P5_HOST', 'PYNCCL'" if p5_first else "'PYNCCL'"
        lines.append(f"INFO [cuda_communicator.py:372] Using [{be}] all-reduce backends (in dispatch order) for group "
                     "'tp:0' out of potential backends: ['P5_HOST', 'FLASHINFER_PCIE_IPC'].")
        return "\n".join(lines)

    def test_labels(self):
        st = [{"comms": [{"p5_executed": 100, "aborted": False}]} for _ in range(4)]
        self.assertEqual(E.p5_arm_label(self._log(), st)[0], "ENGAGED")
        self.assertEqual(E.p5_arm_label(self._log())[0], "ENGAGED")
        self.assertEqual(E.p5_arm_label(self._log(n_ok=3), st)[0], "NOT_INTEGRATED")
        self.assertEqual(E.p5_arm_label(self._log(fail=True), st)[0], "NOT_INTEGRATED")
        self.assertEqual(E.p5_arm_label(self._log(p5_first=False), st)[0], "NOT_INTEGRATED")
        self.assertEqual(E.p5_arm_label(self._log(abort=True), st)[0], "CORRECTNESS_FAIL")
        st0 = [dict(s) for s in st]
        st0[2] = {"comms": [{"p5_executed": 0}]}
        self.assertEqual(E.p5_arm_label(self._log(), st0)[0], "NOT_INTEGRATED")
        self.assertEqual(E.p5_arm_label("no p5 here", None)[0], "NOT_INTEGRATED")

    def test_engaged_per_step(self):
        mk = lambda e, f: {"comms": [{"group": "tp:0", "rank": 0, "p5_executed": e,  # noqa: E731
                                      "counts": {"eager": {"p5": 0, "fallback": f}, "capture": {}}}]}
        r = E.engaged_per_step([mk(1000, 5)], [mk(1000 + 51 * 200, 25)], steps=200, ar_per_forward=51)
        self.assertEqual(r[0]["p5_per_step"], 51.0)
        self.assertEqual(r[0]["nccl_eager_per_step"], 0.1)
        self.assertEqual(r[0]["nccl_in_graph_per_step_est"], 0.0)

    def test_expected_gain(self):
        g = {r["T"]: r for r in E.expected_gain(tokens=(16, 20, 128))}
        self.assertAlmostEqual(g[16]["save_ms_per_microbatch"], 0.934, places=3)
        self.assertAlmostEqual(g[20]["save_ms_per_microbatch"], 1.256, places=3)
        self.assertIn("note", g[128])                      # above the default threshold: NCCL

    def test_sources_compile_and_lf(self):
        for name, (text, sha) in E.payload_files().items():
            compile(text, name, "exec")
            self.assertNotIn("\r", text)
            self.assertEqual(E.sha256_text(text), sha)
        for f in (HERE / "nb4_p5_engine.py", HERE / "nb4_p5_tests.py"):
            self.assertNotIn(b"\r", f.read_bytes(), f.name)
        ns = {}
        exec(compile(E.GATE_SRC, "nb4_p5_gate.py", "exec"), ns)
        self.assertEqual(ns["overlap"]([0, 10], [5, 20]), 0.5)
        self.assertEqual(ns["overlap"]([0, 10], [20, 30]), 0.0)
        x = ns["gen"](torch, torch.device("cpu"), 1, 0, 64, "cancel")
        self.assertEqual(x.dtype, torch.bfloat16)
        base = torch.arange(2 * 100, dtype=torch.float32).view(2, 100).to(torch.bfloat16)
        self.assertTrue(torch.equal(ns["fillv"](base, 3), ns["fillv"](base.clone(), 3)))

    def test_write_payload(self):
        with tempfile.TemporaryDirectory() as td:
            out = E.write_payload(td)
            self.assertEqual(set(out), {"nb4_p5_backend.py", "nb4_p5_worker_ext.py", "nb4_p5_gate.py"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
