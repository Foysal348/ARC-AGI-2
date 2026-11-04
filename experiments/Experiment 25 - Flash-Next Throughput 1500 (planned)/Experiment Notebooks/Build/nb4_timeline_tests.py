"""Quick tests for nb4_timeline.py: synthetic traces with known answers per category + the real NB1 R2 B1 regression.

Run: python nb4_timeline_tests.py        (stdlib unittest; the real-trace test is skipped when the files are absent)
NB4_B1_TRACES=<dir> overrides the real-trace directory.
"""
import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nb4_timeline as T  # noqa: E402

REAL_DIR = Path(os.environ.get("NB4_B1_TRACES", str(
    Path(__file__).resolve().parents[4] / "_scratch" / "e25_gpu_out" / "q27-xhigh-e25-nb1-capacity-r2" / "nb1" /
    "traces" / "profile_b1")))
SCOPE = "execute_context_0(0)_generation_4(16)"
CPU_PID = 100


class Rank:
    """A tiny Chrome-trace writer: GPU events on pid 0 / tid = stream, host events on pid 100 / tid 100."""

    def __init__(self, name, base_ns=1_790_000_000_000_000_000, shift=0.0):
        self.name, self.base_ns, self.shift, self.ev, self.corr = name, base_ns, shift, [], 1000

    def _c(self):
        self.corr += 1
        return self.corr

    def kernel(self, a, b, name="void cutlass::Kernel2<gemm>(x)", corr=None, stream=7, args=None):
        ar = {"correlation": corr if corr is not None else self._c(), "stream": stream}
        ar.update(args or {})
        self.ev.append({"ph": "X", "cat": "kernel", "name": name, "pid": 0, "tid": stream, "ts": a + self.shift,
                        "dur": b - a, "args": ar})
        return ar["correlation"]

    def copy(self, a, b):
        self.ev.append({"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "pid": 0, "tid": 7, "ts": a + self.shift,
                        "dur": b - a, "args": {"correlation": self._c()}})

    def runtime(self, ts, name, corr, dur=10.0):
        self.ev.append({"ph": "X", "cat": "cuda_runtime", "name": name, "pid": CPU_PID, "tid": CPU_PID,
                        "ts": ts + self.shift, "dur": dur, "args": {"correlation": corr}})

    def cpu_op(self, a, b, name="aten::copy_"):
        self.ev.append({"ph": "X", "cat": "cpu_op", "name": name, "pid": CPU_PID, "tid": CPU_PID, "ts": a + self.shift,
                        "dur": b - a, "args": {}})

    def scope(self, a, b, name=SCOPE):
        self.ev.append({"ph": "X", "cat": "user_annotation", "name": name, "pid": CPU_PID, "tid": CPU_PID,
                        "ts": a + self.shift, "dur": b - a, "args": {}})

    def post(self, ts, seq, dtype="Int", nelems=8, coll="broadcast", pg="pp_broadcast"):
        """The host-side record_param_comms op of a collective (what the step posted)."""
        self.ev.append({"ph": "X", "cat": "cpu_op", "name": "record_param_comms", "pid": CPU_PID, "tid": CPU_PID,
                        "ts": ts + self.shift, "dur": 5.0,
                        "args": {"Collective name": coll, "dtype": dtype, "In msg nelems": nelems, "Seq": seq,
                                 "Process Group Description": pg, "Process Group Ranks": "[0, 1]"}})

    def bcast_kernel(self, a, b, seq, dtype="Int", nelems=8, pg="pp_broadcast"):
        return self.kernel(a, b, name=BCAST, stream=31,
                           args={"Collective name": "broadcast", "Process Group Ranks": "[0, 1]", "dtype": dtype,
                                 "In msg nelems": nelems, "Seq": seq, "Process Group Description": pg})

    def graph(self, launch_ts, a, b, nodes):
        """One cudaGraphLaunch whose `nodes` kernels tile [a, b]; returns the correlation id."""
        g = self._c()
        self.runtime(launch_ts, "cudaGraphLaunch", g)
        w = (b - a) / nodes
        for i in range(nodes):
            self.kernel(a + i * w, a + (i + 1) * w, corr=g)
        return g

    def eager_forward(self, sa, sb, launch_ts, ka, kb, name="void cutlass::Kernel2<gemm>(x)"):
        """A scope that launches one eager kernel (kernel at [ka, kb])."""
        self.scope(sa, sb)
        c = self.kernel(ka, kb, name=name)
        self.runtime(launch_ts, "cudaLaunchKernel", c)
        return c

    def trace(self):
        return {"_name": self.name, "baseTimeNanoseconds": str(self.base_ns), "traceEvents": list(self.ev)}


BCAST = "ncclDevKernel_Broadcast_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
ALLRED = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)"


def pp2_tp1(shift1=0.0, base1_delta_ns=0):
    """Two stages, one rank each. Stage-0 tick 0 = [1000, 21000) with known contents (us):
       compute 1000-6000; PP broadcast 6000-10000 while stage 1 computes 5000-10000 (peer wait 4000);
       PP broadcast 10000-12000 while stage 1 is inside an NCCL kernel too (transfer 2000, peer in NCCL);
       idle 12000-15000 with a host op (host launch gap 3000); step scope 15000-17000 with one launch call at
       16500-16510 (python gap 1990, launch 10); idle 17000-21000, no host (idle_no_host 4000)."""
    r0 = Rank("dp0_pp0_tp0_dcp0_ep0_rank0.1.pt.trace.json")
    r1 = Rank("dp0_pp1_tp0_dcp0_ep0_rank1.2.pt.trace.json", base_ns=1_790_000_000_000_000_000 + base1_delta_ns,
              shift=shift1)
    r0.eager_forward(500, 900, 600, 1000, 6000)
    r0.kernel(6000, 10000, name=BCAST, stream=31)
    r0.kernel(10000, 12000, name=BCAST, stream=31)
    r0.cpu_op(12000, 15000)
    r0.scope(15000, 17000)
    c1 = r0.kernel(21000, 22000)
    r0.runtime(16500, "cudaLaunchKernel", c1)
    r0.eager_forward(35000, 36000, 35500, 41000, 42000)
    r1.kernel(5000, 10000)
    r1.kernel(10000, 12000, name=BCAST, stream=31)
    for t in (25000, 27000, 29000, 31000):             # matched short broadcasts (clock anchors) in tick 1
        r0.kernel(t, t + 10, name=BCAST, stream=31)
        r1.kernel(t, t + 10, name=BCAST, stream=31)
    return [r0.trace(), r1.trace()]


TICK0_PP = {"compute": 5.0, "copy": 0.0, "nccl_peer_wait": 4.0, "nccl_tp_skew": 0.0, "nccl_transfer": 2.0,
            "host_launch_gap": 3.01, "host_python_gap": 1.99, "idle_no_host": 4.0}


class Intervals(unittest.TestCase):
    def test_merge_sub_inter_clip(self):
        m = T.merge([[5, 7], [1, 3], [2, 4], [7, 9], [10, 10]])
        self.assertEqual(m, [[1, 4], [5, 9]])
        self.assertEqual(T.sub([[0, 10]], [[2, 3], [5, 6], [9, 12]]), [[0, 2], [3, 5], [6, 9]])
        self.assertEqual(T.sub([[0, 2], [4, 6]], [[1, 5]]), [[0, 1], [5, 6]])
        self.assertEqual(T.inter([[0, 5], [8, 10]], [[3, 9]]), [[3, 5], [8, 9]])
        iv = T.Ivs([[0, 2], [4, 6], [8, 10]])
        self.assertEqual(iv.clip(1, 9), [[1, 2], [4, 6], [8, 9]])
        self.assertEqual(iv.clip(2, 4), [])
        self.assertEqual(T.total(T.union([[0, 1]], [[0.5, 2]], [[3, 4]])), 3)


class SyntheticPP(unittest.TestCase):
    def check_tick0(self, traces):
        rep = T.analyze(traces, tp=1)
        self.assertEqual(len(rep["ticks"]), 2)
        t0 = rep["ticks"][0]
        self.assertAlmostEqual(t0["tick_ms"], 20.0, places=3)
        got = t0["by_rank"][0]
        for c, v in TICK0_PP.items():
            self.assertAlmostEqual(got[c], v, places=3, msg=c)
        self.assertAlmostEqual(sum(got[c] for c in T.CATEGORIES), 20.0, places=3)
        self.assertAlmostEqual(got["nccl_transfer_pp"], 2.0, places=3)
        self.assertAlmostEqual(got["nccl_transfer_peer_in_nccl"], 2.0, places=3)
        # stage 1 in tick 0: compute 5000-10000, NCCL 10000-12000 (stage 0 not computing), idle otherwise
        s1 = t0["by_rank"][1]
        self.assertAlmostEqual(s1["compute"], 5.0, places=3)
        self.assertAlmostEqual(s1["nccl_transfer"], 2.0, places=3)
        self.assertAlmostEqual(t0["stage1_idle_ms"], 15.0, places=3)
        p = t0["pipeline"]
        self.assertAlmostEqual(p["both_busy_ms"], 1.0, places=3)           # 5000-6000
        self.assertAlmostEqual(p["stage0_only_ms"], 4.0, places=3)
        self.assertAlmostEqual(p["other_stage_only_ms"], 4.0, places=3)
        self.assertAlmostEqual(p["neither_ms"], 11.0, places=3)
        self.assertEqual(t0["graph_mode"], "EAGER")
        self.assertAlmostEqual(t0["critical_path"]["ideal_overlapped_tick_ms"], 5.0, places=3)
        return rep

    def test_categories_shared_clock(self):
        rep = self.check_tick0(pp2_tp1())
        self.assertTrue(rep["clock"]["method"].startswith("SHARED_CLOCK"))
        self.assertTrue(rep["cpu_activity"][0]["has_cpu_activity"])
        self.assertFalse(rep["cpu_activity"][0]["with_stack"])

    def test_clock_offset_is_measured_and_corrected(self):
        rep = self.check_tick0(pp2_tp1(shift1=3000.0))
        self.assertTrue(rep["clock"]["method"].startswith("CORRECTED_BY_COLLECTIVES"))
        self.assertAlmostEqual(rep["clock"]["measured_offset_us"][1], 3000.0, places=1)

    def test_base_time_difference_is_applied(self):
        # rank 1's base is 2 ms later and its ts 2 ms smaller: the same instants -> no measured residual
        rep = self.check_tick0(pp2_tp1(shift1=-2000.0, base1_delta_ns=2_000_000))
        self.assertTrue(rep["clock"]["method"].startswith("SHARED_CLOCK"))
        self.assertAlmostEqual(rep["clock"]["base_shift_us"][1], 2000.0, places=3)


class SyntheticTP(unittest.TestCase):
    def test_tp_skew_vs_transfer(self):
        r0 = Rank("dp0_pp0_tp0_dcp0_ep0_rank0.1.pt.trace.json")
        r1 = Rank("dp0_pp0_tp1_dcp0_ep1_rank1.2.pt.trace.json")
        r0.eager_forward(100, 200, 150, 1000, 2000)
        r0.kernel(2000, 5000, name=ALLRED)
        r0.eager_forward(9000, 9100, 9050, 10000, 11000)
        r0.eager_forward(19000, 19100, 19050, 20000, 21000)
        r1.kernel(1000, 4900)
        r1.kernel(4900, 5000, name=ALLRED)
        for t in (12000, 13000, 14000):
            r0.kernel(t, t + 20, name=ALLRED)
            r1.kernel(t, t + 20, name=ALLRED)
        rep = T.analyze([r0.trace(), r1.trace()], tp=2)
        g = rep["ticks"][0]["by_rank"][0]
        self.assertAlmostEqual(g["compute"], 1.0, places=3)
        self.assertAlmostEqual(g["nccl_tp_skew"], 2.9, places=3)
        self.assertAlmostEqual(g["nccl_transfer"], 0.1, places=3)
        self.assertAlmostEqual(g["nccl_transfer_tp"], 0.1, places=3)
        self.assertAlmostEqual(g["host_launch_gap"], 0.01, places=3)
        self.assertAlmostEqual(g["host_python_gap"], 0.09, places=3)
        self.assertAlmostEqual(g["idle_no_host"], 4.9, places=3)
        self.assertNotIn("pipeline", rep["ticks"][0])                   # one stage only
        self.assertEqual(rep["stages"], [0])


class GraphModeAndStall(unittest.TestCase):
    def _graph_rank(self):
        r = Rank("dp0_pp0_tp0_dcp0_ep0_rank0.1.pt.trace.json")
        # forward 0: one small graph (3 nodes) + an eager elementwise input-prep kernel; 6 ms in-graph stall
        r.scope(100, 400)
        g = r._c()
        r.runtime(300, "cudaGraphLaunch", g)
        r.kernel(1000, 2000, corr=g)
        r.kernel(8000, 8100, name=ALLRED, corr=g)                      # resumes after a 6 ms in-graph stall
        r.kernel(8100, 9000, corr=g)
        c = r.kernel(900, 950, name="void at::native::vectorized_elementwise_kernel<4>(x)")
        r.runtime(200, "cudaLaunchKernel", c)
        r.kernel(1500, 7990, name=BCAST, stream=31,
                 args={"Collective name": "broadcast", "Process Group Ranks": "[0, 1]", "dtype": "Int",
                       "In msg nelems": 8})
        # forward 1: FULL by the rev 2 rule (one 600-node graph) although an eager attention kernel runs beside it
        # (the lab rule calls this PIECEWISE)
        r.scope(9100, 9400)
        r.graph(9200, 10000, 16000, 600)
        c2 = r.kernel(16000, 16500, name="flash_fwd_kernel")
        r.runtime(9300, "cudaLaunchKernel", c2)
        # forward 2: EAGER
        r.eager_forward(19000, 19100, 19050, 20000, 21000)
        # forward 3: PIECEWISE (two graph launches)
        r.scope(29000, 29300)
        r.graph(29100, 30000, 30500, 2)
        r.graph(29200, 30600, 31000, 2)
        # forward 4: EMPTY (a 0-token step), with a small graph
        r.scope(39000, 39300, name="execute_context_0(0)_generation_0(0)")
        r.graph(39100, 40000, 40100, 1)
        # forward 5: closes the last tick
        r.eager_forward(49000, 49100, 49050, 50000, 50500)
        return r

    def test_graph_mode_rule(self):
        self.assertEqual(T.graph_mode([], 0), "EMPTY")
        self.assertEqual(T.graph_mode([1037], 0), "EMPTY")
        self.assertEqual(T.graph_mode([], 16), "EAGER")
        self.assertEqual(T.graph_mode([1037], 16), "FULL")
        self.assertEqual(T.graph_mode([499], 16), "SMALL_GRAPH")
        self.assertEqual(T.graph_mode([500], 16), "FULL")
        self.assertEqual(T.graph_mode([600, 40], 16), "PIECEWISE")
        self.assertEqual(T.graph_mode([100], 16, full_min=50), "FULL")
        self.assertEqual(T.scope_kind(2, 0, 1), "PIECEWISE")
        self.assertEqual(T.scope_kind(1, 3, 0), "PIECEWISE")          # the lab rule NB1 used

    def test_graph_modes_in_trace_and_lab_rule_only_in_nb1_reference(self):
        p = Rank("dp0_pp1_tp0_dcp0_ep0_rank1.2.pt.trace.json")
        p.kernel(3000, 7990, name=BCAST, stream=31)      # the peer-side partner of the broadcast (end matches)
        rep = T.analyze([self._graph_rank().trace(), p.trace()], tp=1)
        self.assertEqual([t["graph_mode"] for t in rep["ticks"]],
                         ["SMALL_GRAPH", "FULL", "EAGER", "PIECEWISE", "EMPTY"])
        self.assertEqual(rep["graph_modes"][0], {"SMALL_GRAPH": 1, "FULL": 1, "EAGER": 2, "PIECEWISE": 1, "EMPTY": 1})
        lab = rep["nb1_reference"]["graph_verdict_lab_rule"][0]
        self.assertEqual(lab["verdict"], "MIXED")
        self.assertEqual(lab["forward_kinds"], {"FULL": 3, "PIECEWISE": 1, "EAGER": 2})

    def test_in_graph_stall_bucket_and_release(self):
        p = Rank("dp0_pp1_tp0_dcp0_ep0_rank1.2.pt.trace.json")
        p.kernel(2500, 7900)                              # the peer computes while stage 0 is stalled
        p.kernel(7900, 7990, name=BCAST, stream=31)
        rep = T.analyze([self._graph_rank().trace(), p.trace()], tp=1)
        t0 = rep["ticks"][0]                              # [900, 10000)
        self.assertAlmostEqual(t0["stall"]["in_graph_stall_ms_ref_rank"], 6.0, places=3)   # 2000-8000 in the graph
        self.assertEqual(t0["stall"]["stalls_over_1ms"], 1)
        self.assertIn("broadcast Intx8 group [0, 1]", t0["stall"]["released_by"][0])
        self.assertAlmostEqual(t0["stall"]["lag_us"][0], 10.0, places=1)
        g = t0["by_rank"][0]
        self.assertAlmostEqual(g["in_graph_stall"], 6.0, places=3)     # the exclusive bucket, before NCCL
        self.assertAlmostEqual(g["nccl_peer_wait"], 0.0, places=3)     # the broadcast spin lies inside the stall
        self.assertAlmostEqual(g["in_graph_stall_peer_computing"], 5.4, places=3)   # 2500-7900
        self.assertAlmostEqual(g["in_graph_stall_pp_nccl_running"], 5.99, places=3)  # 2000-7990
        self.assertAlmostEqual(g["nccl_transfer"], 0.1, places=3)      # TP 8000-8100 inside the graph is not a stall
        self.assertAlmostEqual(g["nccl_transfer_pp"], 0.0, places=3)
        self.assertAlmostEqual(g["compute"], 1.95, places=3)
        self.assertAlmostEqual(sum(g[c] for c in T.CATEGORIES), 9.1, places=3)


def pp2_steps(n_steps=6, release="false", parity=(4, 4)):
    """Stage 0 / stage 1 with per-step annotations, PP broadcast posts (Seq 100 + step) and their kernels.
    Step i (base = i * 10000): graph kernels [base+1000, base+2000] and [base+6000, base+7000] -> stall
    [base+2000, base+6000]. release='false': the kernel of step j's receive ends at (j+1)*10000 + 5990 (it releases
    step j+1: delta 1, while step i-2's receive ended a step earlier); release='real': it ends at (j+2)*10000 + 5990
    (it releases step j+2 = its own consumer, delta 2 = pp)."""
    r0 = Rank("dp0_pp0_tp0_dcp0_ep0_rank0.1.pt.trace.json")
    r1 = Rank("dp0_pp1_tp0_dcp0_ep0_rank1.2.pt.trace.json")
    off = {"false": 1, "real": 2}[release]
    for i in range(n_steps):
        base = i * 10000
        n = parity[i % 2]
        nm = f"execute_context_0(0)_generation_{n}({4 * n})"
        r0.scope(base, base + 500, name=nm)
        g = r0._c()
        r0.runtime(base + 100, "cudaGraphLaunch", g)
        r0.kernel(base + 1000, base + 2000, corr=g)
        r0.kernel(base + 6000, base + 7000, corr=g)
        r0.post(base + 400, 100 + i)
        e = (i + off) * 10000 + 5990
        r0.bcast_kernel(e - 4500, e, 100 + i)
        r1.scope(base + 50, base + 450, name=nm)
        r1.post(base + 300, 100 + i)
        r1.kernel(base + 2000, base + 5980)                      # stage 1 computes during the stall
        r1.bcast_kernel(e - 10, e, 100 + i)
    return [r0.trace(), r1.trace()]


class ParityAndDependency(unittest.TestCase):
    def test_balanced_parity_and_step_match(self):
        rep = T.analyze(pp2_steps(), tp=1)
        par = rep["parity"]
        self.assertEqual(par["verdict"], "BALANCED_4+4")
        self.assertEqual(par["stages"]["stage0"]["n_hist_by_parity"], {0: {4: 3}, 1: {4: 3}})
        self.assertEqual(par["stages"]["stage1"]["n_hist_by_parity"], {0: {4: 3}, 1: {4: 3}})
        self.assertEqual(par["stages"]["stage0"]["broadcast_sizes_by_parity"][0], {"Intx8": 3})
        self.assertEqual(par["stage_step_match"]["matched"], 6)

    def test_one_parity(self):
        rep = T.analyze(pp2_steps(parity=(8, 0)), tp=1)
        self.assertEqual(rep["parity"]["verdict"], "ONE_PARITY")
        self.assertEqual(rep["parity"]["stages"]["stage0"]["n_hist_by_parity"], {0: {8: 3}, 1: {0: 3}})
        modes = [t["graph_mode"] for t in rep["ticks"]]
        self.assertEqual(modes[1], "EMPTY")

    def test_false_dependency(self):
        dep = T.analyze(pp2_steps(release="false"), tp=1)["dependency"]
        self.assertEqual(dep["step_delta_hist"], {1: 5})            # steps 1-5 released by the previous step
        self.assertEqual(dep["verdicts"], {"FALSE_NO_REAL_INPUT": 1, "FALSE": 4})
        self.assertEqual(dep["verdict"], "FALSE")
        # step i's real input (step i-2's receive) ended at (i-1)*10000 + 5990: 5.01 ms before graph start
        self.assertAlmostEqual(dep["real_input_ready_before_graph_start_ms_median"], 5.01, places=3)
        self.assertAlmostEqual(dep["stall_ms_median"], 4.0, places=3)

    def test_real_dependency(self):
        dep = T.analyze(pp2_steps(release="real"), tp=1)["dependency"]
        self.assertEqual(dep["step_delta_hist"], {2: 4})
        self.assertEqual(dep["verdicts"], {"REAL": 4})

    def test_ideal_tick(self):
        rep = T.analyze(pp2_steps(n_steps=8), tp=1)
        it = rep["summary"]["ideal_tick"]
        # per 10 ms tick: stage 0 computes 2 ms, stage 1 3.98 ms
        self.assertAlmostEqual(it["per_stage_busy_ms"]["stage0"], 2.0, places=3)
        self.assertAlmostEqual(it["per_stage_busy_ms"]["stage1"], 3.98, places=3)
        self.assertAlmostEqual(it["balanced_parity_ideal_ms"], 3.98, places=3)
        self.assertEqual(it["bound_by"], "stage1 work")
        self.assertAlmostEqual(it["chain_bound_ms"], 2.99, places=3)


class NB1Reference(unittest.TestCase):
    def test_microbatch_costs(self):
        r = Rank("dp0_pp0_tp0_dcp0_ep0_rank0.1.pt.trace.json")
        r.scope(0, 50)
        c = r.kernel(100, 200)
        r.runtime(10, "cudaLaunchKernel", c)
        g = r._c()
        r.runtime(20, "cudaGraphLaunch", g)
        r.kernel(300, 400, corr=g)
        r.kernel(350, 450, corr=g)
        r.kernel(600, 700, corr=g, name=ALLRED)
        agg = T.nb1_microbatch_costs(r.trace()["traceEvents"])[16]
        self.assertEqual(agg["microbatches"], 1)
        self.assertAlmostEqual(agg["critical_path_ms"], 0.6, places=6)
        self.assertAlmostEqual(agg["gpu_busy_union_ms"], 0.35, places=6)
        self.assertAlmostEqual(agg["summed_kernel_ms"], 0.4, places=6)
        pct = T.nb1_category_pct(r.trace()["traceEvents"])
        self.assertEqual(pct["nccl"], 25.0)
        self.assertEqual(T.nb1_cat("void marlin_moe_wna16::Marlin<1>(x)"), "moe_marlin")
        self.assertEqual(T.nb1_cat("_qsa_mqa_paged_uniform_kernel"), "qsa")
        self.assertEqual(T.nb1_cat("ncclDevKernel_SendRecv(x)"), "nccl_sendrecv")

    def test_load_from_directory_gz(self):
        with tempfile.TemporaryDirectory() as d:
            for tr in pp2_tp1():
                with gzip.open(Path(d) / (tr["_name"] + ".gz"), "wt", encoding="utf-8") as fh:
                    json.dump({k: v for k, v in tr.items() if k != "_name"}, fh)
            ranks = T.load_traces(d, tp=1)
            self.assertEqual(sorted(ranks), [0, 1])
            self.assertEqual((ranks[1]["pp"], ranks[1]["tp"]), (1, 0))
            rep = T.analyze(ranks, tp=1)
            self.assertAlmostEqual(rep["ticks"][0]["by_rank"][0]["nccl_peer_wait"], 4.0, places=3)
            self.assertIn("compute", T.format_report(rep))
            json.dumps(T.strip_private(rep))


@unittest.skipUnless(len(list(REAL_DIR.glob("*.pt.trace.json*"))) == 4,
                     f"NB1 R2 B1 traces absent ({REAL_DIR}); the regression runs where the 4 rank files exist")
class RealB1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = T.analyze(str(REAL_DIR), tp=2)

    def test_nb1_numbers_reproduced(self):
        nb = self.rep["nb1_reference"]
        for st, want in (("stage0", (50.6, 27.2, 29.5)), ("stage1", (31.6, 23.3, 25.8))):
            got = (nb[st]["critical_path_ms"], nb[st]["gpu_busy_union_ms"], nb[st]["summed_kernel_ms"])
            for g, w in zip(got, want):
                self.assertAlmostEqual(g, w, delta=0.06, msg=st)
            self.assertEqual(nb[st]["verify_tokens"], 16)
        self.assertEqual(nb["by_rank"][0]["category_pct"]["nccl"], 63.5)

    def test_clock_and_cpu(self):
        self.assertTrue(self.rep["clock"]["method"].startswith("SHARED_CLOCK"))
        for v in self.rep["clock"]["measured_offset_us"].values():
            self.assertLess(abs(v), 10.0)
        for r, a in self.rep["cpu_activity"].items():
            self.assertTrue(a["has_cpu_activity"])
            self.assertFalse(a["with_stack"])

    def test_tick_split(self):
        s = self.rep["summary"]
        self.assertGreaterEqual(s["steady_ticks"], 30)
        self.assertEqual(set(s["graph_modes"]), {"FULL"})
        for t in self.rep["ticks"]:
            for st in t["stages"].values():
                self.assertAlmostEqual(sum(st[c] for c in T.CATEGORIES), t["tick_ms"], delta=0.01)
        s0 = s["stage0"]
        self.assertGreater(s0["in_graph_stall"]["mean_ms"], 15.0)        # the stage-0 gap is the in-graph stall
        self.assertGreater(s0["in_graph_stall_peer_computing"]["mean_ms"], 15.0)
        self.assertLess(s0["host_launch_gap"]["mean_ms"] + s0["idle_no_host"]["mean_ms"], 1.0)
        st = self.rep["stalls"]
        self.assertGreaterEqual(st["ticks_with_stall_over_1ms"], 0.9 * st["steady_ticks"])
        self.assertTrue(all("Broadcast" in k for k in st["released_by"]))

    def test_graph_modes_rev2_and_lab_rule(self):
        for r, m in self.rep["graph_modes"].items():
            self.assertEqual(m, {"FULL": 40}, msg=r)
        for r, v in self.rep["nb1_reference"]["graph_verdict_lab_rule"].items():
            self.assertEqual(v["verdict"], "PIECEWISE", msg=r)               # NB1's verdict reproduced

    def test_parity_and_dependency(self):
        par = self.rep["parity"]
        self.assertEqual(par["verdict"], "BALANCED_4+4")
        self.assertEqual(par["stage_step_match"]["matched"], par["stage_step_match"]["compared"])
        dep = self.rep["dependency"]
        self.assertEqual(dep["verdict"], "FALSE")
        self.assertEqual(set(dep["step_delta_hist"]), {1})
        self.assertGreater(dep["real_input_ready_before_graph_start_ms_min"], 0.0)
        it = self.rep["summary"]["ideal_tick"]
        self.assertLess(it["balanced_parity_ideal_ms"], 0.8 * it["observed_tick_ms"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
