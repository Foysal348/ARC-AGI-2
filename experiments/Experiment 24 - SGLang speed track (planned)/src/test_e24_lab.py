"""Tests for e24_lab (stdlib unittest): /opt/homebrew/bin/python3.12 test_e24_lab.py [-v]

- pure functions: label-aware prom parsing, pval aggregations, Metrics for both engines (absent counters), decode /
  prefill / retract lines on REAL EMS log lines, facts on a real log, compare (alignment, bands, A/A rule, swap,
  cross weights, garbage), tpair, compare_tf bins, decode_vs_prefill rules, degenerate, kl, equal_n, nccl_info /
  nccl_env_proof, block_reason, cache_verdict on a temp tree with fake hook env records, hook counters, flag-check
  JSON parsing + an end-to-end flag check in a temp venv with a stub sglang / vllm package, trace summaries.
- end to end against MINI (below): a small self-contained server that speaks the SGLang 0.5.21 / vLLM 0.32 formats
  the lab relies on (it is test scaffolding, not the WP2 fake).
- integration against the WP2 fakes (src/harness/fake_*_server.py): skipped cleanly when the files are absent.
"""
import gzip
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e24_lab as L  # noqa: E402

E24 = HERE.parent
LOGS = E24 / "Base Engine" / "server_logs"
E14 = E24 / "Base Engine" / "src_e14_modules"
FAKE_SG = HERE / "harness" / "fake_sglang_server.py"
FAKE_VL = HERE / "harness" / "fake_vllm_server.py"
PY = sys.executable
LAB = {"model_name": "/dev/shm/m,0 \\\"x\\\"", "engine_type": "unified", "pp_rank": "0", "moe_ep_rank": "0"}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def prom_line(name, value, **labels):
    lab = ",".join(f'{k}="{v}"' for k, v in labels.items())
    return f"{name}{{{lab}}} {value}" if lab else f"{name} {value}"


# ================================================================== pure functions
class TestProm(unittest.TestCase):
    TXT = "\n".join([
        "# HELP sglang:num_running_reqs The number of running requests.",
        "# TYPE sglang:num_running_reqs gauge",
        'sglang:num_running_reqs{model_name="/dev/shm/m,0 \\"x\\"",engine_type="unified",tp_rank="0",pp_rank="0"} 20.0',
        'sglang:realtime_tokens_total{model_name="a,b",tp_rank="0",mode="decode"} 1500.0',
        'sglang:realtime_tokens_total{model_name="a,b",tp_rank="0",mode="prefill_compute"} 8000.0',
        'sglang:realtime_tokens_total{model_name="a,b",tp_rank="0",mode="prefill_cache"} 64.0',
    ] + [f'sglang:cuda_graph_passes_total{{model_name="a,b",tp_rank="{r}",mode="decode_cuda_graph"}} {100.0 + r}'
         for r in range(4)] + [
        'sglang:fwd_occupancy{tp_rank="0"} NaN',
        'sglang:http_requests_total{endpoint="/generate",method="POST"} 3',
        'sglang:weird{a="x\\ny", b="z"} 1.5 1700000000',
        "sglang:no_labels 7",
        'broken{a="1" 2',
        'vllm:generation_tokens_total{engine="0",model_name="q27"} 12',
        'sglang:queue_time_seconds_bucket{le="+Inf",tp_rank="0"} 4'])

    def setUp(self):
        self.s = L.parse_prom(self.TXT)

    def test_labels_quoted_commas_escapes(self):
        run = [x for x in self.s if x[0] == "sglang:num_running_reqs"][0]
        self.assertEqual(run[1]["model_name"], '/dev/shm/m,0 "x"')
        self.assertEqual(run[1]["tp_rank"], "0")
        self.assertEqual(run[2], 20.0)
        w = [x for x in self.s if x[0] == "sglang:weird"][0]
        self.assertEqual(w[1], {"a": "x\ny", "b": "z"})
        self.assertEqual(w[2], 1.5)
        self.assertIn(("sglang:no_labels", {}, 7.0), self.s)
        self.assertEqual(L.pval(self.s, "sglang:queue_time_seconds_bucket", {"le": "+Inf"}), 4.0)

    def test_nan_and_broken_skipped(self):
        self.assertIsNone(L.pval(self.s, "sglang:fwd_occupancy"))
        self.assertFalse(any(x[0] == "broken" for x in self.s))

    def test_per_rank_aggregations(self):
        n = "sglang:cuda_graph_passes_total"
        self.assertEqual(L.pval(self.s, n, {"mode": "decode_cuda_graph"}, "tp0"), 100.0)
        self.assertEqual(L.pval(self.s, n, None, "max"), 103.0)
        self.assertEqual(L.pval(self.s, n, None, "sum"), 406.0)       # the 4x over-count a summing parser makes
        self.assertEqual(L.pval(self.s, n, None, "first"), 100.0)
        self.assertEqual(L.pval(self.s, n, {"tp_rank": ("1", "2")}, "sum"), 203.0)
        self.assertEqual(L.pval(self.s, n, {"mode": __import__("re").compile(r"decode_.*")}, "tp0"), 100.0)
        self.assertIsNone(L.pval(self.s, n, {"mode": "decode_none"}, "tp0"))
        self.assertIsNone(L.pval(self.s, "nope"))
        # no tp_rank label counts as tp0
        self.assertEqual(L.pval(self.s, "sglang:no_labels", None, "tp0"), 7.0)
        self.assertEqual(L.pval(self.s, "vllm:generation_tokens_total", None, "sum"), 12.0)


def sg_snap(dec=1500.0, pf=8000.0, run=20.0, queue=0.0, passes=100.0, none_passes=None, retr=None, retr_gauge=0.0,
            scheduler=True, spec=4.2):
    lines = []
    if scheduler:
        lines += [prom_line("sglang:num_running_reqs", run, tp_rank="0"), prom_line("sglang:num_queue_reqs", queue, tp_rank="0"),
                  prom_line("sglang:full_token_usage", 0.31, tp_rank="0"),
                  prom_line("sglang:spec_accept_length", spec, tp_rank="0"),
                  prom_line("sglang:spec_accept_rate", 0.64, tp_rank="0"),
                  prom_line("sglang:kv_available_tokens", 1000, tp_rank="0"),
                  prom_line("sglang:kv_evictable_tokens", 500, tp_rank="0"),
                  prom_line("sglang:mamba_available_tokens", 3, tp_rank="0"),
                  prom_line("sglang:mamba_evictable_tokens", 0, tp_rank="0")]
        if retr_gauge is not None:
            lines.append(prom_line("sglang:num_retracted_reqs", retr_gauge, tp_rank="0", pid="123"))
    if dec is not None:
        lines.append(prom_line("sglang:realtime_tokens_total", dec, tp_rank="0", mode="decode"))
    if pf is not None:
        lines.append(prom_line("sglang:realtime_tokens_total", pf, tp_rank="0", mode="prefill_compute"))
    for r in range(4):
        if passes is not None:
            lines.append(prom_line("sglang:cuda_graph_passes_total", passes, tp_rank=str(r), mode="decode_cuda_graph"))
        if none_passes is not None:
            lines.append(prom_line("sglang:cuda_graph_passes_total", none_passes, tp_rank=str(r), mode="decode_none"))
        if retr is not None:
            lines.append(prom_line("sglang:num_retracted_requests_total", retr, tp_rank=str(r)))
    return L.parse_prom("\n".join(lines))


class TestMetrics(unittest.TestCase):
    def test_sglang_counters_tp0(self):
        M = L.Metrics("sglang")
        s = sg_snap(passes=100.0, none_passes=5.0)
        self.assertEqual(M.gen(s), 1500.0)
        self.assertEqual(M.prefill(s), 8000.0)
        self.assertIsNone(M.prefill_cache(s))
        self.assertEqual(M.steps(s), 105.0)                 # tp0 only, decode_cuda_graph + decode_none
        self.assertEqual(M.steps(sg_snap(passes=100.0)), 100.0)   # missing decode_none counts 0
        self.assertEqual(M.steps(sg_snap(passes=None)), 0.0)      # both absent, scheduler live -> 0
        self.assertIsNone(M.steps(sg_snap(passes=None, scheduler=False)))
        self.assertEqual(M.running(s), 20.0)
        self.assertEqual(M.waiting(s), 0.0)
        self.assertEqual(M.kv_usage(s), 0.31)
        self.assertEqual(M.spec(s), {"accept_length": 4.2, "accept_rate": 0.64})
        self.assertTrue(M.has_core(s))
        self.assertFalse(M.has_core(sg_snap(scheduler=False)))
        self.assertIsNone(M.gen(sg_snap(dec=None)))
        c = M.census(s)
        self.assertEqual((c["kv_available_tokens"], c["mamba_available_tokens"], c["running"]), (1000, 3, 20.0))

    def test_sglang_retraction_absence_rule(self):
        M = L.Metrics("sglang")
        self.assertEqual(M.preempt(sg_snap(retr=None)), 0.0)                       # live, gauge 0, no log
        self.assertEqual(M.preempt(sg_snap(retr=None, retr_gauge=None)), 0.0)      # gauge absent: no evidence
        self.assertEqual(M.preempt(sg_snap(retr=None), "[2026-10-26 13:42:34 TP0] Decode batch, ..."), 0.0)
        self.assertIsNone(M.preempt(sg_snap(retr=None, retr_gauge=2.0)))           # gauge says retracted
        line = ("[2026-10-26 13:50:00 TP0] KV cache pool is full. Retract requests. #retracted_reqs: 1, "
                "#new_tokens_gained: 4096, #mamba_num_gained: 4, #new_token_ratio: 0.7000 -> 0.8000")
        self.assertIsNone(M.preempt(sg_snap(retr=None), line))                     # log line disagrees
        self.assertIsNone(M.preempt(sg_snap(retr=None, scheduler=False)))          # scheduler series missing
        self.assertEqual(M.preempt(sg_snap(retr=3.0)), 3.0)                        # present: tp0 value, not 4x
        # window delta: absent at start, present at end -> the end value; None when unknown
        self.assertEqual(L._delta(M.preempt(sg_snap()), M.preempt(sg_snap(retr=2.0)), absent0=False), 2.0)
        self.assertIsNone(L._delta(None, 2.0, absent0=False))
        self.assertEqual(L._delta(None, 50.0), 50.0)
        self.assertIsNone(L._delta(10.0, None))

    def test_vllm(self):
        txt = "\n".join([prom_line("vllm:generation_tokens_total", 100, engine="0", model_name="q27"),
                         prom_line("vllm:prompt_tokens_total", 900, engine="0", model_name="q27"),
                         prom_line("vllm:num_requests_running", 3, engine="0"),
                         prom_line("vllm:num_requests_waiting", 1, engine="0"),
                         prom_line("vllm:num_preemptions_total", 0, engine="0"),
                         prom_line("vllm:kv_cache_usage_perc", 0.5, engine="0"),
                         prom_line("vllm:iteration_tokens_total_count", 40, engine="0"),
                         prom_line("vllm:spec_decode_num_accepted_tokens_total", 30, engine="0"),
                         prom_line("vllm:spec_decode_num_drafts_total", 10, engine="0")])
        M, s = L.Metrics("vllm"), L.parse_prom(txt)
        self.assertEqual((M.gen(s), M.prefill(s), M.running(s), M.waiting(s), M.preempt(s), M.steps(s), M.kv_usage(s)),
                         (100, 900, 3, 1, 0, 40, 0.5))
        self.assertTrue(M.has_core(s))
        self.assertEqual(M.spec(s)["acc"], 30)
        self.assertIsNone(M.spec(s)["dtok"])


class TestLogs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = {p.name: p.read_text(encoding="utf-8", errors="replace") for p in sorted(LOGS.glob("*.log"))}

    def test_real_logs_all_lines_parse(self):
        self.assertEqual(len(self.logs), 7)
        for name, t in self.logs.items():
            tp0_dec = [l for l in t.splitlines() if "Decode batch" in l and ("TP0]" in l or " TP" not in l[:30])]
            tp0_pf = [l for l in t.splitlines() if "Prefill batch" in l and ("TP0]" in l or " TP" not in l[:30])]
            d, p = L.decode_lines(t), L.prefill_lines(t)
            self.assertEqual(len(d), len(tp0_dec), name)
            self.assertEqual(len(p), len(tp0_pf), name)
            self.assertGreater(len(d), 40, name)
            self.assertTrue(all(x["gen_tps"] >= 0 and x["running"] >= 0 for x in d), name)

    def test_real_line_values(self):
        t = self.logs["sglang_server_ems_dflash_w6_s20.log"]
        d = L.decode_lines(t)[0]
        self.assertEqual((d["running"], d["tokens"], d["full_usage"], d["mamba_num"], d["mamba_usage"], d["accept_len"],
                          d["accept_rate"], d["graph"], d["gen_tps"], d["queue"]),
                         (1, 4043, 0.01, 3, 0.04, 3.88, 0.57, True, 16.24, 0))
        self.assertTrue(d["after_prefill"])                       # line 334 is a Prefill batch line
        self.assertTrue(L.decode_lines(t)[1]["after_prefill"])    # line 336 too
        self.assertFalse(L.decode_lines(t)[2]["after_prefill"])   # 338 follows the decode line 337
        self.assertIsNotNone(d["t"])
        p = L.prefill_lines(t)[2]                                 # line 334
        self.assertEqual((p["new_seq"], p["new_token"], p["cached_token"], p["running"], p["queue"], p["pending"],
                          p["graph"] if "graph" in p else False, p["in_tps"]), (1, 3892, 0, 0, 0, 0, False, 1384.69))
        tgt = L.decode_lines(self.logs["sglang_server_ems_target_only.log"])
        self.assertTrue(all(x["accept_len"] is None for x in tgt))
        # slicing by char offset
        off = t.index("[2026-10-26 13:42:37 TP0] Decode batch")
        self.assertEqual(L.decode_lines(t, off)[0]["tokens"], 2196)

    def test_synthetic_variants_and_rank_filter(self):
        lines = "\n".join([
            "[2026-10-26 13:42:34.123 TP0] Decode batch [77], #running-req: 2, #token: 500, token usage: 0.10, "
            "cuda graph: True, gen throughput (token/s): 12.50, #queue-req: 3",
            "[2026-10-26 13:42:34 TP1] Decode batch, #running-req: 9, #full token: 1, full token usage: 0.00, mamba num: 1, "
            "mamba usage: 0.00, cuda graph: True, gen throughput (token/s): 1.00, #queue-req: 0",
            "[2026-10-26 13:42:35] Decode batch, #running-req: 1, #full token: 10, full token usage: 0.00, mamba num: 3, "
            "mamba usage: 0.04, accept len: 4.00, accept rate: 0.60, cap len: 5.00, cuda graph: False, "
            "gen throughput (token/s): 3.00, #queue-req: 0",
            "[2026-10-26 13:42:36 TP0] Prefill batch [5], #new-seq: 2, #new-token: 4096, #cached-token: 89984, full token usage: "
            "0.02, mamba usage: 0.15, #running-req: 1, #queue-req: 7, #pending-token: 42270, #replay-token: 12, "
            "cuda graph: False, input throughput (token/s): 1658.12"])
        d = L.decode_lines(lines)
        self.assertEqual(len(d), 2)
        self.assertEqual((d[0]["running"], d[0]["tokens"], d[0]["token_usage"], d[0]["full_usage"], d[0]["queue"]),
                         (2, 500, 0.10, None, 3))
        self.assertAlmostEqual(d[0]["t"] % 1, 0.123, places=3)
        self.assertEqual(d[1]["accept_len"], 4.0)
        p = L.prefill_lines(lines)[0]
        self.assertEqual((p["cached_token"], p["replay"], p["queue"]), (89984, 12, 7))

    def test_retract_lines(self):
        t = "\n".join([
            "[2026-10-26 13:50:00 TP0] KV cache pool is full. Retract requests. #retracted_reqs: 2, #new_tokens_gained: "
            "8192, #mamba_num_gained: 8, #new_token_ratio: 0.7000 -> 0.9500",
            "[2026-10-26 13:50:00 TP1] KV cache pool is full. Retract requests. #retracted_reqs: 2, #new_tokens_gained: "
            "8192, #mamba_num_gained: 8, #new_token_ratio: 0.7000 -> 0.9500",
            "[2026-10-26 13:51:00 TP0] Testing retraction. #retracted_reqs: 1, #new_tokens_gained: 100",
            "[2026-10-26 13:41:26] server_args={'retraction_policy': 'length'}"])
        r = L.retract_lines(t)
        self.assertEqual([x["n"] for x in r], [2, 1])
        self.assertEqual(r[0]["ratio"], [0.7, 0.95])
        self.assertEqual(r[1]["mamba_gained"], None)
        self.assertEqual(L.retract_lines("\n".join(l for l in self.logs.values())), [])

    def test_facts_real_log(self):
        s = L.Server("x", ["true"], {}, 1, tempfile.mkdtemp())
        s.log_path = LOGS / "sglang_server_ems_dflash_w6_s20.log"
        f = s.facts()
        self.assertEqual(f["server_args"]["context_length"], 155648)
        self.assertEqual(f["server_args"]["max_running_requests"], 21)
        self.assertEqual((f["kv_tokens"], f["max_running"], f["chunked_prefill_size"], f["mamba_cache_size"]),
                         (779192, 21, 4096, 84))
        self.assertEqual(f["available_gpu_mem_gb"], 0.89)
        self.assertEqual(f["cuda_graph_bs"], list(range(1, 22)))
        self.assertEqual([g["kind"] for g in f["graphs"]], ["target verify", "draft verify"])
        self.assertEqual(f["graphs"][0]["elapsed"], 12.72)
        self.assertEqual(f["mamba_sizes_gb"], [0.06, 1.49, 2.32])
        self.assertEqual(len(f["kv_pools"]), 2)
        self.assertEqual([x["type"] for x in f["load_weight"]], ["Qwen3_5ForConditionalGeneration", "DFlash2DraftModel"])
        self.assertEqual(f["startup_timings"]["target_verify"], 12.72)
        self.assertEqual(f["nccl_version"], "2.29.7")
        self.assertTrue(f["ready_line"] and f["custom_ar_disabled"] and f["prefill_graph_disabled"])
        self.assertEqual(f["ready_log_sec"], 68.0)
        # L1062 is the allocator's recovered-OOM warning (the server kept serving, EMS PASS): memory pressure, not an
        # error line (review: real platform #1); the pre-load Triton warnings stay excluded too
        self.assertEqual(f["n_error_lines"], 0)
        self.assertEqual(f["oom_lines"], [])
        self.assertEqual(f["allocator_oom_warnings"], 1)
        self.assertFalse(f["cuda_invalid_kernel"])
        self.assertEqual(f["retract_lines"], 0)
        self.assertEqual(f["fp8_scale_lines"], 2)
        t = L._facts_sglang(self.logs["sglang_server_ems_target_only.log"])
        self.assertEqual([g["kind"] for g in t["graphs"]], ["target decode"])
        self.assertEqual(t["cuda_graph_bs"], list(range(1, 18)))
        self.assertEqual(t["n_error_lines"], 0)

    def test_log_errors_signs(self):
        t = "\n".join(["[x TP0] Triton kernel 'k' device-loaded after serving started (free device mem: 0.67 GiB). Pre-load it "
                       "during engine init to avoid CUDA OOM.",
                       "AssertionError: Can not alloc mamba cache", "[x TP0] Scheduler hit an exception: Traceback",
                       "[x] DFLASH sampling unsupported on this build/device; falling back to greedy argmax verification.",
                       "AssertionError: Mamba release hit lock_ref=0 on node 7",
                       "CUDA error: no kernel image is available for execution on the device",
                       "[x] Subprocess scheduler_0 (pid=9) crashed with exit code 1. Triggering SIGQUIT for cleanup...",
                       "[x] SIGTERM received. signum=None frame=None. Draining requests and shutting down...",
                       "[x] Subprocess detokenizer (pid=11378) crashed with exit code -15. Triggering SIGQUIT for cleanup...",
                       "[x] SIGQUIT received. signum=None, frame=None. It usually means one child failed."])
        e = L.log_errors(t)
        self.assertTrue(e["shutdown_seen"])
        self.assertEqual(e["n_error_lines"], 6)                   # the kill noise after our SIGTERM is not counted
        self.assertEqual(e["mamba_alloc_fail"], 1)
        self.assertEqual(e["greedy_fallback"], 1)
        self.assertTrue(e["cuda_invalid_kernel"])
        self.assertEqual(L._fatal(t), "FAIL_MAMBA_ALLOC")


def row(key, toks, margins, top2=None, text="a b c", error=None):
    return {"key": key, "tokens": toks, "margins": margins, "top2": top2 or [[t, -0.1, t + 1, -0.6] for t in toks],
            "text": text, "error": error}


class TestCompare(unittest.TestCase):
    def test_alignment_truncation_exact(self):
        ref = [row("p1", [1, 2, 3, 4], [1, 1, 1, 1]), row("p2", [5, 6], [1, 1]), row("serial_only", [9], [1])]
        cand = [row("p2", [5, 6, 7, 8], [1, 1, 1, 1]), row("p1", [1, 2, 3], [1, 1, 1]), row("c16_only", [1], [1])]
        c = L.compare(ref, cand)
        self.assertEqual(c["status"], "EXACT_TO_N")
        self.assertEqual(c["n_pairs"], 2)
        self.assertEqual({r["key"]: r["n"] for r in c["rows"]}, {"p1": 3, "p2": 2})
        self.assertEqual(c["positions_examined"], 5)
        self.assertEqual(c["missing_keys"], ["c16_only", "serial_only"])
        self.assertEqual(L.compare([], cand)["status"], "INCONCLUSIVE_PAIRS_MISSING")

    def test_bands_and_swap(self):
        mk = lambda mr, mc: L.compare([row("k", [1, 2, 3], [0.1, mr, 0.1], top2=[[1, -.1, 7, -1], [2, -.1, 9, -1], [3, -.1, 4, -1]])],
                                      [row("k", [1, 9, 3], [0.1, mc, 0.1])], t_pair=0.5)
        self.assertEqual(mk(0.125, 0.25)["rows"][0]["class"], "NEAR_TIE")
        c = mk(0.25, 0.25)                                         # 0.5 is not < 0.5
        self.assertEqual(c["rows"][0]["class"], "FLIP")
        self.assertTrue(c["rows"][0]["swap"])                      # cand token 9 is the ref top-2 at k=1
        self.assertEqual(c["swaps"], 1)
        self.assertEqual(c["rows"][0]["first_diff"], 1)
        self.assertEqual(c["positions_examined"], 1)
        self.assertEqual(mk(2.0, 1.99)["rows"][0]["class"], "FLIP")
        self.assertEqual(mk(2.0, 2.0)["rows"][0]["class"], "STATE_FLIP")
        # calibrated T_pair widens NEAR_TIE; a T_pair below the 0.5 floor is clamped
        c = L.compare([row("k", [1, 2], [0.5, 0.5])], [row("k", [1, 3], [0.5, 0.25])], t_pair=0.875)
        self.assertEqual(c["rows"][0]["class"], "NEAR_TIE")
        self.assertEqual(L.compare([row("k", [1, 2], [0.5, 0.25])], [row("k", [1, 3], [0.5, 0.2])], t_pair=0.1)["t_pair"], 0.5)
        # review (decision logic #7): a calibrated T_pair above 4 nat never turns a STATE_FLIP into a NEAR_TIE
        for tp in (0.5, 3.9, 5.0, 9.0):
            self.assertEqual(mk(2.25, 2.25)["rows"][0]["class"] if tp == 0.5 else
                             L.compare([row("k", [1, 2], [0.1, 2.25])], [row("k", [1, 3], [0.1, 2.25])], t_pair=tp)["rows"][0]["class"],
                             "STATE_FLIP", tp)
        self.assertEqual(L.compare([row("k", [1, 2], [0.1, 2.0])], [row("k", [1, 3], [0.1, 1.5])], t_pair=5.0)["rows"][0]["class"],
                         "NEAR_TIE")                               # 3.5 < T_pair 5.0: still a near tie below 4

    def test_aa_rule_same_weights(self):
        ref = [row(f"p{i}", [1, 2, 3], [1, 1, 1]) for i in range(4)]
        cand = [row(f"p{i}", [1, 5, 3], [1, 1, 1]) for i in range(3)] + [row("p3", [1, 2, 3], [1, 1, 1])]
        self.assertEqual(L.compare(ref, cand, aa_high=None)["status"], "BOUNDARY")        # 3 FLIPs, no A/A
        self.assertEqual(L.compare(ref, cand, aa_high=0)["status"], "FAIL_HIGH_MARGIN")   # 3 > 0 + 1
        self.assertEqual(L.compare(ref, cand, aa_high=2)["status"], "BOUNDARY")           # 3 <= 2 + 1
        c = L.compare(ref, cand, aa_high=1)
        self.assertEqual((c["status"], c["high"], c["counts"]["FLIP"]), ("FAIL_HIGH_MARGIN", 3, 3))
        one = [row("p0", [1, 2], [1, 1])], [row("p0", [1, 3], [1, 1])]
        self.assertEqual(L.compare(*one, aa_high=0)["status"], "BOUNDARY")                 # 1 <= 0 + 1
        self.assertEqual(L.compare([row("k", [1, 2], [1, .1])], [row("k", [1, 3], [1, .1])])["status"], "NEAR_TIE")
        self.assertEqual(L.aa_high_of([c, {"high": 5, "status": "INCONCLUSIVE"}, {"high": 1, "status": "BOUNDARY"}]), 3)
        self.assertIsNone(L.aa_high_of([]))

    def test_cross_weights(self):
        ref = [row("a", [1, 2], [1, 1]), row("b", [1, 2], [3, 3])]
        cand = [row("a", [1, 3], [1, 1]), row("b", [1, 2], [3, 3])]
        self.assertEqual(L.compare(ref, cand, kind="cross_weights")["status"], "MEASURED")
        cand[1] = row("b", [1, 7], [3, 3])
        c = L.compare(ref, cand, kind="cross_weights", aa_high=99)
        self.assertEqual((c["status"], c["state_flips"]), ("FAIL_STATE_FLIP", 1))

    def test_inconclusive_and_garbage(self):
        ref = [row("a", [1, 2], [1, None]), row("b", [1, 2], [1, 1], error="GREEDY_TIMEOUT")]
        cand = [row("a", [1, 3], [1, 1]), row("b", [1, 2], [1, 1])]
        c = L.compare(ref, cand)
        self.assertEqual((c["status"], c["counts"]), ("INCONCLUSIVE", {"INCONCLUSIVE": 2}))
        g = L.compare([row("a", [1, 2], [1, 1], text="the grid is 3 by 3")], [row("a", [1, 2], [1, 1], text="!" * 300)])
        self.assertEqual((g["status"], g["fail_reason"], g["garbage"]), ("FAIL_HIGH_MARGIN", "GARBAGE", 1))
        both = L.compare([row("a", [1, 2], [1, 1], text="")], [row("a", [1, 2], [1, 1], text="")])
        self.assertEqual(both["status"], "EXACT_TO_N")             # the reference itself is not clean
        rows_nokey = [{"tokens": [1, 2], "margins": [1, 1], "text": "x y"}]
        self.assertEqual(L.compare(rows_nokey, rows_nokey)["rows"][0]["key"], "#0")


def tfres(pos, status="MEASURED"):
    return {"status": status, "positions": pos}


class TestTeacherForcedAnalysis(unittest.TestCase):
    def test_tpair(self):
        rnd = random.Random(1)
        a, b = [], []
        for k in range(4000):
            m = rnd.choice([0, 0.0625, 0.125, 0.5, 1.0])
            a.append([-0.1, 1, -0.1, 2, -0.1 - m])
            b.append([-0.1, 1, -0.1, 2, -0.1 - m - (0.0625 if k % 50 == 0 else 0.0)])
        self.assertEqual(L.tpair(tfres(a), tfres(b)), 0.5)          # tiny deltas: the 0.5 floor
        b[0][4], b[1][4], b[2][4], b[3][4], b[4][4] = -5, -5, -5, -5, -5
        t = L.tpair([tfres(a)], [tfres(b)])                        # 5 big deltas out of 4000: q99.9 = 4th largest
        self.assertGreater(t, 3.0)
        self.assertEqual(L.tpair(tfres(a), tfres(None, "INCONCLUSIVE_TF_ALIGN")), 0.5)

    def test_compare_tf_bins(self):
        n = 3000
        ref = [[-0.5, 7, -0.5, 8, -1.5] for _ in range(n)]
        cand = [[-0.5 - (0.2 if k >= 2000 else 0.0), 7 if k < 2500 else 8, -0.5, 8, -1.5] for k in range(n)]
        c = L.compare_tf(tfres(ref), tfres(cand), bins=((0, 2000), (2000, 4000)))
        self.assertEqual(c["status"], "MEASURED")
        self.assertEqual(c["n"], n)
        b0, b1 = c["bins"]
        self.assertEqual((b0["n"], b0["mean_abs_dlogprob"], b0["rank1_agreement"], b0["dnll"]), (2000, 0.0, 1.0, 0.0))
        self.assertEqual((b1["n"], b1["max_abs_dlogprob"], b1["rank1_agreement"], b1["dnll"]), (1000, 0.2, 0.5, 0.2))
        ctx = L.compare_tf(tfres(ref), tfres(cand), bins=((90000, 92000), (92000, 99000)), offset=90000)
        self.assertEqual([b["n"] for b in ctx["bins"]], [2000, 1000])
        self.assertEqual(L.compare_tf(tfres(ref), tfres(ref[:-1]))["status"], "INCONCLUSIVE_TF_MISSING")
        multi = L.compare_tf([tfres(ref), tfres(ref)], [tfres(cand), tfres(ref)])
        self.assertEqual(multi["n"], 2 * n)

    def test_decode_vs_prefill(self):
        rnd = random.Random(7)
        n = 3000

        def trace(noise, gross=0):
            lps = [-1.0 - rnd.random() for _ in range(n)]
            tf = [[x + (rnd.random() - 0.5) * noise, 1, -0.5, 2, -1.0] for x in lps]
            for k in range(gross):
                tf[100 + k][0] = lps[100 + k] - 5.0
            return {"lps": lps}, tfres(tf)
        g, t = trace(0.02)
        base = L.decode_vs_prefill([g], [t], bins=((0, 1000), (1000, 3000)))
        self.assertEqual(base["status"], "MEASURED")
        self.assertEqual(base["bins"][0]["n"], 1000)
        self.assertIsNotNone(base["bins"][0]["se"])
        self.assertAlmostEqual(base["bins"][0]["nll_decode"], 1.5, delta=0.1)
        g2, t2 = trace(0.02, gross=3)
        self.assertEqual(L.decode_vs_prefill(g2, t2)["status"], "STATE_DRIFT")          # gross: 3 positions >= 4 nat
        g3, t3 = trace(0.02, gross=2)
        self.assertEqual(L.decode_vs_prefill(g3, t3)["status"], "MEASURED")
        g4, t4 = trace(0.3)                                       # mean |d| ~0.075 vs base ~0.005
        r = L.decode_vs_prefill(g4, t4, base=base, bins=((0, 1000), (1000, 3000)))
        self.assertEqual(r["status"], "STATE_DRIFT")
        self.assertTrue(any("excess" in x for x in r["reasons"]))
        g5, t5 = trace(0.025)                                     # within 0.02 nat of the base: no drift
        self.assertEqual(L.decode_vs_prefill(g5, t5, base=base, bins=((0, 1000), (1000, 3000)))["status"], "MEASURED")
        dirty = json.loads(json.dumps(base))
        dirty["bins"][0]["ge4"] = 1                               # the base bin is not clean: no excess test there
        r = L.decode_vs_prefill(g4, t4, base=dirty, bins=((0, 1000), (1000, 3000)))
        self.assertTrue(all("0-1000" not in x for x in r["reasons"]))
        bad = L.decode_vs_prefill([g, g], [t, tfres(None, "INCONCLUSIVE_TF_ALIGN")])
        self.assertEqual((bad["status"], bad["traces_bad"]), ("INCONCLUSIVE_TRACES_MISSING", 1))
        self.assertEqual(L.decode_vs_prefill({"lps": [1.0]}, tfres([[1.0], [2.0]]))["status"], "INCONCLUSIVE_TRACES_MISSING")


class TestMisc(unittest.TestCase):
    def test_degenerate(self):
        self.assertEqual(L.degenerate(""), "EMPTY")
        self.assertEqual(L.degenerate("x" + "=" * 201), "REPEATED_CHAR")
        self.assertIsNone(L.degenerate("=" * 150 + " grid row"))
        self.assertEqual(L.degenerate("!" * 50 + "ab"), "BANG_GARBAGE")
        self.assertEqual(L.degenerate("the " * 60), "LOW_VARIETY")
        self.assertIsNone(L.degenerate("The transformation rule maps each 3 to a 4 in the output grid."))
        # review (real platform #3): prose followed by space-separated grids is content, not LOW_VARIETY
        grid = "\n".join(" ".join(str((r * c) % 10) for c in range(15)) for r in range(15))
        self.assertIsNone(L.degenerate("So the output keeps every row and mirrors the colored cells.\n" + grid + "\n\n" + grid))
        self.assertIsNone(L.degenerate(" ".join(["0"] * 900)))               # a grid of zeros is a legal grid
        self.assertEqual(L.degenerate("the cell " * 40 + grid), "LOW_VARIETY")   # repeated words still are garbage

    def test_kl(self):
        p = [math.log(0.5), math.log(0.25), math.log(0.25)]
        self.assertAlmostEqual(L.kl(p, p), 0.0, places=12)
        q = [math.log(1 / 3)] * 3
        exp = 0.5 * math.log(0.5 / (1 / 3)) + 2 * 0.25 * math.log(0.25 / (1 / 3))
        self.assertAlmostEqual(L.kl(p, q), exp, places=12)
        self.assertAlmostEqual(L.kl([x + 3.0 for x in p], [x - 1.0 for x in q]), exp, places=12)   # renormalised
        self.assertIsNone(L.kl(p, q[:2]))
        self.assertIsNone(L.kl(p, [float("nan")] * 3))

    def test_equal_n(self):
        self.assertEqual(L.equal_n(779192, 40000, 5400), 15)
        self.assertEqual(L.equal_n(779192, 8000, 5400, reserve=2900, margin_frac=0.05), 45)
        self.assertIsNone(L.equal_n(None, 8000, 100))
        self.assertEqual(L.equal_n(1000, 8000, 0), 0)

    def test_block_reason(self):
        cg = {"kv_available_tokens": 50000, "kv_evictable_tokens": 10000, "mamba_available_tokens": 8,
              "mamba_evictable_tokens": 0}
        self.assertEqual(L.block_reason(cg, 0, 7, 90000), "ALL_ADMITTED")
        self.assertEqual(L.block_reason(cg, 3, 7, 90000), "KV_BOUND")
        cg2 = dict(cg, kv_available_tokens=85000)
        self.assertEqual(L.block_reason(cg2, 3, 7, 90000), "KV_BOUND_RESERVED")          # 95k < 90k + 8 x 2867
        cg3 = dict(cg, kv_available_tokens=500000, mamba_available_tokens=0)
        self.assertEqual(L.block_reason(cg3, 3, 7, 90000), "MAMBA_BOUND")
        self.assertEqual(L.block_reason(dict(cg3, mamba_available_tokens=9), 3, 21, 90000, max_running=21), "MAX_RUNNING")
        self.assertIsNone(L.block_reason(dict(cg3, mamba_available_tokens=9), 3, 7, 90000))
        self.assertIsNone(L.block_reason(cg, None, 7, 90000))

    def test_nccl_info_and_proof(self):
        t = "\n".join([
            "[2026-10-26 13:41:54 TP0] sglang is using nccl==2.29.7",
            "58d9:395:395 [0] NCCL INFO NCCL version 2.29.7+cuda13.2",
            "58d9:395:395 [0] NCCL INFO NCCL_MIN_CTAS set by environment to 4.",
            "58d9:395:395 [0] NCCL INFO NCCL_MAX_CTAS set by environment to 4.",
            "58d9:395:395 [0] NCCL INFO NCCL_ALGO set by environment to allreduce:ring",
            "58d9:395:395 [0] NCCL INFO Channel 00/04 : 0 1 2 3", "58d9:395:395 [0] NCCL INFO Channel 03/04 : 0 1 2 3",
            "58d9:396:396 [1] NCCL INFO Channel 00 : 1[1] -> 2[2] via SHM/direct/direct",
            "58d9:396:396 [1] NCCL INFO Channel 00/0 : 1[1] -> 2[2] via P2P/IPC",
            "58d9:395:395 [0] NCCL INFO 4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels",
            "58d9:395:395 [0] NCCL WARN Cuda failure 'out of memory'"])
        n = L.nccl_info(t)
        self.assertEqual((n["version"], n["sglang_nccl"], n["channels"], n["coll_channels"]), ("2.29.7+cuda13.2", "2.29.7", 4, [4]))
        self.assertEqual(n["env_applied"], {"NCCL_MIN_CTAS": "4", "NCCL_MAX_CTAS": "4", "NCCL_ALGO": "allreduce:ring"})
        self.assertEqual(n["transport"], {"SHM/direct/direct": 1, "P2P/IPC": 1})
        self.assertEqual(len(n["warnings"]), 1)
        env = {"NCCL_MIN_CTAS": "4", "NCCL_MAX_CTAS": "4", "NCCL_ALGO": "allreduce:ring", "NCCL_DEBUG": "INFO"}
        tr = {"nccl": {"allreduce_algo": ["RING"], "allreduce_proto": ["LL"], "allreduce_grid_x": [4]}}
        self.assertEqual(L.nccl_env_proof(env, n, tr)["status"], "APPLIED")
        tr_bad = {"nccl": {"allreduce_algo": ["TREE"], "allreduce_proto": ["LL"], "allreduce_grid_x": [4]}}
        p = L.nccl_env_proof(env, n, tr_bad)
        self.assertEqual(p["status"], "INVALID_ENV_NOT_APPLIED")
        self.assertEqual([c["var"] for c in p["checks"] if not c["ok"]], ["NCCL_ALGO"])
        tr_grid = {"nccl": {"allreduce_algo": ["RING"], "allreduce_proto": ["LL"], "allreduce_grid_x": [2, 4]}}
        self.assertEqual(L.nccl_env_proof(env, n, tr_grid)["status"], "INVALID_ENV_NOT_APPLIED")   # some launches at 2 CTAs
        self.assertEqual(L.nccl_env_proof({"NCCL_PROTO": "LL"}, n)["status"], "INVALID_ENV_NOT_APPLIED")   # never read
        self.assertEqual(L.nccl_env_proof(env, n)["status"], "INCONCLUSIVE_NO_TRACE_EVIDENCE")          # no trace
        generic = {"nccl": {"allreduce_algo": [], "allreduce_proto": [], "allreduce_grid_x": [4]}}  # names without algo
        self.assertEqual(L.nccl_env_proof(env, n, generic)["status"], "INCONCLUSIVE_NO_TRACE_EVIDENCE")
        self.assertEqual(L.nccl_env_proof({"NCCL_NTHREADS": "512"}, L.nccl_info(
            "h:1:1 [0] NCCL INFO NCCL_NTHREADS set by environment to 512."))["status"], "APPLIED")   # log-only knob
        self.assertEqual(L.nccl_env_proof(env, L.nccl_info(""))["status"], "INCONCLUSIVE_NO_NCCL_LINES")
        n2 = L.nccl_info("h:1:1 [0] NCCL INFO Channel 00/02 : 0 1 2 3\nh:1:1 [0] NCCL INFO Channel 01/02 : 0 1 2 3")
        tr2 = {"nccl": {"allreduce_grid_x": [2]}}
        self.assertEqual(L.nccl_env_proof({"NCCL_MIN_CTAS": "2"}, n2, tr2)["status"], "APPLIED")   # channel count proves CTAs
        # review (real platform #2): NCCL >= 2.19 names the kernel after its specialised variant (E14A: the 11.9 MB
        # all-gather is ncclDevKernel_AllGather_RING_LL), so NCCL_PROTO is proven by its log line alone
        n3 = L.nccl_info("h:1:1 [0] NCCL INFO NCCL_PROTO set by environment to Simple")
        self.assertEqual(L.nccl_env_proof({"NCCL_PROTO": "Simple"}, n3, tr)["status"], "APPLIED")       # trace says LL
        self.assertEqual(L.nccl_env_proof({"NCCL_PROTO": "Simple"}, n3)["status"], "APPLIED")           # no trace needed
        self.assertEqual(L.nccl_env_proof({"NCCL_PROTO": "LL"}, n3)["status"], "INVALID_ENV_NOT_APPLIED")  # log says Simple

    def test_allocator_warning_is_not_an_oom(self):
        """Review (real platform #1): the EMS BASE S20 log's recovered allocator warning (line 1062) is not FAIL_OOM;
        a torch.OutOfMemoryError still is."""
        warn = ("[rank0]:[W1026 13:49:12.417423849 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 "
                "while trying to allocate 209715200 bytes (free: 154533888, total: 23659151360).")
        ok = "[2026-10-26 13:49:13 TP0] Prefill batch, #new-seq: 1, #new-token: 3892, #cached-token: 0"
        self.assertEqual(L.log_errors(warn + "\n" + ok)["oom_lines"], [])
        self.assertIsNone(L._fatal(warn + "\n" + ok))
        real = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 22.03 GiB"
        self.assertEqual(L._fatal(warn + "\n" + real), "FAIL_OOM")
        ems = LOGS / "sglang_server_ems_dflash_w6_s20.log"
        if ems.is_file():                                          # the real log, from the flush before the warning
            t = ems.read_text(encoding="utf-8", errors="replace")
            self.assertIn("CUDACachingAllocator.cpp:3933", t)
            self.assertIsNone(L._fatal(t[t.index("Cache flushed successfully"):]))

    def test_flag_helpers(self):
        a = ["/tmp/sglang-venv/bin/python", "-m", "sglang.launch_server", "--model-path", "/m", "--tp-size", "4"]
        self.assertEqual(L._flags_only("sglang", a), a[3:])
        self.assertEqual(L._flags_only("vllm", ["/v/bin/vllm", "serve", "/fp8", "--seed", "1"]), ["/fp8", "--seed", "1"])
        out = "warning: x\nE24_FLAG_CHECK {\"ran\": true, \"ok\": false, \"unknown\": [\"--x\"]}\n{not json\n"
        self.assertEqual(L._parse_flag_out(out)["unknown"], ["--x"])
        self.assertEqual(L._parse_flag_out('{"ran": true, "ok": true}')["ok"], True)
        self.assertIsNone(L._parse_flag_out("nothing here"))
        help_txt = ("  --max-num-seqs MAX_NUM_SEQS, -n\n  --enable-prefix-caching, --no-enable-prefix-caching\n"
                    "  --cuda-graph-max-bs-decode N  (see also --cuda-graph-backend-prefill=disabled)\n")
        self.assertEqual(L._vllm_help_missing(help_txt, ["--max-num-seqs", "--max-num-seq", "--no-enable-prefix-caching",
                                                         "--enable-prefix-caching", "--cuda-graph-max-bs",
                                                         "--cuda-graph-backend-prefill"]),
                         ["--max-num-seq", "--cuda-graph-max-bs"])
        self.assertEqual(L.prompt_key([1, 2, 3]), L.prompt_key((1, 2, 3)))
        self.assertNotEqual(L.prompt_key([1, 2, 3]), L.prompt_key([1, 2, 4]))
        self.assertEqual(L._q([1, 2, 3, 4], 0.5), 2)
        self.assertEqual(L._q(list(range(1, 1001)), 0.999), 999)
        self.assertIsNone(L._q([], 0.5))

    def test_kill_orphans_guard(self):
        L.LIVE.add("sentinel")
        try:
            self.assertEqual(L.kill_orphans(), [])
        finally:
            L.LIVE.discard("sentinel")


# ================================================================== caches and hook records
def write_rec(d, pid, ppid, env, role_tag=None, **extra):
    """An env record in the e24_hook rev 2 layout (env + cache dicts, flat copies, role dict, role_tag)."""
    rec = {"schema": "e24_hook_env/2", "pid": pid, "ppid": ppid, "t": time.time(), "source": "import", "argv": ["-m"],
           "role": {"pid": pid, "ppid": ppid, "kind": None, "mp_name": "MainProcess", "proctitle": None, "tp_rank": None},
           "role_tag": role_tag, "env": dict(env), "cache": {k: env.get(k) for k in L.CACHE_VARS}, "env_first": None,
           "env_changes": [], "resolved": None}
    rec.update(extra)
    rec.update({k: v for k, v in env.items() if k not in rec})
    (Path(d) / f"env_{pid}.json").write_text(json.dumps(rec))


class TestCache(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="e24lab_cache_"))
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.home)

    def tearDown(self):
        if self.old_home is not None:
            os.environ["HOME"] = self.old_home
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _arm(self, engine, name="s1_c", tagged=True):
        root = self.tmp / "cache" / name
        env = L.cache_env(engine, root)
        hook = self.tmp / f"hook_{name}"
        hook.mkdir()
        tags = (lambda t: t) if tagged else (lambda t: "other" if t != "tracker" else t)
        if engine == "sglang":
            write_rec(hook, 100, 1, env, tags("main"))
            for i in range(4):
                write_rec(hook, 101 + i, 100, env, tags("scheduler"), resolved={"triton": env["TRITON_CACHE_DIR"],
                                                                                "where": "Scheduler.__init__"})
            write_rec(hook, 105, 100, env, tags("detokenizer"))
            write_rec(hook, 106, 100, env, "tracker")
            write_rec(hook, 107, 101, env, "inductor_compile")
        else:
            write_rec(hook, 200, 1, env, tags("main"))
            write_rec(hook, 201, 200, env, tags("engine_core"))
            for i in range(4):
                write_rec(hook, 202 + i, 201, env, tags("worker"), source="fork")
        (Path(env["TRITON_CACHE_DIR"]) / "k1").mkdir(parents=True, exist_ok=True)
        (Path(env["TRITON_CACHE_DIR"]) / "k1" / "kernel.cubin").write_bytes(b"x" * 10)
        if engine == "vllm":
            p = Path(env["VLLM_CACHE_ROOT"]) / "torch_compile_cache" / "abc"
            p.mkdir(parents=True)
            (p / "graph.py").write_text("x")
        return root, env, hook

    def test_cache_env_and_defaults(self):
        env = L.cache_env("vllm", self.tmp / "r")
        self.assertTrue(all(Path(env[k]).is_dir() for k in L.CACHE_VARS + ("VLLM_CACHE_ROOT",)))
        self.assertEqual(env["CUDA_CACHE_DISABLE"], "0")
        self.assertNotIn("VLLM_CACHE_ROOT", L.cache_env("sglang", self.tmp / "r2"))
        d = L.default_cache_dirs()
        self.assertIn(str(self.home / ".cache" / "sglang"), d)
        self.assertIn(str(self.home / ".nv" / "ComputeCache"), d)
        st = L.dir_stats([str(self.tmp / "missing"), env["TRITON_CACHE_DIR"]])
        self.assertEqual(st[str(self.tmp / "missing")], {"files": 0, "bytes": 0, "newest": None})

    def test_isolated_sglang_and_vllm(self):
        for eng in ("sglang", "vllm"):
            for tagged in (True, False):                           # role_tag, else the pid tree (never argv)
                before = L.dir_stats(L.default_cache_dirs())
                root, env, hook = self._arm(eng, name=f"s1_{eng}_{tagged}", tagged=tagged)
                v = L.cache_verdict(eng, root, hook, before, L.dir_stats(L.default_cache_dirs()))
                self.assertEqual(v["status"], "ISOLATED", (eng, tagged, v["reasons"]))
                self.assertEqual((v["roles"]["by_tags"], v["roles"]["by_tree"]), (tagged, True), (eng, v["roles"]))

    def test_not_logged_and_missing_roles(self):
        root, env, hook = self._arm("sglang")
        (hook / "env_103.json").unlink()                           # one scheduler record missing
        (hook / "env_104.json").unlink()
        v = L.cache_verdict("sglang", root, hook, {}, {})
        self.assertEqual(v["status"], "INVALID_NOT_LOGGED")
        v = L.cache_verdict("sglang", root, self.tmp / "nohook", {}, {})
        self.assertEqual((v["status"], v["records"]), ("INVALID_NOT_LOGGED", 0))

    def test_role_tags_without_tree(self):
        root = self.tmp / "cache" / "tags"
        env = L.cache_env("sglang", root)
        hook = self.tmp / "hook_tags"
        hook.mkdir()
        for i, tag in enumerate(["main", "scheduler", "scheduler", "scheduler", "scheduler", "detokenizer"]):
            write_rec(hook, 300 + i, 999 + i, env, tag)                    # unrelated ppids: no tree, tags decide
        (Path(env["TRITON_CACHE_DIR"]) / "f").write_text("x")
        v = L.cache_verdict("sglang", root, hook, {}, {})
        self.assertEqual((v["roles"]["by_tags"], v["roles"]["by_tree"], v["status"]), (True, False, "ISOLATED"), v["reasons"])
        (hook / "env_304.json").unlink()
        self.assertEqual(L.cache_verdict("sglang", root, hook, {}, {})["roles"]["missing"], ["schedulerx4"])

    def test_path_mismatch_prefix_trap(self):
        root, env, hook = self._arm("sglang", name="s1_c")
        other = L.cache_env("sglang", self.tmp / "cache" / "s1_c10")        # "s1_c10" starts with "s1_c" as a string
        good = json.loads((hook / "env_102.json").read_text())
        st = lambda rec: (hook / "env_102.json").write_text(json.dumps(rec)) and L.cache_verdict("sglang", root, hook, {}, {})
        for where in ("env", "cache", None):                           # any one copy outside the root is a mismatch
            rec = json.loads(json.dumps(good))
            (rec[where] if where else rec)["TRITON_CACHE_DIR"] = other["TRITON_CACHE_DIR"]
            v = st(rec)
            self.assertEqual((v["status"], v["path_mismatch"][0]["var"]), ("INVALID_PATH_MISMATCH", "TRITON_CACHE_DIR"), where)
        rec = json.loads(json.dumps(good))
        for d in (rec, rec["env"], rec["cache"]):
            d.pop("SGLANG_JIT_CACHE_DIR")                                  # unset in a process -> default location
        self.assertEqual(st(rec)["status"], "INVALID_PATH_MISMATCH")
        for extra in ({"resolved": {"inductor": "/tmp/torchinductor_root", "where": "Scheduler.__init__"}},
                      {"env_first": {"TRITON_CACHE_DIR": "/root/.triton/cache"}},
                      {"env_changes": [{"t": 1, "var": "TORCHINDUCTOR_CACHE_DIR", "old": env["TORCHINDUCTOR_CACHE_DIR"],
                                        "new": "/tmp/torchinductor_root"}]}):
            self.assertEqual(st(dict(good, **extra))["status"], "INVALID_PATH_MISMATCH", extra)
        ok_change = [{"t": 1, "var": "TORCHINDUCTOR_CACHE_DIR", "old": None, "new": env["TORCHINDUCTOR_CACHE_DIR"]}]
        self.assertEqual(st(dict(good, env_changes=ok_change))["status"], "ISOLATED")

    def test_default_dir_written(self):
        before = L.dir_stats(L.default_cache_dirs())
        root, env, hook = self._arm("sglang")
        leak = self.home / ".cache" / "sglang" / "jit"
        leak.mkdir(parents=True)
        (leak / "k.so").write_bytes(b"1")
        v = L.cache_verdict("sglang", root, hook, before, L.dir_stats(L.default_cache_dirs()))
        self.assertEqual(v["status"], "INVALID_DEFAULT_DIR_WRITTEN")
        self.assertEqual(v["default_dirs_written"], [str(self.home / ".cache" / "sglang")])
        # a directory that appears only after the arm (glob /tmp/torchinductor_*) counts as written
        self.assertEqual(L.cache_verdict("sglang", root, hook, {}, {"/tmp/torchinductor_x": {"files": 2, "newest": 1.0}})
                         ["status"], "INVALID_DEFAULT_DIR_WRITTEN")

    def test_empty_cache_and_dflash_inductor(self):
        root, env, hook = self._arm("sglang")
        v = L.cache_verdict("sglang", root, hook, {}, {}, dflash=True)
        self.assertEqual(v["status"], "INVALID_EMPTY_CACHE")                # Inductor empty on a DFlash arm
        (Path(env["TORCHINDUCTOR_CACHE_DIR"]) / "fx").mkdir()
        (Path(env["TORCHINDUCTOR_CACHE_DIR"]) / "fx" / "a").write_text("x")
        self.assertEqual(L.cache_verdict("sglang", root, hook, {}, {}, dflash=True)["status"], "ISOLATED")
        shutil.rmtree(env["TRITON_CACHE_DIR"])
        self.assertEqual(L.cache_verdict("sglang", root, hook, {}, {})["status"], "INVALID_EMPTY_CACHE")

    def test_vllm_rewritten_dirs_and_missing_compile_cache(self):
        root, env, hook = self._arm("vllm", name="s1_a")
        rw = Path(env["VLLM_CACHE_ROOT"]) / "torch_compile_cache" / "abc" / "inductor_cache"
        rw.mkdir(parents=True)
        for f in hook.glob("env_*.json"):
            rec = json.loads(f.read_text())
            old = rec["TORCHINDUCTOR_CACHE_DIR"]
            for d in (rec, rec["env"], rec["cache"]):                       # vLLM rewrote it: still under the root
                d["TORCHINDUCTOR_CACHE_DIR"] = str(rw)
            rec["env_changes"] = [{"t": 1, "var": "TORCHINDUCTOR_CACHE_DIR", "old": old, "new": str(rw)}]
            f.write_text(json.dumps(rec))
        self.assertEqual(L.cache_verdict("vllm", root, hook, {}, {})["status"], "ISOLATED")
        shutil.rmtree(Path(env["VLLM_CACHE_ROOT"]) / "torch_compile_cache")
        self.assertEqual(L.cache_verdict("vllm", root, hook, {}, {})["status"], "INVALID_EMPTY_CACHE")
        (hook / "env_205.json").unlink()
        (hook / "env_204.json").unlink()
        self.assertEqual(L.cache_verdict("vllm", root, hook, {}, {})["status"], "INVALID_NOT_LOGGED")

    def test_hook_records_and_counters(self):
        d = self.tmp / "hk"
        d.mkdir()
        (d / "env_1.json").write_text('{"pid": 1}\n')
        (d / "env_2.json").write_text('{"pid": 2, "v": 1}\n{"pid": 2, "v": 2}\n')      # appended snapshots: last wins
        (d / "env_3.json").write_text("garbage")
        self.assertEqual([r.get("v") for r in L.hook_records(d)], [None, 2])
        def cnt_file(pid, rank, c, ok=True, tag="scheduler", nested=False):
            rec = {"schema": "e24_hook_counters/2", "pid": pid, "counters": c, "patched": ["a", "b"],
                   "self_test": {"expected": ["a", "b", "c"], "patched": ["a", "b"], "missing": [] if ok else ["c"], "ok": ok}}
            if nested:                                                     # pre-2b layout: rank / kind only in role
                rec["role"] = {"kind": "sglang_scheduler" if tag == "scheduler" else tag, "tp_rank": rank}
            else:
                rec.update(role_tag=tag, tp_rank=rank, role={"kind": "x", "tp_rank": None})
            (d / f"counters_{pid}.json").write_text(json.dumps(rec))
        for r in range(4):
            cnt_file(20 + r, r, {"optional_alloc_failed": 2, "draft_nan_injections": 1 + (r == 3)}, nested=(r == 2))
        cnt_file(30, None, {}, tag="main")                                # a launcher file (pre-2b) is not a rank
        cnt_file(31, None, {}, tag="detokenizer")
        h = L.hook_counters(d, tp=4)
        self.assertEqual((h["files"], h["rank_files"], h["ranks"], h["n_ranks"], h["status"]), (6, 4, [0, 1, 2, 3], 4, "OK"))
        self.assertEqual((h["totals"]["optional_alloc_failed"], h["min"]["draft_nan_injections"],
                          h["max"]["draft_nan_injections"]), (8, 1, 2))
        self.assertEqual(h["equal"], {"draft_nan_injections": False, "optional_alloc_failed": True})   # -> INJECTION_DESYNC
        self.assertTrue(h["self_test_ok"])
        cnt_file(22, 2, {"optional_alloc_failed": 2}, ok=False)            # rank 2 self-test fails
        h = L.hook_counters(d, tp=4)
        self.assertEqual((h["self_test_ok"], h["self_test_missing"]), (False, ["c"]))
        (d / "counters_23.json").unlink()                                  # a missing rank fails closed
        h = L.hook_counters(d, tp=4)
        self.assertEqual((h["status"], h["n_ranks"], h["self_test_ok"]), ("INVALID_RANKS", 3, False))
        cnt_file(23, 1, {})                                                # a duplicate rank fails closed too
        self.assertEqual(L.hook_counters(d, tp=4)["status"], "INVALID_RANKS")
        e = self.tmp / "hk2"
        e.mkdir()
        (e / "counters_1.json").write_text(json.dumps({"pid": 1, "role_tag": "main", "tp_rank": None, "counters": {}}))
        self.assertEqual((L.hook_counters(e)["self_test_ok"], L.hook_counters(e)["status"]), (False, "INVALID_RANKS"))
        self.assertIsNone(L.hook_counters(self.tmp / "none")["self_test_ok"])
        self.assertEqual(L.hook_counters(self.tmp / "none")["files"], 0)



HOOK = HERE / "e24_hook.py"
SG_TREE = r"""
import multiprocessing as mp, os, sys, time
def child(i):
    time.sleep(0.8)                       # the hook thread re-snapshots (mp name set) and flushes
def main():
    ctx = mp.get_context("spawn")
    names = ["sglang::scheduler_TP%d" % i for i in range(4)] + ["sglang::detokenizer"]
    ps = [ctx.Process(target=child, args=(i,), name=n) for i, n in enumerate(names)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    open(os.path.join(os.environ["TRITON_CACHE_DIR"], "k.cubin"), "w").write("x")     # stands for the JIT output
    time.sleep(0.3)
if __name__ == "__main__":
    main()
"""
VL_TREE = r"""
import multiprocessing as mp, os, time
def worker(i):
    time.sleep(0.8)
def engine_core():
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=worker, args=(i,), name="VllmWorker-%d" % i) for i in range(4)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
def main():
    ec = mp.get_context("fork").Process(target=engine_core, name="EngineCore_0")
    ec.start()
    ec.join()
    root = os.environ["VLLM_CACHE_ROOT"]
    os.makedirs(os.path.join(root, "torch_compile_cache", "x"), exist_ok=True)
    open(os.path.join(root, "torch_compile_cache", "x", "g.py"), "w").write("x")
    open(os.path.join(os.environ["TRITON_CACHE_DIR"], "k.cubin"), "w").write("x")
    time.sleep(0.3)
if __name__ == "__main__":
    main()
"""


@unittest.skipUnless(HOOK.exists(), "src/e24_hook.py not present")
class TestRealHookRecords(unittest.TestCase):
    """cache_verdict / hook_records on records written by the REAL e24_hook (WP3) as sitecustomize."""

    def run_tree(self, engine, module, script):
        tmp = Path(tempfile.mkdtemp(prefix="e24lab_hook_"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "site").mkdir()
        shutil.copy(HOOK, tmp / "site" / "sitecustomize.py")
        mod = tmp / "pkg" / Path(*module.split("."))
        mod.parent.mkdir(parents=True)
        for d in [tmp / "pkg" / Path(*module.split(".")[:i]) for i in range(1, len(module.split(".")))]:
            (d / "__init__.py").write_text("")
        mod.with_suffix(".py").write_text(script)
        root = tmp / "cache" / ("s1_c" if engine == "sglang" else "s1_a")
        env = dict(os.environ, **L.cache_env(engine, root), E24_HOOK_DIR=str(tmp / "hook"), E24_HOOK_FLUSH_SEC="0.05",
                   E24_HOOK_CHAIN="0", PYTHONPATH=f"{tmp / 'site'}{os.pathsep}{tmp / 'pkg'}", HOME=str(tmp / "home"),
                   PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONSAFEPATH", None)
        before = L.dir_stats(L.default_cache_dirs())
        r = subprocess.run([PY, "-m", module] + (["serve", "/m"] if engine == "vllm" else []), env=env, cwd=str(tmp),
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        time.sleep(0.3)
        recs = L.hook_records(tmp / "hook")
        return L.cache_verdict(engine, root, tmp / "hook", before, L.dir_stats(L.default_cache_dirs())), recs, root

    def test_sglang_spawn_tree(self):
        v, recs, root = self.run_tree("sglang", "sglang.launch_server", SG_TREE)
        tags = sorted(str(x.get("role_tag")) for x in recs)
        self.assertEqual(v["status"], "ISOLATED", (v["reasons"], tags))
        self.assertTrue(v["roles"]["by_tags"] or v["roles"]["by_tree"], v["roles"])
        self.assertIn("main", tags)
        self.assertTrue(all(L._under(x["TRITON_CACHE_DIR"], root) for x in recs))

    @unittest.skipUnless(hasattr(os, "fork"), "needs fork")
    def test_vllm_fork_tree(self):
        v, recs, root = self.run_tree("vllm", "vllm.entrypoints.cli.main", VL_TREE)
        self.assertEqual(v["status"], "ISOLATED", (v["reasons"], sorted(str(x.get("role_tag")) for x in recs)))


# ================================================================== traces
def write_trace(path, steps=10, label="VERIFY", step_us=30000.0, nccl_name="ncclDevKernel_AllReduce_Sum_bf16_RING_LL",
                grid=(2, 1, 1), nccl_dur=300.0, wrapped=True):
    ev = []
    for k in range(steps):
        t = 1000.0 + k * step_us
        ev.append({"ph": "X", "cat": "user_annotation", "name": f"step[{label} bs=20]", "ts": t, "dur": step_us * 0.8})
        if label == "VERIFY":
            ev.append({"ph": "X", "cat": "user_annotation", "name": "step[DRAFT bs=20]", "ts": t + 10, "dur": 100})
        for j in range(3):
            ev.append({"ph": "X", "cat": "kernel", "name": nccl_name + "(ncclDevKernelArgsStorage<4096ul>)", "ts": t + 50 + j,
                       "dur": nccl_dur, "args": {"grid": list(grid), "block": [512, 1, 1]}})
        ev.append({"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllGather_RING_LL", "ts": t + 60, "dur": 80,
                   "args": {"grid": [2, 1, 1]}})
        ev.append({"ph": "X", "cat": "kernel", "name": "marlin_gemm_kernel<4>", "ts": t + 70, "dur": 1000})
        ev.append({"ph": "X", "cat": "kernel", "name": "fused_recurrent_gated_delta_rule_fwd", "ts": t + 90, "dur": 200})
    doc = {"schemaVersion": 1, "deviceProperties": [{"name": "L4"}], "traceEvents": ev} if wrapped else ev
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(doc, f)


class TestTrace(unittest.TestCase):
    def test_sglang_summary_fallback_and_e14(self):
        d = Path(tempfile.mkdtemp(prefix="e24lab_tr_"))
        try:
            for r in range(4):
                write_trace(d / f"1792778154.61-TP-{r}.trace.json.gz", nccl_dur=300.0 * (1 + 0.1 * r))
            s = L.trace_summary_sglang(d)
            self.assertNotIn("error", s)
            self.assertEqual(s["step_wall_ms"]["label"], "VERIFY")
            self.assertEqual((s["step_wall_ms"]["n"], s["step_wall_ms"]["p50"], s["step_wall_ms"]["p95"]), (9, 30.0, 30.0))
            self.assertEqual((s["nccl"]["allreduce_algo"], s["nccl"]["allreduce_proto"], s["nccl"]["allreduce_grid_x"]),
                             (["RING"], ["LL"], [2]))
            self.assertEqual(s["nccl"]["calls_per_step"], 4.0)
            self.assertAlmostEqual(s["nccl"]["ms_per_step"], 0.98, places=3)
            self.assertAlmostEqual(s["nccl_rank_skew_ms_per_step"], 0.27, places=3)
            self.assertEqual(s["ranks"]["0"]["steps"], 10)
            sys.path.insert(0, str(E14))
            try:
                s2 = L.trace_summary_sglang(d)
            finally:
                sys.path.remove(str(E14))
                sys.modules.pop("e14_trace_summary", None)
            if (E14 / "e14_trace_summary.py").exists():
                self.assertEqual(s2["e14_source"], "e14_trace_summary")
                self.assertEqual(s2["e14"]["groups"]["1792778154.61-TP-*"]["n_ranks"], 4)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_decode_label_and_errors(self):
        d = Path(tempfile.mkdtemp(prefix="e24lab_tr_"))
        try:
            write_trace(d / "p-TP-0.trace.json.gz", steps=5, label="DECODE", step_us=40000.0,
                        nccl_name="ncclDevKernel_AllReduce_Sum_bf16_TREE_SIMPLE", grid=(4, 1, 1), wrapped=False)
            (d / "p-TP-1.trace.json.gz").write_bytes(b"\x1f\x8b broken")
            s = L.trace_summary_sglang(d)
            self.assertEqual((s["step_wall_ms"]["label"], s["step_wall_ms"]["p50"]), ("DECODE", 40.0))
            self.assertEqual((s["nccl"]["allreduce_algo"], s["nccl"]["allreduce_proto"]), (["TREE"], ["SIMPLE"]))
            self.assertIn("error", s["ranks"]["1"])
            self.assertEqual(L.trace_summary_sglang(d / "nothing")["error"], "NO_TRACE_FILE")
            self.assertEqual(len(list(L._iter_events(d / "p-TP-0.trace.json.gz"))), 5 * (1 + 3 + 3))   # json fallback
            sys.path.insert(0, str(E14))
            try:                                                   # e14_trace_summary's streaming reader when importable
                self.assertEqual(len(list(L._iter_events(d / "p-TP-0.trace.json.gz"))), 5 * (1 + 3 + 3))
            finally:
                sys.path.remove(str(E14))
                sys.modules.pop("e14_trace_summary", None)
        finally:
            shutil.rmtree(d, ignore_errors=True)


# ================================================================== flag check in a temp venv with stub packages
STUB_SA = r'''
import argparse
class ServerArgs:
    def __init__(self, ns):
        self.ns = ns
    @staticmethod
    def add_cli_args(p):
        p.add_argument("--model-path")
        p.add_argument("--tp-size", type=int, default=1)
        p.add_argument("--cuda-graph-bs-decode", nargs="+", type=int)
        p.add_argument("--cuda-graph-bs-prefill", nargs="+", type=int)
        p.add_argument("--cuda-graph-max-bs-decode", type=int)
        p.add_argument("--enable-metrics", action="store_true")
        p.add_argument("--mamba-track-interval", type=int, default=256)
        p.add_argument("--default-chat-template-kwargs")
    def resolve_once(self):
        if self.ns.mamba_track_interval < 6:
            raise AssertionError("mamba_track_interval must be >= speculative_num_draft_tokens")
def prepare_server_args(argv):
    p = argparse.ArgumentParser(prog="sglang serve")
    ServerArgs.add_cli_args(p)
    return ServerArgs(p.parse_args(argv))
'''
STUB_VLLM = r'''
import sys
if sys.argv[1:] == ["serve", "--help=all"]:
    print("usage: vllm serve [model_tag] [options]\n" + "\n".join("  --%s X" % n for n in
          ["tensor-parallel-size", "max-num-seqs", "seed", "port", "host", "kv-cache-dtype"]) +
          "\n  --enable-prefix-caching, --no-enable-prefix-caching\n" + "#" * 2500)
'''


class TestFlagCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="e24lab_venv_"))
        r = subprocess.run([PY, "-m", "venv", "--without-pip", str(cls.tmp / "venv")], capture_output=True, text=True)
        if r.returncode:
            raise unittest.SkipTest("venv creation failed: " + r.stderr[-200:])
        cls.py = cls.tmp / "venv" / "bin" / "python"
        site = next((cls.tmp / "venv" / "lib").glob("python3*/site-packages"))
        for rel, txt in (("sglang/__init__.py", ""), ("sglang/srt/__init__.py", ""), ("sglang/srt/server_args.py", STUB_SA),
                         ("vllm/__init__.py", ""), ("vllm/entrypoints/__init__.py", ""),
                         ("vllm/entrypoints/cli/__init__.py", ""), ("vllm/entrypoints/cli/main.py", STUB_VLLM)):
            p = site / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(txt)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_sglang_exact_ambiguous_prefix_unknown(self):
        good = [str(self.py), "-m", "sglang.launch_server", "--model-path", "/m", "--tp-size", "4", "--cuda-graph-bs-decode",
                "1", "2", "3", "--enable-metrics", "--default-chat-template-kwargs", '{"preserve_thinking": true}']
        r = L.flag_check("sglang", self.py, good)
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["status"], r["method"], r["cached"], r["n_flags"]), ("GO", "syntax_only", False, 5))
        self.assertTrue(L.flag_check("sglang", self.py, good)["cached"])
        r = L.flag_check("sglang", self.py, ["--model-path", "/m", "--cuda-graph-bs", "1", "2"])
        self.assertEqual((r["ok"], r["ambiguous"], r["status"]), (False, ["--cuda-graph-bs"], "NO_GO_FLAG_UNSUPPORTED"))
        self.assertIn("argparse exit 2", r["parse_error"])
        r = L.flag_check("sglang", self.py, ["--model-path", "/m", "--tp", "4"])        # argparse would accept the prefix
        self.assertEqual((r["ok"], r["prefix"], r["parse_error"]), (False, ["--tp"], None))
        r = L.flag_check("sglang", self.py, ["--model-path", "/m", "--bogus-flag"])
        self.assertEqual((r["ok"], r["unknown"]), (False, ["--bogus-flag"]))

    def test_sglang_resolution(self):
        argv = ["--model-path", "/m", "--mamba-track-interval", "4"]
        self.assertTrue(L.flag_check("sglang", self.py, argv)["ok"])                     # syntax only passes
        r = L.flag_check("sglang", self.py, argv, resolve=True)
        self.assertEqual((r["ok"], r["status"], r["method"]), (False, "NO_GO_FLAG_RESOLUTION", "resolved"))
        self.assertIn("AssertionError", r["resolve_error"])
        r = L.flag_check("sglang", self.py, ["--model-path", "/m"], resolve=True, model_path="/real/int4")
        self.assertEqual((r["ok"], r["method"]), (True, "resolved"))

    def test_not_run_and_vllm_help(self):
        r = L.flag_check("sglang", PY, ["--model-path", "/m"])                           # no sglang in this python
        self.assertEqual((r["ran"], r["ok"], r["status"]), (False, False, "NOT_RUN"))
        argv = ["/v/bin/vllm", "serve", "/fp8", "--tensor-parallel-size", "4", "--max-num-seqs", "16",
                "--no-enable-prefix-caching", "--seed=1"]
        r = L.flag_check("vllm", self.py, argv)
        self.assertEqual((r["ok"], r["method"], r["unknown"]), (True, "help_all", []))
        r = L.flag_check("vllm", self.py, argv + ["--max-num-seq", "3"])
        self.assertEqual((r["ok"], r["unknown"]), (False, ["--max-num-seq"]))

    def test_fake_server_protocol(self):
        fake = self.tmp / "fake_flags.py"
        fake.write_text("import json, sys\nargv = json.loads(sys.argv[2])\nbad = [a for a in argv if a == '--nope']\n"
                        "print('log line')\nprint(json.dumps({'ran': True, 'ok': not bad, 'unknown': bad, 'ambiguous': [],"
                        " 'parse_error': None, 'method': 'exact'}))\n")
        r = L.flag_check("sglang", PY, ["py", "-m", "sglang.launch_server", "--a", "1"], fake_server=fake)
        self.assertEqual((r["ok"], r["method"], r["status"]), (True, "fake:exact", "GO"))
        r = L.flag_check("sglang", PY, ["--a", "--nope"], fake_server=fake)
        self.assertEqual((r["ok"], r["unknown"], r["status"]), (False, ["--nope"], "NO_GO_FLAG_UNSUPPORTED"))


# ================================================================== MINI: a small server speaking the real formats
MINI = r'''
import gzip, json, math, os, random, sys, threading, time, zlib
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
A = sys.argv[1:]
arg = lambda n, d=None: A[A.index(n) + 1] if n in A else d
PORT, ENGINE = int(arg("--port")), arg("--engine", "sglang")
E = os.environ.get
STEP, ACC, LOGINT = float(E("MINI_STEP_MS", "20")) / 1000, int(E("MINI_ACC", "2")), int(E("MINI_LOGINT", "5"))
MAXRUN, PF_TPS, RETRACT_AT = int(E("MINI_MAX_RUN", "64")), float(E("MINI_PREFILL_TPS", "1e9")), int(E("MINI_RETRACT_AT", "0"))
READY, V, LOCK = time.time() + float(E("MINI_READY_SEC", "0.3")), 64, threading.Lock()
S = {"dec": 0, "pf": 0, "pfc": 0, "passes": 0, "steps": 0, "run_g": 0, "q_g": 0, "retr": None, "gen_fin": 0, "active": [],
     "waiting": [], "seen": set(), "block_until": 0.0}
def log(msg, tp=0):
    print(time.strftime("[%Y-%m-%d %H:%M:%S") + (" TP%d] " % tp if ENGINE == "sglang" else "] ") + msg, flush=True)
def lps(ctx):
    h = zlib.crc32(json.dumps(list(ctx[-3:]) + [len(ctx)]).encode())
    z = [((h * (i + 7) * 2654435761) % 997) / 120.0 for i in range(V)]
    m = max(z)
    s = m + math.log(sum(math.exp(x - m) for x in z))
    return [x - s for x in z]
def pick(l, temp, rnd):
    if not temp:
        return max(range(V), key=lambda i: (l[i], -i))
    w = [math.exp(x) for x in l]
    return rnd.choices(range(V), weights=w)[0]
def topn(l, n):
    return [[l[i], i, None] for i in sorted(range(V), key=lambda i: (-l[i], i))[:n]]
def admit(s):
    P = len(s["ctx"])
    key = json.dumps(s["ctx"][:64]) + str(P)
    cached = ((P - 1) // 64) * 64 if key in S["seen"] else 0
    S["seen"].add(key)
    s["cached"] = cached
    S["pf"] += P - cached
    S["pfc"] += cached
    S["block_until"] = max(S["block_until"], time.time()) + (P - cached) / PF_TPS
    S["active"].append(s)
    S["run_g"], S["q_g"] = len(S["active"]) - 1, len(S["waiting"])
    log("Prefill batch, #new-seq: 1, #new-token: %d, #cached-token: %d, full token usage: 0.01, mamba usage: 0.04, "
        "#running-req: %d, #queue-req: %d, #pending-token: 0, cuda graph: False, input throughput (token/s): 1500.00"
        % (P - cached, cached, len(S["active"]) - 1, len(S["waiting"])))
def loop():
    while True:
        time.sleep(STEP)
        with LOCK:
            if time.time() < S["block_until"]:
                continue
            act = [s for s in S["active"] if s["left"] > 0 and not s["gone"]]
            for s in [s for s in S["active"] if s["gone"] or s["left"] <= 0]:
                S["active"].remove(s)
            while S["waiting"] and len(S["active"]) < MAXRUN:
                admit(S["waiting"].pop(0))
            if not act:
                if S["run_g"] and not S["active"]:
                    S["run_g"], S["q_g"] = 0, len(S["waiting"])
                continue
            S["steps"] += 1
            S["passes"] += 1
            for s in act:
                toks = []
                for _ in range(min(ACC, s["left"])):
                    t = pick(lps(s["ctx"]), s["temp"], s["rnd"])
                    s["ctx"].append(t)
                    toks.append(t)
                s["left"] -= len(toks)
                s["q"].append(toks)
                S["dec"] += len(toks)
            if RETRACT_AT and S["steps"] == RETRACT_AT:
                S["retr"] = (S["retr"] or 0) + 1
                for tp in range(4):
                    log("KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 100, "
                        "#mamba_num_gained: 3, #new_token_ratio: 0.7000 -> 0.8000", tp)
            if S["steps"] % LOGINT == 0:
                S["run_g"], S["q_g"] = len(act), len(S["waiting"])
                log("Decode batch, #running-req: %d, #full token: %d, full token usage: 0.01, mamba num: %d, mamba usage: 0.04, "
                    "accept len: %.2f, accept rate: 0.50, cuda graph: True, gen throughput (token/s): %.2f, #queue-req: %d"
                    % (len(act), sum(len(s["ctx"]) for s in act), 3 * len(act), ACC, len(act) * ACC / STEP, len(S["waiting"])))
def metrics():
    L = 'model_name="/m,0 \\"q\\"",engine_type="unified",tp_rank="%s",pp_rank="0"'
    if ENGINE == "vllm":
        return "\n".join(["vllm:num_requests_running{engine=\"0\"} %d" % len(S["active"]),
                          "vllm:num_requests_waiting{engine=\"0\"} %d" % len(S["waiting"]),
                          "vllm:generation_tokens_total{engine=\"0\"} %d" % S["dec"],
                          "vllm:prompt_tokens_total{engine=\"0\"} %d" % S["pf"], "vllm:num_preemptions_total{engine=\"0\"} 0",
                          "vllm:kv_cache_usage_perc{engine=\"0\"} 0.1",
                          "vllm:iteration_tokens_total_count{engine=\"0\"} %d" % S["passes"]]) + "\n"
    out = ["# TYPE sglang:num_running_reqs gauge", "sglang:num_running_reqs{%s} %d" % (L % 0, S["run_g"]),
           "sglang:num_queue_reqs{%s} %d" % (L % 0, S["q_g"]), "sglang:full_token_usage{%s} 0.01" % (L % 0),
           "sglang:kv_available_tokens{%s} %d" % (L % 0, 1000 if S["waiting"] else 900000),
           "sglang:kv_evictable_tokens{%s} 0" % (L % 0), "sglang:mamba_available_tokens{%s} 40" % (L % 0),
           "sglang:mamba_evictable_tokens{%s} 0" % (L % 0), "sglang:spec_accept_length{%s} %.2f" % (L % 0, ACC),
           "sglang:max_total_num_tokens{%s} 1000000" % (L % 0), "sglang:fwd_occupancy{%s} NaN" % (L % 0),
           "sglang:generation_tokens_total{model_name=\"m\",is_streaming=\"true\"} %d" % S["gen_fin"]]
    if S["dec"]:
        out.append('sglang:realtime_tokens_total{%s,mode="decode"} %d' % (L % 0, S["dec"]))
    out.append('sglang:realtime_tokens_total{%s,mode="prefill_compute"} %d' % (L % 0, S["pf"]))
    out.append('sglang:realtime_tokens_total{%s,mode="prefill_cache"} %d' % (L % 0, S["pfc"]))
    for r in range(4):
        out.append('sglang:cuda_graph_passes_total{%s,mode="decode_cuda_graph"} %d' % (L % r, S["passes"]))
        if S["retr"] is not None:
            out.append("sglang:num_retracted_requests_total{%s} %d" % (L % r, S["retr"]))
    return "\n".join(out) + "\n"
def trace(d, pid, n):
    for r in range(4):
        ev = []
        for k in range(n):
            t = 1000.0 + k * 30000
            ev += [{"ph": "X", "cat": "user_annotation", "name": "step[VERIFY bs=2]", "ts": t, "dur": 25000},
                   {"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(x)", "ts": t + 5,
                    "dur": 300 * (1 + 0.1 * r), "args": {"grid": [2, 1, 1]}},
                   {"ph": "X", "cat": "kernel", "name": "marlin_gemm", "ts": t + 9, "dur": 900}]
        with gzip.open(os.path.join(d, "%s-TP-%d.trace.json.gz" % (pid, r)), "wt") as f:
            json.dump({"traceEvents": ev}, f)
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else (body.encode() if isinstance(body, str) else json.dumps(body).encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/ready", "/health"):
            ok = time.time() >= READY and (p == "/ready") == (ENGINE == "sglang")
            return self.send(200 if ok else (503 if time.time() < READY else 404), "OK" if ok else "starting", "text/plain")
        if p == "/metrics":
            with LOCK:
                return self.send(200, metrics(), "text/plain")
        if p == "/v1/models":
            return self.send(200, {"data": [{"id": "/dev/shm/mini" if ENGINE == "sglang" else "q27"}]})
        if p == "/server_info" and ENGINE == "sglang":
            return self.send(200, {"launch_command": " ".join(A), "max_total_num_tokens": 1000000,
                                   "internal_states": [{"effective_max_running_requests_per_dp": MAXRUN}]})
        self.send(404, {"error": {"message": "no route"}})
    def body(self):
        return json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
    def do_POST(self):
        p, b = self.path.split("?")[0], self.body()
        if p == "/generate" and ENGINE == "sglang":
            return self.generate(b)
        if p == "/v1/completions":
            return self.stream(b) if b.get("stream") else self.complete(b)
        if p == "/v1/chat/completions":
            if b.get("tools"):
                msg = {"content": "", "reasoning_content": None, "tool_calls": [{"id": "c0", "type": "function", "function": {
                       "name": "get_weather", "arguments": json.dumps({"city": "Paris"})}}]}
            else:
                q = b["messages"][0]["content"]
                msg = {"content": str(eval(q.split(":")[1].split("=")[0])), "reasoning_content": ""}
            return self.send(200, {"choices": [{"message": msg, "finish_reason": "stop"}]})
        if p == "/flush_cache":
            with LOCK:
                busy = bool(S["active"] or S["waiting"])
                if not busy:
                    S["seen"].clear()
            return self.send(400 if busy else 200, "busy" if busy else "Cache flushed.\nPlease check backend logs.", "text/plain")
        if p == "/abort_request":
            with LOCK:
                for s in S["active"] + S["waiting"]:
                    if s["rid"] == b.get("rid"):
                        s["gone"] = True
                S["waiting"] = [s for s in S["waiting"] if not s["gone"]]
            return self.send(200, b"", "text/plain")
        if p == "/start_profile":
            d, n = b.get("output_dir") or "/tmp", int(b.get("num_steps") or 5)
            pid = "%.4f" % time.time()
            threading.Timer(0.2, trace, (d, pid, n)).start()
            return self.send(200, "Start profiling.\n", "text/plain")
        self.send(404, {"error": {"message": "no route"}})
    def generate(self, b):
        ids, sp = list(b["input_ids"]), b.get("sampling_params") or {}
        n, temp = int(sp.get("max_new_tokens", 128)), float(sp.get("temperature", 1.0))
        rnd, ctx, top = random.Random(sp.get("sampling_seed", 0)), list(ids), int(b.get("top_logprobs_num") or 0)
        otl, otop, out = [], [], []
        for _ in range(n):
            l = lps(ctx)
            t = pick(l, temp if temp > 1e-6 else 0, rnd)
            otl.append([l[t], t, None])
            otop.append(topn(l, top) if top else None)
            ctx.append(t)
            out.append(t)
        key = json.dumps(ids[:64]) + str(len(ids))
        with LOCK:
            cached = ((len(ids) - 1) // 64) * 64 if key in S["seen"] else 0
            S["seen"].add(key)
            S["pf"] += len(ids) - cached
            S["pfc"] += cached
        meta = {"id": b.get("rid"), "finish_reason": {"type": "length", "length": n}, "prompt_tokens": len(ids),
                "completion_tokens": n, "cached_tokens": cached}
        if b.get("return_logprob"):
            meta.update(output_token_logprobs=otl, output_token_logprobs_length=len(otl), input_token_logprobs=[])
            if top:
                meta["output_top_logprobs"] = otop
            st = int(b.get("logprob_start_len", -1))
            if st >= 0:
                rows = [[lps(ids[:q])[ids[q]], ids[q], None] for q in range(st + 1, len(ids))]
                tops = [topn(lps(ids[:q]), top) for q in range(st + 1, len(ids))] if top else []
                if E("MINI_TF_BROKEN"):
                    meta.update(input_token_logprobs=rows, input_top_logprobs=tops)
                else:
                    meta.update(input_token_logprobs=[[None, ids[st], None]] + rows, input_top_logprobs=[None] + tops)
            tids = b.get("token_ids_logprob")
            if tids:
                l0 = lps(ids)
                meta["output_token_ids_logprobs"] = [[[l0[i] if i < V else -1e4, i, None] for i in tids]]
        self.send(200, {"text": " ".join("w%d" % t for t in out), "output_ids": out, "meta_info": meta})
    def complete(self, b):
        ids = list(b["prompt"])
        ch = {"index": 0, "text": " w0", "token_ids": [0], "finish_reason": "length"}
        if b.get("prompt_logprobs") and ENGINE == "vllm":
            k, pl = int(b["prompt_logprobs"]), [None]
            for q in range(1, len(ids)):
                l = lps(ids[:q])
                rank = sorted(range(V), key=lambda i: (-l[i], i))
                d = {str(i): {"logprob": l[i], "rank": rank.index(i) + 1, "decoded_token": "token_id:%d" % i} for i in rank[:k]}
                d[str(ids[q])] = {"logprob": l[ids[q]], "rank": rank.index(ids[q]) + 1, "decoded_token": "x"}
                pl.append(d)
            ch["prompt_logprobs"] = pl
        with LOCK:
            S["pf"] += len(ids)
        self.send(200, {"choices": [ch], "usage": {"prompt_tokens": len(ids), "completion_tokens": 1,
                                                   "prompt_tokens_details": {"cached_tokens": 0}}})
    def stream(self, b):
        s = {"ctx": list(b["prompt"]), "left": int(b.get("max_tokens", 16)), "temp": float(b.get("temperature", 1.0)),
             "rnd": random.Random(b.get("seed", 0)), "q": [], "gone": False, "rid": b.get("rid"), "cached": 0,
             "lp": int(b.get("logprobs") or 0)}
        with LOCK:
            if len(S["active"]) < MAXRUN:
                admit(s)
            else:
                S["waiting"].append(s)
                S["q_g"] = len(S["waiting"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        def w(obj):
            data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
        try:
            while True:
                with LOCK:
                    chunks, s["q"] = s["q"], []
                    done = (s["left"] <= 0 and not chunks) or s["gone"]
                for toks in chunks:
                    c = {"index": 0, "text": "".join(" w%d" % t for t in toks), "token_ids": toks, "finish_reason": None}
                    if s["lp"]:
                        c["logprobs"] = {"tokens": ["x"] * len(toks), "token_logprobs": [-0.5] * len(toks),
                                         "top_logprobs": [{"token_id:%d" % t: -0.5, "token_id:%d" % ((t + 1) % V): -1.25}
                                                          for t in toks]}
                    w({"object": "text_completion", "choices": [c]})
                if done:
                    break
                time.sleep(0.003)
            w({"object": "text_completion", "choices": [{"index": 0, "text": "", "token_ids": [],
                                                          "finish_reason": "abort" if s["gone"] else "length"}]})
            det = {"cached_tokens": s["cached"]} if s["cached"] else None
            w({"object": "text_completion", "choices": [], "usage": {"prompt_tokens": len(b["prompt"]),
               "completion_tokens": int(b.get("max_tokens", 16)) - s["left"], "prompt_tokens_details": det}})
            w("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            with LOCK:
                S["gen_fin"] += int(b.get("max_tokens", 16)) - s["left"]
        except Exception:
            pass
        finally:
            with LOCK:
                s["gone"] = True
print("[%s] server_args=%r" % (time.strftime("%Y-%m-%d %H:%M:%S"), {"model_path": "/dev/shm/mini", "tp_size": 4,
      "max_running_requests": MAXRUN, "context_length": 155648, "enable_mixed_chunk": False}), flush=True)
log("max_total_num_tokens=1000000, chunked_prefill_size=4096, max_prefill_tokens=16384, max_running_requests=%d, "
    "context_len=155648, available_gpu_mem=0.89 GB" % MAXRUN)
threading.Thread(target=loop, daemon=True).start()
def fired():
    time.sleep(max(0.0, READY - time.time()))
    print("INFO:     Application startup complete." if ENGINE == "vllm" else
          time.strftime("[%Y-%m-%d %H:%M:%S] ") + "The server is fired up and ready to roll!", flush=True)
threading.Thread(target=fired, daemon=True).start()
class Srv(ThreadingHTTPServer):
    daemon_threads = True
    def server_bind(self):                       # no socket.getfqdn(): it can hang without DNS
        import socketserver
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]
Srv(("127.0.0.1", PORT), H).serve_forever()
'''


class MiniBase(unittest.TestCase):
    engine = "sglang"
    env_extra = {}

    @classmethod
    def setUpClass(cls):
        cls.ts, L.TS = L.TS, 0.25                       # the lab's internal floors / idle polls, as in a FAKE run
        cls.tmp = Path(tempfile.mkdtemp(prefix="e24lab_mini_"))
        (cls.tmp / "mini.py").write_text(MINI)
        cls.port = free_port()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", **cls.env_extra)
        cls.srv = L.Server("mini_" + cls.engine, [PY, str(cls.tmp / "mini.py"), "--port", str(cls.port), "--engine",
                                                  cls.engine], env, cls.port, cls.tmp, engine=cls.engine).start()
        ok, msg = cls.srv.wait_ready(30)
        if not ok:
            cls.srv.kill()
            raise RuntimeError("mini server did not start: " + msg + cls.srv.log()[-500:])

    @classmethod
    def tearDownClass(cls):
        L.TS = cls.ts
        cls.srv.kill()
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestMiniSGLang(MiniBase):
    env_extra = {"MINI_PREFILL_TPS": "20000"}

    def test_00_lifecycle_and_facts(self):
        self.assertTrue(self.srv.alive())
        self.assertEqual(self.srv.model_name, "/dev/shm/mini")
        self.assertIsNotNone(self.srv.ready_sec)
        f = self.srv.facts()
        self.assertEqual((f["server_args"]["max_running_requests"], f["kv_tokens"], f["ready_line"]), (64, 1000000, True))
        self.assertEqual(self.srv.server_info()["internal_states"][0]["effective_max_running_requests_per_dp"], 64)
        self.assertTrue(L.wait_idle("sglang", self.port, 10))
        snap = L.snapshot("sglang", self.port)
        self.assertEqual((snap["running"], snap["preempt"]), (0, 0.0))
        self.assertTrue(L.flush_cache("sglang", self.port, 10)["ok"])

    def test_foreign_listener_is_not_ready(self):
        orig = L.port_listening
        L.port_listening = lambda p: False                 # start a second "server" on the mini's (busy) port
        try:
            b = L.Server("foreign", [PY, "-c", "import time; time.sleep(30)"], dict(os.environ), self.port, self.tmp).start()
        finally:
            L.port_listening = orig
        try:
            ok, msg = b.wait_ready(2.0)                     # the mini answers /ready 200; b never logs its ready line
            self.assertEqual((ok, msg), (False, "startup timeout"))
        finally:
            b.kill()
        self.assertFalse(b.alive())

    def test_greedy_rows(self):
        prompts = [[1, 2, 3, 4], [5, 6, 7], [9, 9, 9, 9, 9]]
        rows = L.greedy("sglang", self.port, prompts, n_tok=12)
        self.assertEqual([r["key"] for r in rows], [L.prompt_key(p) for p in prompts])
        for r in rows:
            self.assertIsNone(r["error"])
            self.assertEqual((len(r["tokens"]), len(r["margins"]), len(r["top2"]), len(r["lps"])), (12, 12, 12, 12))
            self.assertEqual(r["top2"][0][0], r["tokens"][0])                        # greedy = top-1
            self.assertGreaterEqual(r["margins"][0], 0)
            self.assertEqual(r["finish_reason"], "length")
        conc = L.greedy("sglang", self.port, prompts, n_tok=12, concurrent=True)
        self.assertEqual(L.compare(rows, conc)["status"], "EXACT_TO_N")
        named = L.greedy("sglang", self.port, {"a": prompts[0]}, n_tok=4)
        self.assertEqual(named[0]["key"], "a")

    def test_teacher_forced_convention(self):
        prompt = [3, 1, 4, 1, 5]
        g = L.generate_logged("sglang", self.port, prompt, 20, temperature=0.0, top=2)
        self.assertEqual((g["status"], g["n"], len(g["lps"])), ("MEASURED", 20, 20))
        tf = L.teacher_forced("sglang", self.port, prompt, g["tokens"], top=2)
        self.assertEqual(tf["status"], "MEASURED", tf.get("error"))
        self.assertEqual(len(tf["positions"]), 20)
        for k, p in enumerate(tf["positions"]):
            self.assertAlmostEqual(p[0], g["lps"][k], places=9)                     # entry k = lp(cont[k] | prompt+cont[:k])
            self.assertEqual(p[1], g["tokens"][k])                                    # greedy: actual = top1
            self.assertEqual(len(p), 5)
        self.assertEqual(L.decode_vs_prefill(g, tf)["status"], "MEASURED")
        self.assertEqual(L.compare_tf(tf, tf)["mean_abs_dlogprob"], 0.0)
        self.assertEqual(L.teacher_forced("sglang", self.port, [], [1])["status"], "INCONCLUSIVE_TF_ALIGN")

    def test_generate_logged_sampling_eos_many(self):
        g = L.generate_logged("sglang", self.port, [1, 2], 30, temperature=1.0, eos_ids=(5, 6), seed=3)
        self.assertEqual(g["status"], "MEASURED")
        self.assertEqual(g["natural_eos_pos"], next((k for k, t in enumerate(g["tokens"]) if t in (5, 6)), None))
        many = L.generate_logged_many("sglang", self.port, [[1, 2], [3, 4], [5]], 10, seeds=[1, 2, 3])
        self.assertEqual([m["status"] for m in many], ["MEASURED"] * 3)
        self.assertEqual(many[0]["key"], L.prompt_key([1, 2]))

    def test_next_token_dist_and_kl(self):
        d = L.next_token_dist(self.port, [1, 2, 3], 64)
        self.assertNotIn("error", d)
        self.assertAlmostEqual(sum(math.exp(x) for x in d["lp"]), 1.0, places=9)
        self.assertAlmostEqual(L.kl(d["lp"], d["lp"]), 0.0, places=12)
        self.assertAlmostEqual(d["sum_raw"], 1.0, places=5)
        part = L.next_token_dist(self.port, [1, 2, 3], 63)                           # a partial vocabulary shows in sum_raw
        self.assertLess(part["sum_raw"], 0.9999)

    def test_stream_chunks_usage_rid(self):
        g = L.Stream(self.port, [1, 2, 3], 9, temperature=0.0, model="default", engine="sglang")
        self.assertTrue(g.rid.startswith("e24-s-"))
        self.assertNotIn("return_tokens_as_token_ids", g.body)
        g.start()
        g.join(20)
        r = g.row()
        self.assertIsNone(r["error"])
        self.assertEqual((len(r["tokens"]), r["finish_reason"], sum(n for _, n in g.chunk_times)), (9, "length", 9))
        self.assertTrue(all(b[0] >= a[0] for a, b in zip(g.chunk_times, g.chunk_times[1:])))
        s = L.Stream(self.port, [1], 5, temperature=1.0)
        self.assertEqual((s.body["top_p"], s.body["top_k"], s.body["temperature"]), (0.95, 20, 1.0))

    def test_probes(self):
        self.assertTrue(L.chat_probes("sglang", self.port)["ok"])
        t = L.tool_probe("sglang", self.port)
        self.assertTrue(t["ok"], t)

    def test_steady_window(self):
        r = L.steady("sglang", self.port, [[i, i + 1, i + 2] for i in range(4)], 4, secs=2.0, max_new=100000, sample=0.3,
                     server=self.srv, min_depth=40, admit=20, gpu_sample=False)
        self.assertEqual(r["status"], "STEADY", r)
        self.assertGreaterEqual(r["depth_at_start"], 40)
        self.assertGreater(r["tok_s"], 0)
        self.assertEqual(r["tokens_per_step"], 8.0)                                  # 4 streams x 2 tokens per step
        self.assertEqual(r["retractions"], 0.0)
        self.assertGreater(r["client_step_ms"]["n"], 10)
        self.assertGreater(r["decode_lines"]["lines"], 0)
        self.assertEqual(r["decode_lines"]["running_values"], [4])
        self.assertEqual(r["decode_lines"]["accept_len_mean"], 2.0)
        self.assertFalse(r["client_leak"])
        self.assertTrue(r["idle_after"])

    def test_census_admitted_all(self):
        r = L.census("sglang", self.port, [[i] * 50 for i in range(3)], 3, secs=1.0, plateau_sec=0.5, min_hold=0.5,
                     server=self.srv, sample=0.2, max_wait=30)
        self.assertEqual(r["status"], "ADMITTED_ALL", r)
        self.assertEqual(r["blocking_reason"], "ALL_ADMITTED")
        self.assertEqual(r["running_gauge"]["max"], 3)
        self.assertIsNotNone(r["gauges_end"]["kv_available_tokens"])

    def test_interference_cold_and_warm(self):
        res = [[i, 7, 7] for i in range(3)]
        d0 = L.interference("sglang", self.port, res, [], 3, server=self.srv, window_sec=1.5, sample=0.2, admit=20)
        self.assertEqual(d0["status"], "MEASURED", d0)
        self.assertFalse(d0["starvation"])
        cold = list(range(2, 64)) * 160                                              # ~10k tokens at 20k tok/s: ~0.5 s freeze
        r = L.interference("sglang", self.port, res, [cold], 3, server=self.srv, window_sec=2.5, sample=0.2, admit=20,
                           d0_median_step=d0["pre_injection_step_ms"]["p50"], gap_sec=0.3)
        self.assertEqual(r["status"], "MEASURED", r)
        self.assertTrue(r["starvation"])
        self.assertGreater(r["max_gap_s"], 0.3)
        self.assertIsNotNone(r["prefill_sec"][0])
        self.assertGreater(r["decode_flat_sec"], 0)
        self.assertTrue(L.flush_cache("sglang", self.port, 10)["ok"])
        warm = list(range(1, 60)) * 100
        w = L.interference("sglang", self.port, res, None, 3, warm_prompt=warm, server=self.srv, window_sec=1.5, sample=0.2,
                           admit=20)
        self.assertEqual(w["status"], "MEASURED", w)
        self.assertEqual(w["warm_check"]["source"], "prefill_line")
        self.assertTrue(w["warm_check"]["ok"])
        self.assertTrue(L.flush_cache("sglang", self.port, 10)["ok"])
        cold2 = list(range(3, 61)) * 100
        miss = L.interference("sglang", self.port, res, [cold2], 3, warm_prompt=list(range(1, 60)) * 100, server=self.srv,
                              window_sec=1.0, sample=0.2, admit=20)
        self.assertEqual(miss["status"], "INVALID_WARM_MISS")

    def test_profile(self):
        out = self.tmp / "prof"
        rec = L.profiled_load("sglang", self.port, [[1, 2], [3, 4]], 2, out, num_steps=6, server=self.srv, sample=0.3)
        self.assertTrue(rec["settled"], rec)
        self.assertEqual(rec["ranks"], [0, 1, 2, 3])
        self.assertNotIn("error", rec)
        self.assertEqual(rec["trace"]["step_wall_ms"]["p50"], 30.0)
        self.assertEqual(rec["trace"]["nccl"]["allreduce_algo"], ["RING"])
        self.assertTrue(rec["idle_after"])


class TestMiniRetraction(MiniBase):
    env_extra = {"MINI_RETRACT_AT": "60"}

    def test_steady_retraction(self):
        r = L.steady("sglang", self.port, [[1, 2, 3], [4, 5, 6]], 2, secs=2.5, max_new=100000, sample=0.3, server=self.srv,
                     min_depth=60, admit=20, gpu_sample=False)
        self.assertEqual(r["status"], "INCONCLUSIVE_RETRACTION", r)
        self.assertEqual(r["retractions"], 1.0)                                      # tp0 value, not 4x
        self.assertEqual(r["retract_log_lines"], 1)                                  # TP0 line only


class TestMiniCapacity(MiniBase):
    env_extra = {"MINI_MAX_RUN": "2"}

    def test_census_capacity_limited(self):
        r = L.census("sglang", self.port, [[i] * 30 for i in range(5)], 5, secs=1.0, plateau_sec=0.6, min_hold=0.6,
                     server=self.srv, sample=0.2, max_wait=30)
        self.assertEqual(r["status"], "CAPACITY_LIMITED", r)
        self.assertEqual(r["blocking_reason"], "KV_BOUND+MAX_RUNNING")             # max_running resolved by the lab
        self.assertEqual((r["max_running"], r["max_running_source"]), (2, "server_info.effective_max_running_requests_per_dp"))
        self.assertEqual(r["queue"]["max"], 3)
        self.assertEqual(r["running_lines"]["p50"], 2)

    def test_max_running_sources_and_idle_guard(self):
        self.assertEqual(L.max_running_of(self.port, self.srv), (2, "server_info.effective_max_running_requests_per_dp"))

        class LogOnly:                                                         # no /server_info: the log decides
            def __init__(self, txt):
                self.txt = txt

            def log(self):
                return self.txt
        dead = free_port()
        cap = "[2026-10-26 13:42:25 TP0] max_running_requests is capped to 17 by the mamba state cache (x)\n"
        line = ("[2026-10-26 13:42:25 TP0] max_total_num_tokens=779192, chunked_prefill_size=4096, max_prefill_tokens=16384, "
                "max_running_requests=21, context_len=155648, available_gpu_mem=0.89 GB\n")
        self.assertEqual(L.max_running_of(dead, LogOnly(line + cap)), (17, "log.capped"))
        self.assertEqual(L.max_running_of(dead, LogOnly(line)), (21, "log.capacity"))
        self.assertEqual(L.max_running_of(dead, LogOnly("")), (None, None))
        self.assertTrue(L.wait_idle("sglang", self.port, 10, self.srv))
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        busy = LogOnly(self.srv.log() + f"[{now} TP0] Prefill batch, #new-seq: 1, #new-token: 4096, #cached-token: 0, "
                       "full token usage: 0.01, mamba usage: 0.04, #running-req: 0, #queue-req: 0, #pending-token: 85904, "
                       "cuda graph: False, input throughput (token/s): 1500.00\n")
        t0 = time.perf_counter()
        self.assertFalse(L.wait_idle("sglang", self.port, 2.5, busy))      # a chunked prefill in flight is not idle
        self.assertGreater(time.perf_counter() - t0, 2.0)
        old = LogOnly(busy.txt.replace(now, "2026-01-01 00:00:00"))        # an old pending line (aborted prefill)
        self.assertTrue(L.wait_idle("sglang", self.port, 5, old))


class TestMiniTFBroken(MiniBase):
    env_extra = {"MINI_TF_BROKEN": "1"}

    def test_tf_align_detected(self):
        tf = L.teacher_forced("sglang", self.port, [1, 2, 3], [4, 5, 6])
        self.assertEqual(tf["status"], "INCONCLUSIVE_TF_ALIGN")


class TestMiniVLLM(MiniBase):
    engine = "vllm"

    def test_vllm_paths(self):
        self.assertEqual(self.srv.model_name, "q27")
        self.assertIsNone(self.srv.server_info())
        rows = L.greedy("vllm", self.port, [[1, 2, 3]], n_tok=6, top=2)
        r = rows[0]
        self.assertIsNone(r["error"])
        self.assertEqual((len(r["tokens"]), r["margins"][0], len(r["lps"])), (6, 0.75, 6))
        self.assertEqual(r["top2"][0][0], r["tokens"][0])
        tf = L.teacher_forced("vllm", self.port, [1, 2, 3], [4, 5, 6, 7], top=2)
        self.assertEqual(tf["status"], "MEASURED", tf.get("error"))
        self.assertEqual(len(tf["positions"]), 4)
        self.assertEqual(len(tf["positions"][0]), 5)
        s = L.Stream(self.port, [1], 4, temperature=0.0, logprobs=2, engine="vllm", model="q27")
        self.assertTrue(s.body["return_tokens_as_token_ids"])
        self.assertNotIn("rid", s.body)
        g = L.generate_logged("vllm", self.port, [1, 2], 8, top=2)
        self.assertEqual((g["status"], g["n"]), ("MEASURED", 8))
        self.assertTrue(L.wait_idle("vllm", self.port, 10))
        st = L.steady("vllm", self.port, [[1, 2], [3, 4]], 2, secs=1.5, max_new=100000, sample=0.3, min_depth=20, admit=20,
                      gpu_sample=False)
        self.assertEqual(st["status"], "STEADY", st)
        self.assertGreater(st["tok_s"], 0)


# ================================================================== WP2 fakes (skipped when absent)
@unittest.skipUnless(FAKE_SG.exists(), "src/harness/fake_sglang_server.py not present")
class TestFakeSGLang(unittest.TestCase):
    ARGV = ["--model-path", "/fake/int4", "--tp-size", "4", "--kv-cache-dtype", "fp8_e4m3", "--context-length", "155648",
            "--chunked-prefill-size", "4096", "--mamba-ssm-dtype", "float16", "--enable-metrics", "--random-seed", "20261005",
            "--host", "127.0.0.1", "--log-level", "info", "--decode-log-interval", "10", "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", "/fake/dflash2", "--speculative-draft-model-quantization", "unquant",
            "--mamba-radix-cache-strategy", "extra_buffer_lazy", "--mem-fraction-static", "0.83",
            "--incremental-streaming-output", "--stream-interval", "1", "--enable-cache-report",
            "--speculative-num-draft-tokens", "6", "--min-free-slots-delay", "1", "--max-running-requests", "9",
            "--cuda-graph-bs-decode", "1", "2", "3", "4", "5", "6", "7", "8", "9", "--max-mamba-cache-size", "36"]

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="e24lab_fake_"))
        cls.port = free_port()
        env = dict(os.environ, FAKE_SG_STARTUP_SEC="0.5", FAKE_SG_STEP_MS="20", SGLANG_LOGPROB_CHUNK_SIZE="128",
                   PYTHONDONTWRITEBYTECODE="1", HOME=str(cls.tmp / "home"))
        env.update(L.cache_env("sglang", cls.tmp / "cache"))
        if HOOK.exists():                                   # the real WP3 hook as sitecustomize, as the notebook does
            (cls.tmp / "site").mkdir()
            shutil.copy(HOOK, cls.tmp / "site" / "sitecustomize.py")
            env.update(E24_HOOK_DIR=str(cls.tmp / "hook"), E24_HOOK_COUNTERS="1", E24_HOOK_FLUSH_SEC="0.05",
                       E24_HOOK_CHAIN="0", PYTHONPATH=str(cls.tmp / "site"))
            env.pop("PYTHONSAFEPATH", None)
        cls.before = L.dir_stats(L.default_cache_dirs())
        cls.srv = L.Server("fake_sg", [PY, str(FAKE_SG)] + cls.ARGV + ["--port", str(cls.port)], env, cls.port, cls.tmp,
                           engine="sglang").start()
        ok, msg = cls.srv.wait_ready(120)
        if not ok:
            cls.srv.kill()
            raise AssertionError("fake SGLang did not start: " + msg + "\n" + cls.srv.log()[-1500:])

    @classmethod
    def tearDownClass(cls):
        cls.srv.kill()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_facts_and_flag_check(self):
        f = self.srv.facts()
        self.assertIsInstance(f["server_args"], dict, f.get("server_args_error"))
        self.assertIsNotNone(f["kv_tokens"])
        self.assertTrue(f["ready_line"])
        r = L.flag_check("sglang", PY, self.ARGV, fake_server=FAKE_SG)
        self.assertTrue(r["ok"], r)
        r = L.flag_check("sglang", PY, self.ARGV + ["--cuda-graph-bs", "1"], fake_server=FAKE_SG)
        self.assertFalse(r["ok"], r)
        self.assertIn("--cuda-graph-bs", r["ambiguous"] + r["unknown"] + r.get("prefix", []))

    def test_greedy_tf_logged(self):
        rows = L.greedy("sglang", self.port, [[11, 12, 13] * 30, [5, 6] * 40], n_tok=32)
        self.assertTrue(all(r["error"] is None and len(r["tokens"]) == 32 for r in rows), rows[0].get("error"))
        self.assertTrue(all(len(r["margins"]) == 32 and r["top2"][0] for r in rows))
        g = L.generate_logged("sglang", self.port, [11, 12, 13] * 30, 64, temperature=0.0, top=2)
        self.assertEqual(g["status"], "MEASURED", g.get("error"))
        tf = L.teacher_forced("sglang", self.port, [11, 12, 13] * 30, g["tokens"], top=2)
        self.assertEqual(tf["status"], "MEASURED", tf.get("error"))
        self.assertEqual([p[1] for p in tf["positions"]][:5], g["tokens"][:5])      # greedy decode = prefill top-1
        dvp = L.decode_vs_prefill(g, tf)
        self.assertIn(dvp["status"], ("MEASURED",), dvp)
        many = L.generate_logged_many("sglang", self.port, [[1, 2, 3] * 20, [4, 5, 6] * 20], 40, temperature=1.0)
        self.assertEqual([m["status"] for m in many], ["MEASURED", "MEASURED"], [m.get("error") for m in many])

    @unittest.skipUnless(HOOK.exists(), "src/e24_hook.py not present")
    def test_hook_ranks_and_cache_isolation(self):
        """Counters files written by the real hook in the fake's 4 spawned TP ranks: rev 2b rank set, self-test rollup,
        roles by tag and by tree, caches under the arm root."""
        hd, end = self.tmp / "hook", time.time() + 30
        while True:
            hc = L.hook_counters(hd, tp=4)
            if hc["n_ranks"] == 4 or time.time() > end:
                break
            time.sleep(0.5)
        self.assertEqual((hc["status"], hc["ranks"], hc["n_ranks"]), ("OK", [0, 1, 2, 3], 4), hc)
        files = [json.loads(p.read_text()) for p in sorted(hd.glob("counters_*.json"))]
        self.assertEqual(sorted(d["tp_rank"] for d in files if d.get("role_tag") == "scheduler"), [0, 1, 2, 3])
        self.assertEqual(hc["self_test_ok"], all((d.get("self_test") or {}).get("ok") for d in files), hc)
        tags = sorted(str(r.get("role_tag")) for r in L.hook_records(hd))
        self.assertEqual((tags.count("main"), tags.count("scheduler"), tags.count("detokenizer")), (1, 4, 1), tags)
        v = L.cache_verdict("sglang", self.tmp / "cache", hd, self.before, L.dir_stats(L.default_cache_dirs()), dflash=True)
        self.assertEqual(v["status"], "ISOLATED", v["reasons"])
        self.assertTrue(v["roles"]["by_tags"] and v["roles"]["by_tree"], v["roles"])

    def test_steady_census_interference_profile(self):
        prompts = [[100 + i] * 300 for i in range(4)]
        st = L.steady("sglang", self.port, prompts, 4, secs=3.0, max_new=100000, sample=0.5, server=self.srv, min_depth=50,
                      admit=60, gpu_sample=False)
        self.assertIn(st["status"], ("STEADY", "CAPACITY_LIMITED"), st)
        self.assertGreater(st["tok_s"] or 0, 0)
        self.assertIsNotNone(st["steps"])
        self.assertTrue(L.flush_cache("sglang", self.port, 30)["ok"])
        ce = L.census("sglang", self.port, [[200 + i] * 400 for i in range(3)], 3, secs=2.0, plateau_sec=1.0, min_hold=1.0,
                      server=self.srv, sample=0.3, max_wait=90)
        self.assertIn(ce["status"], ("ADMITTED_ALL", "CAPACITY_LIMITED"), ce)
        self.assertTrue(L.flush_cache("sglang", self.port, 30)["ok"])
        it = L.interference("sglang", self.port, [[300 + i] * 200 for i in range(3)], [[7] * 3000], 3, server=self.srv,
                            window_sec=4.0, sample=0.3, admit=60)
        self.assertEqual(it["status"], "MEASURED", it)
        self.assertIsNotNone(it["prefill_sec"][0])
        self.assertIsNotNone(it["starvation"])
        pr = L.profiled_load("sglang", self.port, [[400 + i] * 100 for i in range(2)], 2, self.tmp / "prof", num_steps=8,
                             server=self.srv, sample=0.5, tp=4)
        self.assertEqual(pr.get("ranks"), [0, 1, 2, 3], pr)
        self.assertIsNotNone(pr["trace"].get("step_wall_ms", {}).get("p50"), pr["trace"])


@unittest.skipUnless(FAKE_VL.exists(), "src/harness/fake_vllm_server.py not present")
class TestFakeVLLM(unittest.TestCase):
    ARGV = ["/fake/fp8", "--tensor-parallel-size", "4", "--kv-cache-dtype", "fp8_e4m3", "--max-model-len", "131072",
            "--max-num-seqs", "16", "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.90",
            "--disable-custom-all-reduce", "--seed", "20261005", "--host", "127.0.0.1", "--served-model-name", "q27",
            "--attention-backend", "FLASHINFER", "--trust-remote-code", "--language-model-only", "--no-enable-prefix-caching"]

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="e24lab_fakev_"))
        cls.port = free_port()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HOME=str(cls.tmp / "home"))
        cls.srv = L.Server("fake_v", [PY, str(FAKE_VL), "serve"] + cls.ARGV + ["--port", str(cls.port)], env, cls.port,
                           cls.tmp, engine="vllm").start()
        ok, msg = cls.srv.wait_ready(120)
        if not ok:
            cls.srv.kill()
            raise AssertionError("fake vLLM did not start: " + msg + "\n" + cls.srv.log()[-1500:])

    @classmethod
    def tearDownClass(cls):
        cls.srv.kill()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_vllm_greedy_tf_flags(self):
        rows = L.greedy("vllm", self.port, [[11, 12, 13] * 20], n_tok=16)
        self.assertIsNone(rows[0]["error"], rows[0])
        self.assertEqual(len(rows[0]["tokens"]), 16)
        self.assertEqual(len(rows[0]["top2"]), 16)
        tf = L.teacher_forced("vllm", self.port, [11, 12, 13] * 20, rows[0]["tokens"], top=2)
        self.assertEqual(tf["status"], "MEASURED", tf.get("error"))
        self.assertEqual([p[1] for p in tf["positions"]][:4], rows[0]["tokens"][:4])
        r = L.flag_check("vllm", PY, ["/v/bin/vllm", "serve"] + self.ARGV, fake_server=FAKE_VL)
        self.assertTrue(r["ok"], r)


if __name__ == "__main__":
    unittest.main()
