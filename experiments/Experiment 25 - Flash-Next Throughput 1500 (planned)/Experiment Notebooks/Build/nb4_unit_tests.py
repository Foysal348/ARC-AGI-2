"""Quick local unit tests of E25 NB4 TICK: nb4_lib.py (arm planning, budgets, labels, the winner rule, the engaged /
mechanism checks, the read-back, the one OOM policy, the core timer plugin) and a CELL 8 flow test that executes the real
CELL 8 source of nb4_code_export.py with a stub boot runner (no GPU, no server, no end-to-end).

Run: python nb4_unit_tests.py        (stdlib unittest; the real-data tests skip when the NB1 R2 GPU output is absent)
"""
import json
import multiprocessing
import os
import re
import sys
import tempfile
import time
import traceback
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nb4_lib as NL  # noqa: E402

NB1R2 = HERE.parents[3] / "_scratch" / "e25_gpu_out" / "q27-xhigh-e25-nb1-capacity-r2" / "nb1"
EXPORT = HERE / "nb4_code_export.py"


def W(tok, status="STEADY", eq=True, pre=0, split="BALANCED", tick=48.0):
    return {"status": status, "tok_s": tok, "running_eq_level": eq, "preemptions": pre, "tick_ms": tick,
            "split": {"label": split}}


def wins(v8=250.0, v40=245.0, v89=270.0, v409=265.0, **kw):
    return {"8k@8": W(v8, **kw), "40k@8": W(v40, **kw), "8k@9": W(v89, **kw), "40k@9": W(v409, **kw)}


def cg_text(rows):
    """--cudagraph-metrics table text in vLLM's format (compilation/cuda_graph.py)."""
    out = ["**CUDAGraph Stats:**", "", "| Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count |",
           "|-----------------|---------------|--------------|--------------|-------|"]
    for u, p, m, c in rows:
        out.append(f"| {u} | {p} | {p - u} | {m} | {c} |")
    return "\n".join(out) + "\n"


class Arms(unittest.TestCase):
    def test_default_arms_valid_and_order(self):
        a = NL.make_arms()
        self.assertEqual(NL.validate_arms(a), [])
        self.assertEqual(list(NL.ARM_ORDER), ["T0", "S", "S2", "S3", "G", "P", "T0b", "K4", "INT8"])
        self.assertFalse(a["P"]["integrated"])
        self.assertFalse(a["S2"]["integrated"] or a["S3"]["integrated"])
        self.assertEqual((a["S2"]["variant"], a["S3"]["variant"]), ("Q-pack", "Q-defer"))
        self.assertTrue(all(a[k]["integrated"] for k in ("T0", "S", "G", "T0b", "K4", "INT8")))
        for k, x in a.items():
            self.assertEqual(NL.never_do(x), [], k)
            self.assertEqual(len(x["dims"]), 1, k)

    def test_patch_arms_filled_by_install_result(self):
        a = NL.make_arms(s2={"ok": True, "env": {"NB4_PPACK": "1"}, "init_rx": "NB4_PPQ"},
                         p={"ok": True, "env": {"NB4_P5": "1", "NB4_P5_MAX_BYTES": "184320"}, "init_rx": "x", "counters_rpc": "nb4_p5_stats"})
        self.assertTrue(a["P"]["integrated"] and a["S2"]["integrated"])
        self.assertEqual(a["P"]["counters_rpc"], "nb4_p5_stats")
        self.assertEqual(NL.validate_arms(a), [])
        b = NL.make_arms(p={"ok": False, "why": "gate FAIL"})
        self.assertFalse(b["P"]["integrated"])
        self.assertIn("P5_NOT_INTEGRATED", b["P"]["why_not"])

    def test_validate_catches_bad_arms(self):
        a = NL.make_arms()
        a["S"]["dims"] = ["max_connections", "captures"]
        self.assertTrue(any("2 change dimensions" in x for x in NL.validate_arms(a)))
        a = NL.make_arms()
        a["S"]["env"]["NCCL_GRAPH_MIXING_SUPPORT"] = "0"
        self.assertTrue(any("never-do" in x for x in NL.validate_arms(a)))
        a = NL.make_arms()
        a["G"]["extra"] = ["--no-async-scheduling"]
        self.assertTrue(any("never-do" in x for x in NL.validate_arms(a)))
        a = NL.make_arms()
        a["G"]["extra"] = ["--max-num-seqs", "16"]
        self.assertTrue(any("changes a base setting" in x for x in NL.validate_arms(a)))
        a = NL.make_arms()
        a["G"]["env"] = {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}            # two kinds of settings in one arm
        self.assertTrue(any("G: changes" in x for x in NL.validate_arms(a)))
        a = NL.make_arms()
        a["T0b"]["env"] = {"X": "1"}
        self.assertTrue(any("T0b" in x for x in NL.validate_arms(a)))
        for env in ({"NCCL_PROTO": "LL128"}, {"COUNTERS": "1"}, {"NCCL_GRAPH_REGISTER": "1"}, {"P2P": "on"},
                    {"NCCL_SHM_USE_CUDA_MEMCPY": "1"}, {"NCCL_MIN_CTAS": "8"}, {"VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "1"}):
            self.assertTrue(NL.never_do({"env": env}), env)
        self.assertTrue(NL.never_do({"extra": ["--mamba-ssm-cache-dtype", "bfloat16"]}))
        self.assertTrue(NL.never_do({"extra": ["--no-enable-prefix-caching"]}))
        self.assertTrue(NL.never_do({"k": 2}))

    def test_flag_set_identical_on_every_arm(self):
        a = NL.make_arms()
        ext = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt"]
        sets = {k: NL.arm_extra(x, ext, "/tmp/p") for k, x in a.items()}
        for k, v in sets.items():
            want = sets["T0b"] + (["--enable-logging-iteration-details"] if k == "T0" else [])
            self.assertEqual(v, want, k)
        self.assertIn("--cudagraph-metrics", sets["T0b"])
        self.assertIn("--profiler-config", sets["T0b"])
        cfg = json.loads(sets["T0b"][sets["T0b"].index("--profiler-config") + 1])
        self.assertEqual(cfg["torch_profiler_activities"], ["CPU", "CUDA"])
        self.assertFalse(cfg["torch_profiler_record_shapes"] or cfg["torch_profiler_with_stack"] or cfg["torch_profiler_with_memory"])
        e = NL.arm_env(a["T0"], {"VLLM_SERVER_DEV_MODE": "1"}, "/tmp/t")
        self.assertEqual(e["NB4_CORE_TIMER"], "/tmp/t")
        self.assertEqual(e["VLLM_CUSTOM_SCOPES_FOR_PROFILING"], "1")
        self.assertEqual(e["VLLM_RECOVERSSM_COMMIT_GRAPH"], "graph")
        self.assertNotIn("NB4_CORE_TIMER", NL.arm_env(a["S"], {}, "/tmp/t"))
        self.assertEqual(NL.arm_env(a["S"], {}, None)["CUDA_DEVICE_MAX_CONNECTIONS"], "32")

    def test_compile_pins_ports(self):
        a = NL.make_arms()
        self.assertEqual(json.loads(NL.compile_json(a["T0b"]))["cudagraph_capture_sizes"], NL.CAPS16)
        g = json.loads(NL.compile_json(a["G"]))
        self.assertEqual(g["cudagraph_capture_sizes"], [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 28, 32, 36])
        self.assertEqual(g["cudagraph_mode"], "FULL_AND_PIECEWISE")
        self.assertEqual(NL.compile_json(a["T0b"]), '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,6,8,12,16,24,32,40,48,64]}')
        self.assertEqual([NL.pin_for(a["G"], i) for i in range(4)], [3553580524, 3553580524 - 268435456, 2479838700, 2479838700])
        self.assertEqual(NL.pin_for(a["T0"], 2), 3553580524)
        ports = {NL.port_for(k, i) for k in NL.ARM_ORDER for i in range(4)}
        self.assertEqual(len(ports), len(NL.ARM_ORDER) * 4)
        self.assertTrue(all(8000 <= p <= 8355 for p in ports))
        self.assertEqual(NL.launch_name("T0b", 0), "nb4_t0b")
        self.assertEqual(NL.launch_name("G", 2), "nb4_g_t2")
        self.assertEqual(NL.spec_json(4), '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}')


class OomPolicy(unittest.TestCase):
    def setUp(self):
        self.a = NL.make_arms()

    def rd(self, arm, **kw):
        d = dict(status="LAUNCH_FAILED", oom_in_log=False, xid_in_log=False, launch_sec=200, tries=1, rung=0, left_sec=9000,
                 need_sec=1800, log_tail="")
        d.update(kw)
        return NL.retry_decision(self.a[arm], **d)

    def test_policy(self):
        self.assertEqual(self.rd("G", oom_in_log=True)[:2], (True, 1))
        self.assertEqual(self.rd("G", oom_in_log=True, rung=2)[0], False)
        self.assertEqual(self.rd("T0", oom_in_log=True)[0], False)               # a base arm never changes its pin
        self.assertEqual(self.rd("G", xid_in_log=True, oom_in_log=True)[0], False)
        self.assertEqual(self.rd("S")[:3], (True, 0, False))                     # fast non-OOM failure: one retry
        self.assertEqual(self.rd("S", tries=2)[0], False)
        self.assertEqual(self.rd("S", launch_sec=1600)[0], False)                # a readiness timeout is not repeated
        self.assertEqual(self.rd("S", left_sec=100)[0], False)
        self.assertEqual(self.rd("S", status="SERVER_DIED")[0], False)
        r = self.rd("T0", log_tail="... recoverssm-commit capture failed ...", launch_sec=900)
        self.assertEqual(r[:3], (True, 0, True))

    def test_death_labels(self):
        self.assertEqual(NL.death_label(True, True, True, "tf_2k"), "BOOT_CAP_KILL")
        self.assertEqual(NL.death_label(None, True, False, "tf_2k-8k-40k"), "TF_OOM")
        self.assertEqual(NL.death_label(None, True, False, "window_8k@9"), "BODY_OOM_window_8k@9")
        self.assertEqual(NL.death_label(None, False, True, "profile_8k@8"), "DEVICE_FAULT_IN_profile_8k@8")

    def test_free_memory(self):
        smi = NL.free_by_card([[[0, 21000, 23034], [1, 21500, 23034]], [[0, 22000, 23034], [1, 21000, 23034]]])
        self.assertEqual(smi, {0: 1034.0, 1: 1534.0})
        wf = NL.worker_free([{"ranks": [{"rank": 0, "mem": {"free_mib": 639}}, {"rank": 1, "mem": {"free_mib": 2000}}]}])
        f, src = NL.combine_free(smi, wf)
        self.assertEqual(f, {0: 639, 1: 1534.0})
        f2, src2 = NL.combine_free({3: 925.0}, {})
        self.assertEqual(f2, {3: -99.0})
        self.assertIn("no worker", src2[3])


class Readback(unittest.TestCase):
    LINE = None

    @classmethod
    def setUpClass(cls):
        p = NB1R2 / "logs" / "server_nb1_b3_fp8_pin100.log"
        cls.LINE = next((l for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if "non-default args:" in l), None) \
            if p.is_file() else None

    def nd_line(self):
        if not self.LINE:
            self.skipTest("NB1 R2 B3 server log absent")
        l = self.LINE.replace("'max_num_seqs': 8", "'max_num_seqs': 9")
        l = l.replace("'cudagraph_capture_sizes': [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]",
                      "'cudagraph_capture_sizes': [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64]")
        return l.replace("'max_num_batched_tokens': 1024", "'max_num_batched_tokens': 1024, 'cudagraph_metrics': True")

    def test_real_b3_line_parsed(self):
        if not self.LINE:
            self.skipTest("NB1 R2 B3 server log absent")
        nd = NL.parse_nondefault(self.LINE)
        self.assertEqual(nd["kv_cache_dtype"], "fp8_e4m3")
        self.assertEqual(nd["kv_cache_memory_bytes"], "3553580524")
        self.assertEqual(nd["max_num_seqs"], "8")
        self.assertEqual(nd["num_speculative_tokens"], "3")
        self.assertEqual(nd["spec_method"], "mtp")
        self.assertEqual(nd["language_model_only"], "True")
        self.assertEqual(nd["capture_sizes"], [1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
        self.assertEqual(nd["cudagraph_mode"], "FULL_AND_PIECEWISE")
        self.assertEqual(nd["profiler"], "torch")
        ck = NL.readback_checks(nd, {}, NL.make_arms()["T0b"], NL.B3_PIN)
        self.assertFalse(ck["seqs"] or ck["captures"] or ck["cudagraph_metrics"])        # B3 = seqs 8, captures to 32
        self.assertTrue(ck["kv_dtype"] and ck["pin"] and ck["mtp_k"] and ck["profiler"])

    def test_base_line_passes_and_arm_changes_are_checked(self):
        nd = NL.parse_nondefault(self.nd_line())
        a = NL.make_arms()
        env = {"VLLM_SERVER_DEV_MODE": "1", "VLLM_CUSTOM_SCOPES_FOR_PROFILING": "1"}
        ck = NL.readback_checks(nd, env, a["T0b"], NL.B3_PIN)
        self.assertTrue(all(ck.values()), {k: v for k, v in ck.items() if not v})
        ck = NL.readback_checks(nd, dict(env, CUDA_DEVICE_MAX_CONNECTIONS="32"), a["S"], NL.B3_PIN)
        self.assertTrue(all(ck.values()))
        self.assertFalse(NL.readback_checks(nd, env, a["S"], NL.B3_PIN)["env CUDA_DEVICE_MAX_CONNECTIONS"])
        self.assertFalse(NL.readback_checks(nd, dict(env, CUDA_DEVICE_MAX_CONNECTIONS="32"), a["T0b"], NL.B3_PIN)
                         ["env CUDA_DEVICE_MAX_CONNECTIONS absent"])                         # an unplanned change is caught
        self.assertFalse(NL.readback_checks(nd, env, a["G"], NL.B3_PIN)["captures"])         # G's list not in this launch
        self.assertFalse(NL.readback_checks(nd, env, a["T0"], NL.B3_PIN)["iteration_details"])
        self.assertFalse(NL.readback_checks(nd, env, a["T0b"], NL.PIN_LADDER[1])["pin"])     # the launch's own pin is read back

    def test_one_change_vs_ref(self):
        a = NL.make_arms()
        base = ["serve", "--port", "8040", "--max-num-seqs", "9", "--compilation-config", NL.compile_json(a["T0"]),
                "--enable-logging-iteration-details", "--kv-cache-memory-bytes", str(NL.B3_PIN)]
        benv = {"VLLM_RECOVERSSM_COMMIT_GRAPH": "graph", "VLLM_PORT": "34000", "NB4_CORE_TIMER": "/tmp/t", "KV_BYTES": str(NL.B3_PIN)}
        s_argv = [x for x in base if x != "--enable-logging-iteration-details"]
        s_argv[2] = "8060"
        s_env = {"CUDA_DEVICE_MAX_CONNECTIONS": "32", "VLLM_PORT": "36000", "KV_BYTES": str(NL.B3_PIN)}
        d = NL.diff_vs_ref(base, benv, s_argv, s_env)
        self.assertTrue(NL.one_change_ok(d, a["S"], a["T0"])["ok"], d)
        bad = NL.diff_vs_ref(base, benv, s_argv + ["--max-num-batched-tokens", "2048"], s_env)
        self.assertEqual(NL.one_change_ok(bad, a["S"], a["T0"])["unexpected_flags"], ["--max-num-batched-tokens"])
        g_argv = list(s_argv)
        g_argv[g_argv.index("--compilation-config") + 1] = NL.compile_json(a["G"])
        g_argv[g_argv.index("--kv-cache-memory-bytes") + 1] = str(NL.PIN_LADDER[1])
        gd = NL.diff_vs_ref(base, benv, g_argv, {"KV_BYTES": str(NL.PIN_LADDER[1])})
        self.assertFalse(NL.one_change_ok(gd, a["G"], a["T0"])["ok"])                       # a pin step must be declared
        self.assertTrue(NL.one_change_ok(gd, a["G"], a["T0"], pin_differs=True)["ok"])
        i8 = NL.diff_vs_ref(base, benv, s_argv + ["--speculative-config", "x"], {"KV_BYTES": str(NL.B3_PIN)})
        self.assertTrue(NL.one_change_ok(i8, a["INT8"], a["T0"])["ok"])                     # DRAFT_HEAD reaches the spec config

    def test_d8_switches(self):
        d = NL.d8_switches({"VLLM_E2_GUARD_MODE": "warn", "NCCL_P2P_DISABLE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        self.assertEqual(d["VLLM_E2_GUARD_MODE"], "warn")
        self.assertEqual(d["NCCL_P2P_DISABLE"], "1")
        self.assertIsNone(d["VLLM_GDN_DECODE_FASTPATH"])


class CudagraphSplit(unittest.TestCase):
    def test_table_and_split(self):
        txt = "INFO x\n" + cg_text([(16, 16, "FULL", 100), (20, 24, "FULL", 90), (36, 36, "NONE", 80), (1024, 1024, "NONE", 3)])
        t = NL.cudagraph_table(txt)
        self.assertEqual(t["decode_by_n"]["4"]["FULL"], 100)
        self.assertEqual(t["decode_by_n"]["5"]["padded"], {"24": 90})
        self.assertEqual(t["decode_by_n"]["9"]["NONE"], 80)
        self.assertEqual(t["non_decode_steps"], 3)
        self.assertEqual(NL.split_label(NL.cudagraph_table(cg_text([(16, 16, "FULL", 100)])), 8)["label"], "BALANCED")
        self.assertEqual(NL.split_label(NL.cudagraph_table(cg_text([(32, 32, "FULL", 100)])), 8)["label"], "IMBALANCED")   # 8/0 parity lock
        sp = NL.split_label(NL.cudagraph_table(cg_text([(16, 16, "FULL", 50), (20, 24, "FULL", 50)])), 9)
        self.assertEqual(sp["label"], "BALANCED")
        self.assertEqual(NL.split_label(NL.cudagraph_table(""), 9)["label"], "UNKNOWN")
        self.assertEqual(NL.cudagraph_table(cg_text([(20, 20, "FULL", 7)]), k=4)["decode_by_n"]["4"]["FULL"], 7)


class Engaged(unittest.TestCase):
    def setUp(self):
        self.a = NL.make_arms(s2={"ok": True, "env": {"NB4_PPACK": "1"}, "init_rx": r'NB4_PPQ \{.*"event":\s*"init"'},
                              p={"ok": True, "env": {"NB4_P5": "1"}, "init_rx": r'NB4_P5 \{.*"event": "init"', "counters_rpc": "nb4_p5_stats"})

    def test_engaged_log(self):
        self.assertTrue(NL.engaged_log(self.a["T0"], "", {}, None, {"iteration_details": True})["ok"])
        self.assertFalse(NL.engaged_log(self.a["T0"], "", {}, None, {"iteration_details": False})["ok"])
        s_ok = NL.engaged_log(self.a["S"], "", {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}, {"0": "32", "1": "32", "2": "32", "3": "32"})
        self.assertTrue(s_ok["ok"])
        self.assertFalse(NL.engaged_log(self.a["S"], "", {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}, {"0": "32", "1": None})["ok"])
        self.assertFalse(NL.engaged_log(self.a["S"], "", {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}, {})["ok"])
        self.assertTrue(NL.engaged_log(self.a["G"], "", {}, None, {"captures": True})["ok"])
        self.assertTrue(NL.engaged_log(self.a["P"], 'x NB4_P5 {"event": "init", "ok": true}', {"NB4_P5": "1"})["ok"])
        self.assertFalse(NL.engaged_log(self.a["P"], "no init", {"NB4_P5": "1"})["ok"])
        self.assertTrue(NL.engaged_log(self.a["S2"], 'NB4_PPQ {"event":"init"}', {"NB4_PPACK": "1"})["ok"])
        self.assertTrue(NL.engaged_log(self.a["INT8"], "... INT8 weight-only copy of the draft head ...")["ok"])
        self.assertFalse(NL.engaged_log(self.a["INT8"], "nothing")["ok"])
        self.assertTrue(NL.engaged_log(self.a["T0b"], "")["ok"])

    def test_commit_split(self):
        b = [{"mode": "graph", "replays": 0, "staged_steps": 0, "eager_steps": 5, "fallbacks": {"warmup_metadata_type": 18}},
             {"mode": "graph", "replays": 10, "staged_steps": 1, "eager_steps": 2, "fallbacks": {"not_captured": 1}}]
        a = [{"mode": "graph", "replays": 0, "staged_steps": 0, "eager_steps": 105, "fallbacks": {"warmup_metadata_type": 18, "pp_deferred": 100}},
             {"mode": "graph", "replays": 110, "staged_steps": 5, "eager_steps": 2, "fallbacks": {"not_captured": 5}}]
        cs = NL.commit_split(b, a, steps=100)
        self.assertEqual(cs["total"], {"replays": 100, "staged_steps": 4, "eager_steps": 100})
        self.assertEqual(cs["ranks"][0]["fallbacks_delta"], {"pp_deferred": 100})
        self.assertEqual(cs["ranks"][1]["per_step"]["replays"], 1.0)
        m = NL.mechanism({"mechanism": "commit_replays"}, {"commit": cs, "log": "[recoverssm-commit] captured commit graph n=4"})
        self.assertTrue(m["ok"])
        self.assertEqual(m["captured_n"], [4])
        self.assertFalse(NL.mechanism({"mechanism": "commit_replays"}, {"commit": NL.commit_split(b, b)})["ok"])

    def test_mechanisms(self):
        tl_bad = {"stalls": {"steady_ticks": 36, "ticks_with_stall_over_1ms": 36, "graph_stall_ms_mean": 21.2}, "summary": {"tick_ms_mean": 49.7}}
        tl_good = {"stalls": {"steady_ticks": 40, "ticks_with_stall_over_1ms": 1, "graph_stall_ms_mean": 0.3}, "summary": {"tick_ms_mean": 41.0}}
        self.assertFalse(NL.mechanism(self.a["S"], {"tl": tl_bad, "tl_ref": tl_bad})["ok"])
        self.assertTrue(NL.mechanism(self.a["S"], {"tl": tl_good, "tl_ref": tl_bad})["ok"])
        self.assertFalse(NL.mechanism(self.a["S"], {})["ok"])                                  # no trace = not proven
        self.assertFalse(NL.mechanism(self.a["S2"], {"tl": tl_good, "ppack_label": ["NOT_ENGAGED", {}]})["ok"])
        self.assertTrue(NL.mechanism(self.a["S2"], {"tl": tl_good, "ppack_label": ["ENGAGED", {}]})["ok"])
        cg9 = {"8k@9": NL.cudagraph_table(cg_text([(36, 36, "FULL", 50), (16, 16, "FULL", 50)])),
               "40k@9": NL.cudagraph_table(cg_text([(20, 20, "FULL", 80)]))}
        self.assertTrue(NL.mechanism(self.a["G"], {"cg": cg9})["ok"])
        cg9b = {"8k@9": NL.cudagraph_table(cg_text([(36, 36, "NONE", 50)])), "40k@9": NL.cudagraph_table(cg_text([(20, 24, "FULL", 80)]))}
        self.assertFalse(NL.mechanism(self.a["G"], {"cg": cg9b})["ok"])
        self.assertTrue(NL.mechanism(self.a["P"], {"p5_label": ["ENGAGED", []], "p5_delta": {"executed": 5000}})["ok"])
        self.assertFalse(NL.mechanism(self.a["P"], {"p5_label": ["NOT_INTEGRATED", ["inits_ok 2 < 4"]], "p5_delta": {"executed": 5}})["ok"])
        self.assertFalse(NL.mechanism(self.a["P"], {"p5_label": ["ENGAGED", []], "p5_delta": {"executed": 0}})["ok"])
        self.assertTrue(NL.mechanism(self.a["T0"], {"tl": {"ranks": [0, 1, 2, 3], "summary": {"steady_ticks": 30}}})["ok"])
        self.assertFalse(NL.mechanism(self.a["T0"], {"tl": {"ranks": [0, 1], "summary": {}}})["ok"])
        self.assertTrue(NL.mechanism(self.a["K4"], {"accept": {0: 0.8, 1: 0.6, 2: 0.5, 3: 0.4}})["ok"])
        self.assertFalse(NL.mechanism(self.a["K4"], {"accept": {0: 0.8, 1: 0.6, 2: 0.5}})["ok"])
        self.assertTrue(NL.mechanism(self.a["T0b"], {})["ok"])


class Correctness(unittest.TestCase):
    def test_tf_label(self):
        ok = {"status": "MEASURED", "n": 100, "expected": 100, "mean_abs_dlogprob": 0.01, "rank1_agreement": 0.99}
        self.assertEqual(NL.tf_label(ok), "TF_CLOSE")
        self.assertEqual(NL.tf_label(dict(ok, rank1_agreement=0.749)), "TF_DIVERGENT")             # NB0's FP8-vs-BF16 40k
        self.assertEqual(NL.tf_label(dict(ok, mean_abs_dlogprob=1.88, rank1_agreement=0.367)), "TF_DIVERGENT_SEVERE")
        self.assertEqual(NL.tf_label(dict(ok, status="PARTIAL")), "INCONCLUSIVE_PARTIAL")
        self.assertEqual(NL.tf_label(dict(ok, n=99)), "INCONCLUSIVE_INCOMPLETE_POSITIONS")
        self.assertEqual(NL.tf_label(dict(ok, mean_abs_dlogprob=float("nan"))), "INCONCLUSIVE_NONFINITE")
        self.assertEqual(NL.tf_label(None), "INCONCLUSIVE_MISSING")

    def test_correctness_gate(self):
        close = {"2k": "TF_CLOSE", "8k": "TF_CLOSE", "40k": "TF_CLOSE"}
        self.assertEqual(NL.correctness("CLEAN", "EXACT", close)["verdict"], "PASS")
        self.assertEqual(NL.correctness("CLEAN", "NEAR_TIE", close)["verdict"], "PASS")
        self.assertEqual(NL.correctness("CLEAN", "NEAR_TIE", close, need_exact=True)["verdict"], "INCONCLUSIVE")
        self.assertEqual(NL.correctness("CLEAN", "FAIL_HIGH_MARGIN", close)["verdict"], "FAIL")
        self.assertEqual(NL.correctness("NO_GO_ORACLE", "EXACT", close)["verdict"], "FAIL")
        self.assertEqual(NL.correctness("CLEAN", "EXACT", dict(close, **{"40k": "TF_DIVERGENT"}))["verdict"], "FAIL")
        self.assertEqual(NL.correctness("CLEAN", "EXACT", dict(close, **{"40k": "INCONCLUSIVE_MISSING_TF"}))["verdict"], "INCONCLUSIVE")
        self.assertEqual(NL.correctness("INCONCLUSIVE_ORACLE", "INCONCLUSIVE", close)["verdict"], "INCONCLUSIVE")


class Speed(unittest.TestCase):
    def test_aa_spread(self):
        t0 = wins(250, 245, 270, 265)
        t0b = dict(wins(252, 246, 268, 266), **{"8k@8#2": W(247), "40k@8#2": W(246), "8k@9#2": W(269), "40k@9#2": W(266)})
        aa = NL.aa_spread(t0, t0b)
        self.assertAlmostEqual(aa["8k@8"]["t0b"], 249.5)
        self.assertGreaterEqual(aa["8k@8"]["spread_pct"], aa["8k@8"]["within_t0b_pct"])
        self.assertIsNone(NL.aa_spread(t0, {})["8k@8"])

    def test_winner_rule(self):
        t0, t0b = wins(), wins(251, 246, 271, 266)
        self.assertEqual(NL.speed_verdict(wins(300, 290, 320, 315), t0, t0b)["label"], "WIN")
        self.assertEqual(NL.speed_verdict(wins(254, 249, 275, 270), t0, t0b)["label"], "NO_GAIN")       # < 2 %
        self.assertEqual(NL.speed_verdict(wins(300, 290, 320, 200), t0, t0b)["label"], "SLOWER")
        sv = NL.speed_verdict(wins(300, 290, 320, 315), t0, {})
        self.assertEqual(sv["label"], "INCONCLUSIVE_NO_AA")
        self.assertEqual(sv["windows"]["8k@8"]["provisional_vs_2pct"], "WIN")
        noisy = wins(270, 260, 290, 285)                                               # A/A spread 8 %: a 5 % gain is noise
        self.assertEqual(NL.speed_verdict(wins(262, 257, 283, 278), t0, noisy)["label"], "NO_GAIN")
        split = wins(300, 290, 320, 315, split="IMBALANCED")
        self.assertEqual(NL.speed_verdict(split, t0, t0b)["label"], "INCONCLUSIVE_SPEED")
        bad = dict(wins(300, 290, 320, 315), **{"40k@9": W(320, status="INCONCLUSIVE_PREEMPTION")})
        self.assertEqual(NL.speed_verdict(bad, t0, t0b)["label"], "INCONCLUSIVE_SPEED")
        g = wins(250, 245, 300, 295)                                                   # G: only the level-9 windows decide
        self.assertEqual(NL.speed_verdict(g, t0, t0b, decisive=("8k@9", "40k@9"))["label"], "WIN")
        self.assertEqual(NL.speed_verdict(g, t0, t0b)["label"], "NO_GAIN")
        self.assertFalse(NL.window_ok(W(250, eq=False)) or NL.window_ok(W(250, pre=2)))

    def test_c_verdict(self):
        eager = wins()
        graph = {"8k@8": W(251), "40k@8": W(246)}
        commit = NL.commit_split([{"replays": 0}], [{"replays": 400}])
        c = NL.c_verdict(graph, eager, None, {"status": "EXACT"}, commit, "CLEAN")
        self.assertEqual(c["label"], "NO_GAIN")
        self.assertEqual(NL.c_verdict({"8k@8": W(280), "40k@8": W(275)}, eager, None, {"status": "EXACT"}, commit, "CLEAN")["label"], "WIN_SPEED")
        self.assertEqual(NL.c_verdict(graph, eager, None, {"status": "FAIL_HIGH_MARGIN"}, commit, "CLEAN")["label"], "CORRECTNESS_FAIL")
        self.assertEqual(NL.c_verdict(graph, eager, None, {"status": "EXACT"}, NL.commit_split([{}], [{}]), "CLEAN")["label"], "NOT_ENGAGED")
        self.assertEqual(NL.c_verdict(graph, eager, None, {"status": "NEAR_TIE"}, commit, "CLEAN")["label"], "CORRECTNESS_INCONCLUSIVE")


class Labels(unittest.TestCase):
    def setUp(self):
        self.a = NL.make_arms(p={"ok": True, "env": {"NB4_P5": "1"}, "init_rx": "x"})
        self.ok = {"status": "DONE", "correctness": "CLEAN"}
        self.pass_ = {"verdict": "PASS"}
        self.win = {"label": "WIN"}
        self.mech = {"ok": True}

    def test_precedence(self):
        A, ok = self.a, self.ok
        self.assertEqual(NL.final_label(NL.make_arms()["P"], None)["label"], "P5_NOT_INTEGRATED")
        self.assertEqual(NL.final_label(NL.make_arms()["S2"], None)["label"], "NOT_INTEGRATED")
        self.assertEqual(NL.final_label(A["S"], {"status": "SKIPPED_BUDGET"})["label"], "SKIPPED_BUDGET")
        self.assertEqual(NL.final_label(A["S"], {"status": "SKIPPED_NO_BASE"})["label"], "SKIPPED_NO_BASE")
        self.assertEqual(NL.final_label(A["S"], None)["label"], "SKIPPED_BUDGET")
        self.assertEqual(NL.final_label(A["G"], {"status": "LAUNCH_FAILED_OOM"})["label"], "BOOT_FAILED")
        self.assertEqual(NL.final_label(A["P"], {"status": "P5_NOT_INTEGRATED_INIT_FAILED"})["label"], "P5_NOT_INTEGRATED")
        self.assertEqual(NL.final_label(A["S"], {"status": "CONFIG_MISMATCH", "early_stop": "CONFIG_MISMATCH"})["label"], "CONFIG_MISMATCH")
        self.assertEqual(NL.final_label(A["S"], dict(ok, early_stop="NOT_ENGAGED"))["label"], "NOT_ENGAGED")
        self.assertEqual(NL.final_label(A["S"], ok, self.pass_, self.win, {"ok": False, "evidence": {}}, self.mech)["label"], "NOT_ENGAGED")
        self.assertEqual(NL.final_label(A["S"], ok, {"verdict": "FAIL"}, self.win, None, self.mech)["label"], "CORRECTNESS_FAIL")
        self.assertEqual(NL.final_label(A["S"], ok, self.pass_, self.win, None, {"ok": False, "why": "x"})["label"], "NOT_ENGAGED")
        self.assertEqual(NL.final_label(A["S"], ok, {"verdict": "INCONCLUSIVE"}, self.win, None, self.mech)["label"], "CORRECTNESS_INCONCLUSIVE")
        w = NL.final_label(A["S"], ok, self.pass_, self.win, None, self.mech)
        self.assertEqual(w["label"], "WIN_SPEED")
        self.assertIn("paired ARC accuracy", w["why"])
        self.assertEqual(NL.final_label(A["P"], ok, self.pass_, self.win, None, self.mech)["label"], "WIN_DIAGNOSTIC")
        self.assertEqual(NL.final_label(A["G"], ok, self.pass_, {"label": "NO_GAIN"}, None, self.mech)["label"], "NO_GAIN")
        self.assertEqual(NL.final_label(A["T0"], dict(ok))["label"], "BASE")
        self.assertEqual(NL.final_label(A["T0"], {"status": "SERVER_DIED", "death": "BODY_OOM_window_40k@9"})["label"], "BASE_FAILED")
        self.assertEqual(NL.final_label(A["T0b"], dict(ok))["label"], "AA")
        self.assertEqual(NL.final_label(A["T0b"], dict(ok, early_stop="SHORT_CORRECTNESS_GATE_FAILED"))["label"], "AA_FAILED")


class Planner(unittest.TestCase):
    BASE = {"status": "DONE", "correctness": "CLEAN", "reference_stored": True}

    def arms(self, **kw):
        a = NL.make_arms(s2={"ok": True, "env": {"NB4_PPACK": "1"}}, p={"ok": True, "env": {"NB4_P5": "1"}},
                         s3={"ok": True, "env": {"NB4_PPDEFER": "1"}})
        for k, v in kw.items():
            a[k]["integrated"] = v
        return a

    def test_full_order_with_budget(self):
        a, done, seq, left = self.arms(), {}, [], 400 * 60
        while True:
            k, why = NL.next_arm(a, done, left)
            if k is None:
                break
            seq.append(k)
            done[k] = dict(self.BASE)
            left -= NL.EST_MIN[k] * 60
        self.assertEqual(seq, ["T0", "S", "S2", "S3", "G", "P", "T0b", "K4", "INT8"])

    def test_t0b_protected_and_optional_skipped(self):
        a, done, seq, left = self.arms(), {}, [], 150 * 60
        while True:
            k, why = NL.next_arm(a, done, left)
            if k is None:
                break
            seq.append(k)
            done[k] = dict(self.BASE)
            left -= NL.EST_MIN[k] * 60
        self.assertIn("T0b", seq)
        self.assertNotIn("K4", seq)
        self.assertLess(seq.index("T0b"), len(seq))
        self.assertEqual(seq[:2], ["T0", "S"])
        self.assertTrue(all(left_ok for left_ok in [True]))

    def test_t0b_moves_before_p(self):
        a = self.arms()
        done = {k: dict(self.BASE) for k in ("T0", "S", "S2", "S3", "G")}
        k, why = NL.next_arm(a, done, 50 * 60)
        self.assertEqual(k, "T0b")
        self.assertIn("protected", why)
        self.assertEqual(NL.next_arm(a, done, 90 * 60)[0], "P")

    def test_bad_base_paths(self):
        a = self.arms()
        self.assertEqual(NL.next_arm(a, {"T0": {"status": "LAUNCH_FAILED"}}, 9000)[0], "T0b")
        k, why = NL.next_arm(a, {"T0": {"status": "LAUNCH_FAILED"}, "T0b": {"status": "LAUNCH_FAILED"}}, 9000)
        self.assertIsNone(k)
        self.assertIn("no clean base", why)
        done = {"T0": {"status": "LAUNCH_FAILED"}, "T0b": dict(self.BASE)}
        self.assertEqual(NL.next_arm(a, done, 9000)[0], "S")
        self.assertEqual(NL.ref_arm(done), "T0b")
        self.assertEqual(NL.ref_arm({"T0": dict(self.BASE), "T0b": dict(self.BASE)}), "T0")

    def test_not_integrated_never_booted(self):
        a = self.arms(S2=False, S3=False, P=False, G=False)
        done, seq = {}, []
        while True:
            k, _ = NL.next_arm(a, done, 9000)
            if k is None:
                break
            seq.append(k)
            done[k] = dict(self.BASE)
        self.assertEqual(seq, ["T0", "S", "T0b", "K4", "INT8"])


class EngineCoreTimer(unittest.TestCase):
    def test_plugin_dormant_and_active(self):
        with tempfile.TemporaryDirectory() as d:
            mod = types.ModuleType("nb4_core_timer_t")
            exec(compile(NL.CORE_TIMER_SRC, "nb4_core_timer.py", "exec"), mod.__dict__)
            os.environ.pop("NB4_CORE_TIMER", None)

            class Sched:
                def schedule(self):
                    time.sleep(0.001)
                    return "so"

                def update_from_output(self, so, out):
                    return {0: "eco"}

            class EngineCore:
                def __init__(self):
                    self.scheduler, self.batch_queue = Sched(), []

                def step_with_batch_queue(self):
                    so = self.scheduler.schedule()
                    return self.scheduler.update_from_output(so, None), True

                def step(self):
                    return {}, False

            fake = types.ModuleType("vllm.v1.engine.core")
            fake.EngineCore = EngineCore
            sys.modules["vllm.v1.engine.core"] = fake
            try:
                mod.register()                                         # dormant: no env
                self.assertFalse(getattr(EngineCore, "_nb4_timer", False))
                os.environ["NB4_CORE_TIMER"] = str(Path(d) / "t")
                mod.register()                                         # not the EngineCore process: still dormant
                self.assertFalse(getattr(EngineCore, "_nb4_timer", False))
                cp = multiprocessing.current_process()
                old = cp.name
                cp.name = "EngineCore"
                try:
                    mod.register()
                finally:
                    cp.name = old
                self.assertTrue(EngineCore._nb4_timer)
                ec = EngineCore()
                for _ in range(5):
                    ec.step_with_batch_queue()
                ec.step()
                mod._flush()
                files = list(Path(d).glob("t.*.jsonl"))
                self.assertEqual(len(files), 1)
                rows = [json.loads(l) for l in files[0].read_text().splitlines()]
                self.assertEqual(len(rows), 6)
                self.assertGreater(rows[0][3], 0)                       # schedule time measured
                s = NL.core_timer_summary(rows)
                self.assertEqual(s["rows"], 6)
                self.assertEqual(s["kinds"], ["batch_queue", "step"])
            finally:
                os.environ.pop("NB4_CORE_TIMER", None)
                sys.modules.pop("vllm.v1.engine.core", None)
        self.assertEqual(NL.CORE_TIMER_DIST["nb4_core_timer-0.1.dist-info/entry_points.txt"].split("\n")[0], "[vllm.general_plugins]")


class T0Extras(unittest.TestCase):
    def test_iteration_summary(self):
        txt = "\n".join(f"INFO Iteration({i}): 0 context requests, 0 context tokens, {n} generation requests, {4 * n} generation tokens, "
                        f"iteration elapsed time: {20 + n}.50 ms, GPU KV cache usage: 5.0%" for i, n in enumerate([4, 4, 5, 4, 8, 0, 8, 0]))
        s = NL.iteration_summary(txt)
        self.assertEqual(s["lines"], 8)
        self.assertEqual(s["decode_by_gen_requests"]["4"]["n"], 3)
        self.assertEqual(s["parity"]["odd_mean_n"], 2.0)                 # the 8/0 tail shows on one parity

    def test_op_placement_and_cpu(self):
        ev = [{"ph": "X", "cat": "kernel", "name": "ncclDevKernel_Broadcast_RING_LL", "dur": 27900.0},
              {"ph": "X", "cat": "kernel", "name": "_ple_conv_kernel", "dur": 100.0},
              {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD (Pinned -> Device)", "dur": 5.0},
              {"ph": "X", "cat": "user_annotation", "name": "propose_draft", "dur": 1.0},
              {"ph": "X", "cat": "kernel", "name": "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "dur": 51.0}]
        p = NL.op_placement(ev)
        self.assertEqual(p["pp_broadcast"]["gpu_ms"], 27.9)
        self.assertEqual(p["ple"]["gpu_n"], 1)
        self.assertEqual(p["h2d_copy"]["gpu_n"], 1)
        self.assertEqual(p["draft_mtp"]["host_n"], 1)
        self.assertEqual(p["tp_allreduce"]["gpu_n"], 1)
        c = NL.cpu_per_process([{"pid": 1, "utime": 100, "stime": 0}], [{"pid": 1, "utime": 1100, "stime": 0, "comm": "py"}], 10.0, {1: "Worker_PP0_TP0"})
        self.assertEqual(c["procs"][0], {"pid": 1, "role": "Worker_PP0_TP0", "cores": 1.0})


@unittest.skipUnless((NB1R2 / "traces" / "profile_b1").is_dir() and (HERE / "nb4_timeline.py").is_file(), "real B1 traces / timeline absent")
class RealB1Stall(unittest.TestCase):
    def test_base_stall_reads_not_removed(self):
        import nb4_timeline as T
        rep = T.analyze(str(NB1R2 / "traces" / "profile_b1"), tp=2)
        f = NL.stall_fraction(rep)
        self.assertIsNotNone(f)
        self.assertGreater(f["frac_over_1ms"], 0.9)                       # B1: the stall in every steady graph
        self.assertFalse(NL.mechanism(NL.make_arms()["S"], {"tl": rep})["ok"])


class Cell8Flow(unittest.TestCase):
    """The real CELL 8 source (from nb4_code_export.py) executed with a stub boot runner: the gates, the planner loop, the
    try/finally verdict, the fallback submission."""

    @classmethod
    def setUpClass(cls):
        if not EXPORT.is_file():
            raise unittest.SkipTest("nb4_code_export.py not built")
        txt = EXPORT.read_text(encoding="utf-8")
        i = txt.index("    # ==================================================================== CELL 8 — E25 NB4 main")
        body = txt[i:]
        cls.src = "\n".join(l[4:] if l.startswith("    ") else l for l in body.splitlines())

    def run_cell8(self, arm_status=None, raise_in=None, minutes=150, merge_ok=True, p_integrated=True, disabled=()):
        out = Path(tempfile.mkdtemp())
        clock = {"t": 0.0}
        calls = {"arms": [], "fallback": 0, "saves": 0}
        arms = NL.make_arms(s2={"ok": True, "env": {"NB4_PPACK": "1"}}, p={"ok": True, "env": {"NB4_P5": "1"}} if p_integrated else None,
                            s3={"ok": True, "env": {"NB4_PPDEFER": "1"}}, disabled=disabled)
        R = {"sections": {"A0": {"merge": {"ok": merge_ok, "why": None if merge_ok else "missing"}}}, "inputs": {}, "setup": {}, "errors": []}
        NB4 = {"arm_problems": [], "embedded_sha_ok": {"nb4_lib.py": True}, "plan_log": []}
        DONE, RECS, GATES = {}, {}, {}

        def step(sec, name, est, fn):
            try:
                r = fn() or {}
                r.setdefault("status", "DONE")
            except Exception as e:
                r = {"status": "ERROR", "error": repr(e)}
            R["sections"].setdefault(sec, {})[name] = r
            return r

        def nb4_arm_step(k):
            calls["arms"].append(k)
            if raise_in == k:
                raise RuntimeError("stub boot crash in " + k)
            clock["t"] += NL.EST_MIN[k] * 60
            st = (arm_status or {}).get(k, "DONE")
            r = {"status": st, "correctness": "CLEAN", "reference_stored": st == "DONE"}
            RECS[k] = r
            return r

        def nb4_analysis():
            return {"arms": {k: {"label": NL.final_label(arms[k], RECS.get(k))} for k in NL.ARM_ORDER}, "winners": [], "C": {"label": "NO_GAIN"}}

        ns = {"RERUN": False, "os": os, "json": json, "time": time, "traceback": traceback, "T_START": time.time(), "R": R, "NB4": NB4,
              "GATES": GATES, "DONE": DONE, "RECS": RECS, "ARMS": arms, "NL": NL, "MERLIN_OK": True, "ABORT": {"why": None},
              "HOST_PHASE": {}, "OUT": out, "FAKE": False, "step": step, "merlin_install": lambda: {"kv": {"MERLIN_TREE": "/t"}},
              "apply_60490": lambda: {"ok": True}, "core_timer_install": lambda: {"ok": True}, "sha_gate": lambda t: (True, "ok"),
              "save": lambda: calls.__setitem__("saves", calls["saves"] + 1), "left": lambda: minutes * 60 - 5 * 60 - clock["t"],
              "log": lambda *a: None, "nb4_arm_step": nb4_arm_step, "nb4_analysis": nb4_analysis,
              "write_fallback_submission": lambda: calls.__setitem__("fallback", calls["fallback"] + 1), "NB4_PIN": {}}
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            exec(compile(self.src, "<nb4 CELL 8>", "exec"), ns)
        v = json.loads((out / "nb4_verdict.json").read_text())
        return v, calls, buf.getvalue(), ns

    def test_normal_run(self):
        v, calls, outp, ns = self.run_cell8(minutes=400)
        self.assertEqual(calls["arms"], ["T0", "S", "S2", "S3", "G", "P", "T0b", "K4", "INT8"])
        self.assertEqual(calls["fallback"], 1)
        self.assertIsNone(v["main_error"])
        self.assertEqual(v["labels"]["T0"], "BASE")
        self.assertIn("@@NB4 ", outp)
        self.assertTrue(all(g["ok"] for g in v["gates"].values()))

    def test_150_min_budget_protects_t0b(self):
        v, calls, _, _ = self.run_cell8(minutes=150)
        self.assertIn("T0b", calls["arms"])
        self.assertNotIn("K4", calls["arms"])
        self.assertEqual(calls["arms"][:2], ["T0", "S"])

    def test_crash_still_writes_verdict_and_fallback(self):
        v, calls, _, ns = self.run_cell8(raise_in="S", minutes=400)
        self.assertIn("stub boot crash in S", v["main_error"]["error"])
        self.assertEqual(calls["fallback"], 1)
        self.assertEqual(calls["arms"], ["T0", "S"])
        self.assertEqual(v["labels"]["S2"], "SKIPPED_BUDGET")             # never run: no measurement claimed

    def test_gate_failure_boots_nothing(self):
        v, calls, _, _ = self.run_cell8(merge_ok=False)
        self.assertEqual(calls["arms"], [])
        self.assertFalse(v["gates"]["MERGE"]["ok"])
        self.assertEqual(calls["fallback"], 1)

    def test_bad_base(self):
        v, calls, _, _ = self.run_cell8(arm_status={"T0": "LAUNCH_FAILED", "T0b": "LAUNCH_FAILED"}, minutes=400)
        self.assertEqual(calls["arms"], ["T0", "T0b"])
        self.assertEqual(v["labels"]["S"], "SKIPPED_NO_BASE")
        self.assertEqual(v["labels"]["T0"], "BASE_FAILED")

    def test_disabled_arms_by_flag(self):
        v, calls, _, _ = self.run_cell8(minutes=400, disabled=("S3", "K4"))
        self.assertEqual(calls["arms"], ["T0", "S", "S2", "G", "P", "T0b", "INT8"])
        self.assertEqual(v["labels"]["S3"], "DISABLED")
        self.assertEqual(v["labels"]["K4"], "DISABLED")

    def test_p_not_integrated_skipped(self):
        v, calls, _, _ = self.run_cell8(p_integrated=False, minutes=400)
        self.assertNotIn("P", calls["arms"])
        self.assertEqual(v["labels"]["P"], "P5_NOT_INTEGRATED")


class Built(unittest.TestCase):
    def test_notebook_and_export(self):
        nb = HERE.parent / "Q27-XHIGH-E25_NB4_TICK.ipynb"
        if not nb.is_file() or not EXPORT.is_file():
            self.skipTest("not built")
        b = nb.read_bytes()
        self.assertLess(len(b), 400_000)
        self.assertNotIn(b"\r", b.replace(b"\\r", b""))
        cells = ["".join(c["source"]) for c in json.loads(b)["cells"] if c["cell_type"] == "code"]
        self.assertEqual("\n\n".join(cells), EXPORT.read_text(encoding="utf-8"))
        self.assertEqual(len(cells), 8)
        for i, c in enumerate(cells):
            compile(c, f"cell{i + 1}", "exec")
        self.assertTrue(cells[1].startswith("if not RERUN:"))
        self.assertIn("write_fallback_submission()", cells[0])           # dual-mode CELL 0/1: a rerun writes the fallback only
        meta = json.loads((Path(r"D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb4-tick__foysalemonshanto") / "kernel-metadata.json").read_text())
        self.assertEqual(meta["machine_shape"], "NvidiaL4")
        self.assertTrue(meta["is_private"] and meta["enable_gpu"] and not meta["enable_internet"])
        self.assertEqual(len(meta["model_sources"]), 7)


if __name__ == "__main__":
    unittest.main(verbosity=1)
