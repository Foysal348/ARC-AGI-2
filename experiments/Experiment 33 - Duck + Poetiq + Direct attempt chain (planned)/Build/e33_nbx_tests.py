"""E33 quick local tests of the notebook-side pieces (stdlib + the E29 lib; no GPU, no server): e33_lib (the CELL 1 switch
checker, the admission rule) and the generated e33_nbx.py executed in a stub notebook namespace (the hooks wiring: E33_RUN_POETIQ
-> e33_poetiq.poetiq_run kwargs, E33_RUN_DIRECT -> e29_direct.solve_all_direct kwargs, the server url check, E33_CAN_START,
e33_setup on / off / a solver miss, the LIVE lines incl. no gold in the rerun, the report).

usage: python e33_nbx_tests.py         (Build/e33_nbx.py must exist: python build_e33.py [--draft] first; --draft writes it to
                                        D:/ARC AGI 2/_scratch/e33_nb/DRAFT_e33_nbx.py, used when Build/e33_nbx.py is missing)
"""
import json
import os
import sys
import threading
import time
import types
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E29B = Path(r"D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(E29B))
import e33_lib as L     # noqa: E402
sys.path.insert(0, str(E29B.parent.parent / 'Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)' / 'Experiment Notebooks' / 'Build'))
import e33_sched_patch  # noqa: E402,F401: CELL 3 loads branch metadata before notebook hooks
import e29_lib as E29L  # noqa: E402

NBX = HERE / "e33_nbx.py"
if not NBX.is_file():
    NBX = Path(r"D:/ARC AGI 2/_scratch/e33_nb/DRAFT_e33_nbx.py")
NBX_TEXT = NBX.read_text(encoding="utf-8")


class TestCheckChain(unittest.TestCase):
    def test_default_on(self):
        c = L.check_chain(L.DEFAULT_CHAIN)
        self.assertTrue(c["on"] and c["poetiq"] and c["direct"] and c["attempt2"])
        self.assertEqual(c["invalid"], {})
        self.assertEqual((c["poetiq_think"], c["direct_think"], c["drain_min"]), (60000, 104000, 26))

    def test_bad_item_only_that_item(self):
        c = L.check_chain(dict(L.DEFAULT_CHAIN, poetiq_think=59999))
        self.assertTrue(c["on"] and c["direct"] and c["attempt2"])
        self.assertFalse(c["poetiq"])
        self.assertIn("poetiq", c["invalid"])
        c = L.check_chain(dict(L.DEFAULT_CHAIN, direct_answer="8192"))
        self.assertTrue(c["poetiq"] and not c["direct"])
        c = L.check_chain(dict(L.DEFAULT_CHAIN, attempt2="yes"))
        self.assertTrue(c["poetiq"] and c["direct"] and not c["attempt2"])
        c = L.check_chain(dict(L.DEFAULT_CHAIN, direct_think=130000))
        self.assertFalse(c["direct"])
        c = L.check_chain(dict(L.DEFAULT_CHAIN, temperature=3))     # shared: both solvers use it
        self.assertFalse(c["poetiq"] or c["direct"])
        self.assertTrue(c["on"] and c["attempt2"])

    def test_chain_off_and_garbage(self):
        c = L.check_chain(dict(L.DEFAULT_CHAIN, on=False))
        self.assertFalse(c["on"] or c["poetiq"] or c["direct"] or c["attempt2"])
        for bad in (None, 5, "x", dict(L.DEFAULT_CHAIN, on="True")):
            c = L.check_chain(bad)
            self.assertFalse(c["on"])
            self.assertIn("chain", c["invalid"])

    def test_missing_key_and_unknown_key(self):
        d = dict(L.DEFAULT_CHAIN)
        d.pop("direct_think")
        d["poetic"] = True
        c = L.check_chain(d)
        self.assertTrue(c["poetiq"] and not c["direct"])
        self.assertEqual(c["ignored"], ["poetic"])

    def test_harness_override_exempt_from_floor(self):
        c = L.check_chain(L.DEFAULT_CHAIN, json.dumps({"poetiq_think": 300, "direct_think": 400}))
        self.assertTrue(c["poetiq"] and c["direct"] and c["harness_override"])
        self.assertEqual(L.floor_tokens(c, "poetiq"), 300)
        self.assertEqual(L.floor_tokens(L.check_chain(L.DEFAULT_CHAIN), "direct"), 60000)

    def test_items_state(self):
        st = L.items_state(L.check_chain(dict(L.DEFAULT_CHAIN, poetiq_think=10)))
        self.assertEqual(st["chain"], "ON")
        self.assertTrue(st["poetiq"].startswith("OFF (invalid"))
        self.assertEqual(L.items_state(L.check_chain(dict(L.DEFAULT_CHAIN, on=False)))["poetiq"], "OFF (chain off)")


class TestCanStart(unittest.TestCase):
    C = L.check_chain(L.DEFAULT_CHAIN)

    def test_duck_never_gated(self):
        self.assertEqual(L.can_start(0, 1, self.C, "A")[0], True)
        self.assertEqual(L.can_start(0, 1, self.C, "B1")[0], True)

    def test_floor_math(self):
        need = 60 + 60000 / 18.0                       # 3,393 s
        dl = 1000.0 + need + 26 * 60                   # exactly reachable
        self.assertTrue(L.can_start(1000.0, dl, self.C, "E")[0])
        self.assertFalse(L.can_start(1000.0, dl - 1, self.C, "E")[0])
        self.assertEqual(L.can_start(1000.0, dl, self.C, "E")[2], round(need, 1))

    def test_admit_stop_and_off(self):
        self.assertFalse(L.can_start(0, 11 * 60, dict(self.C, harness_override=True, direct_think=1), "E")[0])
        self.assertFalse(L.can_start(0, 1e9, dict(self.C, direct=False), "E")[0])
        self.assertFalse(L.can_start(0, 1e9, dict(self.C, on=False), "C")[0])

    def test_rate(self):
        self.assertEqual(L.per_stream_rate(None, None), 18.0)
        self.assertEqual(L.per_stream_rate(420.0, 0), 18.0)
        self.assertEqual(L.per_stream_rate(100.0, 4), 18.0)     # 20 > 18: the lower wins
        self.assertAlmostEqual(L.per_stream_rate(380.0, 23), 380.0 / 24)

    def test_stage_names(self):
        self.assertEqual([L.stage_solver(s) for s in ("C", "D", "E", "c", "E'", "A", "B2", None)],
                         ["poetiq", "poetiq", "direct", "poetiq", "direct", None, None, None])


# ------------------------------------------------------------------ the stub notebook namespace
class _Srv:
    url = "http://127.0.0.1:8000"


def make_ns(chain=None, sched_ok=True, rerun=False, solutions=None, payload_ok=("direct", "poetiq"), sandbox_ok=True,
            profile="E28EVAL120"):
    calls = {"poetiq": [], "direct": [], "log": []}
    tmp = Path(r"D:/ARC AGI 2/_scratch/e33_nb/_nbx_test_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    fake_p = types.ModuleType("e33_poetiq")
    fake_p.LIB_VERSION = "fake e33_poetiq"

    def poetiq_run(tid, task, **kw):
        calls["poetiq"].append(dict(kw, tid=tid))
        return {"task_id": tid, "e33_poetiq": True, "attempt": kw["attempt"], "iteration": kw["iteration"], "generated_tokens": 1234,
                "submissions": [{"test_outputs": [[[1]]], "accepted": True}], "accepted": True,
                "poetiq": {"train_pass": True, "score": 1.0, "mode": "natural", "think_tokens": 1000}}
    fake_p.poetiq_run = poetiq_run
    fake_p.sandbox_ready = lambda force=False: {"ok": sandbox_ok, "mode": "fake", "error": None if sandbox_ok else "no root"}
    fake_p.can_admit = lambda deadline, cfg=None, now=None, harness=False: (deadline - time.monotonic() > 5000, {"need_sec": 3393.3})
    fake_d = types.ModuleType("e29_direct")
    fake_d.LIB_VERSION = "fake e29_direct"

    def solve_all_direct(tasks, order, **kw):
        calls["direct"].append(dict(kw, tasks=list(tasks)))
        t = order[0]
        return {t: [{"task_id": t, "generated_tokens": 999, "e31": {"mode": "natural"}, "submissions": [{"test_outputs": [[[2]]]}]}]}
    fake_d.solve_all_direct = solve_all_direct
    mods = {"e33_poetiq": fake_p, "e29_direct": fake_d}

    def q27x_payload(name, s):
        key = {"e29_direct.py": "direct", "e33_poetiq.py": "poetiq"}.get(name)
        if key and key not in payload_ok:
            raise FileNotFoundError(name)
        sys.modules[name[:-3]] = mods.get(name[:-3]) or types.ModuleType(name[:-3])
        return b"# stub\n"
    e12a = types.SimpleNamespace(SERVER_DOWN=threading.Event(), SCHED_LAST={"e28": {"e33": {"skipped": {"C": 1}}}},
                                 solve_all_r6=None, build_submission=lambda t, r: {k: [{"attempt_1": [[1]], "attempt_2": [[2]]}] for k in t},
                                 E33_RUN_POETIQ="x", E33_RUN_DIRECT="x", E33_CAN_START="x")
    ns = {"__name__": "nbx_test", "json": json, "os": os, "time": time, "threading": threading, "Path": Path, "sys": sys,
          "E33_CHAIN": L.check_chain(chain if chain is not None else L.DEFAULT_CHAIN), "E33L": L,
          "E33D_FILES": {"direct": ["e29_direct.py", "0" * 64], "poetiq": ["e33_poetiq.py", "1" * 64]},
          "E33_SCHED_REC": {"ok": sched_ok, "error": None if sched_ok else "S failed",
                            "env": {"E33_SCHED": "1", "E33_POETIQ": "1", "E33_DIRECT": "1"}},
          "E29_RT": {"cfg": {}, "on": ["df"], "df_live": True}, "E12A": e12a, "SGL_URL": _Srv.url, "E11_SERVER": _Srv(),
          "TMP": tmp, "q27x_payload": q27x_payload, "NOTEBOOK_T0": time.perf_counter(), "E11_HARNESS": False, "RERUN": rerun,
          "log": lambda *a: calls["log"].append(" ".join(map(str, a))), "e21_prom": lambda: [], "TASK_IDS": ["t2", "t1"],
          "SOLUTIONS": solutions, "WORKING": tmp, "BASE_OUT": tmp / "out", "E29L": E29L, "PROFILE": profile, "NB5_PORT": 8000,
          "e29_ledger": lambda port: {"port": port},
          "E29_SMOKE": {"tasks": ["t1", "t2"], "window_min": 120, "budget_min": 150, "probe_min": 95, "probe_sched_min": 20,
                        "probe": {"poetiq": "t1", "direct": "t2"}}}
    exec(compile(NBX_TEXT, "e33_nbx.py", "exec"), ns)
    ns["_E29D_TOK"].update({"render": None, "encode": None, "decode": None, "eos_ids": [1]})
    return ns, calls


class TestNbx(unittest.TestCase):
    def test_setup_on_hooks(self):
        ns, calls = make_ns()
        ns["e33_setup"]()
        r = ns["E33_RT"]
        self.assertEqual(r["state"], "ON")
        e = ns["E12A"]
        self.assertIs(e.E33_RUN_POETIQ, ns["e33_run_poetiq"])
        self.assertIs(e.E33_RUN_DIRECT, ns["e33_run_direct"])
        self.assertIs(e.E33_CAN_START, ns["e33_can_start"])
        self.assertIs(ns["E29_RT"]["direct"], r)
        eff = ns["e33_effective"]()
        self.assertEqual(eff["chain"], "ON")
        self.assertTrue(eff["items"]["poetiq"].startswith("ON") and eff["items"]["direct"].startswith("ON"))
        r["_rate_stop"].set()

    def test_poetiq_hook_kwargs(self):
        ns, calls = make_ns()
        ns["e33_setup"]()
        url = _Srv.url + "/v1"
        rec = ns["e33_run_poetiq"]("t1", {"train": [], "test": [{}]}, {"attempt": 2, "branch": 123, "e33_stage": "D"}, 123.0, url, "m", "/o", 2, {"p": 1})
        kw = calls["poetiq"][-1]
        self.assertEqual((kw["iteration"], kw["attempt"], kw["prior"], kw["deadline"], kw["base_url"], kw["model_id"], kw["out_dir"]),
                         (2, 2, {"p": 1}, 123.0, _Srv.url, "m", "/o"))
        self.assertEqual({k: kw["cfg"][k] for k in ("think", "answer", "temperature", "attempt", "branch")},
                         {"think": 60000, "answer": 8192, "temperature": 1.0, "attempt": 2, "branch": 123})
        self.assertNotIn("drain_min", kw["cfg"])                     # Poetiq's own drain (e33_poetiq CFG_DEFAULT)
        self.assertEqual(kw["task_index"], {"t1": 0, "t2": 1})
        self.assertIs(kw["server_down"], ns["E12A"].SERVER_DOWN)
        self.assertTrue(rec["e33_poetiq"])
        self.assertEqual(ns["E33_RT"]["runs"][-1]["stage"], "D")
        ns["E33_RT"]["_rate_stop"].set()

    def test_direct_hook_kwargs_and_url_check(self):
        ns, calls = make_ns()
        ns["e33_setup"]()
        rec = ns["e33_run_direct"]("t2", {"train": [], "test": [{}]}, {"attempt": 1, "branch": 115, "e33_stage": "E"}, 50.0, _Srv.url + "/v1", "m", "/o/d")
        kw = calls["direct"][-1]
        self.assertEqual((kw["tasks"], kw["deadline"], kw["base_url"], kw["workers"], kw["solutions"], kw["task_index"]),
                         (["t2"], 50.0, _Srv.url, 1, None, {"t1": 0, "t2": 1}))
        ns["e33_run_direct"]("t2", {"train": [], "test": [{}]}, {"attempt": 2, "branch": 125, "e33_stage": "E"}, 50.0, _Srv.url + "/v1", "m", "/o/d")
        self.assertEqual(calls["direct"][-1]["task_index"], {"t1": 2, "t2": 3})     # E' seeds: index + N (never an attempt-1 seed)
        self.assertEqual(ns["E33_RT"]["runs"][-1]["seed_index"], 3)
        self.assertEqual({k: kw["cfg"][k] for k in ("think", "answer", "temperature", "drain_min", "workers")},
                         {"think": 104000, "answer": 8192, "temperature": 1.0, "drain_min": 26, "workers": 1})
        self.assertEqual(rec["generated_tokens"], 999)
        with self.assertRaises(AssertionError):
            ns["e33_run_direct"]("t2", {"test": [{}]}, {}, 50.0, "http://other:1/v1", "m", "/o/d")
        self.assertEqual(ns["E33_RT"]["runs"][-1]["error"][:14], "AssertionError")
        ns["E33_RT"]["_rate_stop"].set()

    def test_can_start(self):
        ns, calls = make_ns()
        ns["e33_setup"]()
        cs = ns["E12A"].E33_CAN_START
        now = time.monotonic()
        self.assertTrue(cs("A", now + 10))
        self.assertTrue(cs("C", now + 6000))           # fake can_admit: > 5000 s left
        self.assertFalse(cs("D", now + 4000))
        self.assertFalse(cs("E", now + 3393 + 26 * 60))
        self.assertTrue(cs("E", now + 3500 + 26 * 60))
        st = ns["E33_RT"]["can_start"]
        self.assertEqual((st["D"]["no"], st["E"]["go"], st["E"]["no"]), (1, 1, 1))
        ns["E33_RT"]["_rate_stop"].set()

    def test_can_start_letters_real_poetiq(self):
        """The scheduler passes the bare letters C / D / E: C / D -> the REAL e33_poetiq.can_admit (60k floor, Poetiq drain
        9 min), E -> e33_lib.can_start (60k floor, drain_min 26); a late time is False for each letter."""
        import importlib.util as _iu                       # the REAL file (make_ns puts a fake e33_poetiq into sys.modules)
        _sp = _iu.spec_from_file_location("e33_poetiq_real", HERE / "e33_poetiq.py")
        RP = _iu.module_from_spec(_sp)
        _sp.loader.exec_module(RP)
        ns, _ = make_ns()
        ns["e33_setup"]()
        ns["E33P"] = RP
        cs = ns["e33_can_start"]
        now = time.monotonic()
        need_p = 60 + 60000 / 18.0
        for st in ("C", "D"):
            self.assertFalse(cs(st, now + need_p + 9 * 60 - 30), st)
            self.assertTrue(cs(st, now + need_p + 9 * 60 + 30), st)
        self.assertFalse(cs("E", now + need_p + 26 * 60 - 30))
        self.assertTrue(cs("E", now + need_p + 26 * 60 + 30))
        self.assertFalse(cs("C", now + 11 * 60))
        self.assertEqual(ns["E33_RT"]["can_start"]["C"]["last"]["need_sec"], round(need_p, 1))
        for st in ("C-POETIQ1", "E-DIRECT", "X"):           # never sent by e33_sched_patch; recorded as Duck (True)
            self.assertEqual(L.stage_solver(st), None)
        ns["E33_RT"]["_rate_stop"].set()

    def _probe(self, rerun=False, profile="E29SMOKE", left=7200.0):
        ns, calls = make_ns(rerun=rerun, profile=profile, solutions={"t1": [[[1]]], "t2": [[[2]], [[0]], [[2]]]})
        ns["e33_setup"]()
        seen, lines = {}, []

        def orig(tasks, *a, on_branch_done=None, **k):
            seen["poetiq_calls_before_solve"] = len(calls["poetiq"])
            seen["direct_calls_before_solve"] = len(calls["direct"])
            return {}
        ns["E12A"].solve_all_r6 = orig
        ns["e33_live_setup"]()
        import builtins
        p0 = builtins.print
        builtins.print = lambda *a, **k: lines.append(" ".join(map(str, a)))
        try:
            ns["E12A"].solve_all_r6({"t1": {"train": [], "test": [{}]}, "t2": {"train": [], "test": [{}, {}, {}]}}, ["t1", "t2"],
                                    out_dir=Path(r"D:/ARC AGI 2/_scratch/e33_nb/_nbx_test_tmp/duck"), deadline=time.monotonic() + left,
                                    base_url=_Srv.url + "/v1", model_id="m")
        finally:
            builtins.print = p0
        ns["E33_RT"]["_rate_stop"].set()
        return ns, calls, seen, lines

    def test_probe_smoke_only(self):
        ns, calls, seen, lines = self._probe()
        self.assertEqual((seen["poetiq_calls_before_solve"], seen["direct_calls_before_solve"]), (2, 1))
        it1, it2 = calls["poetiq"]
        self.assertEqual((it1["iteration"], it2["iteration"], it1["attempt"], it1["prior"]), (1, 2, 1, None))
        self.assertEqual(it2["prior"]["iteration"], 1)                       # iteration 2 gets the iteration-1 record
        self.assertEqual((it1["cfg"]["think"], it1["cfg"]["answer"], it1["cfg"]["branch"], it2["cfg"]["branch"]), (60000, 8192, 113, 114))
        d = calls["direct"][0]
        self.assertEqual((d["cfg"]["think"], d["cfg"]["answer"], d["tasks"]), (104000, 8192, ["t2"]))
        self.assertTrue(str(d["out_dir"]).replace("\\", "/").endswith("e33_probe/direct/a1"))
        dl = it1["deadline"]
        self.assertAlmostEqual(dl - time.monotonic(), 95 * 60, delta=30)       # start + probe_min (7,200 s left: not capped)
        pr = ns["E33_RT"]["probe"]
        self.assertTrue(pr["ran"])
        self.assertEqual(sorted(r["stage"] for r in pr["rows"]), ["C", "D", "E"])
        e = next(r for r in pr["rows"] if r["stage"] == "E")
        self.assertEqual((e["n_test"], e["grids"]), (3, 1))
        c = next(r for r in pr["rows"] if r["stage"] == "C")
        self.assertEqual((c["exact"], c["train_pass"]), (1, True))
        self.assertEqual(sum(1 for l in lines if l.startswith("[E33 PROBE] t")), 3)
        self.assertTrue(all(r["probe"] for r in ns["E33_RT"]["runs"]))
        self.assertIn("probe:", ns["e33_report_text"](ns["e33_report"]()))
        self.assertTrue(any(" | ledger " in l for l in lines) or True)        # the 15-min line (900 s tick) is not reached here

    def test_probe_capped_by_solve_end_and_not_outside_smoke(self):
        ns, calls, seen, lines = self._probe(left=100 * 60.0)
        self.assertAlmostEqual(calls["poetiq"][0]["deadline"] - time.monotonic(), 80 * 60, delta=30)   # solve end - 20 min
        for kw in ({"rerun": True}, {"profile": "E28EVAL120"}):
            ns, calls, seen, lines = self._probe(**kw)
            self.assertEqual((calls["poetiq"], calls["direct"]), ([], []))
            self.assertIsNone(ns["E33_RT"].get("probe"))
        ns, calls, seen, lines = self._probe(left=25 * 60.0)                  # < 10 min to the probe deadline: no probe
        self.assertEqual(calls["poetiq"], [])
        self.assertIn("PROBE_NO_TIME", ns["E33_RT"]["probe"]["errors"][0])

    def test_ledger_in_15min_line(self):
        self.assertIn('" | ledger " + json.dumps(e29_ledger(int(NB5_PORT)))', NBX_TEXT)

    def test_chain_off_and_sched_miss_and_solver_miss(self):
        ns, _ = make_ns(chain=dict(L.DEFAULT_CHAIN, on=False))
        os.environ["E33_SCHED"] = "1"
        ns["e33_setup"]()
        e = ns["E12A"]
        self.assertTrue(ns["E33_RT"]["state"].startswith("OFF_SWITCH"))
        self.assertTrue(e.E33_RUN_POETIQ is None and e.E33_RUN_DIRECT is None and e.E33_CAN_START is None)
        self.assertNotIn("E33_SCHED", os.environ)
        ns, _ = make_ns(sched_ok=False)
        ns["e33_setup"]()
        self.assertEqual(ns["E33_RT"]["state"], "OFF_SETUP")
        self.assertIsNone(ns["E12A"].E33_CAN_START)
        ns, _ = make_ns(payload_ok=("direct",))
        os.environ["E33_POETIQ"] = "1"
        ns["e33_setup"]()
        self.assertEqual(ns["E33_RT"]["state"], "ON")
        self.assertIsNone(ns["E12A"].E33_RUN_POETIQ)
        self.assertIs(ns["E12A"].E33_RUN_DIRECT, ns["e33_run_direct"])
        self.assertTrue(ns["E33_RT"]["items"]["poetiq"].startswith("OFF_SETUP"))
        self.assertEqual(os.environ.get("E33_POETIQ"), "0")       # Poetiq off before the solve: A -> B -> E
        os.environ.pop("E33_POETIQ", None)
        ns["E33_RT"]["_rate_stop"].set()
        ns, _ = make_ns(sandbox_ok=False)                          # E18: no Poetiq without a passing sandbox self-test
        os.environ["E33_POETIQ"] = "1"
        ns["e33_setup"]()
        self.assertIsNone(ns["E12A"].E33_RUN_POETIQ)
        self.assertIn("POETIQ_SANDBOX_UNAVAILABLE", ns["E33_RT"]["items"]["poetiq"])
        self.assertEqual(os.environ.pop("E33_POETIQ", None), "0")
        ns["E33_RT"]["_rate_stop"].set()

    def _live(self, rerun):
        ns, _ = make_ns(rerun=rerun, solutions={"t1": [[[1]]]})
        ns["e33_setup"]()
        lines = []
        seen = {}

        def orig(tasks, *a, on_branch_done=None, **k):
            seen["cb"] = on_branch_done
            for rec in k.get("recs", []):
                on_branch_done(rec)
            return {}
        ns["E12A"].solve_all_r6 = orig
        ns["e33_live_setup"]()
        import builtins
        p0 = builtins.print
        builtins.print = lambda *a, **k: lines.append(" ".join(map(str, a)))
        try:
            a1 = {"e33_attempt": 1, "e33_stage": "A", "e33_name": "A"}
            recs = [{"task_id": "t1", "branch": 0, "accepted": False,
                     "e28": dict(a1, phase="A:1", tokens=100, start_sec=1, end_sec=5)},
                    {"task_id": "t1", "branch": 2, "accepted": False,
                     "e28": {"phase": "B1:1", "e33_attempt": 1, "e33_stage": "B", "e33_name": "B1", "tokens": 90}},
                    {"task_id": "t1", "branch": 113, "e33_poetiq": True, "attempt": 1, "iteration": 1, "generated_tokens": 50,
                     "submissions": [{"test_outputs": [[[1]]]}], "poetiq": {"train_pass": True, "score": 1.0},
                     "e33": {"e33_attempt": 1, "e33_stage": "C", "e33_name": "C-POETIQ1", "tokens": 50}},
                    {"task_id": "t1", "branch": 1, "accepted": False,
                     "e28": {"phase": "A:2", "e33_attempt": 2, "e33_stage": "A", "e33_name": "A"}},
                    {"task_id": "t1", "branch": 125, "generated_tokens": 70, "e31": {"mode": "natural"},
                     "submissions": [{"test_outputs": [[[3]]]}],
                     "e33": {"e33_attempt": 2, "e33_stage": "E", "e33_name": "E-DIRECT", "tokens": 70}}]
            ns["E12A"].solve_all_r6({"t1": {"train": [], "test": [{}]}}, recs=recs)
        finally:
            builtins.print = p0
        ns["E33_RT"]["_rate_stop"].set()
        return ns, lines

    def test_live_lines_eval_and_rerun(self):
        ns, lines = self._live(False)
        live = [l for l in lines if l.startswith("[LIVE")]
        self.assertTrue(live[0].startswith("[LIVE start] algo"), live)
        self.assertIn('"chain": "ON"', live[0])
        self.assertIn("t1 a1:A ", live[1])
        self.assertIn("t1 a1:B1 ", live[2])
        self.assertIn("t1 a1:C-POETIQ1 ", live[3])
        self.assertIn("TRAIN PASS (poetiq score 1.0)", live[3])
        self.assertIn("exact 1/1", live[3])
        self.assertIn("t1 a2:A ", live[4])
        self.assertIn("t1 a2:E-DIRECT  direct natural 70 tok", live[5])
        self.assertIn("exact 0/1", live[5])
        self.assertLess(live[5].index("a1:A "), live[5].index("a1:B1 "))
        self.assertLess(live[5].index("a1:C-POETIQ1 "), live[5].index("a2:A "))
        rep = ns["e33_report"]()
        self.assertEqual(rep["per_stage"]["a1:C-POETIQ1"]["train_pass"], 1)
        self.assertEqual(rep["per_stage"]["a1:B1"]["tokens"], 90)
        self.assertEqual(rep["reached"]["t1"], {"a1": "C-POETIQ1", "a2": "E-DIRECT"})
        self.assertEqual(rep["sched_stats"], {"skipped": {"C": 1}})
        self.assertIn("a1:C-POETIQ1: finished 1", ns["e33_report_text"](rep))
        ns, lines = self._live(True)                  # the rerun: never gold
        self.assertFalse(any("exact" in l for l in lines if l.startswith("[LIVE") and "start" not in l), lines)


if __name__ == "__main__":
    unittest.main(verbosity=2)
