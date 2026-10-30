"""E24 decision-logic tests (stdlib unittest): the verdict functions of the notebook cells on synthetic data.

The cells are pulled out of `Experiment Notebooks/Build/build_e24_notebook.py` with ast (the builder is never executed),
then ARGV_CODE + CELL4 + CELL5 run in one namespace with the real `e24_lab.py` and mocked I/O (no server, no Kaggle).
Each case is a scenario from the independent decision-logic review (2026-10-30) with the corrected verdict:
section 2 eligibility / confirmation, section 5 partial loads, section 1 A/A sources + scaled allowances + verdicts that
reach the arm status + unevaluated chain links, section 3 ranking vs the control, section 6 verdicts, the hard stop, the
launch guard. Run: python3 test_e24_decisions.py
"""
import ast
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER = Path(os.environ.get("E24_DEC_BUILDER") or HERE.parent / "Experiment Notebooks" / "Build" / "build_e24_notebook.py")
sys.dont_write_bytecode = True


def _cells():
    env = {}
    for node in ast.parse(BUILDER.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("ARGV_CODE", "CELL4", "CELL5"):
            env[node.targets[0].id] = eval(compile(ast.Expression(node.value), "<cell>", "eval"), {}, env)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "CELL5":
            env["CELL5"] += eval(compile(ast.Expression(node.value), "<cell>", "eval"), {}, env)
    return env


CELLS = _cells()


def _lab():
    spec = importlib.util.spec_from_file_location("e24_lab_dec", HERE / "e24_lab.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LAB = _lab()


def namespace(variant="FULL_B", sm=False):
    work = Path(tempfile.mkdtemp(prefix="e24dec_"))
    ns = {"__name__": "e24_cells", "os": os, "sys": sys, "json": json, "time": time, "re": re, "shutil": shutil, "math": math,
          "Path": Path, "L": LAB}
    exec("import hashlib, threading, traceback, inspect, random, signal, socket, tarfile, base64, gzip, subprocess\n"
         "import urllib.request as _U", ns)
    ns.update(SM=sm, FAKE=False, TS=1.0, FORCE_COND=False, MODE="SMOKE" if sm else "FULL", VARIANT=variant,
              PART={"FULL_A": "A", "FULL_B": "B"}.get(variant, "ALL"),
              WORK=work, OUT=work, ARMS_OUT=work / "arms", LOGS=work / "logs", T_START=time.time(), HARD_STOP=time.time() + 1e6,
              ABORT={"why": None}, PATCH_BROKEN=None, ENGINES={"sglang": {}, "vllm": {}})
    ns["R"] = {"sections": {s: {} for s in "123456"}, "comparisons": {}, "argv": {}, "errors": [], "order": [], "api_drift": {},
               "gates": {}}
    ns.update(save=lambda: None, left=lambda: 1e6, log=lambda *a: None)
    ns["err"] = lambda where, e=None, **kw: ns["R"]["errors"].append({"where": where, "error": repr(e), **kw})
    exec(compile(CELLS["ARGV_CODE"], "<ARGV_CODE>", "exec"), ns)
    exec(compile(CELLS["CELL4"], "<CELL4>", "exec"), ns)
    ns.update(PLAN=[], TRACES={}, ORACLE=[], C16=[], PROMPTS={}, CAT={})
    ns["ABORT"]["why"] = "test harness"            # CELL5 ends with run_plan() unless ABORT: an empty PLAN either way
    exec(compile(CELLS["CELL5"], "<CELL5>", "exec"), ns)
    ns["ABORT"]["why"] = None
    ns["ENGINES"].update(sglang={}, vllm={})
    return ns


# ------------------------------------------------------------------------------------------------ synthetic data
def win(tok, st="STEADY", n=20, running=None):
    w = {"tok_s": tok, "status": st, "n_offered": n, "tokens_per_step": 4.0, "step_ms": 140.0}
    if running is not None:
        w["decode_lines"] = {"running": {"p50": running}}
    return w


def arm(ws, status="GO", **gates):
    g = {"flags": "GO", "launch": "GO", "hook": "GO", "parity": "GO", "windows": "GO"}
    g.update(gates)
    return {"status": status, "gates": g, "windows": ws}


N = 64
ORK = ["8k#0", "8k#1", "8k#2", "8k#3", "40k#4", "40k#5", "90k#6", "90k#7", "120k#8"]
C16K = ["8k#%d" % v for v in (0, 1, 2, 3) + tuple(range(9, 21))]


def row(key, flips=None, err=None, cached=0):
    """flips: {position: m_sum}; the candidate takes another token there (m_ref = m_cand = m_sum / 2)."""
    toks, marg = list(range(1000, 1000 + N)), [2.0] * N
    if err:
        return {"key": key, "tokens": [], "margins": [], "top2": [], "text": "", "error": err, "cached_tokens": None}
    for d, s in (flips or {}).items():
        toks = toks[:d] + [toks[d] + 1] + [t + 7 for t in toks[d + 1:]]
        marg = marg[:d] + [s / 2] + marg[d + 1:]
    return {"key": key, "tokens": toks, "margins": marg, "top2": [[t, -0.1, t + 1, -2.1] for t in toks],
            "text": "The rule maps each colored cell to its mirror; rows keep order while columns swap %s." % key, "error": None,
            "cached_tokens": cached}


def rows(prefix, keys, flipmap=None, err=None, cached=0):
    return [row(prefix + k, (flipmap or {}).get(k), err, cached) for k in keys]


def ref_fix(rs, flipmap):
    """The reference needs the same half margin at each flip position (m_ref)."""
    for r in rs:
        for d, s in (flipmap.get(r["key"].split(":", 1)[1]) or {}).items():
            r["margins"][d] = s / 2
    return rs


ALL1 = ["s1_a", "s1_b", "s1_c32", "s1_c", "s1_d6", "s1_d4"]


def s1_setup(ns, arms=ALL1):
    for a in arms:
        ns["R"]["sections"]["1"][a] = {"status": "GO", "gates": {"cache_isolation": "GO"}}
        ns["ROWS"][a] = {"m1": rows("s:", ORK), "c16a": rows("c:", C16K), "c16b": rows("c:", C16K)}
        ns["C16F"][a] = [True, True] if a not in ("s1_a", "s1_b") else [None, None]
        if a not in ("s1_a", "s1_b"):
            ns["ROWS"][a]["p2"] = rows("s:", ORK, cached=4096)


# ------------------------------------------------------------------------------------------------ section 2
class _S2Base:
    def base(self):
        ns = namespace("FULL_B")
        sec = ns["R"]["sections"]["2"]
        sec["s2_w6s20"] = arm({"8k": win(600), "40k": win(540, n=16), "90k": win(364, n=7, running=7), "120k": win(300, n=5, running=5)})
        sec["s2_w6s20_aa"] = arm({"8k": win(597), "40k": win(538, n=16)})
        return ns, sec

    def challenger(self, sec, long90=380.0, long120=310.0, rep=None):
        sec["s2_w4s20"] = arm({"8k": win(660), "40k": win(594, n=16)})        # +10 % at 8k and 40k
        sec["s2_best_long__s2_w4s20"] = arm({"90k": win(long90, "CAPACITY_LIMITED", n=7, running=7),
                                             "120k": win(long120, "CAPACITY_LIMITED", n=5, running=5)})
        for i, r in enumerate([] if rep is None else rep if isinstance(rep, list) else [rep], 1):   # one A/A round per ratio
            sec[f"aa_ctrl_r{i}"] = arm({"8k": win(601), "40k": win(541, n=16)})
            sec[f"aa_best_r{i}__s2_w4s20"] = arm({"8k": win(r * 601), "40k": win(r * 541, n=16)})


class TestSection2(_S2Base, unittest.TestCase):
    def test_parity_gate_never_go_by_default(self):
        ns = namespace("FULL_B")
        pg = ns["parity_gate"]
        self.assertEqual(pg({"status": "FAIL_HIGH_MARGIN"}), "NO_GO_PARITY")
        self.assertEqual(pg({"status": "EXACT_TO_N"}), "GO")
        self.assertEqual(pg({"status": "MEASURED"}), "GO")                                   # the control saved its rows
        self.assertEqual(pg({"status": "BOUNDARY", "aa_high": 1}), "GO")                    # within the A/A allowance
        self.assertEqual(pg({"status": "BOUNDARY", "aa_high": None}), "INCONCLUSIVE_PARITY_BOUNDARY")
        self.assertEqual(pg({"status": "INCONCLUSIVE_NO_CONTROL_ROWS"}), "INCONCLUSIVE_PARITY_NO_CONTROL_ROWS")
        self.assertEqual(pg({}), "INCONCLUSIVE_PARITY_MISSING")

    def test_gated_challenger_is_never_best(self):
        """Review #1 scenario A: startup 640 s + greedy fallback + retraction windows + no parity, 10 % faster."""
        ns, sec = self.base()
        sec["s2_w4s20"] = arm({"8k": win(660, "INCONCLUSIVE_RETRACTION"), "40k": win(600, "INCONCLUSIVE_RETRACTION", n=16)},
                              status="NO_GO_GREEDY_FALLBACK", launch="NO_GO_STARTUP_600", greedy_fallback="NO_GO_GREEDY_FALLBACK",
                              parity="INCONCLUSIVE_PARITY_NO_CONTROL_ROWS", windows="INCONCLUSIVE_RETRACTION")
        # clean windows, 10 % faster, but a gate blocks adoption: the startup gate, the DFlash greedy fallback
        sec["s2_w6s16"] = arm({"8k": win(660, n=16), "40k": win(594, n=16)}, status="NO_GO_STARTUP_600", launch="NO_GO_STARTUP_600")
        sec["s2_w4s16"] = arm({"8k": win(660, n=16), "40k": win(594, n=16)}, status="NO_GO_GREEDY_FALLBACK",
                              greedy_fallback="NO_GO_GREEDY_FALLBACK")
        o = ns["s2_compare"]()
        for a in ("s2_w4s20", "s2_w6s16", "s2_w4s16"):
            self.assertFalse(o["arms"][a]["eligible"], a)
            self.assertIsNone(o["arms"][a]["better"], a)
        self.assertEqual((o["best"], o["best_candidate"]), (None, None))
        self.assertEqual(o["ranking"], [])

    def test_spread_needs_valid_control_windows(self):
        """Review #1 scenario B: the control's 40k window depressed (prefill in window) and the A/A's 40k not GO."""
        ns = namespace("FULL_B")
        sec = ns["R"]["sections"]["2"]
        sec["s2_w6s20"] = arm({"8k": win(600), "40k": win(470, "INCONCLUSIVE_PREFILL_IN_WINDOW", n=16)},
                              status="INCONCLUSIVE_PREFILL_IN_WINDOW", windows="INCONCLUSIVE_PREFILL_IN_WINDOW")
        sec["s2_w6s20_aa"] = arm({"8k": win(597), "40k": win(500, "INCONCLUSIVE_RETRACTION", n=16)})
        sec["s2_w6s16"] = arm({"8k": win(590, n=16), "40k": win(536, n=16)})
        o = ns["s2_compare"]()
        self.assertEqual(o["aa_points"], 1)                                   # 8k only: the 40k pair is not valid on both
        self.assertFalse(o["arms"]["s2_w6s16"]["eligible"])                   # the control's 40k window is not usable
        self.assertIsNone(o["best"])

    def test_best_needs_long_bands_to_hold(self):
        """Review #2: an XL collapse at 90k / 116k of the 8k / 40k winner revisits `best`."""
        ns, sec = self.base()
        self.challenger(sec, long90=150.0, long120=90.0, rep=1.09)            # reproduced: only the long bands stop it
        o = ns["s2_compare"]()
        self.assertEqual(o["best_candidate"], "s2_w4s20")
        self.assertIsNone(o["best"])
        self.assertEqual(o["best_check"]["long_verdict"], "STOP_XL_COLLAPSE")

    def test_best_long_not_run_is_not_best(self):
        ns, sec = self.base()
        self.challenger(sec)
        del sec["s2_best_long__s2_w4s20"]
        o = ns["s2_compare"]()
        self.assertIsNone(o["best"])
        self.assertEqual(o["best_check"]["verdict"], "INCONCLUSIVE_LONG_BANDS_NOT_RUN")

    def test_best_needs_every_repeat(self):
        """Review #2: the A/A round's repeat of the winner at ratio 0.995 is not ignored."""
        ns, sec = self.base()
        self.challenger(sec, rep=[1.09, 0.995])                                # one round reproduces, one does not
        o = ns["s2_compare"]()
        self.assertIsNone(o["best"])
        self.assertEqual(o["best_check"]["verdict"], "NOT_REPRODUCED")
        self.assertEqual(o["best_check"]["repeats"]["aa_best_r2__s2_w4s20"]["vs"], "aa_ctrl_r2")

    def test_confirmed_best(self):
        ns, sec = self.base()
        self.challenger(sec, rep=1.09)
        o = ns["s2_compare"]()
        self.assertEqual(o["best"], "s2_w4s20")
        self.assertEqual(o["best_check"]["verdict"], "CONFIRMED")
        self.assertAlmostEqual(o["arms"]["s2_w4s20"]["threshold"], o["aa_spread"])    # "beyond the A/A spread", not 2x

    def test_branch_rule_needs_usable_windows(self):
        ns, sec = self.base()
        sec["s2_w4s20"] = arm({"8k": win(580, "INCONCLUSIVE_RETRACTION"), "40k": win(500, n=16)})
        self.assertIsNone(ns["s2_compare"]()["branch"]["run"])
        sec["s2_w4s20"] = arm({"8k": win(500), "40k": win(450, n=16)}, launch="NO_GO_STARTUP_600")   # windows count, not adoption
        self.assertEqual(ns["s2_compare"]()["branch"]["run"], "s2_w7s20")


class TestSection2Round2(_S2Base, unittest.TestCase):
    def test_best_needs_a_reproduced_repeat(self):
        ns, sec = self.base()
        self.challenger(sec)                                                   # long bands hold, no A/A-round repeat
        o = ns["s2_compare"]()
        self.assertEqual((o["best_candidate"], o["best"]), ("s2_w4s20", None))
        self.assertEqual(o["best_check"]["verdict"], "UNCONFIRMED_NO_REPEAT")

    def test_a_repeat_that_failed_a_gate_did_not_reproduce(self):
        """Review 2 #1: the winner's own repeat crashed (engine error) -> not CONFIRMED."""
        ns, sec = self.base()
        self.challenger(sec, rep=1.09)
        sec["aa_best_r1__s2_w4s20"]["gates"]["engine_errors"] = "NO_GO_ENGINE_ERROR"
        sec["aa_best_r1__s2_w4s20"]["status"] = "NO_GO_ENGINE_ERROR"
        o = ns["s2_compare"]()
        self.assertIsNone(o["best"])
        self.assertEqual(o["best_check"]["repeats"]["aa_best_r1__s2_w4s20"]["verdict"], "NOT_REPRODUCED_GATE")
        self.assertEqual(o["best_check"]["verdict"], "NOT_REPRODUCED")

    def test_capacity_limited_window_is_not_equal_n(self):
        """Review 2 audit #2: an 8k / 40k window running fewer streams than offered is no gain claim."""
        ns, sec = self.base()
        self.challenger(sec, rep=1.09)
        sec["s2_w4s20"]["windows"]["40k"]["status"] = "CAPACITY_LIMITED"
        o = ns["s2_compare"]()
        self.assertFalse(o["arms"]["s2_w4s20"]["eligible"])
        self.assertIsNone(o["best"])


# ------------------------------------------------------------------------------------------------ section 5
def ld(tok, st="MEASURED", starv=False):
    return {"tok_s_window": tok, "status": st, "starvation": starv, "max_gap_s": 1.2, "prefill_sec": [70.0], "tpot_ms": {"p95": 160.0}}


class TestSection5(unittest.TestCase):
    def test_not_worse_needs_every_load(self):
        """Review #3: partial or invalid loads never read NOT_WORSE."""
        ns = namespace("FULL_B")
        sec = ns["R"]["sections"]["5"]
        sec["s5_c4096"] = {"status": "GO", "gates": {"launch": "GO"},
                           "measured": {"D0": ld(600), "D1": ld(560), "D2": ld(420), "D3": ld(300)}}
        sec["s5_pdi4"] = {"status": "NO_GO_ENGINE_ERROR", "gates": {"launch": "GO", "engine_errors": "NO_GO_ENGINE_ERROR"},
                          "measured": {"D0": ld(598), "D1": ld(565), "D2": {"status": "FAIL_SERVER_DIED"},
                                       "D3": {"status": "INCONCLUSIVE_NOT_IDLE_AT_START"}}}
        sec["s5_pdi1"] = {"status": "INCONCLUSIVE_INJECT_NO_FIRST_TOKEN", "gates": {"launch": "GO", "load_D3": "INCONCLUSIVE_INJECT_NO_FIRST_TOKEN"},
                          "measured": {"D0": ld(600), "D1": ld(561), "D2": ld(440), "D3": ld(560, "INCONCLUSIVE_INJECT_NO_FIRST_TOKEN")}}
        sec["s5_c2048"] = {"status": "GO", "gates": {"launch": "GO"}, "measured": {"D0": ld(600), "D1": ld(559), "D2": ld(425), "D3": ld(301)}}
        sec["s5_c8192"] = {"status": "GO", "gates": {"launch": "GO"}, "measured": {"D0": ld(600), "D1": ld(500), "D2": ld(380), "D3": ld(270)}}
        sec["s5_mixed"] = {"status": "NO_GO_STARVATION", "gates": {"launch": "GO", "load_D2": "NO_GO_STARVATION"},
                           "measured": {"D0": ld(600), "D1": ld(560), "D2": ld(430, starv=True), "D3": ld(310)}}
        st = ns["s5_compare"]()["stop"]
        self.assertEqual(st["s5_pdi4"]["verdict"], "STOP_NO_GO_ENGINE_ERROR")
        self.assertEqual(st["s5_pdi1"]["verdict"], "INCONCLUSIVE_LOADS_D3")
        self.assertIsNone(st["s5_pdi1"]["mixed_ratio_d1_d3"])
        self.assertEqual(st["s5_c2048"]["verdict"], "NOT_WORSE")
        self.assertEqual(st["s5_c8192"]["verdict"], "STOP_WORSE_MIXED")
        self.assertEqual(st["s5_mixed"]["verdict"], "STOP_STARVATION")
        self.assertEqual(st["s5_c4096"]["verdict"], "CONTROL")

    def test_control_validity_and_startup(self):
        ns = namespace("FULL_B")
        sec = ns["R"]["sections"]["5"]
        good = lambda t: {"D0": ld(600), "D1": ld(t), "D2": ld(t), "D3": ld(t)}
        sec["s5_c4096"] = {"status": "NO_GO_GREEDY_FALLBACK", "gates": {"greedy_fallback": "NO_GO_GREEDY_FALLBACK"}, "measured": good(500)}
        sec["s5_c2048"] = {"status": "GO", "gates": {"launch": "GO"}, "measured": good(500)}
        st = ns["s5_compare"]()["stop"]
        self.assertTrue(st["s5_c2048"]["verdict"].startswith("INCONCLUSIVE_CONTROL"), st)     # review 2 #6
        sec["s5_c4096"] = {"status": "NO_GO_STARVATION", "gates": {"load_D2": "NO_GO_STARVATION"},
                           "measured": dict(good(500), D2=ld(500, starv=True))}             # DESIGN §4.5: the control starves
        sec["s5_c2048"] = {"status": "NO_GO_STARTUP_600", "gates": {"launch": "NO_GO_STARTUP_600"}, "measured": good(500)}
        st = ns["s5_compare"]()["stop"]
        self.assertEqual(st["s5_c4096"]["verdict"], "STOP_STARVATION")
        self.assertEqual((st["s5_c2048"]["verdict"], st["s5_c2048"]["adoption_blocked"]), ("NOT_WORSE", "NO_GO_STARTUP_600"))


# ------------------------------------------------------------------------------------------------ section 1
class TestSection1(unittest.TestCase):
    def test_state_flips_reach_the_arm_status(self):
        """Review #5: two 8-nat STATE_FLIPs in c <-> d6 against a FLIP-only A/A of s1_c."""
        ns = namespace("FULL_A")
        s1_setup(ns)
        fl = {"8k#9": {10: 0.6}, "8k#10": {20: 0.6}}
        ns["ROWS"]["s1_c"]["c16b"] = rows("c:", C16K, fl)
        ref_fix(ns["ROWS"]["s1_c"]["c16a"], fl)
        sf = {"8k#0": {5: 8.0}, "40k#4": {7: 8.0}}
        ns["ROWS"]["s1_d6"]["m1"] = rows("s:", ORK, sf)
        ns["ROWS"]["s1_d6"]["p2"] = rows("s:", ORK, sf, cached=4096)
        ref_fix(ns["ROWS"]["s1_c"]["m1"], sf)
        o = ns["s1_compare_all"](final=True)
        self.assertIn("NO_GO_STATE_FLIP vs s1_c", o["arm_verdicts"].get("s1_d6", []))
        self.assertEqual(ns["R"]["sections"]["1"]["s1_d6"]["status"], "NO_GO_STATE_FLIP")
        self.assertEqual(ns["R"]["sections"]["1"]["s1_d6"]["status_measured"], "GO")
        self.assertEqual(ns["R"]["sections"]["1"]["s1_c"]["status"], "GO")

    def test_unevaluated_link_is_never_clean(self):
        """Review #6: every M1 row of s1_d4 errored -> the chain is incomplete, not NO_HIGH_MARGIN_LINK."""
        ns = namespace("FULL_A")
        s1_setup(ns)
        ns["ROWS"]["s1_d4"] = {"m1": rows("s:", ORK, err="RemoteDisconnected: server died")}
        ns["C16F"].pop("s1_d4")
        o = ns["s1_compare_all"](final=True)
        link = next(x for x in o["chain"] if x["pair"] == "s1_c~s1_d4")
        self.assertIsNone(link["normalised"])
        self.assertEqual(o["decisive"]["verdict"], "INCONCLUSIVE_CHAIN_INCOMPLETE")
        # no common key at all: INCONCLUSIVE_PAIRS_MISSING, 0 positions, no row to call inconclusive
        ns["ROWS"]["s1_d4"] = {"m1": rows("s:", ["other#%d" % i for i in range(9)])}
        o = ns["s1_compare_all"](final=True)
        self.assertEqual(o["pairs"]["s1_c~s1_d4"]["status"], "INCONCLUSIVE_PAIRS_MISSING")
        self.assertIsNone(next(x for x in o["chain"] if x["pair"] == "s1_c~s1_d4")["normalised"])
        self.assertEqual(o["decisive"]["verdict"], "INCONCLUSIVE_CHAIN_INCOMPLETE")

    def test_restore_errors_are_a_verdict_not_allowance(self):
        """Review #4: 3 nine-nat STATE_FLIPs in d6's pass 2 (restore exercised) + 4 genuine FLIPs in c <-> d6."""
        ns = namespace("FULL_A")
        s1_setup(ns)
        f4 = {"8k#0": {11: 1.5}, "8k#3": {12: 1.5}, "90k#6": {13: 1.5}, "120k#8": {14: 1.5}}
        rf = {"8k#1": {3: 9.0}, "8k#2": {4: 9.0}, "40k#5": {6: 9.0}}
        ns["ROWS"]["s1_d6"]["m1"] = ref_fix(rows("s:", ORK, f4), rf)
        ns["ROWS"]["s1_d6"]["p2"] = rows("s:", ORK, {**f4, **rf}, cached=4096)
        ref_fix(ns["ROWS"]["s1_c"]["m1"], f4)
        o = ns["s1_compare_all"](final=True)
        self.assertEqual(o["pairs"]["s1_c~s1_d6"]["status"], "FAIL_HIGH_MARGIN")
        self.assertEqual(o["decisive"]["pair"], "s1_c~s1_d6")
        self.assertIn("NO_GO_STATE_FLIP (pass1_pass2)", o["arm_verdicts"]["s1_d6"])
        self.assertEqual(o["within"]["s1_d6"]["pass2_restore"], "EXERCISED")

    def test_restore_not_exercised_is_no_verdict(self):
        ns = namespace("FULL_A")
        s1_setup(ns)
        rf = {"8k#1": {3: 9.0}}
        ns["ROWS"]["s1_d6"]["p2"] = rows("s:", ORK, rf, cached=0)          # no radix hit: NOT_EXERCISED, never PASS / FAIL
        ref_fix(ns["ROWS"]["s1_d6"]["m1"], rf)
        o = ns["s1_compare_all"](final=True)
        self.assertEqual(o["within"]["s1_d6"]["pass2_restore"], "NOT_EXERCISED")
        self.assertNotIn("NO_GO_STATE_FLIP (pass1_pass2)", o["arm_verdicts"].get("s1_d6", []))

    def test_warm_relaunch_is_a_test_pair(self):
        """Review #4: c <-> c_warm with 3 STATE_FLIPs fails c_warm and no longer inflates every s1_c pair."""
        ns = namespace("FULL_A")
        s1_setup(ns, ALL1 + ["s1_c_warm"])
        ns["ROWS"]["s1_c_warm"] = {"m1": rows("s:", ORK, {"8k#0": {3: 9.0}, "40k#4": {4: 9.0}, "90k#7": {5: 9.0}})}
        ns["C16F"].pop("s1_c_warm")
        f3 = {"8k#1": {11: 1.5}, "8k#2": {12: 1.5}, "90k#6": {13: 1.5}}
        ns["ROWS"]["s1_d6"]["m1"] = rows("s:", ORK, f3)
        ns["ROWS"]["s1_d6"]["p2"] = rows("s:", ORK, f3, cached=4096)
        ref_fix(ns["ROWS"]["s1_c"]["m1"], {"8k#0": {3: 9.0}, "40k#4": {4: 9.0}, "90k#7": {5: 9.0}, **f3})
        o = ns["s1_compare_all"](final=True)
        self.assertIn("NO_GO_STATE_FLIP vs s1_c", o["arm_verdicts"]["s1_c_warm"])
        self.assertEqual(o["pairs"]["s1_c~s1_d6"]["status"], "FAIL_HIGH_MARGIN")
        self.assertNotIn("s1_c~s1_c_warm", o["aa"])

    def test_allowance_scales_with_rows(self):
        """Review #12: 2 FLIPs in a 16-row A/A allow 4 in a 25-row pair (+1): 3 FLIPs are BOUNDARY, 6 are a finding."""
        ns = namespace("FULL_A")
        s1_setup(ns)
        fl = {"8k#9": {10: 1.0}, "8k#10": {20: 1.0}}
        for a in ("s1_c", "s1_d6"):
            ns["ROWS"][a]["c16b"] = rows("c:", C16K, fl)
            ref_fix(ns["ROWS"][a]["c16a"], fl)
        f3 = {"8k#1": {11: 1.5}, "8k#2": {12: 1.5}, "90k#6": {13: 1.5}}
        ns["ROWS"]["s1_d6"]["m1"] = rows("s:", ORK, f3)
        ref_fix(ns["ROWS"]["s1_c"]["m1"], f3)
        p = ns["s1_compare_all"](final=True)["pairs"]["s1_c~s1_d6"]
        self.assertEqual((p["aa_high"], p["rows_judged"], p["status"]), (4, 25, "BOUNDARY"))
        f6 = {k: {11 + i: 1.5} for i, k in enumerate(ORK[:6])}
        ns["ROWS"]["s1_d6"]["m1"] = rows("s:", ORK, f6)
        ns["ROWS"]["s1_c"]["m1"] = ref_fix(rows("s:", ORK), f6)
        self.assertEqual(ns["s1_compare_all"](final=True)["pairs"]["s1_c~s1_d6"]["status"], "FAIL_HIGH_MARGIN")

    def test_c16_aa_needs_its_flushes(self):
        ns = namespace("FULL_A")
        s1_setup(ns)
        ns["C16F"]["s1_c"] = [True, False]                 # c16b ran on c16a's radix entries: not a pure repeat
        o = ns["s1_compare_all"](final=True)
        self.assertFalse(o["within"]["s1_c"]["c16_aa_pure"])
        self.assertIsNone(o["aa_rate"].get("s1_c"))

    def test_errored_rows_are_not_garbage(self):
        ns = namespace("FULL_A")
        s = ns["row_summary"](rows("s:", ORK[:3], err="URLError: refused"))
        self.assertEqual(s["garbage"], [None, None, None])


class TestSection1Round2(unittest.TestCase):
    """Second validation round: Fisher significance, restore-state-matched A/A, unjudged rows, garbage without an A/A."""

    def test_fisher_gates_the_decisive_pair(self):
        ns = namespace("FULL_A")
        s1_setup(ns)
        f3 = {"8k#1": {11: 1.5}, "8k#2": {12: 1.5}, "90k#6": {13: 1.5}}         # EMS: 3 > 0 + 1 -> FAIL_HIGH_MARGIN
        ns["ROWS"]["s1_d6"]["m1"] = rows("s:", ORK, f3)
        ref_fix(ns["ROWS"]["s1_c"]["m1"], f3)
        o = ns["s1_compare_all"](final=True)
        p = o["pairs"]["s1_c~s1_d6"]
        self.assertEqual(p["status"], "FAIL_HIGH_MARGIN")
        self.assertGreater(p["p_value"], ns["S1_ALPHA"])                      # 3 of 25 vs 0 of 32: p ~ 0.08
        self.assertEqual((o["decisive"]["pair"], o["decisive"]["suggestive_pair"]), (None, "s1_c~s1_d6"))
        self.assertTrue(o["decisive"]["verdict"].startswith("DECISIVE_PAIR_SUGGESTIVE"))
        f6 = {k: {11 + i: 1.5} for i, k in enumerate(ORK[:6])}
        ns["ROWS"]["s1_d6"]["m1"] = rows("s:", ORK, f6)
        ns["ROWS"]["s1_c"]["m1"] = ref_fix(rows("s:", ORK), f6)
        o = ns["s1_compare_all"](final=True)
        self.assertLess(o["pairs"]["s1_c~s1_d6"]["p_value"], ns["S1_ALPHA"])
        self.assertEqual(o["decisive"]["pair"], "s1_c~s1_d6")
        self.assertTrue(o["decisive"]["verdict"].startswith("DECISIVE_PAIR_FOUND"))

    def test_target_only_drift_needs_significance(self):
        ns = namespace("FULL_A")
        s1_setup(ns)
        f3 = {"8k#1": {11: 1.5}, "8k#2": {12: 1.5}, "90k#6": {13: 1.5}}
        ns["ROWS"]["s1_c"]["m1"] = rows("s:", ORK, f3)                       # c32 -> c: the SSM dtype link
        ns["ROWS"]["s1_c"]["p2"] = rows("s:", ORK, f3, cached=4096)           # pass 2 = pass 1 (no restore drift)
        ref_fix(ns["ROWS"]["s1_c32"]["m1"], f3)
        o = ns["s1_compare_all"](final=True)
        self.assertEqual(o["pairs"]["s1_c32~s1_c"]["status"], "FAIL_HIGH_MARGIN")
        self.assertFalse(any("HIGH_MARGIN_DRIFT" in v for v in o["arm_verdicts"].get("s1_c", [])))
        self.assertTrue(any(v.startswith("SUGGESTIVE_HIGH_MARGIN_DRIFT") for v in o["arm_notes"].get("s1_c", [])))
        self.assertEqual(ns["R"]["sections"]["1"]["s1_c"]["status"], "GO")
        f6 = {k: {11 + i: 1.5} for i, k in enumerate(ORK[:6])}
        ns["ROWS"]["s1_c"]["m1"] = rows("s:", ORK, f6)
        ns["ROWS"]["s1_c"]["p2"] = rows("s:", ORK, f6, cached=4096)
        ns["ROWS"]["s1_c32"]["m1"] = ref_fix(rows("s:", ORK), f6)
        o = ns["s1_compare_all"](final=True)
        self.assertIn("NO_GO_HIGH_MARGIN_DRIFT vs s1_c32", o["arm_verdicts"]["s1_c"])

    def test_aa_rows_need_the_same_restore_state(self):
        """FAKE restore_corruption: rows that hit the 64-token shared header in one batch only are not A/A noise."""
        ns = namespace("FULL_A")
        s1_setup(ns)
        a, b = rows("c:", C16K), rows("c:", C16K, {"8k#1": {9: 9.0}})
        a[1]["cached_tokens"], b[1]["cached_tokens"] = 0, 64                  # admission order differed for this row
        ref_fix(a, {"8k#1": {9: 9.0}})
        ns["ROWS"]["s1_d6"]["c16a"], ns["ROWS"]["s1_d6"]["c16b"] = a, b
        o = ns["s1_compare_all"](final=True)
        w = o["within"]["s1_d6"]
        self.assertEqual((w["c16_aa_rows_restore_mismatch"], w["c16_aa"]["high"]), (1, 0))
        self.assertNotIn("NO_GO_STATE_FLIP (c16 A/A)", o["arm_verdicts"].get("s1_d6", []))

    def test_clean_aa_with_one_unjudged_row_still_counts(self):
        """Review 2 #4: a timed-out c16b row must not drop a clean A/A (which made a cleaner A/A give a weaker verdict)."""
        ns = namespace("FULL_A")
        s1_setup(ns)
        for arm in ("s1_c", "s1_d6"):
            ns["ROWS"][arm]["c16b"][5] = row("c:" + C16K[5], err="timeout")
        f6 = {k: {11 + i: 1.5} for i, k in enumerate(ORK[:6])}
        ns["ROWS"]["s1_d6"]["m1"] = rows("s:", ORK, f6)
        ns["ROWS"]["s1_c"]["m1"] = ref_fix(rows("s:", ORK), f6)
        o = ns["s1_compare_all"](final=True)
        self.assertEqual(o["aa_rate"]["s1_d6"], 0.0)
        self.assertEqual(o["pairs"]["s1_c~s1_d6"]["status"], "FAIL_HIGH_MARGIN")
        self.assertEqual(o["decisive"]["pair"], "s1_c~s1_d6")

    def test_garbage_link_without_an_aa(self):
        ns = namespace("FULL_A")
        s1_setup(ns)
        for arm in ("s1_c", "s1_d4"):
            ns["C16F"][arm] = [False, False]                                  # no pure A/A on either arm
        g = rows("s:", ORK)
        for r in g[:2]:
            r["text"] = "!" * 300
        ns["ROWS"]["s1_d4"] = {"m1": g}
        o = ns["s1_compare_all"](final=True)
        link = next(x for x in o["chain"] if x["pair"] == "s1_c~s1_d4")
        self.assertEqual((link["aa_high"], link["normalised"], link["significant"]), (None, 2, True))


# ------------------------------------------------------------------------------------------------ section 3
class TestSection3(unittest.TestCase):
    def test_best_two_need_go_and_faster_than_control(self):
        ns = namespace("FULL_B")

        def w(ms, status="GO", **g):
            r = arm({"8k": {"status": "STEADY", "client_step_ms": {"mean": ms, "p95": ms * 1.3}, "tok_s": 84000 / ms,
                            "tokens_per_step": 84}}, status=status, env_proof="GO", **g)
            r["gates"].pop("parity")
            return r
        sec = ns["R"]["sections"]["3"]
        sec.update(s3_ctrl=w(140.0), s3_ll=w(132.0, "NO_GO_STARTUP_600", launch="NO_GO_STARTUP_600"),
                   s3_tree=w(131.0, "NO_GO_ENGINE_ERROR", engine_errors="NO_GO_ENGINE_ERROR"), s3_cta4=w(150.0), s3_cta8=w(136.0),
                   s3_ring=w(145.0))
        o = ns["s3_compare"]()
        self.assertEqual(o["best_two"], ["s3_cta8"])                          # ring / cta4 are slower than the control
        sec["s3_ring"] = w(138.0)
        self.assertEqual(ns["s3_compare"]()["best_two"], ["s3_cta8", "s3_ring"])

    def test_confirmation_verdicts(self):
        ns = namespace("FULL_B")
        ns["R"]["comparisons"]["2"] = {"aa_spread": 0.01}
        ns["R"]["sections"]["2"]["s2_w6s20"] = arm({"90k": win(364, n=7)})
        sec = ns["R"]["sections"]["3"]
        sec["s3_ctrl_rerun"] = arm({"40k": win(540, n=16)})
        sec["s3_conf__s3_cta8"] = arm({"40k": win(570, n=16), "90k": win(385, n=7)})
        sec["s3_conf__s3_ring"] = arm({"40k": win(520, n=16), "90k": win(366, n=7)})
        c = ns["s3_compare"]()["confirmations"]
        self.assertEqual(c["s3_conf__s3_cta8"]["verdict"], "GAIN_CONFIRMED")
        self.assertEqual(c["s3_conf__s3_ring"]["verdict"], "STOP_SLOWER")


# ------------------------------------------------------------------------------------------------ section 6
class TestSection6(unittest.TestCase):
    def test_inject_verdicts(self):
        ns = namespace("FULL_A")
        v = ns["s6_inject_verdict"]
        flip = {"c8": {"status": "BOUNDARY", "high": 1, "state_flips": 0, "garbage": 0}}
        state = {"c8": {"status": "BOUNDARY", "high": 1, "state_flips": 1, "garbage": 0}}
        clean = {"c8": {"status": "EXACT_TO_N", "high": 0, "state_flips": 0, "garbage": 0}}
        self.assertEqual(v("sent", True, True, 0, None, None, True, flip, True, {}), "INCONCLUSIVE_FLIP_NO_AA")   # review #9
        self.assertEqual(v("sent", True, True, 0, None, None, True, state, True, {}), "FAIL")
        self.assertEqual(v("sent", True, True, 2, None, None, True, clean, True, {}), "FAIL")
        self.assertEqual(v("nan", True, True, None, 0, 3, False, clean, False, {}), "INCONCLUSIVE_SERVER_DIED")
        self.assertEqual(v("nan", True, True, None, 0, 3, False, clean, False, {"Scheduler hit an exception": 1}), "FAIL")
        self.assertEqual(v("nan", True, True, None, 0, 3, False, clean, True, {}), "CLEAN")
        self.assertEqual(v("nan", True, True, None, 0, 0, False, clean, True, {}), "INCONCLUSIVE_NOT_EXERCISED")
        self.assertEqual(v("nan", False, True, None, 4, 0, False, clean, True, {}), "BUG_EXPOSED")
        self.assertEqual(v("nan", False, True, None, 4, 0, False, clean, True, {}, agree=False), "INCONCLUSIVE_COUNTERS_DESYNC")
        self.assertEqual(v("nan", True, True, None, 0, 3, False, clean, True, {}, agree=False), "INCONCLUSIVE_COUNTERS_DESYNC")
        agree = ns["_ranks_agree"]
        self.assertTrue(agree({"counters": [{"x": 2}] * 4}, ("x",)))
        self.assertFalse(agree({"counters": [{"x": 2}] * 3 + [{"x": 0}]}, ("x",)))         # one rank only: a desync
        self.assertFalse(agree({"counters": [{"x": 2}] * 3}, ("x",)))                       # a missing rank

    def test_pairs_keep_arms_without_a_verdict(self):
        """Review #8a: the patched natural arm died on the assert -> its row stays in the table."""
        ns = namespace("FULL_A")
        ns["R"]["sections"]["6"] = {"s6_ffu": {"verdict": "BUG_EXPOSED"}, "s6_ffp": {"verdict": "CLEAN"},
                                    "s6_np": {"status": "NO_GO_ASSERT_43559"}, "s6_gu_df": {"status": "SKIPPED_CONDITION"},
                                    "s6_combined": {"verdict": "PASS"}}
        o = ns["s6_pairs"]()
        np_row = next(v for k, v in o.items() if k.startswith("43559 natural"))
        self.assertEqual(np_row["patched"], "NO_VERDICT (NO_GO_ASSERT_43559)")
        self.assertEqual(np_row["verdict"], "INCONCLUSIVE")
        self.assertFalse(any("fallback" in k for k in o))                    # a skipped conditional stays out

    def test_patched_needs_safety_and_allocator(self):
        ns = namespace("FULL_A")
        vp = ns["s6_verdict_patched"]
        t = {"reference_valid": True, "safe_depth": 128,
             "cycles": [{"flush_valid": True, "seed_rid": "s0", "probe_kl": 0.0001, "probe_cached": 128}]}
        ev = {"confirmed": {}, "failed": {"s0"}, "counters": [{}]}
        self.assertEqual(vp([t], ev, None, need_safety=False), "CLEAN")
        self.assertEqual(vp([t], ev, None, need_safety=True), "INCONCLUSIVE_SAFETY_NOT_RUN")
        self.assertEqual(vp([t], ev, {"status": "INCONCLUSIVE_NO_ALLOCATOR_SNAPSHOT"}, need_safety=True),
                         "INCONCLUSIVE_SAFETY_NO_ALLOCATOR_SNAPSHOT")
        self.assertEqual(vp([t], ev, {"status": "OK"}, need_safety=True), "CLEAN")
        self.assertEqual(vp([t], ev, {"status": "INCONCLUSIVE_NO_ALLOCATOR_SNAPSHOT", "leaks": 1}, need_safety=True),
                         "INCONCLUSIVE_SAFETY_NO_ALLOCATOR_SNAPSHOT")             # review 2 #2a: no false FAIL
        self.assertEqual(vp([t], ev, {"status": "OK", "leaks": 1}, need_safety=True), "FAIL")
        self.assertEqual(vp([t], ev, {"status": "INCONCLUSIVE_FLUSH_INVALID", "stale": 1}, need_safety=True), "FAIL")
        self.assertEqual(vp([t], dict(ev, ranks_agree=False), {"status": "OK"}, need_safety=True), "INCONCLUSIVE_COUNTERS_DESYNC")

    def test_ngram_fallback_trigger(self):
        ns = namespace("FULL_A")
        f = ns["ngram_launch_failed"]
        self.assertTrue(f({"status": "NO_GO_FLAG_UNSUPPORTED", "gates": {"flags": "NO_GO_FLAG_UNSUPPORTED"}}))
        self.assertTrue(f({"status": "NO_GO_OOM", "gates": {"flags": "GO", "launch": "NO_GO_OOM"}}))
        self.assertTrue(f({"status": "NO_GO_SM89_KERNEL", "gates": {"flags": "GO", "launch": "NO_GO_SM89_KERNEL"}}))
        self.assertFalse(f({"status": "GO", "verdict": "CLEAN", "gates": {"flags": "GO", "launch": "GO"}}))
        self.assertFalse(f({"status": "SKIPPED_BUDGET"}))
        self.assertFalse(f(None))


# ------------------------------------------------------------------------------------------------ hard stop, launch guard
class TestRunControl(unittest.TestCase):
    def test_hard_stop_gates(self):
        """Review #10: gates produced by the watchdog kill never stand; a NO_GO measured before the stop does."""
        ns = namespace("FULL_B")

        def fn(gates):
            def f():
                ns["ABORT"]["why"] = "HARD_STOP reached: watchdog killed live servers"
                return {"gates": dict(gates), "status": ns["worst"](gates.values())}
            return f
        cases = [({"launch": "GO", "census_90k": "GO", "census_116k": "NO_GO_SERVER_DIED", "alive": "ERROR_SERVER_DIED"},
                  "INCONCLUSIVE_HARD_STOP"),
                 ({"flags": "GO", "launch": "NO_GO_LAUNCH"}, "INCONCLUSIVE_HARD_STOP"),
                 ({"launch": "GO", "load_D2": "NO_GO_STARVATION", "load_D3": "NO_GO_SERVER_DIED"}, "NO_GO_STARVATION"),
                 ({"launch": "GO", "m6_tool": "NO_GO_TOOL_PROBE", "alive": "ERROR_SERVER_DIED"}, "INCONCLUSIVE_HARD_STOP")]
        for i, (gates, want) in enumerate(cases):
            ns["ABORT"]["why"] = None
            self.assertEqual(ns["step"]("4", f"arm{i}", 1, fn(gates))["status"], want, gates)

    def test_launch_guard(self):
        ns = namespace("FULL_B")
        ls, to = ns["launch_streak"], "startup timeout"
        for a in ("s3_ring", "s3_tree", "s4_mem086", "s6_gu"):              # risky knobs never count
            ls("sglang", a, "NO_GO_LAUNCH", to, [a])
        ls("sglang", "s2_w6s20", "NO_GO_OOM", to, ["x"])                    # a named failure is the arm's own
        for a in ("s6_ffu", "s6_ffp", "s6_np"):                             # one config failing three times (review 2 #5)
            ls("sglang", a, "NO_GO_LAUNCH", to, ["same-argv"])
        ls("sglang", "s2_w6s20", "NO_GO_LAUNCH", "exited rc=1", ["fast-exit"])  # a fast exit costs seconds
        self.assertFalse(ns["ENGINES"]["sglang"].get("stopped"))
        ls("sglang", "s2_w6s20_aa", "NO_GO_LAUNCH", to, ["argv-2"])
        self.assertFalse(ns["ENGINES"]["sglang"].get("stopped"))
        ls("sglang", "s2_w4s20", "NO_GO_LAUNCH", to, ["argv-3"])           # three distinct argvs timed out
        self.assertIn("three distinct argvs", ns["ENGINES"]["sglang"]["stopped"])
        self.assertIn("launches stopped", ns["dep_engine"]("sglang")())


# ------------------------------------------------------------------------------------------------ fuzz (properties)
ST = ["GO", "GO", "GO", "NO_GO_STARTUP_600", "NO_GO_ENGINE_ERROR", "NO_GO_STARVATION", "INCONCLUSIVE_RETRACTION",
      "ERROR_SERVER_DIED", "SKIPPED_BUDGET", "INVALID_CACHE_ISOLATION", "INCONCLUSIVE_HARD_STOP", None]
WST = ["STEADY", "STEADY", "STEADY", "CAPACITY_LIMITED", "MEASURED", "INCONCLUSIVE_NO_SETTLE", "INCONCLUSIVE_RETRACTION",
       "FAIL_SERVER_DIED", "INVALID_WARM_MISS", "INCONCLUSIVE_INJECT_NO_FIRST_TOKEN", None]


class TestFuzz(unittest.TestCase):
    """Seeded random, partial and broken records: the verdict functions never raise, and their fail-closed properties
    hold on every draw (a crash in the report cell would lose that section's verdicts)."""
    N = 300

    def rng(self, k):
        import random
        return random.Random(20261030 + k)

    def rec(self, g, bands, tok=600.0, spread=0.12):
        """A valid record (every gate GO, STEADY windows at the common N) with 0-2 random corruptions: a bad gate, a bad
        window status, a missing / zero tok/s, NOT_EQUAL_N, a dropped band, a non-GO status, no windows at all."""
        ws = {b: {"status": "STEADY", "tok_s": tok * {"8k": 1.0, "40k": 0.9, "90k": 0.6, "120k": 0.5}.get(b, 1.0)
                  * (1 + g.uniform(-spread, spread)), "n_offered": 20, "equal_n": {}, "decode_lines": {"running": {"p50": 7}}}
              for b in bands}
        r = {"status": "GO", "gates": {k: "GO" for k in ("flags", "launch", "hook", "parity", "windows")}, "windows": ws}
        for _ in range(g.choice([0, 0, 0, 1, 1, 2])):
            c = g.randrange(7)
            if c == 0:
                r["gates"][g.choice(list(r["gates"]))] = g.choice(["NO_GO_STARTUP_600", "INCONCLUSIVE_X", "NO_GO_PARITY", "ERROR_BODY_X"])
            elif c == 1 and ws:
                ws[g.choice(list(ws))]["status"] = g.choice(WST)
            elif c == 2 and ws:
                ws[g.choice(list(ws))]["tok_s"] = g.choice([None, 0])
            elif c == 3 and ws:
                ws[g.choice(list(ws))]["equal_n"] = {"label": "NOT_EQUAL_N"}
            elif c == 4 and ws:
                ws.pop(g.choice(list(ws)))
            elif c == 5:
                r["status"] = g.choice(ST)
            else:
                r.pop("windows", None)
        return r

    def test_s2_properties(self):
        for k in range(self.N):
            g, ns = self.rng(k), namespace("FULL_B")
            sec = ns["R"]["sections"]["2"]
            sec["s2_w6s20"] = self.rec(g, ("8k", "40k", "90k", "120k"), spread=0.02)
            for a in ["s2_w6s20_aa", "aa_ctrl_r1", "aa_ctrl_r2"]:
                if g.random() < 0.85:
                    sec[a] = self.rec(g, ("8k", "40k"), spread=0.02)
            for a in ns["S2_CHALLENGERS"]:
                gain = g.choice([1.0, 1.0, 1.1, 1.2])               # some challengers really are faster
                if g.random() < 0.8:
                    sec[a] = self.rec(g, ("8k", "40k"), tok=600 * gain, spread=0.02)
                if g.random() < 0.6:
                    sec["s2_best_long__" + a] = self.rec(g, ("90k", "120k"), tok=600 * g.choice([gain, 0.5]), spread=0.02)
                    for i in (1, 2):
                        if g.random() < 0.6:
                            sec[f"aa_best_r{i}__{a}"] = self.rec(g, ("8k", "40k"), tok=600 * g.choice([gain, 1.0]), spread=0.02)
            o = ns["s2_compare"]()
            sp = o["aa_spread"]
            for a, v in o["arms"].items():
                if v["better"] is not None:
                    self.assertTrue(v["eligible"], (k, a))
                if v["eligible"]:
                    self.assertTrue(o["control_ok"] and v["mean_ratio_8k_40k"] is not None and sp is not None, (k, a))
            b = o["best"]
            if b:
                v, chk = o["arms"][b], o["best_check"]
                self.assertIn(b, ns["S2_CHALLENGERS"])
                self.assertTrue(v["eligible"] and v["better"], k)
                self.assertTrue(ns["_s2_gates_ok"](sec[b]), k)
                self.assertEqual(chk["verdict"], "CONFIRMED")
                self.assertTrue(all(x["ratio"] is not None and x["ratio"] >= 1 - sp for x in chk["long"].values()), k)
                self.assertFalse(any(x["verdict"].startswith("NOT_REPRODUCED") for x in chk["repeats"].values()), k)
                self.assertTrue(any(x["verdict"] == "REPRODUCED" for x in chk["repeats"].values()), k)

    def test_s5_properties(self):
        for k in range(self.N):
            g, ns = self.rng(1000 + k), namespace("FULL_B")
            sec = ns["R"]["sections"]["5"]

            def ld(tok):
                d = {"tok_s_window": tok * g.uniform(0.9, 1.1), "status": "MEASURED", "starvation": False, "prefill_sec": [70.0]}
                c = g.randrange(10)                                  # 0-3: one corruption, else a clean load
                if c == 0:
                    return None
                if c == 1:
                    d["status"] = g.choice(WST)
                elif c == 2:
                    d["starvation"] = g.choice([True, None])
                elif c == 3:
                    d["tok_s_window"] = g.choice([None, 0])
                return d
            for a in ("s5_c4096", "s5_c2048", "s5_c8192", "s5_pdi1", "s5_pdi4"):
                if g.random() < 0.9:
                    tok = 500 * g.choice([1.0, 1.0, 0.9])
                    sec[a] = {"status": g.choice(["GO", "GO", "GO"] + ST), "gates": {"launch": g.choice(["GO", "GO", "NO_GO_STARTUP_600"])},
                              "measured": {l: ld(tok) for l in ("D0", "D1", "D2", "D3")}}
            o = ns["s5_compare"]()
            for a, v in o["stop"].items():
                L5 = o["arms"][a]
                if v["verdict"] == "NOT_WORSE":
                    for l in ("D1", "D2", "D3"):
                        self.assertEqual(ns["mgate"](L5[l]["status"]), "GO", (k, a, l))
                        self.assertEqual(ns["mgate"](L5[l]["ctrl_status"]), "GO", (k, a, l))
                        self.assertIs(L5[l]["starvation"], False, (k, a, l))
                    self.assertGreaterEqual(v["mixed_ratio_d1_d3"], 0.97)
                    self.assertIn(o["stop"]["s5_c4096"]["verdict"], ("CONTROL", "STOP_STARVATION"), (k, a))
                if v["mixed_ratio_d1_d3"] is not None:
                    self.assertFalse(v["unusable_loads"], (k, a))
                if any(x["starvation"] for x in L5.values()):
                    self.assertEqual(v["verdict"], "STOP_STARVATION", (k, a))

    def test_s3_properties(self):
        for k in range(self.N):
            g, ns = self.rng(2000 + k), namespace("FULL_B")
            sec = ns["R"]["sections"]["3"]
            for a in ns["S3_ARMS"]:
                if g.random() < 0.9:
                    r = self.rec(g, ())
                    r["windows"] = {"8k": {"status": g.choice(["STEADY"] * 6 + WST),
                                           "client_step_ms": g.choice([{"mean": g.uniform(25, 30)}] * 5 + [None]),
                                           "tok_s": g.uniform(2000, 3000), "tokens_per_step": 84}}
                    r["gates"]["env_proof"] = g.choice(["GO"] * 5 + ["INVALID_ENV_NOT_APPLIED"])
                    sec[a] = r
            o = ns["s3_compare"]()
            for a in o["best_two"]:
                self.assertTrue(ns["_s2_gates_ok"](sec[a]), (k, a))
                self.assertLess(o["arms"][a]["client_step_ms_mean"], o["control_step_ms_mean"], (k, a))

    def test_s1_properties(self):
        arms = ALL1 + ["s1_c_warm", "s1_c_nocache", "s1_d6fr", "s1_d6_r2", "s1_c_r2"]
        for k in range(120):
            g, ns = self.rng(3000 + k), namespace("FULL_A")
            for a in arms:
                if g.random() < 0.15:
                    continue
                ns["R"]["sections"]["1"][a] = {"status": g.choice(ST), "gates": {"cache_isolation": g.choice(["GO", "GO", "INVALID_CACHE_ISOLATION"])}}
                fm = lambda: {g.choice(ORK): {g.randint(1, 60): g.choice([0.2, 1.0, 1.5, 3.0, 8.0])}} if g.random() < 0.4 else None
                tags = {"m1": rows("s:", ORK, fm(), err="died" if g.random() < 0.05 else None)}
                if g.random() < 0.7:
                    tags["c16a"], tags["c16b"] = rows("c:", C16K, fm()), rows("c:", C16K, fm())
                    ns["C16F"][a] = [g.choice([True, True, False, None]), g.choice([True, True, False, None])]
                if g.random() < 0.6:
                    tags["p2"] = rows("s:", ORK, fm(), cached=g.choice([0, 4096]))
                ns["ROWS"][a] = tags
            o1 = ns["s1_compare_all"](final=True)
            st1 = {a: r.get("status") for a, r in ns["R"]["sections"]["1"].items()}
            o2 = ns["s1_compare_all"](final=True)                           # idempotent: recomputed from status_measured
            self.assertEqual(st1, {a: r.get("status") for a, r in ns["R"]["sections"]["1"].items()}, k)
            self.assertEqual(o1["arm_verdicts"], o2["arm_verdicts"], k)
            dec = o1["decisive"]
            if dec.get("pair"):
                self.assertGreater(dec["link"]["normalised"], 0, k)
                self.assertFalse(str(dec["link"]["status"]).startswith("INCONCLUSIVE"), k)
            for a, v in o1["arm_verdicts"].items():
                r = ns["R"]["sections"]["1"].get(a)
                if r and r.get("status_measured") and not str(r["status_measured"]).startswith(("ERROR", "SKIPPED")):
                    self.assertTrue(str(r["status"]).startswith(("NO_GO", "ERROR")), (k, a, r["status"], v))

    def test_hard_stop_properties(self):
        pool = ["GO", "GO", "NO_GO_LAUNCH", "NO_GO_SERVER_DIED", "NO_GO_LOGPROB_PREFLIGHT", "NO_GO_STARVATION", "NO_GO_ENGINE_ERROR",
                "ERROR_SERVER_DIED", "ERROR_BODY_URLError", "INCONCLUSIVE_X", "NO_GO_STARTUP_600"]
        for k in range(self.N):
            g, ns = self.rng(4000 + k), namespace("FULL_B")
            gates = {f"g{i}": g.choice(pool) for i in range(g.randint(0, 6))}

            def f():
                ns["ABORT"]["why"] = "HARD_STOP reached: watchdog killed live servers"
                return {"gates": dict(gates), "status": ns["worst"](gates.values())}
            st = ns["step"]("4", f"a{k}", 1, f)["status"]
            self.assertFalse(str(st).startswith("ERROR") or st in ns["DEAD_GATES"], (k, gates, st))
            real = [v for v in gates.values() if str(v).startswith("NO_GO") and v not in ns["DEAD_GATES"]]
            self.assertEqual(str(st).startswith("NO_GO"), bool(real), (k, gates, st))

    def test_s6_pairs_properties(self):
        vs = ["BUG_EXPOSED", "CLEAN", "FAIL", "INCONCLUSIVE_NOT_EXERCISED", None]
        for k in range(self.N):
            g, ns = self.rng(5000 + k), namespace("FULL_A")
            sec = ns["R"]["sections"]["6"]
            for a in ("s6_ffu", "s6_ffp", "s6_np", "s6_ffud", "s6_ffpd", "s6_gu", "s6_gp", "s6_sent_u", "s6_sent_p", "s6_nan_u", "s6_nan_p"):
                if g.random() < 0.8:
                    v = g.choice(vs)
                    sec[a] = {"status": g.choice(ST)} | ({"verdict": v} if v else {})
            for name, row in ns["s6_pairs"]().items():
                if row["verdict"] == "PASS" and "natural" not in name:
                    self.assertEqual((row["unpatched"], row["patched"]), ("BUG_EXPOSED", "CLEAN"), (k, name))
                if row.get("patched") == "FAIL":
                    self.assertEqual(row["verdict"], "FAIL", (k, name))


if __name__ == "__main__":
    unittest.main(verbosity=1)
