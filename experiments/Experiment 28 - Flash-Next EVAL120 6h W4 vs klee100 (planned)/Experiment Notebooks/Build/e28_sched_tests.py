"""E28 SCHED local unit tests (quick units, no GPU, no server): python e28_sched_tests.py [-v]

Covers: the patch (apply / verify / revert, shas, three states, strict diff), the B1 / B2 / time-cut / train-pass
classification on the REAL recorded attempts (K100Q klee100 Q_K100_FP8_K3: 60 attempts; NB5 W4 Q_FP8_K3: 30), the queue
order (phase order, phase-C priority and the ascending-token sort, no ORDER2, slots), a scheduler simulation with a fake
engine (120 tasks, 21 workers, KV gate, time cut, admission window), env-off identity against the original arc2_duck.py,
the answer-slot rule, the real Arc2ToolAgent budget / window hook, and the notebook text edits N1-N3 on the NB6Q and K100Q
code exports.

Inputs (read only): the Duck payload q27x_e15b_duck_9e4b89285136.bin (sha checked, extracted into a temp dir), the
recorded run folders under D:/ARC AGI 2/_scratch/e2{5,6}_gpu_out, the ARC eval challenges (Exp 10 data copy), the E25
code exports. Override paths with E28_TEST_PAYLOAD / E28_TEST_K100 / E28_TEST_W4 / E28_TEST_EVAL / E28_TEST_EXPORTS.
"""
import copy
import glob
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import unittest
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e28_sched_patch as E28S  # noqa: E402

ROOT = Path(r"D:/ARC AGI 2")
PAYLOAD = Path(os.environ.get("E28_TEST_PAYLOAD", ROOT / "experiments/Experiment 15 - Duck 2x35k independent attempts (planned)"
                                                       "/Experiment Notebooks/Build/payload/q27x_e15b_duck_9e4b89285136.bin"))
PAYLOAD_SHA = "9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566"
K100 = Path(os.environ.get("E28_TEST_K100", ROOT / "_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x"))
W4 = Path(os.environ.get("E28_TEST_W4", ROOT / "_scratch/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x"))
EVAL = Path(os.environ.get("E28_TEST_EVAL", ROOT / "experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)"
                                                    "/Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_challenges.json"))
EXPORTS = Path(os.environ.get("E28_TEST_EXPORTS", ROOT / "experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)"
                                                         "/Experiment Notebooks/Build"))
K100_PHASE = K100 / "phase_Q_K100_FP8_K3"
W4_PHASE = W4 / "phase_Q_FP8_K3"
E28_KEYS = list(E28S.server_env())


# ------------------------------------------------------------------ fixtures (one extraction for the whole run)
def _tmp_root():
    for c in (os.environ.get("E28_TEST_TMP"), str(ROOT / "_scratch")):
        if c and Path(c).is_dir():
            return tempfile.mkdtemp(prefix="e28_sched_tests_", dir=c)
    return tempfile.mkdtemp(prefix="e28_sched_tests_")


FIX = {}


def setUpModule():
    data = PAYLOAD.read_bytes()
    assert hashlib.sha256(data).hexdigest() == PAYLOAD_SHA, "Duck payload sha mismatch"
    tmp = Path(_tmp_root())
    FIX["tmp"] = tmp
    for name in ("orig", "e28"):
        with zipfile.ZipFile(PAYLOAD) as z:
            z.extractall(tmp / name)
    rec = E28S.apply_e28_sched(tmp / "e28")
    assert rec["ok"] and rec["applied"] == "applied", rec
    for k in E28_KEYS + ["ARC3_MAX_ACTIVE_STREAMS"]:
        os.environ.pop(k, None)
    os.environ.update({"E12_ARC2_REV5": "1"})
    sys.path[:0] = [str(tmp / "orig" / "vendor")]          # the vendor tree is byte-identical in both copies
    FIX["orig"] = _load(tmp / "orig" / "arc2_duck.py", "arc2_duck_orig")
    FIX["e28"] = _load(tmp / "e28" / "arc2_duck.py", "arc2_duck_e28")
    FIX["eval"] = json.loads(EVAL.read_bytes().decode("utf-8")) if EVAL.is_file() else {}


def tearDownModule():
    for k in E28_KEYS:
        os.environ.pop(k, None)
    shutil.rmtree(FIX.get("tmp", ""), ignore_errors=True)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class EnvE28:
    """E28 env on (server_env) for a with-block; restores the previous values."""
    def __init__(self, **extra):
        self.env = dict(E28S.server_env(), **{k: str(v) for k, v in extra.items()})

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.env}
        os.environ.update(self.env)
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _raw_records(phase_dir):
    out = {}
    for f in sorted(glob.glob(str(phase_dir / "duck" / "*" / "branch_result.json"))):
        r = json.loads(Path(f).read_bytes().decode("utf-8"))
        out[(r["task_id"], int(r["branch"]))] = r
    return out


def _solve_end(phase_dir):
    return json.loads((phase_dir / "e12_run_report.json").read_bytes().decode("utf-8"))["e12_run"]["solve_end_epoch"]


# ------------------------------------------------------------------ 1. the patch
class TestPatch(unittest.TestCase):
    def test_shas_and_constants(self):
        self.assertEqual(E28S.DUCK_PRE, "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128")
        self.assertEqual(E28S.E28_SCHED_SHA["arc2_duck.py"], {"pre": E28S.DUCK_PRE, "post": E28S.DUCK_POST})
        self.assertEqual(E28S.E28_SCHED_SHA["patch"], hashlib.sha256(E28S.DUCK_PATCH.encode()).hexdigest())
        self.assertEqual(set(E28S.PATCHES), {"arc2_duck.py"})
        self.assertEqual(E28S.ENV_NAMES, ("E28_SCHED", "E28_T1", "E28_B2_TOKENS", "E28_B2_WINDOW", "E28_B2_REQUEST_TIMEOUT"))
        self.assertEqual(E28S.server_env(), {"E28_SCHED": "1", "E28_T1": "104000", "E28_B2_TOKENS": "120000",
                                             "E28_B2_WINDOW": "130560"})
        self.assertEqual((E28S.T1, E28S.B2_TOKENS, E28S.MAX_MODEL_LEN, E28S.B2_WINDOW), (104000, 120000, 131072, 130560))
        self.assertNotIn("\r", E28S.DUCK_PATCH)
        self.assertNotIn(b"\r", (HERE / "e28_sched_patch.py").read_bytes())

    def test_apply_verify_revert_dir(self):
        d = FIX["tmp"] / "t_dir"
        shutil.rmtree(d, ignore_errors=True)
        with zipfile.ZipFile(PAYLOAD) as z:
            z.extract("arc2_duck.py", d)
        orig = (d / "arc2_duck.py").read_bytes()
        self.assertEqual(E28S.tree_state(d), "ORIGINAL")
        self.assertEqual(E28S.verify_e28_sched(d)[0], False)
        r = E28S.apply_e28_sched(d)
        self.assertEqual((r["applied"], r["before"], r["after"], r["py_compile"]["rc"]), ("applied", "ORIGINAL", "FULLY_PATCHED", 0))
        self.assertEqual(hashlib.sha256((d / "arc2_duck.py").read_bytes()).hexdigest(), E28S.DUCK_POST)
        ok, why = E28S.verify_e28_sched(d)
        self.assertTrue(ok, why)
        self.assertEqual(E28S.apply_e28_sched(d / "arc2_duck.py")["applied"], "already_applied")   # file path form
        self.assertEqual(list(d.glob("*.e28tmp")), [])
        rv = E28S.revert_e28_sched(d)
        self.assertEqual((rv["reverted"], rv["after"]), ("reverted", "ORIGINAL"))
        self.assertEqual((d / "arc2_duck.py").read_bytes(), orig)
        self.assertEqual(E28S.revert_e28_sched(d)["reverted"], "not_applied")

    def test_invalid_mixed_refused(self):
        d = FIX["tmp"] / "t_bad"
        shutil.rmtree(d, ignore_errors=True)
        with zipfile.ZipFile(PAYLOAD) as z:
            z.extract("arc2_duck.py", d)
        f = d / "arc2_duck.py"
        bad = f.read_bytes().replace(b"def _rev6()", b"def _rev6( )", 1)
        f.write_bytes(bad)
        self.assertEqual(E28S.tree_state(d), "INVALID_MIXED")
        with self.assertRaises(RuntimeError):
            E28S.apply_e28_sched(d)
        with self.assertRaises(RuntimeError):
            E28S.revert_e28_sched(d)
        self.assertEqual(f.read_bytes(), bad)                         # nothing written
        self.assertEqual(E28S.tree_state(FIX["tmp"] / "no_such_dir_e28"), "INVALID_MIXED")
        crlf = (FIX["tmp"] / "orig" / "arc2_duck.py").read_bytes().replace(b"\n", b"\r\n")
        self.assertEqual(E28S.tree_state(crlf.decode("utf-8")), "INVALID_MIXED")   # CRLF is refused, never patched

    def test_text_mode_and_strict_diff(self):
        t = (FIX["tmp"] / "orig" / "arc2_duck.py").read_bytes().decode("utf-8")
        r = E28S.apply_e28_sched(t)
        self.assertEqual((r["applied"], r["after"]), ("applied", "FULLY_PATCHED"))
        self.assertEqual(r["text"], (FIX["tmp"] / "e28" / "arc2_duck.py").read_bytes().decode("utf-8"))
        self.assertEqual(E28S.apply_e28_sched(r["text"])["applied"], "already_applied")
        self.assertEqual(E28S.revert_e28_sched(r["text"])["text"], t)
        self.assertTrue(E28S.verify_e28_sched(r["text"])[0])
        # strict: one changed context line -> ValueError (no fuzz, no offset)
        lines = t.splitlines(True)
        i = next(i for i, ln in enumerate(lines) if "self._arc2_rev6 = _rev6()" in ln and i > 1400)
        self.assertEqual(i + 1, 1445)                              # a context line of the first hunk (@@ -1444)
        lines[i] = lines[i].rstrip(chr(10)) + "  " + chr(10)
        with self.assertRaises(ValueError):
            E28S.apply_unified("".join(lines), E28S.DUCK_PATCH)
        shifted = "\n" + t                                            # a one-line offset is not tolerated either
        with self.assertRaises(ValueError):
            E28S.apply_unified(shifted, E28S.DUCK_PATCH)

    def test_hunks(self):
        heads = [ln for ln in E28S.DUCK_PATCH.splitlines() if ln.startswith("@@")]
        self.assertEqual(len(heads), 8, heads)
        removed = [ln for ln in E28S.DUCK_PATCH.splitlines() if ln.startswith("-") and not ln.startswith("---")]
        self.assertEqual(len(removed), 3, removed)            # 3 changed lines; everything else is added
        self.assertTrue(all(("b_attempt_budget(branch)" in ln or "def select_attempts_r5" in ln) for ln in removed))


# ------------------------------------------------------------------ 2. classification on the real recorded attempts
class TestClassifyReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = FIX["e28"]
        cls.k100 = _raw_records(K100_PHASE)
        cls.w4 = _raw_records(W4_PHASE)
        cls.k100_end = _solve_end(K100_PHASE)
        cls.w4_end = _solve_end(W4_PHASE)
        cls.qrep = json.loads((K100 / "k100_quality_report.json").read_bytes().decode("utf-8"))

    def _cls(self, recs, end):
        return {k: self.M.e28_classify(r, budget=104000, window_end_epoch=end) for k, r in recs.items()}

    def test_counts(self):
        self.assertEqual(len(self.k100), 60)
        self.assertEqual(len(self.w4), 30)

    def test_b2_b1_time_train_pass(self):
        ck, cw = self._cls(self.k100, self.k100_end), self._cls(self.w4, self.w4_end)
        allc = {("k100",) + k: v for k, v in ck.items()}
        allc.update({("w4",) + k: v for k, v in cw.items()})
        by = {}
        for k, v in allc.items():
            by.setdefault(v["kind"], set()).add(k)
        self.assertEqual(by.get("B2"), {("k100", "16b78196", 1), ("k100", "20270e3b", 0), ("w4", "2b83f449", 0)})
        self.assertEqual(by.get("B1", set()), set())
        self.assertEqual(by.get("FAIL", set()), set())
        self.assertEqual(by.get("TIME"), {("k100", "2d0172a1", 1), ("k100", "3dc255db", 1), ("w4", "16b78196", 1),
                                          ("w4", "2b83f449", 1), ("w4", "64efde09", 1)})
        self.assertEqual(len(by["TRAIN_PASS"]), 90 - 3 - 5)
        # README section 2: klee100 54 accepted + 2 train pass cut before submitting, 4 never passed train; W4 26 + 4
        self.assertEqual(sum(1 for k in ck.values() if k["kind"] == "TRAIN_PASS"), 56)
        self.assertEqual(sum(1 for k in ck.values() if k["end_reason"] == "accepted_submission"), 54)
        self.assertEqual(sum(1 for k in ck.values() if k["end_reason"] == "train_pass_check"), 2)
        self.assertEqual(sum(1 for k in cw.values() if k["end_reason"] == "accepted_submission"), 26)
        # the three B2 cases and their reasons / tokens
        self.assertEqual((ck[("16b78196", 1)]["end_reason"], ck[("16b78196", 1)]["tokens"]), ("CONTEXT_EXHAUSTED", 95825))
        self.assertEqual((ck[("20270e3b", 0)]["end_reason"], ck[("20270e3b", 0)]["tokens"]), ("commit_done_tiny_rest", 103427))
        self.assertEqual((cw[("2b83f449", 0)]["end_reason"], cw[("2b83f449", 0)]["tokens"]), ("commit_done_tiny_rest", 103761))
        # best train match of a B2 case (host-checked candidates): 16b78196 b1 = 1 of 2 pairs; 2b83f449 W4 a1 = 1 of 2
        self.assertEqual({k: ck[("16b78196", 1)]["train"][k] for k in ("full", "k", "n")}, {"full": False, "k": 1, "n": 2})
        self.assertEqual({k: cw[("2b83f449", 0)]["train"][k] for k in ("full", "k", "n")}, {"full": False, "k": 1, "n": 2})

    def test_phase_b_queue_from_attempt1_only(self):
        """Phase B is decided on attempt 1 (branch 0): klee100 -> 20270e3b B2; W4 -> 2b83f449 B2; everything else C2."""
        for recs, end, exp in ((self.k100, self.k100_end, {"20270e3b": "B2"}), (self.w4, self.w4_end, {"2b83f449": "B2"})):
            a1 = {t: self.M.e28_classify(r, budget=104000, window_end_epoch=end) for (t, b), r in recs.items() if b == 0}
            got = {t: self.M.e28_b_phase(c["kind"]) for t, c in a1.items() if self.M.e28_b_phase(c["kind"])}
            self.assertEqual(got, exp)
            self.assertTrue(all(c["kind"] == "TRAIN_PASS" for t, c in a1.items() if t not in exp))

    def test_matches_recorded_end_reasons(self):
        """The recorded k100_quality_report end_reason (nb5_lib.attempt_end) agrees on every attempt; where it says
        running_at_window_end (t1 = the final freeze time) E28 says TIME or, with a host-verified check, TRAIN_PASS."""
        sides = [("k100", self.k100, self.k100_end, self.qrep["quality"]["k100"]["tasks"]),
                 ("w4", self.w4, self.w4_end, self.qrep["quality"]["ref"]["w4_fp8"]["tasks"])]
        n = 0
        for side, recs, end, tasks in sides:
            for tid, row in tasks.items():
                for a in row["attempts"]:
                    r = recs[(tid, a["attempt"] - 1)]
                    c = self.M.e28_classify(r, budget=104000, window_end_epoch=end)
                    n += 1
                    if a["end_reason"] == "running_at_window_end":
                        self.assertIn((c["kind"], c["end_reason"]), {("TIME", "running_at_window_end"),
                                                                     ("TRAIN_PASS", "train_pass_check")}, (side, tid, a))
                    else:
                        self.assertEqual(c["end_reason"], a["end_reason"], (side, tid, a))
        self.assertEqual(n, 90)

    def test_c2_ascending_tokens_from_real_attempt1(self):
        M = self.M
        order = list(self.qrep["quality"]["k100"]["order"])
        idx = {t: i for i, t in enumerate(order)}
        keys = []
        for t in order:
            c = M.e28_classify(self.k100[(t, 0)], budget=104000, window_end_epoch=self.k100_end)
            keys.append((M.e28_c_key(c["kind"] == "TRAIN_PASS", c["tokens"], idx[t]), t))
        c2 = [t for k, t in sorted(keys) if k[0] == 2]
        toks = [self.k100[(t, 0)]["generated_tokens"] for t in c2]
        self.assertEqual(toks, sorted(toks))
        self.assertEqual(c2[0], "1818057f")                     # 3,065 tokens: the cheapest train-pass attempt 1 first
        self.assertEqual(len(c2), 29)                           # 20270e3b goes to B2, not to C2


# ------------------------------------------------------------------ 3. classification rules (synthetic edge cases)
class TestClassifyRules(unittest.TestCase):
    def setUp(self):
        self.M = FIX["e28"]

    def rec(self, **kw):
        r = {"task_id": "t", "branch": 0, "generated_tokens": 50000, "accepted": False, "submissions": [], "checked": [],
             "note": "", "status": "done", "b_budget": 104000, "b_events": {}, "t0": 1000.0, "runs": [{"sec": 100.0}]}
        r.update(kw)
        return r

    def test_rules(self):
        c = self.M.e28_classify
        self.assertEqual(c(self.rec(), at_deadline=False)["kind"], "B1")                          # ended on its own
        self.assertEqual(c(self.rec(), at_deadline=False)["end_reason"], "self_ended")
        self.assertEqual(c(self.rec(), at_deadline=True)["kind"], "TIME")
        self.assertEqual(c(self.rec(), window_end_epoch=1100.0 + 30)["kind"], "TIME")             # within 90 s
        self.assertEqual(c(self.rec(), window_end_epoch=1100.0 + 600)["kind"], "B1")
        self.assertEqual(c(self.rec(generated_tokens=103000))["kind"], "B2")                     # budget spent
        self.assertEqual(c(self.rec(nb5_verdict="BUDGET_LIMITED"))["end_reason"], "BUDGET_LIMITED")
        self.assertEqual(c(self.rec(nb5_verdict="CONTEXT_EXHAUSTED"), at_deadline=True)["kind"], "B2")   # cap first
        self.assertEqual(c(self.rec(b_events={"commit_at_tokens": 99904}))["end_reason"], "forced_commit")
        self.assertEqual(c(self.rec(b_events={"tiny_request_skipped": 1}))["end_reason"], "commit_done_tiny_rest")
        self.assertEqual(c(self.rec(note="analyzer failed 11 times consecutively"))["kind"], "FAIL")
        self.assertEqual(c(self.rec(nb5_verdict="KV_NEED_ABOVE_POOL", note="nb5: KV_NEED_ABOVE_POOL"))["kind"], "FAIL")
        self.assertEqual(c(self.rec(note="nb5: no KV reservation before the admit deadline"))["kind"], "TIME")
        self.assertEqual(c(self.rec(status="paused"))["kind"], "TIME")
        self.assertEqual(c(self.rec(generated_tokens=0), at_deadline=False)["kind"], "FAIL")
        acc = {"accepted": True, "train_exact": [True, True, True], "test_outputs": [[[1]]]}
        self.assertEqual(c(self.rec(checked=[acc], generated_tokens=104000))["kind"], "TRAIN_PASS")   # pass beats cap
        self.assertEqual(c(self.rec(accepted=True, submissions=[acc]))["end_reason"], "accepted_submission")
        self.assertEqual(self.M.e28_b_phase("FAIL"), "B1")
        self.assertEqual(self.M.e28_b_phase("TIME"), None)
        self.assertEqual(self.M.e28_b_phase("TRAIN_PASS"), None)

    def test_priority_order_audits(self):
        """GPT audits 2026-11-01: explicit budget evidence beats a note; a time cut beats the near-budget heuristic;
        INCONCLUSIVE_CONTEXT_LENGTH is a cap verdict."""
        c = self.M.e28_classify
        r = c(self.rec(generated_tokens=103427, b_events={"tiny_request_skipped": 1, "commit_at_tokens": 99904},
                       note="analyzer returned no result"), at_deadline=False)
        self.assertEqual((r["kind"], r["end_reason"]), ("B2", "commit_done_tiny_rest"))      # budget cut + note
        r = c(self.rec(b_events={"commit_at_tokens": 99904}, note="RuntimeError: x"), at_deadline=False)
        self.assertEqual((r["kind"], r["end_reason"]), ("B2", "forced_commit"))              # forced commit + note
        r = c(self.rec(nb5_verdict="INCONCLUSIVE_CONTEXT_LENGTH", note="nb5: tokenize failed"), at_deadline=True)
        self.assertEqual((r["kind"], r["end_reason"]), ("B2", "INCONCLUSIVE_CONTEXT_LENGTH"))
        self.assertEqual(self.M.e28_b_phase(r["kind"]), "B2")
        self.assertIn("INCONCLUSIVE_CONTEXT_LENGTH", self.M.E28_CAP_VERDICTS)
        r = c(self.rec(generated_tokens=103500, status="paused"), at_deadline=False)
        self.assertEqual((r["kind"], r["end_reason"]), ("TIME", "paused_at_window_end"))     # paused near budget
        r = c(self.rec(generated_tokens=103500), at_deadline=True)
        self.assertEqual((r["kind"], r["end_reason"]), ("TIME", "running_at_window_end"))    # at deadline near budget
        r = c(self.rec(generated_tokens=103500), window_end_epoch=1100.0 + 30)
        self.assertEqual(r["kind"], "TIME")                                                  # recorded window end
        r = c(self.rec(generated_tokens=103500), at_deadline=False)
        self.assertEqual((r["kind"], r["end_reason"]), ("B2", "budget_spent"))               # heuristic, no time cut
        r = c(self.rec(note="HTTPError: 500 Server Error"), at_deadline=False)
        self.assertEqual(r["kind"], "FAIL")                                                  # explicit server error
        r = c(self.rec(nb5_verdict="SERVER_ERROR"), at_deadline=False)
        self.assertEqual(r["kind"], "FAIL")
        acc = {"accepted": True, "train_exact": [True, True], "test_outputs": [[[1]]]}
        r = c(self.rec(accepted=True, submissions=[acc], nb5_verdict="CONTEXT_EXHAUSTED", note="x"), at_deadline=True)
        self.assertEqual(r["kind"], "TRAIN_PASS")                                            # accepted wins
        r = c(self.rec(), at_deadline=False)
        self.assertEqual((r["kind"], r["end_reason"]), ("B1", "self_ended"))                 # normal non-accepted end

    def test_inconclusive_context_plans_b2(self):
        """An attempt 1 ending with nb5_verdict INCONCLUSIVE_CONTEXT_LENGTH (set by the notebook wrapper) gets a B2 re-run:
        120k plan, window 130,560; request timeout = E28_B2_REQUEST_TIMEOUT (the notebooks: 9,000) or, unset,
        7,200 x 120k / 104k."""
        M = FIX["e28"]
        tasks = _tasks(2)
        SCRIPTS.clear()
        run0 = M.BranchSession.run

        def wrapped(s, pause_at_tokens=None, tag=None):
            st = run0(s, pause_at_tokens=pause_at_tokens, tag=tag)
            if s.task_id == "t000" and (getattr(s, "e28_plan", None) or {}).get("phase") == "A":
                s.record.update(accepted=False, nb5_verdict="INCONCLUSIVE_CONTEXT_LENGTH")
                s.agent.arc2_submissions.clear()
                s.agent.arc2_accepted = None
            return st
        M.BranchSession.run = wrapped
        try:
            with EnvE28(E28_B2_REQUEST_TIMEOUT=9000):
                recs, el, sl = _solve(M, tasks, list(tasks), workers=1, analyzer_timeout=7200.0)
            with EnvE28():
                recs, el, sl2 = _solve(M, tasks, list(tasks), workers=1, analyzer_timeout=7200.0)
        finally:
            M.BranchSession.run = run0
        rows = {(r["task"], r["phase"]): r for r in sl["e28"]["attempts"]}
        self.assertEqual(rows[("t000", "A")]["end_reason"], "INCONCLUSIVE_CONTEXT_LENGTH")
        b2 = rows[("t000", "B2")]
        self.assertEqual((b2["budget"], b2["window"], b2["session_cap"], b2["request_timeout"]), (120000, 130560, 120000, 9000.0))
        self.assertEqual((b2["agent_budget"], rows[("t000", "A")]["agent_budget"]), (120000, 104000))
        self.assertEqual(rows[("t000", "A")]["request_timeout"], 7200.0)
        self.assertEqual(b2["b_class"], "B2")
        b2d = {(r["task"], r["phase"]): r for r in sl2["e28"]["attempts"]}[("t000", "B2")]
        self.assertAlmostEqual(b2d["request_timeout"], 7200.0 * 120000 / 104000, places=3)

    def test_train_match(self):
        tr = self.M.e28_train
        p = {"accepted": False, "train_exact": [True, False, True], "test_outputs": [[[1]]]}
        self.assertEqual({k: tr(self.rec(checked=[p]))[k] for k in ("full", "k", "n", "source")},
                         {"full": False, "k": 2, "n": 3, "source": "host"})
        self.assertEqual(tr(self.rec(progress={"best": {"k": 1, "n": 4, "acc": 0.5}}))["source"], "progress")
        self.assertEqual(tr(self.rec())["frac"], 0.0)


# ------------------------------------------------------------------ fake Duck agent (the scheduler sees the real BranchSession)
class FakeResult:
    retryable_failure = False
    yielded_control = False
    step_executed = True


SCRIPTS = {}            # (task, phase) -> script; phase in A / B1 / B2 / C


def _grid(v):
    return [[int(v)]]


def make_fake(mod, step=1000, sleep=0.0005):
    class FakeAgent:
        def __init__(self, *, task, task_id, branch, view="facts", model=None, timeout=None, base_url=None,
                     provider=None, save_request_logs=False, dispatch_index=0):
            self.task, self.task_id, self.branch, self.view = task, task_id, branch, view
            pf = getattr(mod, "_e28_plan_for", None)
            self._e28_plan = pf(task_id, branch) if pf else None           # = the patched __init__ hook
            bud = getattr(mod, "_e28_budget", None)
            self._arc2_b_budget = bud(self._e28_plan, branch) if bud else mod.b_attempt_budget(branch)
            self._arc2_window = int(self._e28_plan["window"]) if self._e28_plan and self._e28_plan.get("window") else 130560
            self.timeout, self.dispatch_index = timeout, dispatch_index
            self.generated_tokens = self.total_tokens = 0
            self.arc2_accepted, self.arc2_usage, self.arc2_submissions, self.arc2_checked = None, [], [], []
            self.arc2_progress = {"best": None, "runs": 0}
            self.arc2_rev5_events, self.arc2_rev6_events = {}, {}
            self._arc2_b = False
            ph = (self._e28_plan or {}).get("phase") or {0: "A", 1: "C", 2: "B"}.get(branch, "X")
            self.script = SCRIPTS.get((task_id, ph)) or SCRIPTS.get((task_id, ph[:1])) or {"kind": "pass", "at": 5000}
            self.nq = len(task.get("test") or [1])

        def _cand(self, accepted, value, exact):
            return {"name": "transform", "accepted": accepted, "train_exact": list(exact), "t": time.time(),
                    "test_outputs": [_grid(value)] * self.nq, "branch": self.branch, "origin": "submit"}

        def analyze(self, state_path, _i, *, valid_actions, step_env, transcript_path, analysis_step,
                    request_timeout_seconds, should_stop):
            sc = self.script
            if sc["kind"] == "hang":                      # thinks until the solve window ends (time cut)
                while not should_stop():
                    time.sleep(0.01)
                    self.generated_tokens += 1
                return FakeResult()
            if sc["kind"] == "fail":
                return None                               # Duck: "analyzer returned no result" -> a failure note
            time.sleep(sc.get("sleep", sleep))
            self.generated_tokens += step
            self.total_tokens += step
            self.arc2_usage.append({"prompt": 1000, "completion": step, "cached": 0})
            if sc["kind"] == "pass" and self.generated_tokens >= sc["at"]:
                c = self._cand(True, sc.get("value", 1), [True, True])
                self.arc2_submissions.append(c)
                self.arc2_accepted = c
                self.arc2_progress["best"] = {"name": "transform", "k": 2, "n": 2, "acc": 1.0}
            elif sc["kind"] == "cap" and not self.arc2_checked:
                self.arc2_checked.append(self._cand(False, sc.get("value", 5), sc.get("exact", [True, False])))
                self.arc2_progress["best"] = {"name": "transform", "k": 1, "n": 2, "acc": 0.5}
            return FakeResult()
    return FakeAgent


def _tasks(n, nq=1):
    return {f"t{i:03d}": {"train": [{"input": [[0]], "output": [[0]]}] * 2,
                          "test": [{"input": [[9]]} for _ in range(nq)]} for i in range(n)}


def _solve(mod, tasks, order, workers, deadline_sec=60.0, min_admit=0.2, probe=None, kv_poll=0.05, analyzer_timeout=30.0,
           **kw):
    out = FIX["tmp"] / f"sim_{random.randrange(1 << 30)}"
    t0 = time.monotonic()
    recs = mod.solve_all_r6(tasks, order, out_dir=out, deadline=time.monotonic() + deadline_sec, base_url="http://127.0.0.1:9/v1",
                            model_id="fake", branches=2, workers=workers, agent_cls=make_fake(mod),
                            first_exposure_tokens=None, min_admit_sec=min_admit, capacity_probe=probe, kv_poll_sec=kv_poll,
                            fair_share=False, join_grace_sec=2.0, analyzer_timeout=analyzer_timeout, **kw)
    return recs, time.monotonic() - t0, copy.deepcopy(mod.SCHED_LAST)


def _nb5_lib():
    sys.path.insert(0, str(EXPORTS))
    try:
        import nb5_lib
    finally:
        sys.path.remove(str(EXPORTS))
    return nb5_lib


class ResKit:
    """The real NB5L.Reservation + the N4 hooks exec'd from NOTEBOOK_EDITS + a BranchSession.run wrapper that does what
    _nb5_session_run does (acquire the N2 blocks with the admit deadline, run, release in finally)."""

    def __init__(self, M, cap, mode="projected", f=6, block=4096, log_calls=True):
        L = _nb5_lib()
        self.M, self.L, self.res = M, L, L.Reservation(cap)
        lab, old, new = E28S.NOTEBOOK_EDITS[3]
        self.ns = {"NB5_RES": self.res, "NB5_ADMIT_MODE": mode, "NB5L": L, "MAX_LEN": 131072, "NB5_PROMPT_ALLOW": 8192,
                   "NB5_TOOL_ALLOW": 8192, "NB5_BLOCKS": {"b0": 40, "b1": 40, "start": 8, "f": f, "block": block},
                   "E12A": M, "time": time}
        exec(new[len(old):], self.ns)
        self.calls = []
        if log_calls:
            hook = M.E28_TRY_RESERVE

            def logged(key, plan):
                u0 = self.res.used                     # the blocks in use BEFORE this check
                ok = hook(key, plan)
                self.calls.append((key, plan.get("phase"), ok, u0))
                return ok
            M.E28_TRY_RESERVE = logged
        self.run0 = M.BranchSession.run
        kit = self

        def wrapped(s, pause_at_tokens=None, tag=None):
            p = getattr(s, "e28_plan", None) or {}
            need = L.cdiv(min(131072, 8192 + int(p.get("budget") or 104000) + 8192), block) + f
            key = (s.task_id, s.branch)
            if not kit.res.acquire(key, need, s.deadline - 0.2):
                s.status = "done"
                s.record.update(note="nb5: no KV reservation before the admit deadline")
                s.write()
                return s.status
            try:
                return kit.run0(s, pause_at_tokens=pause_at_tokens, tag=tag)
            finally:
                kit.res.release(key)
        M.BranchSession.run = wrapped

    def close(self):
        self.M.BranchSession.run = self.run0
        self.M.E28_TRY_RESERVE = self.M.E28_RELEASE = None


# ------------------------------------------------------------------ 4. queue order (deterministic: one worker)
class TestQueueOrder(unittest.TestCase):
    def test_single_worker_exact_order(self):
        M = FIX["e28"]
        tasks = _tasks(8)
        order = list(tasks)[::-1]                             # a given order is kept (never re-sorted)
        SCRIPTS.clear()
        SCRIPTS.update({("t007", "A"): {"kind": "pass", "at": 50000}, ("t006", "A"): {"kind": "cap"},
                        ("t006", "B"): {"kind": "pass", "at": 30000, "value": 2},
                        ("t005", "A"): {"kind": "pass", "at": 20000}, ("t004", "A"): {"kind": "fail"},
                        ("t004", "B"): {"kind": "pass", "at": 40000, "value": 2},
                        ("t003", "A"): {"kind": "cap"}, ("t003", "B"): {"kind": "cap"},
                        ("t002", "A"): {"kind": "pass", "at": 90000}, ("t001", "A"): {"kind": "pass", "at": 10000},
                        ("t000", "A"): {"kind": "pass", "at": 20000}})
        for t in tasks:
            SCRIPTS.setdefault((t, "C"), {"kind": "pass", "at": 5000, "value": 3})
        with EnvE28():
            recs, el, sl = _solve(M, tasks, order, workers=1)
        rows = sl["e28"]["attempts"]
        seq = [(r["phase"], r["task"]) for r in sorted(rows, key=lambda r: r["start_seq"])]
        self.assertEqual(sorted(r["start_seq"] for r in rows), list(range(1, len(rows) + 1)))
        # A in the given order; B (task order) = t006 (B2, cap), t004 (B1, failed), t003 (B2); C1 = t003 (still no
        # train pass); C2 by tokens ascending: t001 10k, t005 20k, t000 20k (task order breaks the tie: t005 first),
        # t006 30k (its B), t004 40k (its B), t007 50k, t002 90k
        exp = ([("A", t) for t in order] + [("B2", "t006"), ("B1", "t004"), ("B2", "t003")] +
               [("C", t) for t in ("t003", "t001", "t005", "t000", "t006", "t004", "t007", "t002")])
        self.assertEqual([p for p, _ in seq], [p for p, _ in exp])
        self.assertEqual(seq, exp)
        by = {(r["task"], r["phase"]): r for r in rows}
        self.assertEqual(by[("t003", "C")]["c_priority"], 1)
        self.assertEqual(by[("t006", "C")]["c_key_tokens"], 30000)
        self.assertEqual(by[("t004", "B1")]["b_class"], "FAIL")
        self.assertEqual(by[("t006", "B2")]["budget"], 120000)
        self.assertEqual(by[("t006", "B2")]["window"], 130560)
        self.assertEqual(by[("t003", "B2")]["tokens"], 120000)        # the B2 session cap is 120k
        self.assertEqual(by[("t007", "A")]["budget"], 104000)
        self.assertEqual(by[("t003", "A")]["tokens"], 104000)        # the A session cap is T1
        self.assertEqual(sl["e28"]["started"], {"A": 8, "B1": 1, "B2": 2, "C": 8})
        self.assertEqual(sl["e28"]["queued"], {"A": 8, "B1": 1, "B2": 2, "C1": 1, "C2": 7})
        self.assertEqual(sl["e28"]["ignored_args"]["branches"], 2)
        # answer slots (the user rule) via build_submission
        with EnvE28():
            sub = M.build_submission(tasks, recs)
        self.assertEqual(sub["t006"][0], {"attempt_1": _grid(2), "attempt_2": _grid(3)})   # full B replaces A; C second
        self.assertEqual(sub["t007"][0], {"attempt_1": _grid(1), "attempt_2": _grid(3)})   # A full; C second
        self.assertEqual(sub["t003"][0], {"attempt_1": _grid(5), "attempt_2": _grid(3)})   # A / B partial (tie -> A); C
        slots = sl["e28"]["slots"]
        self.assertEqual((slots["t006"][0]["attempt_1"], slots["t006"][0]["attempt_2"]), ("B", "C"))
        self.assertEqual((slots["t003"][0]["attempt_1"], slots["t003"][0]["attempt_2"]), ("A", "C"))
        self.assertEqual(by[("t006", "B2")]["answer_slots"], ["attempt_1"])
        self.assertEqual(by[("t006", "C")]["answer_slots"], ["attempt_2"])
        self.assertEqual(by[("t006", "A")]["answer_slots"], [None])

    def test_c_key(self):
        k = FIX["e28"].e28_c_key
        keys = [k(True, 40000, 0), k(False, 0, 5), k(True, 30000, 9), k(False, 0, 1), k(True, 30000, 2)]
        self.assertEqual(sorted(keys), [(1, 0, 1), (1, 0, 5), (2, 30000, 2), (2, 30000, 9), (2, 40000, 0)])

    def test_no_order2_notebook_edit(self):
        """N3: with E28 on the order is TASK_IDS as given (no ORDER2 sort); off = the ORDER2 expression unchanged."""
        lab, old, new = E28S.NOTEBOOK_EDITS[2]
        ns = {"os": os, "TASK_IDS": ["c", "a", "b"], "_E12_INDEX": {"c": 0, "a": 1, "b": 2},
              "_e12_order2_score": lambda t: {"a": 2, "b": 1, "c": 0}[t], "_e12_total_cells": lambda t: 0}
        with EnvE28():
            exec(textwrap.dedent(new), ns)
            self.assertEqual(ns["E12_ORDER"], ["c", "a", "b"])
        exec(textwrap.dedent(new), ns)
        self.assertEqual(ns["E12_ORDER"], ["a", "b", "c"])
        exec(textwrap.dedent(old), ns)
        self.assertEqual(ns["E12_ORDER"], ["a", "b", "c"])


# ------------------------------------------------------------------ 5. scheduler simulation (fake engine)
class TestSimulation(unittest.TestCase):
    def test_120_tasks_21_workers(self):
        M = FIX["e28"]
        rnd = random.Random(20261101)
        tasks = _tasks(120, nq=1)
        order = list(tasks)
        SCRIPTS.clear()
        kinds = {}
        for t in order:
            u = rnd.random()
            if u < 0.70:
                SCRIPTS[(t, "A")] = {"kind": "pass", "at": rnd.randrange(3, 100) * 1000}
                kinds[t] = "pass"
            elif u < 0.82:
                SCRIPTS[(t, "A")] = {"kind": "cap"}
                SCRIPTS[(t, "B")] = {"kind": "pass", "at": rnd.randrange(20, 115) * 1000, "value": 2}
                kinds[t] = "cap_bpass"
            elif u < 0.90:
                SCRIPTS[(t, "A")] = {"kind": "cap"}
                SCRIPTS[(t, "B")] = {"kind": "cap"}
                kinds[t] = "cap_bcap"
            else:
                SCRIPTS[(t, "A")] = {"kind": "fail"}
                SCRIPTS[(t, "B")] = {"kind": "pass", "at": 30000, "value": 2}
                kinds[t] = "fail_bpass"
            SCRIPTS[(t, "C")] = {"kind": "pass", "at": rnd.randrange(3, 60) * 1000, "value": 3}
        with EnvE28():
            recs, el, sl = _solve(M, tasks, order, workers=21, deadline_sec=90.0)
            sub = M.build_submission(tasks, recs)
        e = sl["e28"]
        rows = e["attempts"]
        self.assertLess(el, 60.0, "the scheduler must finish (no deadlock) well before the deadline")
        self.assertEqual((sl["stragglers"], sl["worker_errors"], sl["setup_failed"]), (0, 0, 0))
        self.assertEqual(e["left"], {"A": 0, "B": 0, "C": 0})
        by = {}
        for r in rows:
            by.setdefault(r["task"], {})[r["phase"]] = r
        for t in order:
            ph = by[t]
            self.assertIn("A", ph)
            self.assertIn("C", ph)                                    # every task got its attempt 2
            bph = {"pass": None, "cap_bpass": "B2", "cap_bcap": "B2", "fail_bpass": "B1"}[kinds[t]]
            self.assertEqual([p for p in ph if p.startswith("B")], [bph] if bph else [], t)
            self.assertEqual(ph["A"]["budget"], 104000)
            self.assertEqual(ph["C"]["budget"], 104000)
            if bph == "B2":
                self.assertEqual((ph["B2"]["budget"], ph["B2"]["window"], ph["B2"]["session_cap"]), (120000, 130560, 120000))
                self.assertGreater(ph["B2"]["request_timeout"], 30.0)  # B2's request timeout scales with its budget
            if bph == "B1":
                self.assertEqual((ph["B1"]["budget"], ph["B1"]["window_plan"]), (104000, None))
            exp_cp = 1 if kinds[t] == "cap_bcap" else 2
            self.assertEqual(ph["C"]["c_priority"], exp_cp, t)
            # answer slots
            a1 = {"pass": 1, "cap_bpass": 2, "cap_bcap": 5, "fail_bpass": 2}[kinds[t]]
            self.assertEqual(sub[t][0], {"attempt_1": _grid(a1), "attempt_2": _grid(3)}, t)
        # A drains before B, B before C: every B started with an empty A queue, every C with empty A and B queues
        for r in rows:
            q = r["queues_at_start"]
            if r["phase"].startswith("B"):
                self.assertEqual(q[0], 0, r)
            if r["phase"] == "C":
                self.assertEqual(q[:2], [0, 0], r)
        last_a = max(r["start_seq"] for r in rows if r["phase"] == "A")
        self.assertGreater(min(r["start_seq"] for r in rows if r["phase"] != "A"), last_a)
        self.assertEqual(last_a, 120)                              # the 120 attempt-1 starts are the first 120 admissions
        self.assertEqual(sum(e["started"].values()), len(rows))
        self.assertTrue(set(r["worker_slot"] for r in rows) <= set(range(21)))
        self.assertGreater(len(set(r["worker_slot"] for r in rows)), 10)   # the slots were really used in parallel

    def test_kv_gate_waits_never_deadlocks(self):
        M = FIX["e28"]
        tasks = _tasks(6)
        SCRIPTS.clear()
        t_on = time.monotonic() + 0.6
        probe_hi = lambda: {"kv_usage": 0.99}                         # always above the gate: idle override only
        probe_mid = lambda: {"kv_usage": 0.99 if time.monotonic() < t_on else 0.1}
        with EnvE28():
            recs, el, sl = _solve(M, tasks, list(tasks), workers=4, probe=probe_hi, kv_admit_limit=0.95)
            self.assertEqual(sl["e28"]["started"]["A"], 6)
            self.assertGreaterEqual(sl["kv_idle_overrides"], 1)       # nothing runs + the gate holds: admitted anyway
            self.assertGreater(sl["kv_holds"], 0)
            self.assertLess(el, 30.0)
            recs, el, sl = _solve(M, tasks, list(tasks), workers=4, probe=probe_mid, kv_admit_limit=0.95)
            self.assertEqual(sl["e28"]["done"]["C"], 6)

    def test_time_cut_is_not_b(self):
        M = FIX["e28"]
        tasks = _tasks(5)
        SCRIPTS.clear()
        SCRIPTS[("t002", "A")] = {"kind": "hang"}
        with EnvE28():
            recs, el, sl = _solve(M, tasks, list(tasks), workers=3, deadline_sec=4.0, min_admit=0.5)
        rows = {(r["task"], r["phase"]): r for r in sl["e28"]["attempts"]}
        self.assertEqual(rows[("t002", "A")]["kind"], "TIME")
        self.assertEqual(rows[("t002", "A")]["end_reason"], "running_at_window_end")
        self.assertFalse(any(t == "t002" and p != "A" for t, p in rows))     # no B, no C for the time-cut task
        self.assertEqual(sl["e28"]["a_class"]["t002"]["kind"], "TIME")
        self.assertLess(el, 4.0 + 2.0 + 3.0)

    def test_backfill_b2_waits_c_starts(self):
        """KV is the constraint (real Reservation, capacity 73 blocks: two T1 attempts (36 each) fit, a B2 (38) fits only
        alone): while the B2 waits, a free worker starts a C attempt (backfill, no idle); the B2 is checked FIRST at every
        admission check and starts at the first check where it fits; no reservation leaks."""
        M = FIX["e28"]
        tasks = _tasks(6)
        SCRIPTS.clear()
        SCRIPTS[("t000", "A")] = {"kind": "cap", "sleep": 0.0}
        SCRIPTS[("t000", "B")] = {"kind": "pass", "at": 20000, "value": 2}
        SCRIPTS[("t000", "C")] = {"kind": "pass", "at": 5000, "value": 3}
        for t_ in list(tasks)[1:]:
            SCRIPTS[(t_, "A")] = {"kind": "pass", "at": 10000, "sleep": 0.01}
            SCRIPTS[(t_, "C")] = {"kind": "pass", "at": 5000, "sleep": 0.01, "value": 3}
        # t001's attempt 1 keeps 36 blocks busy for ~2 s: the B2 (38) cannot fit beside it (74 > 73) while the short
        # attempt 1s end and their C items (36) can (72 <= 73): the backfill is deterministic, not a timing race
        SCRIPTS[("t001", "A")] = {"kind": "pass", "at": 100000, "sleep": 0.02}
        kit = ResKit(M, cap=73)
        try:
            with EnvE28():
                recs, el, sl = _solve(M, tasks, list(tasks), workers=3, deadline_sec=60.0)
                sub = M.build_submission(tasks, recs)
        finally:
            kit.close()
        e = sl["e28"]
        rows = {(r["task"], r["phase"]): r for r in e["attempts"]}
        b2 = rows[("t000", "B2")]
        self.assertEqual(e["started"], {"A": 6, "B1": 0, "B2": 1, "C": 6})
        self.assertGreater(b2["backfill_starts_while_waiting"], 0)
        self.assertGreater(b2["fit_fails"], 0)
        self.assertGreater(b2["wait_sec"], 0.0)
        self.assertGreater(e["backfill_starts"], 0)
        self.assertTrue(any(x["started"][0] == "C" and ["t000", 2] in x["skipped"] for x in e["backfill_log"]),
                        e["backfill_log"])
        # B2 checked first at every check once it waited; it started at the first check where it fitted
        key = ("t000", 2)
        b2_calls = [i for i, c in enumerate(kit.calls) if c[0] == key]
        first_fail = b2_calls[0]
        self.assertFalse(kit.calls[first_fail][2])
        oks = [i for i in b2_calls if kit.calls[i][2]]
        self.assertEqual(len(oks), 1)                          # granted once = started at that check
        for i in range(first_fail + 1, oks[0]):
            c = kit.calls[i]
            if c[0] != key and c[2]:                         # a backfill grant: this check tried the B2 first
                self.assertEqual(kit.calls[i - 1][0], key)
                self.assertFalse(kit.calls[i - 1][2])
        self.assertLessEqual(kit.calls[oks[0]][3] + 38, 73)  # it fitted at that check
        self.assertEqual((kit.res.used, kit.res.held), (0, {}))
        self.assertEqual(sub["t000"][0], {"attempt_1": _grid(2), "attempt_2": _grid(3)})

    def test_no_kv_constraint_no_idle_and_abc_order(self):
        """No fit hook (= KV not the constraint) with 6 workers and 40 tasks: idle worker time is exactly 0, nothing is
        backfilled, every B starts with an empty A queue and every C with empty A and B queues."""
        M = FIX["e28"]
        rnd = random.Random(7)
        tasks = _tasks(40)
        SCRIPTS.clear()
        for t_ in tasks:
            SCRIPTS[(t_, "A")] = {"kind": "cap"} if rnd.random() < 0.2 else {"kind": "pass", "at": rnd.randrange(3, 60) * 1000}
            SCRIPTS[(t_, "B")] = {"kind": "pass", "at": 30000, "value": 2}
        with EnvE28():
            recs, el, sl = _solve(M, tasks, list(tasks), workers=6, deadline_sec=60.0)
        e = sl["e28"]
        self.assertEqual(e["idle_worker_seconds"], 0.0)
        self.assertEqual((e["idle_sec_by_reason"]["nothing_fits"], e["idle_sec_by_reason"]["kv_gate"]), (0.0, 0.0))
        self.assertEqual((e["backfill_starts"], e["fit_checks"], e["fit_hook"]), (0, 0, False))
        for r in e["attempts"]:
            if r["phase"].startswith("B"):
                self.assertEqual(r["queues_at_start"][0], 0, r)
            if r["phase"] == "C":
                self.assertEqual(r["queues_at_start"][:2], [0, 0], r)

    def test_slots_mode_no_idle_b2_at_its_turn(self):
        """The notebooks' mode ("slots", as K100Q): Reservation capacity = workers, 1 slot per attempt, the KV gate =
        the reservation fraction (limit 1.0, a 5-s probe like the notebooks). Every free worker takes the next item at
        once: no idle while the queues are non-empty, no fit failure, no backfill; the B2 starts at its turn (right
        after the last attempt 1, before every C); B -> C order as without KV; no slot leaks."""
        M = FIX["e28"]
        rnd = random.Random(5)
        tasks = _tasks(40)
        SCRIPTS.clear()
        b2 = []
        for t_ in tasks:
            if rnd.random() < 0.15:
                SCRIPTS[(t_, "A")] = {"kind": "cap"}
                b2.append(t_)
            else:
                SCRIPTS[(t_, "A")] = {"kind": "pass", "at": rnd.randrange(3, 100) * 1000}
            SCRIPTS[(t_, "B")] = {"kind": "pass", "at": rnd.randrange(20, 150) * 1000, "value": 2}
        W = 6
        kit = ResKit(M, cap=W, mode="slots", log_calls=False)
        probe = lambda: {"kv_usage": kit.res.frac()}
        try:
            with EnvE28():
                recs, el, sl = _solve(M, tasks, list(tasks), workers=W, deadline_sec=90.0, probe=probe, kv_poll=5.0,
                                      kv_admit_limit=1.0)
        finally:
            kit.close()
        e = sl["e28"]
        self.assertGreater(len(b2), 2)
        self.assertEqual(e["left"], {"A": 0, "B": 0, "C": 0})
        self.assertEqual((e["fit_fails"], e["backfill_starts"], e["forced_admits"]), (0, 0, 0))
        self.assertLess(e["idle_worker_seconds"], 0.05, e["idle_sec_by_reason"])
        self.assertEqual(e["idle_sec_by_reason"]["nothing_fits"], 0.0)
        self.assertEqual((kit.res.used, kit.res.held), (0, {}))
        self.assertEqual(kit.res.stats["e28_try_grants"], sum(e["started"].values()))
        rows = sorted(e["attempts"], key=lambda r: r["start_seq"])
        seq = [r["phase"] for r in rows]
        nA = seq.count("A")
        self.assertEqual(seq[:nA], ["A"] * nA)                       # A first
        self.assertEqual(seq[nA:nA + len(b2)], ["B2"] * len(b2))     # the B2s at their turn, before any C
        self.assertEqual([r["task"] for r in rows[nA:nA + len(b2)]], b2)   # task order
        for r in rows:
            if r["phase"] == "B2":
                self.assertEqual((r["fit_fails"], r["backfill_starts_while_waiting"]), (0, 0))
                self.assertEqual(r["queues_at_start"][0], 0)
            if r["phase"] == "C":
                self.assertEqual(r["queues_at_start"][:2], [0, 0], r)

    def test_reservation_sim_no_deadlock_no_leak(self):
        """60 tasks, 12 workers, the real Reservation (projected, 400 blocks = 10 T1 or a mix with B2s): everything
        finishes, every B2 starts, idle happens only while nothing fits, no reservation is left held."""
        M = FIX["e28"]
        rnd = random.Random(11)
        tasks = _tasks(60)
        SCRIPTS.clear()
        b2_tasks = set()
        for t_ in tasks:
            if rnd.random() < 0.2:
                SCRIPTS[(t_, "A")] = {"kind": "cap"}
                b2_tasks.add(t_)
            else:
                SCRIPTS[(t_, "A")] = {"kind": "pass", "at": rnd.randrange(3, 100) * 1000}
            SCRIPTS[(t_, "B")] = {"kind": "pass", "at": rnd.randrange(20, 150) * 1000, "value": 2}
        kit = ResKit(M, cap=400, log_calls=False)
        try:
            with EnvE28():
                recs, el, sl = _solve(M, tasks, list(tasks), workers=12, deadline_sec=90.0)
        finally:
            kit.close()
        e = sl["e28"]
        self.assertLess(el, 60.0)
        self.assertEqual(e["left"], {"A": 0, "B": 0, "C": 0})
        self.assertEqual(e["started"]["B2"], len(b2_tasks))
        self.assertEqual(e["started"]["C"], 60)
        self.assertEqual((kit.res.used, kit.res.held), (0, {}))
        self.assertEqual(kit.res.stats["e28_try_grants"] + e["forced_admits"], sum(e["started"].values()))
        self.assertEqual(e["idle_sec_by_reason"]["kv_gate"], 0.0)
        self.assertGreaterEqual(e["fit_fails"], 1)
        b2_rows = [r for r in e["attempts"] if r["phase"] == "B2"]
        self.assertTrue(all(r["wait_sec"] is not None and r["backfill_starts_while_waiting"] >= 0 for r in b2_rows))

    def test_admission_window(self):
        M = FIX["e28"]
        tasks = _tasks(3)
        with EnvE28():
            recs, el, sl = _solve(M, tasks, list(tasks), workers=2, deadline_sec=1.0, min_admit=5.0)
        self.assertEqual(sl["e28"]["started"]["A"], 0)                      # less than min_admit_sec left: nothing starts
        self.assertEqual(sl["e28"]["left"]["A"], 3)
        self.assertLess(el, 5.0)


# ------------------------------------------------------------------ 6. env-off identity
def _norm(recs):
    drop = {"t0", "t1", "sec", "publish_seq", "t"}

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k not in drop}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    return {t: sorted((clean(r) for r in rs), key=lambda r: r.get("branch", 0)) for t, rs in recs.items()}


class TestEnvOffIdentity(unittest.TestCase):
    def test_scheduler_identical(self):
        tasks = _tasks(10)
        SCRIPTS.clear()
        for i, t in enumerate(tasks):
            SCRIPTS[(t, "A")] = {"kind": "pass", "at": (i + 1) * 7000} if i % 3 else {"kind": "cap"}
            SCRIPTS[(t, "C")] = {"kind": "pass", "at": 4000, "value": 3}
        for k in E28_KEYS:
            os.environ.pop(k, None)
        out = []
        for name in ("orig", "e28"):
            recs, el, sl = _solve(FIX[name], tasks, list(tasks), workers=1, deadline_sec=60.0, max_generated_tokens=50000)
            out.append((_norm(recs), {k: v for k, v in sl.items() if k not in ("durations", "gauges", "events")}))
        self.assertEqual(out[0][0], out[1][0])
        self.assertEqual(out[0][1], out[1][1])
        self.assertNotIn("e28", out[1][1])
        self.assertEqual(sorted(len(v) for v in out[1][0].values()), [2] * 10)    # b0 + b1, the rev 6 scheduler

    def test_selection_identical_on_real_records(self):
        if not FIX["eval"]:
            self.skipTest("evaluation challenges not found")
        for k in E28_KEYS:
            os.environ.pop(k, None)
        for phase_dir in (K100_PHASE, W4_PHASE):
            raw = _raw_records(phase_dir)
            recs = {}
            for (t, b), r in raw.items():
                recs.setdefault(t, []).append(r)
            tasks = {t: FIX["eval"][t] for t in recs}
            a = FIX["orig"].build_submission(tasks, recs)
            b = FIX["e28"].build_submission(tasks, recs)
            self.assertEqual(a, b)
            for t in tasks:
                lists = []
                self.assertEqual(FIX["orig"].select_attempts_r5(tasks[t], recs[t]),
                                 FIX["e28"].select_attempts_r5(tasks[t], recs[t], _e28_lists=lists))
                self.assertEqual(len(lists), len(tasks[t]["test"]))

    def test_e28_selection_on_real_records(self):
        """E28 on, real K100Q records (b0 = A, b1 = C): every task where both attempts are train-pass keeps A first."""
        if not FIX["eval"]:
            self.skipTest("evaluation challenges not found")
        raw = _raw_records(K100_PHASE)
        recs = {}
        for (t, b), r in raw.items():
            recs.setdefault(t, []).append(r)
        tasks = {t: FIX["eval"][t] for t in recs}
        with EnvE28():
            sub = FIX["e28"].build_submission(tasks, recs)
            tab = FIX["e28"].e28_slot_table(tasks, recs)
        self.assertEqual(set(sub), set(tasks))
        for t in tasks:
            self.assertEqual(len(sub[t]), len(tasks[t]["test"]))
            for x in tab[t]:
                self.assertIn(x["attempt_1"], ("A", "r5"))
        # 20270e3b: A (b0) only partial, C (b1) train pass -> attempt_1 = A's partial, attempt_2 = C's accepted check
        self.assertEqual((tab["20270e3b"][0]["attempt_1"], tab["20270e3b"][0]["attempt_2"]), ("A", "C"))


# ------------------------------------------------------------------ 7. answer-slot rule (synthetic records)
class TestSlots(unittest.TestCase):
    def setUp(self):
        self.M = FIX["e28"]
        self.task = {"train": [{"input": [[0]], "output": [[0]]}] * 2, "test": [{"input": [[9]]}]}

    def r(self, b, phase=None, acc=None, part=None):
        subs = [{"accepted": True, "train_exact": [True, True], "test_outputs": [_grid(acc)], "t": 10 + b}] if acc is not None else []
        chk = [{"accepted": False, "train_exact": part[1], "test_outputs": [_grid(part[0])], "t": 5 + b}] if part else []
        r = {"task_id": "x", "branch": b, "submissions": subs, "checked": chk, "accepted": acc is not None}
        if phase:
            r["e28"] = {"phase": phase}
        return r

    def sel(self, recs):
        rows = []
        out = self.M.select_attempts_e28(self.task, recs, _slots_out=rows)
        return (out[0]["attempt_1"][0][0], out[0]["attempt_2"][0][0], rows[0]["attempt_1"], rows[0]["attempt_2"])

    def test_rules(self):
        r = self.r
        self.assertEqual(self.sel([r(0, acc=1), r(1, acc=3)]), (1, 3, "A", "C"))
        self.assertEqual(self.sel([r(0, acc=1), r(1, acc=1)]), (1, 9, "A", "input"))                # C same -> next
        self.assertEqual(self.sel([r(0, part=(5, [True, False])), r(2, "B2", acc=2), r(1, acc=3)]), (2, 3, "B", "C"))
        self.assertEqual(self.sel([r(0, part=(5, [True, False])), r(2, "B2", acc=2)]), (2, 5, "B", "A"))   # fallback
        self.assertEqual(self.sel([r(0, part=(5, [True, False])), r(2, "B1", part=(6, [False, False]))]), (5, 6, "A", "B"))
        self.assertEqual(self.sel([r(0, part=(5, [False, False])), r(2, "B1", part=(6, [True, False]))]), (6, 5, "B", "A"))
        self.assertEqual(self.sel([r(0, part=(5, [True, False])), r(2, "B1", part=(6, [True, False]))]), (5, 6, "A", "B"))  # tie
        self.assertEqual(self.sel([r(0, part=(5, [True, False])), r(1, part=(7, [False, False]))]), (5, 7, "A", "C"))
        self.assertEqual(self.sel([r(1, acc=3)]), (3, 9, "r5", "input"))
        self.assertEqual(self.sel([]), (9, 9, "r5", "same"))
        # without e28 fields the phase comes from the branch: 0 = A, 1 = C, 2 = B
        self.assertEqual(self.sel([r(0, part=(5, [True, False])), r(2, acc=2), r(1, acc=3)]), (2, 3, "B", "C"))


# ------------------------------------------------------------------ 8. the real Arc2ToolAgent hook (budget, window, prompt)
class TestAgentHook(unittest.TestCase):
    def test_init_budget_window_prompt(self):
        if not FIX["eval"]:
            self.skipTest("evaluation challenges not found")
        task = FIX["eval"]["16b78196"]
        env = {"E12_ARC2_REV6": "1", "E12_B_MODE": "1", "E12_B_ATTEMPT_TOKENS": "104000", "E12_B_ATTEMPT_TOKENS_B1": "104000",
               "E12_DUCK_CONTEXT_WINDOW": "130560"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            mk = lambda mod, b: mod.make_agent_class()(task=task, task_id="16b78196", branch=b, view="facts", model="fake",
                                                       timeout=30, base_url="http://127.0.0.1:9/v1", provider="vllm",
                                                       save_request_logs=False, dispatch_index=0)
            M = FIX["e28"]
            for b in (0, 1, 2):
                a0, a1 = mk(FIX["orig"], b), mk(M, b)           # E28 off: identical
                self.assertEqual((a0._system_prompt, a0._arc2_b_budget, a0._arc2_window),
                                 (a1._system_prompt, a1._arc2_b_budget, a1._arc2_window))
                self.assertIsNone(a1._e28_plan)
            with EnvE28():
                M.E28_PLANS.clear()
                a = mk(M, 2)                                     # E28 on, no plan (not created by the scheduler): original
                self.assertEqual((a._arc2_b_budget, a._arc2_window), (104000, 130560))
                M.E28_PLANS[("16b78196", 2)] = {"phase": "B2", "budget": 120000, "window": 130560, "branch": 2}
                M.E28_PLANS[("16b78196", 1)] = {"phase": "C", "budget": 104000, "window": None, "branch": 1}
                b2, c = mk(M, 2), mk(M, 1)
                self.assertEqual((b2._arc2_b_budget, b2._arc2_window), (120000, 130560))
                self.assertIn("about 120,000 generated tokens", b2._system_prompt)
                self.assertEqual((c._arc2_b_budget, c._arc2_window), (104000, 130560))
                self.assertIn("about 104,000 generated tokens", c._system_prompt)
                self.assertEqual(b2.generated_tokens, 0)
                self.assertEqual(b2._arc2_apply_cap(), 120000 - 4096)        # the first B2 request may think 115,904
                M.E28_PLANS.clear()
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# ------------------------------------------------------------------ 9. notebook text edits N1-N3 on the real exports
class TestNotebookEdits(unittest.TestCase):
    def test_exports(self):
        for name in ("nb6q_code_export.py", "e26_k100q_code_export.py"):
            f = EXPORTS / name
            if not f.is_file():
                self.skipTest(f"{name} not found")
            t = f.read_bytes().decode("utf-8").replace("\r\n", "\n")
            new, done = E28S.apply_notebook_edits(t)
            self.assertEqual(len(done), 4, name)
            compile(new, name, "exec")
            with self.assertRaises(ValueError):                  # each edit must match exactly once (already applied)
                E28S.apply_notebook_edits(new)

    def test_n4_try_reserve(self):
        """N4: grants now or refuses now (never waits), the attempt's own acquire then finds its key held; idle grant;
        live hold; a need above the capacity is left to _nb5_session_run (C-7); slots mode = 1 block."""
        M = FIX["e28"]
        kit = ResKit(M, cap=73, log_calls=False)
        try:
            hook, rel, res = M.E28_TRY_RESERVE, M.E28_RELEASE, kit.res
            t1, b2 = {"budget": 104000, "branch": 0}, {"budget": 120000, "branch": 2}
            t_ = time.monotonic()
            self.assertTrue(hook(("a", 0), t1))                      # 36
            self.assertTrue(hook(("b", 0), t1))                      # 72
            self.assertFalse(hook(("c", 2), b2))                     # 72 + 38 > 73: refused at once
            self.assertLess(time.monotonic() - t_, 0.5)
            self.assertEqual(res.used, 72)
            self.assertTrue(res.acquire(("a", 0), 36, time.monotonic()))   # the wrapper's acquire: already held
            self.assertEqual(res.used, 72)
            rel(("a", 0))
            rel(("a", 0))                                            # idempotent
            self.assertFalse(hook(("c", 2), b2))                     # 36 + 38 > 73
            rel(("b", 0))
            self.assertTrue(hook(("c", 2), b2))                      # idle grant
            self.assertEqual(res.used, 38)
            small = {"budget": 1000, "branch": 0}                    # cdiv(17,384, 4,096) + 6 = 11 blocks
            res.set_live_hold("server waiting")
            self.assertFalse(hook(("d", 0), small))                  # fits, but the live hold refuses
            res.set_live_hold(None)
            self.assertTrue(hook(("d", 0), small))                   # 38 + 11 = 49
            self.assertFalse(hook(("h", 0), t1))                     # 49 + 36 > 73
            kit.ns["MAX_LEN"] = 10 ** 7
            self.assertTrue(hook(("e", 2), {"budget": 900000, "branch": 2}))   # above the capacity: C-7 denies later
            kit.ns["MAX_LEN"] = 131072
            self.assertNotIn(("e", 2), res.held)
            kit.ns["NB5_ADMIT_MODE"] = "slots"
            rel(("c", 2))
            rel(("d", 0))
            self.assertTrue(hook(("f", 2), b2))
            self.assertEqual(res.held[("f", 2)], 1)
            kit.ns["NB5_RES"] = None
            self.assertTrue(hook(("g", 0), t1))                      # no reservation: everything fits
        finally:
            kit.close()

    def test_n1_n2_values(self):
        _, n1_old, n1_new = E28S.NOTEBOOK_EDITS[0]
        fn = "def _nb5_peak_len(self):\n" + n1_new + "    return int(min(MAX_LEN, pk)) if pk else MAX_LEN\n"
        ns = {"NB5_BLOCKS": {"peak_b0": 120384, "peak_b1": 120384}, "NB5_PROMPT_ALLOW": 8192, "NB5_TOOL_ALLOW": 8192,
              "MAX_LEN": 131072}
        exec(fn, ns)

        class A:
            arc2_branch = 2
        a = A()
        self.assertEqual(ns["_nb5_peak_len"](a), 120384)                       # no plan: NB5's peak (b1 key)
        a._e28_plan = {"budget": 120000}
        self.assertEqual(ns["_nb5_peak_len"](a), 131072)                       # B2: min(131,072, 8,192 + 120,000 + 8,192)
        a._e28_plan = {"budget": 104000}
        self.assertEqual(ns["_nb5_peak_len"](a), 120384)                       # T1 attempts: unchanged value
        _, n2_old, n2_new = E28S.NOTEBOOK_EDITS[1]

        class L:
            @staticmethod
            def cdiv(x, y):
                return -(-int(x) // int(y))

        class S:
            branch = 2
        s = S()
        ns2 = {"NB5L": L, "MAX_LEN": 131072, "NB5_PROMPT_ALLOW": 8192, "NB5_TOOL_ALLOW": 8192, "NB5_ADMIT_MODE": "projected",
               "NB5_BLOCKS": {"b0": 40, "b1": 40, "start": 8, "f": 6, "block": 4096}, "self": s}
        exec(textwrap.dedent(n2_new), ns2)
        self.assertEqual(ns2["need"], 40)                                       # no plan: NB5's b1 blocks
        s.e28_plan = {"budget": 120000}
        exec(textwrap.dedent(n2_new), ns2)
        self.assertEqual(ns2["need"], 32 + 6)                                   # cdiv(131072, 4096) = 32, + F 6
        ns2["NB5_ADMIT_MODE"] = "live"
        exec(textwrap.dedent(n2_new), ns2)
        self.assertEqual(ns2["need"], 8)                                        # live mode unchanged
        ns2["NB5_ADMIT_MODE"] = "projected"
        s.e28_plan = {"budget": 104000}
        exec(textwrap.dedent(n2_new), ns2)
        self.assertEqual(ns2["need"], 30 + 6)                                   # T1: cdiv(120384, 4096) = 30 + 6


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1, argv=[a for a in sys.argv if a != "-v"])
