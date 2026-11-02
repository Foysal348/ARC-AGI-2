"""E33 ATTEMPT CHAIN scheduler: the MERGED suite  python e33_sched_tests_all.py [-v]   (exit 0 = all pass)

= S's e33_sched_tests.py (38 tests, run first, unchanged scenarios; they run under E33_A1_ORDER=e28, the first-build answer
rule) + every unique scenario of the three GPT scheduler builds (GPT/gpt_sched_v1, v2, v3: e33_sched_tests.py and v3's
stress_e33.py) ported to S's real API (env names E33_SCHED / E33_POETIQ / E33_DIRECT / E33_ATTEMPT2 / budgets / E33_A1_ORDER,
BRANCH_MAP / SUPPLY_BRANCH_BASE, E33_CAN_START gets "C" / "D" / "E", S's report keys) + the tests of the user's answer-order
decision of 2026-11-02 (E33_A1_ORDER = direct_first default | e28) + the stress runs (120 tasks x 24 workers, all stages,
both attempts; again with the real NB5 Reservation; again with the solve window ending mid-run).
Harness = the E29 ST harness with the REAL Duck payload (through e33_sched_tests: run33, the stub hooks, Env33).
Source tags in every docstring: [S] [v1] [v2] [v3] = the test file the scenario comes from. Where a GPT expectation follows a
GPT-only design choice, the test asserts the spec (DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md) / S behaviour and says why.
Negative control: E33_TEST_NEGCTL=1 (the unpatched E29 post file is the module under test; the E33 tests must fail).
"""
import copy
import os
import random
import sys
import threading
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e33_sched_tests as T      # noqa: E402  S's suite + the harness (ST, FIX, run33, hooks, Env33, record builders)

ST, FIX, S33 = T.ST, T.FIX, T.S33
g, rows, chain_of, run33, Env33, BR, ATT = T.g, T.rows, T.chain_of, T.run33, T.Env33, T.BR, T.ATT
NEGCTL = T.NEGCTL


def setUpModule():
    T.setUpModule()


def tearDownModule():
    T.tearDownModule()


def load_tests(loader, tests, pattern):
    """S's 38 tests first (their own module fixture), then the merged ones (this module's fixture)."""
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(T))
    for name in MERGED:
        suite.addTests(loader.loadTestsFromTestCase(globals()[name]))
    return suite


# ------------------------------------------------------------------ shared checks
def invariants(tc, r, window=False):
    """starts == done == releases per (task, branch); never two stages of one (task, attempt) at once; no straggler; no
    worker / route error; no lost record; the queue empty at the end (unless the window ended mid-run)."""
    e = r.e
    tc.assertEqual(e["setup_failed"], [])
    tc.assertEqual((r.sl["worker_errors"], r.sl.get("stage2_errors", 0), r.sl["stragglers"]), (0, 0, 0))
    taken = [(x["task"], x["branch"]) for x in e["attempts"]]
    tc.assertEqual(len(taken), len(set(taken)))
    starts, ends = {}, {}
    for ev, tid, b, ph, t in r.runs:
        d = starts if ev == "start" else ends
        d[(tid, b)] = d.get((tid, b), 0) + 1
    for c in r.hooks:
        k = (c["tid"], c["plan"]["branch"])
        starts[k] = starts.get(k, 0) + 1
        if c["t1"] is not None:
            ends[k] = ends.get(k, 0) + 1
    tc.assertTrue(all(v == 1 for v in starts.values()), {k: v for k, v in starts.items() if v != 1})
    tc.assertEqual(starts, ends)                                                  # every start ended
    ran = {(x["task"], x["branch"]) for x in e["attempts"] if x.get("status") != "skipped" or x.get("skipped_after_start")}
    tc.assertEqual(set(starts), ran)
    for k in taken:                                                               # one release per taken item
        tc.assertEqual(r.rel.count(k), 1, (k, r.rel.count(k)))
    tc.assertEqual(set(r.rel), set(taken))
    for ph in S33.BRANCH_MAP:
        tc.assertEqual(e["started"][ph], e["done"][ph], ph)
        st = e["e33"]["stages"][ph]
        tc.assertEqual(st["taken"], st["started"] + sum(st["skipped"].values()) - st["skipped_after_start"], ph)
        if not window:
            tc.assertEqual(st["left"], 0, ph)
    for (tid, a), iv in T._intervals(r).items():                                  # strictly sequential per attempt
        iv.sort()
        for (s0, e0, b0), (s1, e1, b1) in zip(iv, iv[1:]):
            tc.assertGreaterEqual(s1, e0, (tid, a, b0, b1))
    for t, rs in r.recs.items():                                                  # no lost / duplicated record
        bs = [x["branch"] for x in rs]
        tc.assertEqual(len(bs), len(set(bs)), t)
        done_rows = {x["branch"] for x in e["attempts"] if x["task"] == t and x.get("status") not in ("skipped", "no_record")}
        tc.assertEqual(set(bs) & set(BR.values()), done_rows & set(BR.values()), t)
    if not window:
        tc.assertEqual((r.sl["heap_left"], r.sl["stage1_left"]), (0, 0))
        tc.assertEqual(sum(s["left"] for s in e["e33"]["stages"].values()), 0)


class HookKit:
    """A scripted E28_TRY_RESERVE (fit hook) for one run (run33's kit=)."""
    def __init__(self, M, fn):
        self.M, self.calls = M, []

        def hook(k, p):
            self.calls.append((tuple(k), p.get("phase")))
            return fn(k, p)
        M.E28_TRY_RESERVE = hook

    def close(self):
        self.M.E28_TRY_RESERVE = None


# ------------------------------------------------------------------ A. ported chain / scheduler scenarios
class TestPortedChain(unittest.TestCase):
    def test_full_chain_via_b2_both_attempts(self):
        """[v3 ChainTests.test_full_chain_and_attempt2] A cap -> B2 -> C -> D -> E on both attempts, 1 worker: 10 records,
        10 releases (one per branch), B2 on branches 2 / 122 with the E28 B2 budget and window."""
        sc = {("t000", "A"): {"kind": "cap", "sleep": 0.0}, ("t000", "B"): {"kind": "partial", "at": 2000}}
        r = run33(sc, n=1)
        for a in (1, 2):
            self.assertEqual(chain_of(r, "t000", a), [f"{p}:{a}" for p in ("A", "B2", "C-POETIQ1", "D-POETIQ2", "E-DIRECT")])
        self.assertEqual(len(r.recs["t000"]), 10)
        self.assertEqual(sorted(r.rel), sorted(("t000", b) for b in BR.values()))
        by = {x["phase"]: x for x in rows(r)}
        for ph, b in (("B2:1", 2), ("B2:2", 122)):
            self.assertEqual((by[ph]["branch"], by[ph]["budget"], by[ph]["window_plan"]), (b, 120000, 130560))
        invariants(self, r)

    def test_real_time_cut_at_a_queues_nothing(self):
        """[v2 test_attempt_a_time_queues_nothing, v3 test_window_cut_mid_chain] a Duck A that really runs until the solve
        window ends (fake 'hang' script; S's own test fakes TIME through the classifier) is TIME: nothing queued for that
        task (no B / C, no attempt 2), while another task's chains run to the end."""
        sc = {("t000", "A"): {"kind": "hang"}}
        sc.update(T._fail_all(["t001"]))
        r = run33(sc, n=2, workers=2, deadline_sec=7.0, min_admit=1.0)
        self.assertEqual(chain_of(r, "t000", 1), ["A:1"])
        self.assertEqual(chain_of(r, "t000", 2), [])
        self.assertEqual(r.e["a_class"]["t000"]["kind"], "TIME")
        self.assertEqual(r.e["e33"]["reached"]["t000"]["a1"]["end"], "TIME at A:1")
        self.assertNotIn("t000", [x[0] for x in r.e["e33"]["a2_queued"]])
        rec = next(x for x in r.recs["t000"] if x["branch"] == 0)
        self.assertEqual(rec["e28"]["kind"], "TIME")
        self.assertEqual([c["tid"] for c in r.hooks if c["tid"] == "t000"], [])
        for a in (1, 2):
            self.assertEqual(chain_of(r, "t001", a)[-1], f"E-DIRECT:{a}")
        invariants(self, r)

    def test_every_hook_failure_mode_ends_both_chains(self):
        """[v1 test_no_hook_skip_terminates / test_hook_exception_skip / test_false_admission_skip /
        test_hooks_none_and_release_one_per_stage; v2 test_hooks_missing_none_false_raise_end_chain; v3
        test_hook_none_exception_and_late_end_chain / test_hooks_missing_skip_and_stop / test_callable_hook_false_ends_chain]
        At stage C, D and E, with attempt 2 on: no hook, a None / False return, a raising run hook, CAN_START False, a
        raising CAN_START. Both chains end at that stage, nothing is published for it, one release per taken item, never a
        Duck fallback, no worker error. Spec / S: an admission refusal is a skip (row SKIPPED, reason); a failed run hook
        is row NO_RECORD (GPT v1-v3 call every case "skipped": a GPT naming choice; the chain end is the spec behaviour)."""
        label = {"C": "C-POETIQ1", "D": "D-POETIQ2", "E": "E-DIRECT"}
        for st in ("C", "D", "E"):
            modes = ["none", "false", "raise", "gate_false", "gate_raise"] + (["no_hook"] if st != "D" else [])
            for mode in modes:
                with self.subTest(stage=st, mode=mode):
                    beh, hooks, can = {}, ("poetiq", "direct"), True
                    if mode in ("none", "false"):
                        beh[("t000", st)] = {"ret": None if mode == "none" else False}
                    elif mode == "raise":
                        beh[("t000", st)] = {"raise": True}
                    elif mode == "gate_false":
                        can = lambda s_, dl, st=st: s_ != st
                    elif mode == "gate_raise":
                        def can(s_, dl, st=st):
                            if s_ == st:
                                raise ZeroDivisionError("gate")
                            return True
                    else:
                        hooks = ("direct",) if st == "C" else ("poetiq",)
                    r = run33(T._fail_all(["t000"]), beh, n=1, hooks=hooks, can=can)
                    for a in (1, 2):
                        ch = chain_of(r, "t000", a)
                        self.assertEqual(ch[-1], f"{label[st]}:{a}", ch)
                        x = next(x for x in rows(r) if x["phase"] == f"{label[st]}:{a}")
                        want = {"none": ("no_record", "NO_RECORD", "no_record"), "false": ("no_record", "NO_RECORD", "no_record"),
                                "raise": ("no_record", "NO_RECORD", "error"), "gate_false": ("skipped", "SKIPPED", "late"),
                                "gate_raise": ("skipped", "SKIPPED", "hook_error ZeroDivisionError"),
                                "no_hook": ("skipped", "SKIPPED", "no_hook")}[mode]
                        self.assertEqual((x["status"], x["kind"], x["end_reason"]), want)
                        self.assertFalse([y for y in r.recs["t000"] if y["branch"] == x["branch"]])
                    self.assertTrue({b for _, _, b, _, _ in r.runs} <= {0, 2, 1, 122})
                    invariants(self, r)

    def test_poetiq_train_pass_without_accepted_flag(self):
        """[v2 test_bad_poetiq_train_pass_evidence_fail_closed] a Poetiq record with poetiq.train_pass True but accepted
        False. GPT v2 fail-closes it ("inconsistent_train_pass" skip, its own safety rule). Spec / S: the train pass is the
        solver's statement that the program is exact on every train example (agent P: train pass = poetiq.train_pass): the
        chain stops at C and its grid is attempt_1 (step 1-pass)."""
        ret = {"task_id": "t000", "status": "done", "accepted": False, "generated_tokens": 60001,
               "submissions": [{"test_outputs": [g(44)], "accepted": False, "train_exact": [True, True]}],
               "poetiq": {"train_pass": True, "train_score": 1.0}}
        r = run33(T._fail_all(["t000"]), {("t000", 1, "C"): {"ret": ret}}, n=1)
        self.assertEqual(chain_of(r, "t000", 1), ["A:1", "B1:1", "C-POETIQ1:1"])
        self.assertEqual(r.e["e33"]["stages"]["C-POETIQ1:1"]["train_pass"], 1)
        self.assertEqual(r.subs["t000"][0]["attempt_1"], g(44))
        self.assertEqual(r.slots["t000"][0]["a1_step"], "1-pass")

    def test_kv_refusal_backfill_and_force_when_idle(self):
        """[v3 AdditionalChainTests.test_reservation_backfill] a fit hook that always refuses t000's A:1 and t001's
        C-POETIQ1:1: lower items backfill (fit_fails, backfill_starts), the refused ones start only when nothing runs
        (force-when-idle, E28 unchanged); every chain still runs to the end; the invariants hold."""
        refuse = lambda k, p: not ((k[0], p["phase"]) in (("t000", "A:1"), ("t001", "C-POETIQ1:1")))
        r = run33(T._fail_all(["t000", "t001", "t002"]), n=3, workers=3, kit=lambda M: HookKit(M, refuse))
        e = r.e
        self.assertGreater(e["fit_fails"], 0)
        self.assertGreater(e["backfill_starts"], 0)
        self.assertGreaterEqual(e["forced_admits"], 2)
        for t in ("t000", "t001", "t002"):
            for a in (1, 2):
                self.assertEqual(chain_of(r, t, a)[-1], f"E-DIRECT:{a}", (t, a))
        a1 = next(x for x in rows(r) if x["task"] == "t000" and x["phase"] == "A:1")
        self.assertGreater(a1["start_seq"], min(x["start_seq"] for x in rows(r) if x["task"] != "t000"))
        self.assertIn((("t001", 113), "C-POETIQ1:1"), r.kit.calls)
        invariants(self, r)

    def test_kv_hook_raising_is_fail_open(self):
        """[v1 report: 'hook exceptions fail closed (a deliberate difference from E28)'] Spec section 3: the E28 KV hook and
        force-when-idle 'stay as they are' = E28's fail-open (_fits counts fit_errors and admits). A fit hook raising for
        every Poetiq / Direct item: those items still run, fit_errors counts them, the chains complete."""
        def boom(k, p):
            if p.get("e33_kind") in ("poetiq", "direct"):
                raise RuntimeError("fit hook broken")
            return True
        r = run33(T._fail_all(["t000", "t001"]), n=2, workers=2, kit=lambda M: HookKit(M, boom))
        n_pd = sum(1 for _, ph in r.kit.calls if ph.split(":")[0] in ("C-POETIQ1", "D-POETIQ2", "E-DIRECT"))
        self.assertEqual(r.e["fit_errors"], n_pd)
        self.assertEqual(n_pd, 12)
        for t in ("t000", "t001"):
            for a in (1, 2):
                self.assertEqual(chain_of(r, t, a)[-1], f"E-DIRECT:{a}")
        invariants(self, r)

    def test_supply_with_poetiq_and_attempt2_off(self):
        """[v2 test_supply_only_after_attempt_queues_empty] Poetiq off, attempt 2 off, supply on: chain A -> B1 -> E, then
        supply S1 on branches 200+ only after the chain queue is empty. S counts every E33 start (Direct too) toward
        E29_SUPPLY_MAX_ATTEMPTS (GPT v2 counts Duck A / B / S only; the spec says nothing on the cap): cap 6 = 3 chain + 3
        supply attempts."""
        sc = T._fail_all(["t000"])
        sc[("t000", "S")] = {"kind": "partial", "at": 3000}
        r = run33(sc, n=1, poetiq=False, attempt2=False, supply=True, env={"E29_SUPPLY_MAX_ATTEMPTS": 6})
        self.assertEqual(chain_of(r, "t000", 1), ["A:1", "B1:1", "E-DIRECT:1"])
        srows = [x for x in rows(r) if str(x["phase"]).startswith("S")]
        self.assertEqual([x["branch"] for x in srows], [200, 201, 202])
        last = max(x["start_seq"] for x in rows(r) if x["branch"] in ATT)
        self.assertTrue(all(x["start_seq"] > last and x["queues_at_start"] == [0, 0, 0, 0] for x in srows))
        self.assertEqual(r.e["supply"]["attempts_by_task"]["t000"], 6)

    def test_real_branchsession_first_pass_four_workers(self):
        """[v3 RealBranchSessionContractTests.test_real_branchsession_first_pass_stops_chain] two tasks pass at A on 4
        workers (real BranchSession): exactly A:1 + A:2 each, no B, tasks_with_request_sent = 2."""
        sc = {("t000", "A"): {"kind": "pass", "at": 3000}, ("t001", "A"): {"kind": "pass", "at": 4000}}
        r = run33(sc, n=2, workers=4)
        for t in ("t000", "t001"):
            self.assertEqual(sorted(x["branch"] for x in r.recs[t]), [0, 1])
        self.assertFalse([x for x in rows(r) if x["phase"].startswith("B")])
        self.assertEqual(r.sl["tasks_with_request_sent"], 2)
        invariants(self, r)

    def test_all_attempt1_a_before_any_attempt2_four_workers(self):
        """[v1 test_priority_attempt1_before_attempt2, v3 test_priority_and_no_same_attempt_overlap] 6 tasks, 4 workers,
        no KV hook: every A:1 starts before any attempt-2 item; attempt-2 starts see no queued attempt-1 item."""
        sc = {(t, "A"): {"kind": "partial", "at": 3000, "sleep": 0.002} for t in ST._tasks(6)}
        r = run33(sc, n=6, workers=4)
        rs = rows(r, skipped=False)
        last_a1 = max(x["start_seq"] for x in rs if x["phase"] == "A:1")
        self.assertTrue(all(x["start_seq"] > last_a1 for x in rs if ATT[x["branch"]] == 2))
        self.assertTrue(all(x["e33_q1_left"] == 0 for x in rs if ATT[x["branch"]] == 2))
        invariants(self, r)

    def test_branch_ids_and_supply_base(self):
        """[v2 test_branch_ids_no_supply_collision] every (attempt, stage) has its own branch (B1 / B2 share one per
        attempt), the module map = BRANCH_MAP, and supply (>= SUPPLY_BRANCH_BASE) never meets an E33 branch."""
        M = FIX["e33"]
        self.assertEqual(len(set(M.E33_BRANCH.values())), len(M.E33_BRANCH))
        self.assertEqual(M.E33_SRC, S33.BRANCH_MAP)
        self.assertEqual(M.E33_SUPPLY_B0, S33.SUPPLY_BRANCH_BASE)
        self.assertGreater(S33.SUPPLY_BRANCH_BASE, max(S33.BRANCH_MAP.values()))
        self.assertEqual(S33.BRANCH_MAP["B1:1"], S33.BRANCH_MAP["B2:1"])
        self.assertEqual(S33.BRANCH_MAP["B1:2"], S33.BRANCH_MAP["B2:2"])

    def test_switch_values(self):
        """[v3 PatchTests.test_flags_invalid_and_off, v1 test_off_gate] a switch is on only for exactly "1" (spaces
        stripped); E28 off keeps E33 off even with E33_SCHED=1."""
        M = FIX["e33"]
        for v, want in (("1", True), (" 1 ", True), ("0", False), ("", False), ("garbage", False), ("true", False),
                        ("yes", False), ("01", False), ("2", False)):
            with Env33(on=True, poetiq=v, direct=v, attempt2=v, sched_value=v):
                cfg = M.e33_config()
                self.assertEqual((cfg["poetiq"], cfg["direct"], cfg["attempt2"], M._e33_on()), (want,) * 4, v)
        with Env33(on=True, e28=False):
            self.assertFalse(M._e33_on())

    def test_cell3_wrong_path_clears_every_name(self):
        """[v2 test_cell3_fail_closed, v3 test_cell3_fails_closed_when_e29_is_not_selected] cell3 with a non-E29
        scheduler record (path OTHER / E28 / not applied): an error, nothing written, every E33 name (E33_A1_ORDER too)
        removed."""
        d = FIX["tmp"] / "t33_cell3_paths"
        d.mkdir(exist_ok=True)
        e29 = T._e29_text().encode()
        (d / "arc2_duck.py").write_bytes(e29)
        for rec in ({"applied": True, "path": "OTHER"}, {"applied": True, "path": "E28"}, {"applied": False, "path": "E29"}):
            os.environ.update({k: "1" for k in S33.ENV_NAMES})
            r = S33.cell3(d, rec, {})
            self.assertIn("error", r)
            self.assertTrue(all(os.environ.get(k) is None for k in S33.ENV_NAMES), rec)
            self.assertEqual((d / "arc2_duck.py").read_bytes(), e29)


# ------------------------------------------------------------------ B. ported answer-rule scenarios (E33_A1_ORDER=e28 = S's rule)
def mk(a, st, grids, acc=False, exact=None, score=None, tp=None, seq=1, name=None, branch=None):
    """A tagged E33 record with per-query grids: Duck (A / B: accepted -> submission, else a non-accepted check),
    Poetiq (C / D), Direct (E)."""
    lab = {"A": "A", "B": name or "B1", "C": "C-POETIQ1", "D": "D-POETIQ2", "E": "E-DIRECT"}[st]
    r = {"task_id": "x", "branch": branch if branch is not None else BR[(a, st)], "publish_seq": seq,
         "e28": {"phase": f"{lab}:{a}", "e33_attempt": a, "e33_stage": st}}
    if st in ("A", "B"):
        s = {"accepted": acc, "train_exact": list(exact if exact is not None else ([True, True] if acc else [])),
             "test_outputs": grids, "t": 5}
        r.update(submissions=[s] if acc else [], checked=[] if acc else ([s] if grids else []), accepted=acc)
    elif st in ("C", "D"):
        tp = acc if tp is None else tp
        r.update(accepted=acc, submissions=[{"test_outputs": grids, "accepted": acc,
                                             "train_exact": list(exact if exact is not None else ([True, True] if acc else []))}],
                 poetiq={"train_pass": tp, **({"train_score": score} if score is not None else {})})
    else:
        r.update(accepted=False, submissions=[{"test_outputs": grids, "accepted": False, "train_exact": []}])
    return r


def task_n(nq, n_train=2):
    return {"train": [{"input": [[0]], "output": [[0]]}] * n_train, "test": [{"input": [[9]]} for _ in range(nq)]}


def sel(recs, task, order="e28", direct=False):
    """-> [(attempt_1, attempt_2, a1 source, a1 step, a2 source, a2 step)] per query (through the select_attempts_e28
    dispatch, or select_attempts_e33 itself with direct=True)."""
    M = FIX["e33"]
    rows_ = []
    with Env33(on=True, a1_order=order):
        out = (M.select_attempts_e33 if direct else M.select_attempts_e28)(task, recs, rows_)
        if not direct:
            T.TestCase = unittest.TestCase
            assert M.build_submission({"x": task}, {"x": recs})["x"] == out
    for r_ in rows_:
        assert r_.get("a1_order") == (order if order == "e28" else "direct_first"), r_
    return [(o["attempt_1"], o["attempt_2"], r_["attempt_1"], r_["a1_step"], r_["attempt_2"], r_["a2_step"])
            for o, r_ in zip(out, rows_)]


class TestPortedAnswers(unittest.TestCase):
    def test_two_slots_per_query_multi_test(self):
        """[v1 test_answer_distinct_multi_test, v3 SlotTests.test_first_train_pass_and_two_slots] attempt-1 and attempt-2
        Duck passes with different grids per query: grid K -> test K for both slots, sources A:1 / A:2."""
        for nq in (2, 3):
            recs = [mk(1, "A", [g(i + 1) for i in range(nq)], acc=True), mk(2, "A", [g(i + 4) for i in range(nq)], acc=True)]
            out = sel(recs, task_n(nq))
            self.assertEqual([(o[0], o[1]) for o in out], [(g(i + 1), g(i + 4)) for i in range(nq)])
            self.assertTrue(all(o[2:] == ("A:1", "1-pass", "A:2", "a2:1-pass") for o in out))

    def test_three_test_poetiq_direct_cases(self):
        """[v2 TestAnswers.test_grid_order_multitest_and_distinct_slots] 3 test inputs: a Poetiq pass beats Direct; a
        higher Poetiq train score wins (score-only records); Direct alone; attempt 2's pass equal to attempt_1 -> attempt_2
        = attempt 1's Direct grid (a1-next; GPT v2's slot label '1E' = S's 'E-DIRECT:1')."""
        tk = task_n(3)
        G = lambda v: [g(v + i) for i in range(3)]
        out = sel([mk(1, "A", []), mk(1, "C", G(4), acc=True), mk(1, "E", G(7))], tk)
        self.assertEqual([o[0] for o in out], G(4))
        out = sel([mk(1, "A", []), mk(1, "C", G(4), score=0.3, exact=[]), mk(1, "D", G(5), score=0.8, exact=[])], tk)
        self.assertEqual([o[0] for o in out], G(5))
        out = sel([mk(1, "A", []), mk(1, "E", G(7))], tk)
        self.assertEqual([o[0] for o in out], G(7))
        out = sel([mk(1, "A", []), mk(1, "C", G(4), acc=True), mk(1, "E", G(7)), mk(2, "A", []),
                   mk(2, "C", G(4), acc=True)], tk)
        self.assertEqual([(o[0], o[1], o[4], o[5]) for o in out], [(g(4 + i), g(7 + i), "E-DIRECT:1", "a1-next") for i in range(3)])

    def test_invalid_grid_at_one_query(self):
        """[v2 test_invalid_grids_and_input_fallback] a Poetiq pass whose grid is invalid at query 0 and valid at query 1:
        query 0 -> the test input, query 1 -> the grid."""
        out = sel([mk(1, "A", []), mk(1, "C", [[[100]], g(2)], acc=True)], task_n(2))
        self.assertEqual((out[0][0], out[0][3]), ([[9]], "5-input"))
        self.assertEqual((out[1][0], out[1][3]), (g(2), "1-pass"))

    def test_poetiq_score_vs_duck_exact_fraction(self):
        """[v3 SlotTests.test_best_train_match_poetiq_over_duck] Duck A 1/2 exact vs Poetiq C train_exact 0/2 with
        train_score 0.75. GPT v3 (max of the two) / v2 (score) pick C. S (e28 and direct_first): the Poetiq match is its
        train_exact fraction, the train score only when train_exact is empty -> A. OPEN POINT (reported, not changed):
        spec 4 says 'Poetiq's train score as the match'; the user's 2026-11-02 order says 'train-match fraction'. With C
        PASSING it is C in every reading."""
        tk = task_n(1)
        recs = [mk(1, "A", [g(2)], exact=[True, False]), mk(1, "C", [g(5)], score=0.75, exact=[False, False])]
        self.assertEqual(sel(recs, tk)[0][:4], (g(2), g(5), "A:1", "2-best"))
        recs.append(mk(1, "D", [g(6)], acc=True, score=1.0))
        self.assertEqual(sel(recs, tk)[0][:4], (g(6), g(2), "D-POETIQ2:1", "1-pass"))

    def test_direct_alone_and_no_records(self):
        """[v3 SlotTests.test_direct_and_input_fallback] Direct alone -> attempt_1 = its grid; no record at all ->
        select_attempts_e33 gives the test input (and 'same')."""
        tk = task_n(3)
        out = sel([mk(1, "E", [g(9 - i) for i in range(3)])], tk)
        self.assertEqual([o[0] for o in out], [g(9), g(8), g(7)])
        out = sel([], tk, direct=True)
        self.assertEqual([o[:6] for o in out], [([[9]], [[9]], "input", "5-input", "same", "same")] * 3)

    def test_attempt_separation_and_next_distinct(self):
        """[v3 SlotTests.test_strict_attempt_separation_and_next_distinct] a1 A pass [1] + B1 partial [2], a2 A pass [1]:
        attempt_2 = [2] (a1-next). Only an attempt-2 record: GPT v3 never lets it fill attempt_1 (input / A:2); S: E28
        fallback = select_attempts_r5 over every Duck record (E28 used all branches) -> A:2's grid in attempt_1 and the
        input in attempt_2: the same two answers, slots swapped (pass@2 identical)."""
        tk = task_n(1)
        out = sel([mk(1, "A", [g(1)], acc=True), mk(1, "B", [g(2)], exact=[True, False]), mk(2, "A", [g(1)], acc=True)], tk)
        self.assertEqual(out[0][:2], (g(1), g(2)))
        self.assertEqual(out[0][4:], ("B1:1", "a1-next"))
        out = sel([mk(2, "A", [g(1)], acc=True)], tk, direct=True)
        self.assertEqual(out[0], (g(1), [[9]], "r5", "4-fallback", "input", "5-input"))

    def test_r5_fallback_stub(self):
        """[v3 SlotTests.test_fallback_rules_preserved_but_not_attempt2] select_attempts_r5 stubbed to give [[7]]; only an
        attempt-2 Duck pass [8]: attempt_1 = [[7]] (E28 fallback), attempt_2 = [[8]] (the attempt-2 chain)."""
        M = FIX["e33"]
        old = M.select_attempts_r5

        def stub(task, records, _e28_lists=None):
            if _e28_lists is not None:
                _e28_lists.extend([[[[7]]] for _ in task["test"]])
            return [{"attempt_1": [[7]], "attempt_2": [[9]]} for _ in task["test"]]
        M.select_attempts_r5 = stub
        try:
            for order in ("e28", "direct_first"):
                out = sel([mk(2, "A", [g(8)] * 3, acc=True)], task_n(3), order=order, direct=True)
                self.assertEqual([o[:2] for o in out], [([[7]], [[8]])] * 3)
        finally:
            M.select_attempts_r5 = old

    def test_invalid_grids_never_selected(self):
        """[v3 SlotTests.test_bad_grids_never_selected] an accepted Duck A whose three test grids are all invalid (10, 12,
        999): every query gets the test input (both orders)."""
        for order in ("e28", "direct_first"):
            out = sel([mk(1, "A", [[[10]], [[12]], [[999]]], acc=True)], task_n(3), order=order)
            self.assertEqual([o[0] for o in out], [[[9]]] * 3)


# ------------------------------------------------------------------ C. the user's answer order of 2026-11-02 (E33_A1_ORDER)
class TestA1Order(unittest.TestCase):
    P, H, Z = [True, False], [True, True, False], [False, False]

    def test_switch_value(self):
        """direct_first when unset or unknown; e28 only for 'e28' (case / spaces ignored); ENV_NAMES / server_env / cell3
        carry the name; the report and every slot row record the effective order."""
        M = FIX["e33"]
        for v, want in ((None, "direct_first"), ("", "direct_first"), ("direct_first", "direct_first"), ("e28", "e28"),
                        (" E28 ", "e28"), ("old", "direct_first"), ("1", "direct_first"), ("e29", "direct_first")):
            with Env33(on=True, a1_order=v):
                self.assertEqual(M._e33_a1_order(), want, v)
        self.assertIn("E33_A1_ORDER", S33.ENV_NAMES)
        self.assertIn("E33_A1_ORDER", M.E33_ENV_NAMES)
        self.assertEqual(S33.A1_ORDERS, ("direct_first", "e28"))
        base = {"on": True, "poetiq": True, "direct": True, "attempt2": True, "poetiq_think": 60000, "poetiq_answer": 8192,
                "direct_think": 104000, "direct_answer": 8192}
        self.assertNotIn("E33_A1_ORDER", S33.server_env(base))
        for v, want in (("e28", "e28"), ("E28", "e28"), ("direct_first", "direct_first"), ("bad", "direct_first"),
                        (None, "direct_first")):
            self.assertEqual(S33.server_env(dict(base, a1_order=v))["E33_A1_ORDER"], want)
        self.assertEqual(S33.server_env(dict(base, on=False, a1_order="e28")), {})
        d = FIX["tmp"] / "t33_cell3_a1"
        d.mkdir(exist_ok=True)
        (d / "arc2_duck.py").write_bytes(T._e29_text().encode())
        os.environ["E33_A1_ORDER"] = "stale"
        r = S33.cell3(d, {"applied": True, "path": "E29"}, {}, dict(base, a1_order="e28"))
        self.assertTrue(r["ok"], r)
        self.assertEqual(os.environ.get("E33_A1_ORDER"), "e28")
        r = S33.cell3(d, {"applied": True, "path": "E29"}, {}, base)
        self.assertIsNone(os.environ.get("E33_A1_ORDER"))
        for k in S33.ENV_NAMES:
            os.environ.pop(k, None)

    def test_attempt1_every_case(self):
        """attempt_1 under direct_first: 1 pass (beats Direct); 2 Direct (beats partial and zero); 3 partial > 0, higher
        first, ties earlier; 4 zero-match after partial; then the E28 fallbacks; then the input. Multi-test: an invalid
        Direct grid at query K -> the next step for THAT query. Supply records join attempt 1's pass / partial groups."""
        t1, t3 = task_n(1), task_n(1, n_train=3)
        D = "direct_first"
        # 1 pass beats Direct (Duck and Poetiq passes)
        self.assertEqual(sel([mk(1, "A", [g(1)], acc=True), mk(1, "E", [g(7)])], t1, D)[0][:6],
                         (g(1), g(7), "A:1", "1-pass", "E-DIRECT:1", "a1-next"))
        self.assertEqual(sel([mk(1, "A", [g(5)], exact=self.P), mk(1, "C", [g(3)], acc=True), mk(1, "E", [g(7)])], t1, D)[0][:6],
                         (g(3), g(7), "C-POETIQ1:1", "1-pass", "E-DIRECT:1", "a1-next"))
        # 2 Direct beats partial; attempt_2 = the best partial (ties: earlier = A)
        self.assertEqual(sel([mk(1, "A", [g(5)], exact=self.P), mk(1, "C", [g(3)], exact=self.P, score=0.5),
                              mk(1, "E", [g(7)])], t1, D)[0][:6],
                         (g(7), g(5), "E-DIRECT:1", "2-direct", "A:1", "a1-next"))
        # 2 Direct beats zero-match (e28 kept the zero-match Duck first)
        recs = [mk(1, "A", [g(5)], exact=self.Z), mk(1, "E", [g(7)])]
        self.assertEqual(sel(recs, t1, D)[0][:6], (g(7), g(5), "E-DIRECT:1", "2-direct", "A:1", "a1-next"))
        self.assertEqual(sel(recs, t1, "e28")[0][:6], (g(5), g(7), "A:1", "2-best", "E-DIRECT:1", "a1-next"))
        # 3 partial: higher fraction first (C 2/3 over A 1/3), the zero-match B after them
        recs = [mk(1, "A", [g(5)], exact=[True, False, False]), mk(1, "B", [g(6)], exact=[False] * 3),
                mk(1, "C", [g(3)], exact=self.H, score=0.7)]
        o = sel(recs, t3, D)[0]
        self.assertEqual(o[:6], (g(3), g(5), "C-POETIQ1:1", "3-partial", "A:1", "a1-next"))
        M = FIX["e33"]
        with Env33(on=True, a1_order=D):
            rr = []
            M.select_attempts_e33(t3, recs, rr)
        self.assertEqual(rr[0]["a1_chain"], ["C-POETIQ1:1", "A:1", "B1:1"])
        # 3 ties keep the earlier stage
        self.assertEqual(sel([mk(1, "A", [g(5)], exact=self.P), mk(1, "D", [g(4)], exact=self.P, score=0.5)], t1, D)[0][2:4],
                         ("A:1", "3-partial"))
        # Poetiq score-only partial (no train_exact) counts as partial by its score; 0 score = zero-match
        self.assertEqual(sel([mk(1, "A", [g(5)], exact=self.Z), mk(1, "C", [g(3)], exact=[], score=0.4)], t1, D)[0][2:6],
                         ("C-POETIQ1:1", "3-partial", "A:1", "a1-next"))
        self.assertEqual(sel([mk(1, "A", [g(5)], exact=self.Z), mk(1, "C", [g(3)], exact=[], score=0.0)], t1, D)[0][2:6],
                         ("A:1", "4-zero", "C-POETIQ1:1", "a1-next"))
        # 4 zero only; then the E28 fallbacks (here an attempt-2 Duck grid via select_attempts_r5); then the input
        self.assertEqual(sel([mk(1, "A", [g(5)], exact=self.Z)], t1, D)[0][2:6], ("A:1", "4-zero", "input", "5-input"))
        self.assertEqual(sel([mk(1, "A", []), mk(2, "A", [], acc=False)], t1, D)[0], ([[9]], [[9]], "input", "5-input", "same", "same"))
        self.assertEqual(sel([mk(1, "A", []), mk(1, "E", [[[10]]]), mk(2, "B", [g(8)], exact=self.P)], t1, D)[0][:4],
                         (g(8), g(8), "r5", "4-fallback")[:2] + ("r5", "4-fallback"))
        # multi-test: an invalid Direct grid at query 1 -> that query takes the partial
        o = sel([mk(1, "A", [g(5), g(6)], exact=self.P), mk(1, "E", [g(7), [[10]]])], task_n(2), D)
        self.assertEqual([x[:4] for x in o], [(g(7), g(5), "E-DIRECT:1", "2-direct"), (g(6), [[9]], "A:1", "3-partial")])
        # supply: a supply pass is a pass (beats Direct); a supply partial is a partial (after Direct)
        self.assertEqual(sel([mk(1, "A", [g(5)], exact=self.Z), mk(1, "E", [g(7)]), T.srec(200, 6)], t1, D)[0][2:6],
                         ("S200", "1-pass", "E-DIRECT:1", "a1-next"))

    def test_attempt2_every_case(self):
        """attempt_2 under direct_first: the attempt-2 chain's head by the same order when it differs from attempt_1
        (pass / Direct / partial / zero); else attempt 1's next distinct candidate (the best partial program usually);
        else the fallbacks; else the input; else 'same'."""
        t1 = task_n(1)
        D = "direct_first"
        a1 = [mk(1, "A", [g(5)], exact=self.P), mk(1, "E", [g(7)])]                 # attempt_1 = Direct 7
        self.assertEqual(sel(a1 + [mk(2, "A", [g(2)], acc=True), mk(2, "E", [g(6)])], t1, D)[0][4:], ("A:2", "a2:1-pass"))
        self.assertEqual(sel(a1 + [mk(2, "A", [g(8)], exact=self.P), mk(2, "E", [g(6)])], t1, D)[0][1:],
                         (g(6), "E-DIRECT:1", "2-direct", "E-DIRECT:2", "a2:2-direct"))
        self.assertEqual(sel(a1 + [mk(2, "A", [g(8)], exact=self.P)], t1, D)[0][4:], ("A:2", "a2:3-partial"))
        self.assertEqual(sel(a1 + [mk(2, "A", [g(8)], exact=self.Z)], t1, D)[0][4:], ("A:2", "a2:4-zero"))
        # the attempt-2 head equals attempt_1 -> attempt 1's next distinct = its best partial program
        self.assertEqual(sel(a1 + [mk(2, "A", [g(8)], exact=self.P), mk(2, "E", [g(7)])], t1, D)[0][1:],
                         (g(5), "E-DIRECT:1", "2-direct", "A:1", "a1-next"))
        # nothing distinct in either chain -> the E28 fallbacks (an older accepted Duck grid), else the input
        self.assertEqual(sel([mk(1, "E", [g(7)]), mk(1, "A", [g(7)], exact=self.P), mk(2, "E", [g(7)])], t1, D)[0][4:],
                         ("input", "5-input"))
        self.assertEqual(sel([mk(1, "A", []), mk(1, "E", [[[9]]])], t1, D)[0][4:], ("same", "same"))

    def test_end_to_end_direct_first(self):
        """S's full-chain scenario (two tasks fail every stage: Poetiq C / D partial 0.5, Direct grids) under the new
        default order: attempt_1 = E-DIRECT:1, attempt_2 = E-DIRECT:2 (its chain head differs); slot rows and the report
        say direct_first. The same run under e28 keeps S's original slots (C-POETIQ1:1 / C-POETIQ1:2)."""
        r = run33(T._fail_all(["t000", "t001"]), n=2, env={"a1_order": None})
        self.assertEqual(r.e["e33"]["a1_order"], "direct_first")
        for t in ("t000", "t001"):
            self.assertEqual(r.subs[t], [{"attempt_1": g(71), "attempt_2": g(72)}])
            s = r.slots[t][0]
            self.assertEqual((s["attempt_1"], s["a1_step"], s["attempt_2"], s["a2_step"], s["a1_order"]),
                             ("E-DIRECT:1", "2-direct", "E-DIRECT:2", "a2:2-direct", "direct_first"))
            by = {x["branch"]: x for x in r.e["attempts"] if x["task"] == t}
            self.assertEqual((by[115]["answer_slots"], by[125]["answer_slots"]), (["attempt_1"], ["attempt_2"]))
        r = run33(T._fail_all(["t000"]), n=1)
        self.assertEqual(r.e["e33"]["a1_order"], "e28")
        self.assertEqual((r.slots["t000"][0]["attempt_1"], r.slots["t000"][0]["a1_order"]), ("C-POETIQ1:1", "e28"))


# ------------------------------------------------------------------ D. stress (port of GPT v3 stress_e33.py)
def stress_scen(n, seed):
    """Random scripts over every stage: Duck A pass / fail / cap / partial, B pass / partial, Poetiq pass at C or D
    sometimes, Direct; small sleeps (real BranchSession, scripted fake Duck)."""
    rnd = random.Random(seed)
    sc, beh = {}, {}
    for i in range(n):
        t = f"t{i:03d}"
        u = rnd.random()
        sc[(t, "A")] = ({"kind": "cap", "sleep": 0.0} if u < 0.15 else {"kind": "fail"} if u < 0.4 else
                        {"kind": "partial", "at": 3000} if u < 0.75 else {"kind": "pass", "at": rnd.randrange(2, 12) * 1000})
        sc[(t, "B")] = {"kind": "pass", "at": 3000, "value": 2} if rnd.random() < 0.2 else {"kind": "partial", "at": 2000}
        for st in ("C", "D"):
            beh[(t, st)] = {"pass": rnd.random() < 0.2, "sleep": rnd.choice([0.001, 0.004, 0.01])}
        beh[(t, "E")] = {"sleep": rnd.choice([0.001, 0.006])}
    return sc, beh


STRESS = {}


class TestStress(unittest.TestCase):
    def test_120_tasks_24_workers(self):
        """[v3 stress_e33.py] 120 tasks, 24 workers, every stage, both attempts: starts == done == releases per (task,
        branch), never two stages of one (task, attempt) at once, no straggler, the queue empty, zero worker errors."""
        sc, beh = stress_scen(120, 33)
        t0 = time.monotonic()
        r = run33(sc, beh, n=120, workers=24, deadline_sec=600.0)
        invariants(self, r)
        self.assertEqual(len(r.recs), 120)
        STRESS["plain"] = _stress_row(r, time.monotonic() - t0)
        self.assertGreater(STRESS["plain"]["started"], 700)
        self.assertGreater(r.e["started"]["E-DIRECT:2"], 0)

    def test_120_tasks_24_workers_real_reservation(self):
        """The same with the real NB5 Reservation (projected KV blocks, 4 x 73 blocks): nothing leaks, every grant /
        forced admit is one start or one skip."""
        sc, beh = stress_scen(120, 34)
        t0 = time.monotonic()
        r = run33(sc, beh, n=120, workers=24, deadline_sec=600.0, kit=lambda M: ST.ResKit(M, cap=292))
        invariants(self, r)
        self.assertEqual((r.kit.res.used, r.kit.res.held), (0, {}))
        self.assertEqual(r.kit.res.stats["e28_try_grants"] + r.e["forced_admits"],
                         sum(r.e["started"].values()) + sum(sum(s["skipped"].values()) for s in r.e["e33"]["stages"].values()))
        self.assertGreater(r.e["backfill_starts"] + r.e["fit_fails"], 0)
        STRESS["reservation"] = dict(_stress_row(r, time.monotonic() - t0), fit_checks=r.e["fit_checks"],
                                     fit_fails=r.e["fit_fails"], backfill_starts=r.e["backfill_starts"],
                                     forced_admits=r.e["forced_admits"], res_used_end=r.kit.res.used)

    def test_120_tasks_24_workers_window_ends_mid_run(self):
        """The solve window ends mid-run (deadline 9 s, min_admit 6 s: admission closes at 3 s; Poetiq / Direct sleep
        0.3-0.5 s): no start after the window, items left in the queue (counted in left / skipped_for_time / heap_left),
        everything started ends and releases once, no straggler, no worker error."""
        sc, beh = stress_scen(120, 35)
        for k in list(beh):
            beh[k] = dict(beh[k], sleep=0.3 if k[1] != "E" else 0.5)
        t0 = time.monotonic()
        r = run33(sc, beh, n=120, workers=24, deadline_sec=9.0, min_admit=6.0)
        invariants(self, r, window=True)
        st = r.e["e33"]["stages"]
        left = sum(s["left"] for s in st.values())
        self.assertGreater(left, 0)
        self.assertEqual(r.sl["heap_left"], left)
        self.assertTrue(all((x.get("start_sec") or 0) <= 3.0 + 0.3 for x in rows(r)), max(x.get("start_sec") or 0 for x in rows(r)))
        self.assertLess(r.el, 9.0)
        STRESS["window"] = dict(_stress_row(r, time.monotonic() - t0), left=left,
                                skipped_for_time=sum(s["skipped_for_time"] for s in st.values()),
                                last_start_sec=max(x.get("start_sec") or 0 for x in rows(r)))


def _stress_row(r, sec):
    e = r.e
    starts = sum(1 for ev in r.runs if ev[0] == "start") + len(r.hooks)
    return {"tasks": len(r.recs), "workers": r.sl["workers"], "taken": len(e["attempts"]), "started": starts,
            "done": sum(e["done"][ph] for ph in S33.BRANCH_MAP), "released": len(r.rel),
            "skipped": sum(sum(s["skipped"].values()) for s in e["e33"]["stages"].values()),
            "worker_errors": r.sl["worker_errors"], "stage2_errors": r.sl.get("stage2_errors", 0),
            "stragglers": r.sl["stragglers"], "queue_left": sum(s["left"] for s in e["e33"]["stages"].values()),
            "records": sum(len(v) for v in r.recs.values()), "sec": round(sec, 1),
            "by_stage_started": {ph: e["started"][ph] for ph in S33.BRANCH_MAP if e["started"][ph]}}


MERGED = ["TestPortedChain", "TestPortedAnswers", "TestA1Order", "TestStress"]

if __name__ == "__main__":
    import json
    prog = unittest.main(verbosity=2 if "-v" in sys.argv else 1, argv=[a for a in sys.argv if a != "-v"], exit=False)
    res = prog.result
    if STRESS:
        print("STRESS " + json.dumps(STRESS, sort_keys=True))
    print(f"E33 SCHED merged tests{' (NEGATIVE CONTROL: the unpatched E29 post file)' if NEGCTL else ''}: {res.testsRun} run, "
          f"{len(res.failures)} failures, {len(res.errors)} errors, {len(res.skipped)} skipped")
    sys.exit(0 if res.wasSuccessful() else 1)
