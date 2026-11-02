"""E33 ATTEMPT CHAIN scheduler tests:  python e33_sched_tests.py [-v]   (exit 0 = all pass)

Harness = Experiment 29's e29_sched_tests.py (ST: the REAL arc2_duck.py from the pinned Duck payload, the real BranchSession and
E28 scheduler, the scripted fake Duck agent SCRIPTS, Env, RunLog, ResKit = the real NB5 Reservation, _solve, the identity
normalisers _norm_recs / _norm_e28) + the E33 diff (e33_sched_patch.py, this folder) on the E29 post file, with stub Poetiq /
Direct / admission hooks (E33_RUN_POETIQ / E33_RUN_DIRECT / E33_CAN_START).
Covers (DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md sections 3-4): the patch (shas, apply / verify / revert, refusals, strict diff,
hunks, cell3, server_env, the generator reproduces the file); the full chain order on one worker; the stop at the first train
pass of every stage; B1 vs B2 vs straight to C vs TIME; the attempt-2 groups and order; attempt 2 never ahead of a runnable
attempt-1 item; attempt 1 and attempt 2 of a task concurrently; 4-worker invariants (one start / done / release per (task,
branch), never two stages of one (task, attempt) at once; the real Reservation leaks nothing); hooks None / False / raising /
no record = skip + chain end (never a Duck fallback); the window end mid-chain (left in the queue, late skips); the answer
rule cases 1-5 for both slots, multi-test, supply records; the switches (Poetiq / Direct / attempt 2 off); switches off =
strict parity with the E29 post file; supply only when every chain queue is empty (branches >= 200, the cap, a Poetiq pass
stops supply); the real Arc2ToolAgent budget / window on the E33 branches; the E29 LIVE lines on E33 records.
Negative control: E33_TEST_NEGCTL=1 loads the UNPATCHED E29 post file as the module under test (the E33 tests must fail).
Inputs (read only): the Duck payload, the E28 Build folder, the E29 Build folder (E33_TEST_E29_DIR), the eval challenges and
K100 / W4 run records when present (else those tests skip).
"""
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import threading
import time
import unittest
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E29B = Path(os.environ.get("E33_TEST_E29_DIR", r"D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build"))
sys.path[:0] = [str(HERE), str(E29B)]
import e29_sched_tests as ST      # noqa: E402  the E29 scheduler harness (fixtures set up below)
import e33_sched_patch as S33     # noqa: E402

E29S, E28S, FIX = ST.E29S, ST.E28S, ST.FIX
NEGCTL = os.environ.get("E33_TEST_NEGCTL", "").strip() == "1"
E33_ENVS = S33.ENV_NAMES
BR = {(1, "A"): 0, (1, "B"): 2, (1, "C"): 113, (1, "D"): 114, (1, "E"): 115,
      (2, "A"): 1, (2, "B"): 122, (2, "C"): 123, (2, "D"): 124, (2, "E"): 125}
ATT = {b: a for (a, _), b in BR.items()}


def setUpModule():
    ST.setUpModule()
    tmp = FIX["tmp"]
    for name in ("e33", "e29d_alt"):
        with zipfile.ZipFile(ST.PAYLOAD) as z:
            z.extractall(tmp / name)
        assert E29S.apply_e29_sched(tmp / name)["ok"]
    if not NEGCTL:
        r = S33.apply_e33(tmp / "e33")
        assert r["ok"] and r["applied"] == "applied", r
    FIX["e33"] = ST._load(tmp / "e33" / "arc2_duck.py", "arc2_duck_e33")
    for k in E33_ENVS:
        os.environ.pop(k, None)


def tearDownModule():
    for k in E33_ENVS:
        os.environ.pop(k, None)
    ST.tearDownModule()


def _e29_text():
    return (FIX["tmp"] / "e29" / "arc2_duck.py").read_bytes().decode("utf-8")


class Env33:
    """ST.Env (E28 / E29 names) + exactly these E33 names (every other E33 name removed); restored after.
    FIX-A1ORDER (2026-11-02): these tests were written for the first-build answer rule, so E33_A1_ORDER=e28 is set by
    default (a1_order=None leaves it unset = the new default direct_first)."""
    def __init__(self, on=True, poetiq=True, direct=True, attempt2=True, supply=False, e28=True, sched_value="1",
                 a1_order="e28", **extra):
        self.st = ST.Env(supply=supply, e28=e28, **extra)
        self.e = {"E33_SCHED": sched_value} if on else {}
        if a1_order is not None:
            self.e["E33_A1_ORDER"] = a1_order
        for k, v in (("E33_POETIQ", poetiq), ("E33_DIRECT", direct), ("E33_ATTEMPT2", attempt2)):
            if v is not None:
                self.e[k] = "1" if v is True else ("0" if v is False else str(v))

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in E33_ENVS}
        for k in E33_ENVS:
            os.environ.pop(k, None)
        os.environ.update(self.e)
        self.st.__enter__()
        return self

    def __exit__(self, *a):
        self.st.__exit__(*a)
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------------ stub Poetiq / Direct hooks (record shapes of the E29 DIRECT record)
HOOK = {"log": [], "lock": threading.Lock(), "beh": {}}


def _beh(tid, a, st):
    b = HOOK["beh"]
    return b.get((tid, a, st)) or b.get((tid, st)) or {}


def _log_call(ent):
    with HOOK["lock"]:
        HOOK["log"].append(ent)
    return ent


def poetiq_hook(tid, task, plan, deadline, base_url, model_id, out_dir, iteration, prior_record):
    a, st = plan.get("e33_attempt"), plan.get("e33_stage")
    b = _beh(tid, a, st)
    ent = _log_call({"tid": tid, "a": a, "st": st, "kind": "poetiq", "it": iteration, "prior": copy.deepcopy(prior_record),
                     "plan": dict(plan), "out_dir": str(out_dir), "deadline": deadline, "t0": time.monotonic(), "t1": None})
    try:
        time.sleep(b.get("sleep", 0.002))
        if b.get("raise"):
            raise RuntimeError("stub poetiq failed")
        if "ret" in b:
            return b["ret"]
        n = len(task["test"])
        tp = bool(b.get("pass"))
        grids = b.get("grids") or [g(b.get("grid", 40 + iteration + 10 * (a - 1)))] * n
        return {"task_id": tid, "status": "done", "accepted": tp, "generated_tokens": int(b.get("tokens", 60000 + iteration)),
                "submissions": [{"test_outputs": grids, "accepted": tp, "t": time.time(),
                                 "train_exact": [True, True] if tp else list(b.get("exact", [True, False]))}],
                "poetiq": {"train_pass": tp, "train_score": 1.0 if tp else float(b.get("score", 0.5)), "iteration": iteration,
                           "solutions": [{"code": f"it{iteration}", "score": 0.5}]}}
    finally:
        ent["t1"] = time.monotonic()


def direct_hook(tid, task, plan, deadline, base_url, model_id, out_dir):
    a, st = plan.get("e33_attempt"), plan.get("e33_stage")
    b = _beh(tid, a, st)
    ent = _log_call({"tid": tid, "a": a, "st": st, "kind": "direct", "it": None, "prior": None, "plan": dict(plan),
                     "out_dir": str(out_dir), "deadline": deadline, "t0": time.monotonic(), "t1": None})
    try:
        time.sleep(b.get("sleep", 0.002))
        if b.get("raise"):
            raise RuntimeError("stub direct failed")
        if "ret" in b:
            return b["ret"]
        n = len(task["test"])
        grids = b.get("grids") or [g(b.get("grid", 70 + a))] * n
        return {"task_id": tid, "status": "done", "accepted": False, "generated_tokens": 104000 + 8192,
                "submissions": [{"test_outputs": grids, "accepted": False, "train_exact": [], "t": time.time()}],
                "e31": {"mode": "natural", "end_reason": "natural"}}
    finally:
        ent["t1"] = time.monotonic()


class R(dict):
    __getattr__ = dict.__getitem__


def run33(scripts, beh=None, n=2, nq=1, tasks=None, workers=1, on=True, poetiq=True, direct=True, attempt2=True, supply=False,
          can=True, hooks=("poetiq", "direct"), deadline_sec=60.0, min_admit=0.2, kit=None, classify=None, env=None, **kw):
    """One scheduler run on the E33 module with these Duck scripts, hook behaviours, switches and hooks."""
    M = FIX["e33"]
    tasks = tasks or ST._tasks(n, nq)
    order = list(tasks)
    ST.SCRIPTS.clear()
    ST.SCRIPTS.update(scripts)
    HOOK["log"].clear()
    HOOK["beh"] = dict(beh or {})
    cans = []

    def can_log(st, dl):
        cans.append((st, dl, time.monotonic()))
        return can(st, dl) if callable(can) else True
    M.E33_RUN_POETIQ = poetiq_hook if "poetiq" in hooks else None
    M.E33_RUN_DIRECT = direct_hook if "direct" in hooks else None
    M.E33_CAN_START = None if can is None else can_log
    kobj = kit(M) if kit else None
    rel, rel0 = [], M.E28_RELEASE

    def rel_fn(k):
        rel.append(tuple(k))
        if rel0 is not None:
            rel0(k)
    M.E28_RELEASE = rel_fn
    cls0 = M.e28_classify
    if classify:
        M.e28_classify = lambda rec, **k: classify(rec, cls0(rec, **k))
    log = ST.RunLog(M)
    try:
        with Env33(on=on, poetiq=poetiq, direct=direct, attempt2=attempt2, supply=supply, **(env or {})):
            recs, el, sl = ST._solve(M, tasks, order, workers, deadline_sec=deadline_sec, min_admit=min_admit, **kw)
            subs = M.build_submission(tasks, recs)
            slots = M.e28_slot_table(tasks, recs)
    finally:
        log.close()
        runs = list(ST.RUNLOG)
        M.e28_classify = cls0
        if kobj is not None:
            kobj.close()
        M.E28_RELEASE = None
        M.E33_RUN_POETIQ = M.E33_RUN_DIRECT = M.E33_CAN_START = None
    return R(tasks=tasks, recs=recs, sl=sl, e=sl["e28"], subs=subs, slots=slots, runs=runs, hooks=list(HOOK["log"]),
             rel=rel, cans=cans, el=el, kit=kobj, peak=dict(log.peak))


def rows(r, skipped=True):
    rs = [x for x in r.e["attempts"] if skipped or x.get("status") != "skipped"]
    return sorted(rs, key=lambda x: (x.get("start_seq") or 1 << 30))


def seq_of(r):
    return [(x["phase"], x["task"]) for x in rows(r, skipped=False)]


def chain_of(r, tid, a):
    return [x["phase"] for x in rows(r) if x["task"] == tid and ATT.get(x["branch"]) == a]


def g(v):
    """A valid 0-9 grid that encodes v: one cell below 10 (= the fake Duck's ST._grid), else one row of v's digits."""
    v = int(v)
    return [[v]] if v < 10 else [[int(c) for c in str(v)]]


def _fail_all(tids):
    out = {}
    for t in tids:
        out[(t, "A")] = {"kind": "fail"}
        out[(t, "B")] = {"kind": "fail"}
    return out


# ------------------------------------------------------------------ 1. the patch
class TestPatch(unittest.TestCase):
    def test_shas_states_strict(self):
        e29 = _e29_text()
        self.assertEqual(S33.PRE, E29S.E29_POST)
        self.assertEqual(S33.PRE, "d8603109f9530896e9e4a56e52806202fc2cd4c58bd664d9a25222fe6f7592c2")
        self.assertEqual(S33.SHA["patch"], hashlib.sha256(S33.PATCH.encode()).hexdigest())
        self.assertEqual(S33.SHA["applier"], E28S.E28_SCHED_SHA["applier"])
        post = S33.apply_e33(e29)["text"]
        self.assertEqual(hashlib.sha256(post.encode()).hexdigest(), S33.POST)
        self.assertEqual((S33.tree_state(e29), S33.tree_state(post)), ("E29_POST", "E33_PATCHED"))
        self.assertEqual(S33.revert_e33(post)["text"], e29)
        self.assertEqual(S33.apply_e33(post)["applied"], "already")
        self.assertEqual(S33.revert_e33(e29)["reverted"], "not_applied")
        self.assertTrue(S33.verify_e33(post)[0])
        self.assertFalse(S33.verify_e33(e29)[0])
        self.assertEqual(post.count(S33.MARKER), 1)
        self.assertEqual(E29S.apply_unified(e29, S33.PATCH), post)
        self.assertEqual(E29S.apply_unified(post, E29S.reverse_unified(S33.PATCH)), e29)
        lines = e29.splitlines(True)                                    # strict: no fuzz, no offset
        i = next(i for i, ln in enumerate(lines) if "has = bool(qa or qb or qc)" in ln)
        lines[i - 1] = lines[i - 1].rstrip("\n") + "  \n"
        with self.assertRaises(ValueError):
            E29S.apply_unified("".join(lines), S33.PATCH)
        with self.assertRaises(ValueError):
            E29S.apply_unified("\n" + e29, S33.PATCH)

    def test_dir_and_file(self):
        d = FIX["tmp"] / "t33_dir"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        e29 = _e29_text().encode()
        (d / "arc2_duck.py").write_bytes(e29)
        r = S33.apply_e33(d)
        self.assertEqual((r["applied"], r["before"], r["after"], r["py_compile"]["rc"], r["ok"]),
                         ("applied", "E29_POST", "E33_PATCHED", 0, True))
        self.assertEqual(hashlib.sha256((d / "arc2_duck.py").read_bytes()).hexdigest(), S33.POST)
        self.assertTrue(S33.verify_e33(d)[0])
        self.assertEqual(S33.apply_e33(d / "arc2_duck.py")["applied"], "already")
        self.assertEqual(list(d.glob("*tmp")), [])
        rv = S33.revert_e33(d)
        self.assertEqual((rv["reverted"], rv["after"]), ("reverted", "E29_POST"))
        self.assertEqual((d / "arc2_duck.py").read_bytes(), e29)
        self.assertEqual(S33.tree_state(FIX["tmp"] / "no_such_dir_e33"), "OTHER")
        self.assertFalse(S33.verify_e33(FIX["tmp"] / "no_such_dir_e33")[0])

    def test_refusals(self):
        """ORIGINAL, the E28 post, the E29 DIRECT post (the alternative diff), a CRLF copy and a one-character change are
        refused (nothing written)."""
        e29 = _e29_text()
        bad = [ST._orig_text(), E28S.patched_text(ST._orig_text()), e29.replace("\n", "\r\n"),
               e29.replace("def _sec() -> float:", "def _sec( ) -> float:", 1)]
        try:
            import e29d_sched_patch as DS29
            bad.append(DS29.apply_direct(e29)["text"])
            self.assertEqual(S33.tree_state(bad[-1]), "OTHER")
        except ImportError:
            pass
        for t in bad:
            with self.assertRaises(RuntimeError):
                S33.apply_e33(t)
            with self.assertRaises(RuntimeError):
                S33.revert_e33(t)
        d = FIX["tmp"] / "t33_bad"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        (d / "arc2_duck.py").write_bytes(bad[3].encode())
        with self.assertRaises(RuntimeError):
            S33.apply_e33(d)
        self.assertEqual((d / "arc2_duck.py").read_bytes(), bad[3].encode())

    def test_hunks_compile_lf(self):
        heads = [ln for ln in S33.PATCH.splitlines() if ln.startswith("@@")]
        self.assertEqual(len(heads), 15, heads)
        self.assertTrue(all(int(h.split()[1].split(",")[0][1:]) > 3685 for h in heads), heads)   # the E28 / E29 block only
        body = S33.PATCH.splitlines()
        rem = [ln for ln in body if ln.startswith("-") and not ln.startswith("---")]
        self.assertEqual(len(rem), 4, rem)            # has / qlen / the gauge / heap_left: each gets an always-empty qe
        for ln in rem:
            self.assertTrue(any(k in ln for k in ("has = bool(", "qlen = [", "len(qa) + len(qb) + len(qc)])", "heap_left=")), ln)
        for name in ("e33_sched_patch.py", "e33_sched_tests.py", "e33_sched_build.py"):
            b = (HERE / name).read_bytes()
            self.assertNotIn(b"\r", b, name)
            compile(b.decode("utf-8"), name, "exec")
        self.assertNotIn("\r", S33.PATCH)

    def test_generator_reproduces_the_file(self):
        import e33_sched_build as B
        out = B.regen(write=False)
        self.assertEqual(out["e33_sched_patch.py"], hashlib.sha256((HERE / "e33_sched_patch.py").read_bytes()).hexdigest())
        self.assertEqual((out["pre"], out["post"], out["patch"]), (S33.PRE, S33.POST, S33.SHA["patch"]))

    def test_server_env(self):
        self.assertEqual(S33.server_env(None), {"E33_SCHED": "1"})
        self.assertEqual(S33.server_env({"on": False, "poetiq": True}), {})
        c = {"on": True, "poetiq": True, "direct": False, "attempt2": True, "poetiq_think": 60000, "poetiq_answer": 8192,
             "direct_think": None, "direct_answer": 8192}
        self.assertEqual(S33.server_env(c), {"E33_SCHED": "1", "E33_POETIQ": "1", "E33_DIRECT": "0", "E33_ATTEMPT2": "1",
                                             "E33_POETIQ_BUDGET": "68192"})
        self.assertEqual(S33.BRANCH_MAP, {"A:1": 0, "B1:1": 2, "B2:1": 2, "C-POETIQ1:1": 113, "D-POETIQ2:1": 114,
                                          "E-DIRECT:1": 115, "A:2": 1, "B1:2": 122, "B2:2": 122, "C-POETIQ1:2": 123,
                                          "D-POETIQ2:2": 124, "E-DIRECT:2": 125})
        self.assertEqual(S33.SUPPLY_BRANCH_BASE, 200)

    def test_cell3(self):
        e29 = _e29_text().encode()
        d = FIX["tmp"] / "t33_cell3"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        (d / "arc2_duck.py").write_bytes(e29)
        os.environ["E33_POETIQ"] = "1"                                         # stale names are always removed first
        r = S33.cell3(d, {"applied": True, "path": "E28"}, {"want": True, "ok": False})
        self.assertFalse(r.get("ok"))
        self.assertIn("not the E29 scheduler", r["error"])
        self.assertTrue(all(os.environ.get(k) is None for k in E33_ENVS))
        self.assertEqual((d / "arc2_duck.py").read_bytes(), e29)
        chain = {"on": True, "poetiq": True, "direct": True, "attempt2": False, "poetiq_think": 60000, "poetiq_answer": 8192,
                 "direct_think": 104000, "direct_answer": 8192}
        r = S33.cell3(d, {"applied": True, "path": "E29"}, {"want": True, "ok": False}, chain)
        self.assertTrue(r["ok"] and r["after"] == "E33_PATCHED", r)
        self.assertEqual({k: os.environ.get(k) for k in E33_ENVS},
                         {"E33_SCHED": "1", "E33_POETIQ": "1", "E33_DIRECT": "1", "E33_ATTEMPT2": "0",
                          "E33_POETIQ_BUDGET": "68192", "E33_DIRECT_BUDGET": "112192",
                          "E33_A1_ORDER": None})              # FIX-A1ORDER: the chain has no a1_order key
        (d / "arc2_duck.py").write_bytes(e29.replace(b"def _sec() -> float:", b"def _sec( ) -> float:", 1))
        r = S33.cell3(d, {"applied": True, "path": "E29"}, {"want": True, "ok": False}, chain)
        self.assertFalse(r.get("ok"))
        self.assertIn("revert_error", r)                                       # an unknown file is never written
        self.assertTrue(all(os.environ.get(k) is None for k in E33_ENVS))


# ------------------------------------------------------------------ 2. the chain order, stops, B kinds
class TestChain(unittest.TestCase):
    def test_full_chain_order_one_worker(self):
        r = run33(_fail_all(["t000", "t001"]), n=2)
        exp = []
        for a in (1, 2):
            for ph in ("A", "B1", "C-POETIQ1", "D-POETIQ2", "E-DIRECT"):
                exp += [(f"{ph}:{a}", "t000"), (f"{ph}:{a}", "t001")]
        self.assertEqual(seq_of(r), exp)
        for x in rows(r):
            self.assertEqual(x["branch"], S33.BRANCH_MAP[x["phase"]], x["phase"])
            self.assertEqual(x["status"], "done", x)
        self.assertEqual(r.e["setup_failed"], [])
        h = r.hooks
        self.assertEqual([(c["tid"], c["a"], c["st"], c["it"]) for c in h if c["kind"] == "poetiq"],
                         [("t000", 1, "C", 1), ("t001", 1, "C", 1), ("t000", 1, "D", 2), ("t001", 1, "D", 2),
                          ("t000", 2, "C", 1), ("t001", 2, "C", 1), ("t000", 2, "D", 2), ("t001", 2, "D", 2)])
        for c in h:                                          # D receives iteration 1's record as the C hook returned it
            if c["st"] == "C":
                self.assertIsNone(c["prior"])
            if c["st"] == "D":
                self.assertEqual(c["prior"]["submissions"][0]["test_outputs"], [g(41 + 10 * (c["a"] - 1))])
                self.assertEqual(c["prior"]["poetiq"]["iteration"], 1)
            self.assertEqual(c["plan"]["budget"], 68192 if c["kind"] == "poetiq" else 112192)
            self.assertEqual(c["plan"]["branch"], BR[(c["a"], c["st"])])
            self.assertEqual((c["plan"]["attempt"], c["plan"]["e33_attempt"]), (c["a"], c["a"]))   # Poetiq attempt-2 seeds
            want = ("poetiq", f"a{c['a']}_it{c['it']}") if c["kind"] == "poetiq" else ("direct", f"a{c['a']}")
            self.assertEqual(Path(c["out_dir"]).parts[-2:], want)
            self.assertTrue(0 < c["deadline"] - time.monotonic() < 120)
        self.assertEqual([c[0] for c in r.cans], ["C", "C", "D", "D", "E", "E"] * 2)   # bare stage letters (e33_lib)
        for x in rows(r):
            if x["phase"].startswith(("A:", "B1:")):
                self.assertEqual(x["budget"], 104000)
        e33 = r.e["e33"]
        for ph, st in e33["stages"].items():
            want = 0 if ph.startswith("B2") else 2
            self.assertEqual((st["queued"], st["taken"], st["started"], st["done"], st["ended"], st["train_pass"], st["left"]),
                             (want, want, want, want, want, 0, 0), ph)
        self.assertEqual(e33["reached"]["t000"]["a1"]["stages"],
                         [["A:1", "FAIL"], ["B1:1", "FAIL"], ["C-POETIQ1:1", "POETIQ"], ["D-POETIQ2:1", "POETIQ"],
                          ["E-DIRECT:1", "DIRECT"]])
        self.assertEqual(e33["reached"]["t000"]["a1"]["end"], "DIRECT at E-DIRECT:1")
        self.assertEqual(e33["reached"]["t001"]["a2"]["reached"], "E-DIRECT:2")
        self.assertEqual(e33["tokens"]["C-POETIQ1:1"], 2 * 60001)
        self.assertEqual(e33["switches"], {"poetiq": True, "direct": True, "attempt2": True})
        self.assertEqual(e33["hooks"], {"poetiq": True, "direct": True, "can_start": True})
        # answers: attempt_1 = the best train match of attempt 1 (C ties D at 0.5: the earlier); attempt_2 = attempt 2's
        for t in ("t000", "t001"):
            self.assertEqual(r.subs[t], [{"attempt_1": g(41), "attempt_2": g(51)}])
            s = r.slots[t][0]
            self.assertEqual((s["attempt_1"], s["a1_step"], s["attempt_2"], s["a2_step"]),
                             ("C-POETIQ1:1", "2-best", "C-POETIQ1:2", "a2:2-best"))
            self.assertEqual(len(r.recs[t]), 10)
            by = {x["branch"]: x for x in r.e["attempts"] if x["task"] == t}
            self.assertEqual((by[113]["answer_slots"], by[123]["answer_slots"], by[0]["answer_slots"]),
                             (["attempt_1"], ["attempt_2"], [None]))
        rec = next(x for x in r.recs["t000"] if x["branch"] == 113)
        self.assertEqual((rec["e33"]["phase"], rec["e28"]["phase"], rec["e33"]["e33_attempt"], rec["submissions"][0]["branch"]),
                         ("C-POETIQ1:1", "C-POETIQ1:1", 1, 113))
        self.assertIn("poetiq", rec)
        self.assertEqual(r.peak, {"A:1": 1, "B1:1": 1, "A:2": 1, "B1:2": 1})

    def test_stop_at_first_pass_each_stage(self):
        sc = {("t000", "A"): {"kind": "pass", "at": 3000}}
        sc.update(_fail_all(["t001", "t002", "t003", "t004"]))
        sc[("t001", "B")] = {"kind": "pass", "at": 2000, "value": 2}
        beh = {("t002", "C"): {"pass": True}, ("t003", "D"): {"pass": True}}
        r = run33(sc, beh, n=5)
        want = {"t000": ["A"], "t001": ["A", "B1"], "t002": ["A", "B1", "C-POETIQ1"],
                "t003": ["A", "B1", "C-POETIQ1", "D-POETIQ2"], "t004": ["A", "B1", "C-POETIQ1", "D-POETIQ2", "E-DIRECT"]}
        for t, w in want.items():
            for a in (1, 2):
                self.assertEqual(chain_of(r, t, a), [f"{p}:{a}" for p in w], (t, a))
        st = r.e["e33"]["stages"]
        for ph in ("A", "B1", "C-POETIQ1", "D-POETIQ2"):
            for a in (1, 2):
                self.assertEqual(st[f"{ph}:{a}"]["train_pass"], 1, ph)
        self.assertEqual(r.subs["t002"][0], {"attempt_1": g(41), "attempt_2": g(51)})
        self.assertEqual(r.subs["t003"][0], {"attempt_1": g(42), "attempt_2": g(52)})
        self.assertEqual(r.slots["t003"][0]["a1_step"], "1-pass")
        self.assertEqual(r.slots["t003"][0]["attempt_1"], "D-POETIQ2:1")
        self.assertEqual(r.subs["t000"][0], {"attempt_1": g(1), "attempt_2": [[9]]})        # attempt 2 same -> the input
        self.assertEqual(r.slots["t000"][0]["a2_step"], "5-input")
        self.assertEqual(r.e["e33"]["reached"]["t002"]["a1"]["end"], "TRAIN_PASS at C-POETIQ1:1")

    def test_b1_b2_straight_to_c_and_time(self):
        sc = {("t000", "A"): {"kind": "fail"}, ("t000", "B"): {"kind": "pass", "at": 2000, "value": 2},
              ("t001", "A"): {"kind": "cap", "sleep": 0.0}, ("t001", "B"): {"kind": "pass", "at": 2000, "value": 3},
              ("t002", "A"): {"kind": "partial", "at": 2000}, ("t003", "A"): {"kind": "partial", "at": 2000}}
        beh = {(t, "C"): {"pass": True} for t in ("t000", "t001", "t002", "t003")}

        def cls(rec, out):
            ph = (rec.get("e28") or {}).get("phase")
            if rec.get("task_id") == "t002" and ph in ("A:1", "A:2"):
                out = dict(out, kind="ODD", end_reason="test")         # a non-pass A kind without a B phase
            if rec.get("task_id") == "t003" and ph == "A:1":
                out = dict(out, kind="TIME", end_reason="running_at_window_end")
            return out
        r = run33(sc, beh, n=4, classify=cls, analyzer_timeout=7200.0)
        self.assertEqual(chain_of(r, "t000", 1), ["A:1", "B1:1"])
        self.assertEqual(chain_of(r, "t001", 1), ["A:1", "B2:1"])
        self.assertEqual(chain_of(r, "t001", 2), ["A:2", "B2:2"])
        self.assertEqual(chain_of(r, "t002", 1), ["A:1", "C-POETIQ1:1"])     # straight to C
        self.assertEqual(chain_of(r, "t002", 2), ["A:2", "C-POETIQ1:2"])
        self.assertEqual(chain_of(r, "t003", 1), ["A:1"])                    # TIME: nothing queued, no attempt 2
        self.assertEqual(chain_of(r, "t003", 2), [])
        by = {(x["task"], x["phase"]): x for x in rows(r)}
        b1, b2, b2b = by[("t000", "B1:1")], by[("t001", "B2:1")], by[("t001", "B2:2")]
        self.assertEqual((b1["budget"], b1["window_plan"], b1["agent_budget"], b1["b_class"]), (104000, None, 104000, "FAIL"))
        self.assertEqual((b2["budget"], b2["window_plan"], b2["window"], b2["agent_budget"], b2["b_class"], b2["branch"]),
                         (120000, 130560, 130560, 120000, "B2", 2))
        self.assertEqual((b2b["budget"], b2b["window"], b2b["branch"]), (120000, 130560, 122))
        self.assertAlmostEqual(b2["request_timeout"], 7200.0 * 120000 / 104000, places=3)
        self.assertEqual(r.e["a_class"]["t003"]["kind"], "TIME")
        self.assertEqual(r.e["e33"]["reached"]["t003"]["a1"]["end"], "TIME at A:1")
        self.assertEqual([x[0] for x in r.e["e33"]["a2_queued"]], ["t000", "t001", "t002"])

    def test_attempt2_groups_and_order(self):
        sc = {("t000", "A"): {"kind": "pass", "at": 30000}, ("t001", "A"): {"kind": "fail"},
              ("t002", "A"): {"kind": "pass", "at": 10000}, ("t003", "A"): {"kind": "fail"},
              ("t004", "A"): {"kind": "pass", "at": 20000}, ("t001", "B"): {"kind": "fail"}, ("t003", "B"): {"kind": "fail"}}
        r = run33(sc, n=5, poetiq=False, direct=False)
        a1 = [("A:1", f"t00{i}") for i in range(5)]
        self.assertEqual(seq_of(r), a1 + [("B1:1", "t001"), ("B1:1", "t003"),
                                          ("A:2", "t001"), ("A:2", "t003"), ("A:2", "t002"), ("A:2", "t004"), ("A:2", "t000"),
                                          ("B1:2", "t001"), ("B1:2", "t003")])
        self.assertEqual(r.e["e33"]["a2_queued"], [["t000", [2, 30000, 0]], ["t001", [1, 0, 1]], ["t002", [2, 10000, 2]],
                                                   ["t003", [1, 0, 3]], ["t004", [2, 20000, 4]]])
        by = {(x["task"], x["phase"]): x for x in rows(r)}
        self.assertEqual(by[("t002", "A:2")]["e33_group"], [2, 10000, 2])
        self.assertEqual(by[("t003", "B1:2")]["e33_group"], [1, 0, 3])          # the group stays with the task's attempt 2
        self.assertEqual(r.hooks, [])                                             # Poetiq / Direct off: never called
        self.assertEqual(r.cans, [])
        self.assertEqual(r.e["e33"]["switches"], {"poetiq": False, "direct": False, "attempt2": True})
        self.assertEqual(r.e["e33"]["reached"]["t001"]["a1"]["end"], "FAIL at B1:1")

    def test_switch_items(self):
        """Poetiq off: A -> B -> E; Direct off: ... -> D ends; attempt 2 off: no attempt-2 item; a bad value = off."""
        sc = _fail_all(["t000"])
        r = run33(sc, n=1, poetiq=False)
        self.assertEqual(chain_of(r, "t000", 1), ["A:1", "B1:1", "E-DIRECT:1"])
        r = run33(sc, n=1, direct=False, attempt2="yes")
        self.assertEqual(chain_of(r, "t000", 1), ["A:1", "B1:1", "C-POETIQ1:1", "D-POETIQ2:1"])
        self.assertEqual(chain_of(r, "t000", 2), [])
        self.assertEqual(r.e["e33"]["a2_queued"], [])
        self.assertEqual(r.e["e33"]["switches"]["attempt2"], False)
        r = run33(sc, n=1, attempt2=False, env={"E33_POETIQ_BUDGET": "70000", "E33_DIRECT_BUDGET": "bad"})
        self.assertEqual(chain_of(r, "t000", 2), [])
        self.assertEqual([c["plan"]["budget"] for c in r.hooks], [70000, 70000, 112192])

    def test_b2_waiting_first_and_candidate_order(self):
        """_e33_candidates (pure): attempt 1 before attempt 2; a B2 that waited before every other item of its attempt;
        then A, B, C, D, E; inside a stage the group key; a busy (task, attempt) is never offered."""
        M = FIX["e33"]
        mk = lambda tid, a, st, grp, name=None, seq=1: ((a, 0), seq, tid, {"e33_attempt": a, "e33_stage": st, "e33_group": grp,
                                                                            "e33_name": name or st, "branch": BR[(a, st)]})
        qe = [mk("t5", 2, "A", [1, 0, 5], seq=1), mk("t1", 1, "E", [0, 0, 1], seq=2), mk("t2", 1, "B", [0, 0, 2], "B2", 3),
              mk("t0", 1, "A", [0, 0, 0], seq=4), mk("t3", 1, "C", [0, 0, 3], seq=5), mk("t4", 2, "B", [1, 0, 4], "B2", 6),
              mk("t6", 2, "A", [2, 500, 6], seq=7), mk("t7", 2, "A", [2, 100, 7], seq=8), mk("t8", 1, "D", [0, 0, 8], seq=9),
              mk("t9", 1, "B", [0, 0, 9], "B1", 10)]
        e33 = {"busy": set()}
        order = lambda meta: [(c[2], c[3]["e33_attempt"], c[3]["e33_stage"]) for c in M._e33_candidates(qe, meta, e33)]
        self.assertEqual(order({}), [("t0", 1, "A"), ("t2", 1, "B"), ("t9", 1, "B"), ("t3", 1, "C"), ("t8", 1, "D"),
                                     ("t1", 1, "E"), ("t5", 2, "A"), ("t7", 2, "A"), ("t6", 2, "A"), ("t4", 2, "B")])
        meta = {("t2", 2): {"waiting": True}, ("t4", 122): {"waiting": True}}
        self.assertEqual(order(meta)[:2], [("t2", 1, "B"), ("t0", 1, "A")])
        self.assertEqual(order(meta)[6:8], [("t4", 2, "B"), ("t5", 2, "A")])     # attempt 2's waiting B2 after attempt 1
        e33["busy"] = {("t5", 2), ("t0", 1)}
        self.assertNotIn(("t5", 2, "A"), order({}))
        self.assertNotIn(("t0", 1, "A"), order({}))

    def test_next_stage_rule(self):
        M = FIX["e33"]
        on = {"poetiq": True, "direct": True}
        cases = [("A", "TRAIN_PASS", None), ("A", "TIME", None), ("A", "B1", ("B", "B1")), ("A", "FAIL", ("B", "B1")),
                 ("A", "B2", ("B", "B2")), ("A", "ODD", ("C", None)), ("B", "B2", ("C", None)), ("B", "FAIL", ("C", None)),
                 ("B", "TIME", None), ("C", "POETIQ", ("D", None)), ("C", "TRAIN_PASS", None), ("C", "NO_RECORD", None),
                 ("D", "POETIQ", ("E", None)), ("E", "DIRECT", None), ("D", "SKIPPED", None)]
        for st, kind, want in cases:
            self.assertEqual(M._e33_next(on, st, kind), want, (st, kind))
        self.assertEqual(M._e33_next({"poetiq": False, "direct": True}, "B", "FAIL"), ("E", None))
        self.assertEqual(M._e33_next({"poetiq": False, "direct": False}, "A", "ODD"), None)
        self.assertEqual(M._e33_next({"poetiq": True, "direct": False}, "D", "POETIQ"), None)


# ------------------------------------------------------------------ 3. concurrency
def _intervals(r):
    """(task, attempt) -> [(t0, t1, branch)] from the Duck run log and the hook log."""
    out, open_ = {}, {}
    for ev, tid, b, ph, t in r.runs:
        if ev == "start":
            open_[(tid, b)] = t
        else:
            out.setdefault((tid, ATT[b]), []).append((open_.pop((tid, b)), t, b))
    for c in r.hooks:
        out.setdefault((c["tid"], c["a"]), []).append((c["t0"], c["t1"], c["plan"]["branch"]))
    return out


def _rand_scen(n, seed):
    rnd = random.Random(seed)
    sc, beh = {}, {}
    for i in range(n):
        t = f"t{i:03d}"
        u = rnd.random()
        sc[(t, "A")] = ({"kind": "cap", "sleep": 0.0} if u < 0.2 else {"kind": "fail"} if u < 0.4 else
                        {"kind": "partial", "at": 4000} if u < 0.6 else {"kind": "pass", "at": rnd.randrange(3, 40) * 1000})
        sc[(t, "B")] = {"kind": "pass", "at": 20000, "value": 2} if rnd.random() < 0.3 else {"kind": "partial", "at": 3000}
        for st in ("C", "D"):
            beh[(t, st)] = {"pass": rnd.random() < 0.25, "sleep": rnd.choice([0.002, 0.01, 0.03])}
        beh[(t, "E")] = {"sleep": rnd.choice([0.002, 0.02])}
    return sc, beh


class TestConcurrency(unittest.TestCase):
    def _invariants(self, r):
        e = r.e
        self.assertEqual(e["setup_failed"], [])
        self.assertEqual((r.sl["worker_errors"], r.sl.get("stage2_errors", 0)), (0, 0))
        taken = [(x["task"], x["branch"]) for x in e["attempts"]]
        self.assertEqual(len(taken), len(set(taken)))
        starts = {}
        for ev, tid, b, ph, t in r.runs:
            if ev == "start":
                starts[(tid, b)] = starts.get((tid, b), 0) + 1
        for c in r.hooks:
            k = (c["tid"], c["plan"]["branch"])
            starts[k] = starts.get(k, 0) + 1
        self.assertTrue(all(v == 1 for v in starts.values()), starts)                     # one start per (task, branch)
        ran = {(x["task"], x["branch"]) for x in e["attempts"] if x.get("status") != "skipped"}
        self.assertEqual(set(starts), ran)
        for k in taken:                                                                    # one release per taken item
            self.assertEqual(r.rel.count(k), 1, (k, r.rel.count(k)))
        self.assertEqual(set(r.rel), set(taken))
        for ph in S33.BRANCH_MAP:
            self.assertEqual(e["started"][ph], e["done"][ph], ph)                         # one done per start
            st = e["e33"]["stages"][ph]
            self.assertEqual(st["taken"], st["started"] + sum(st["skipped"].values()) - st["skipped_after_start"], ph)
            self.assertEqual(st["left"], 0, ph)
        for (tid, a), iv in _intervals(r).items():                                        # strictly sequential per attempt
            iv.sort()
            for (s0, e0, b0), (s1, e1, b1) in zip(iv, iv[1:]):
                self.assertGreaterEqual(s1, e0, (tid, a, b0, b1))
        for t, rs in r.recs.items():
            bs = [x["branch"] for x in rs]
            self.assertEqual(len(bs), len(set(bs)), t)

    def test_four_workers_invariants(self):
        sc, beh = _rand_scen(16, 11)
        r = run33(sc, beh, n=16, workers=4)
        self._invariants(r)
        self.assertGreater(r.e["started"]["E-DIRECT:1"], 0)
        for x in rows(r, skipped=False):                       # attempt 2 never ahead of a runnable attempt-1 item
            if ATT[x["branch"]] == 2:
                self.assertEqual(x["e33_q1_left"], 0, x)
        self.assertGreater(sum(1 for x in rows(r) if ATT[x["branch"]] == 2), 0)

    def test_four_workers_real_reservation(self):
        """The real NB5 Reservation (projected, 73 blocks): Duck 36 / B2 38 / Poetiq 27 / Direct 34 blocks; backfill across
        stages; nothing leaks."""
        sc, beh = _rand_scen(10, 5)
        r = run33(sc, beh, n=10, workers=4, kit=lambda M: ST.ResKit(M, cap=73))
        self._invariants(r)
        self.assertEqual((r.kit.res.used, r.kit.res.held), (0, {}))
        self.assertGreater(r.e["fit_checks"], 0)
        self.assertEqual(r.kit.res.stats["e28_try_grants"] + r.e["forced_admits"],
                         sum(v for k, v in r.e["started"].items()) + sum(sum(s["skipped"].values()) for s in r.e["e33"]["stages"].values()))

    def test_attempts_of_one_task_run_concurrently(self):
        sc = {("t000", "A:1"): {"kind": "fail"}, ("t000", "B1:1"): {"kind": "partial", "at": 30000, "sleep": 0.01},
              ("t000", "A:2"): {"kind": "pass", "at": 2000}}
        r = run33(sc, n=1, workers=2)
        iv = _intervals(r)
        b1 = next(x for x in iv[("t000", 1)] if x[2] == 2)
        a2 = next(x for x in iv[("t000", 2)] if x[2] == 1)
        self.assertLess(a2[0], b1[1])                                                       # A:2 started while B1:1 ran
        self.assertGreater(a2[1], b1[0])
        self._invariants(r)


# ------------------------------------------------------------------ 4. hooks: no hook / False / raising / no record
class TestHooks(unittest.TestCase):
    def _duck_only(self, r):
        self.assertTrue({b for _, _, b, _, _ in r.runs} <= {0, 2, 1, 122}, {b for _, _, b, _, _ in r.runs})

    def _skipped(self, r, tid, label, why):
        x = next(x for x in rows(r) if x["task"] == tid and x["phase"] == label)
        self.assertEqual((x["status"], x["kind"], x["end_reason"], x["tokens"]), ("skipped", "SKIPPED", why, 0))
        self.assertEqual(r.rel.count((tid, x["branch"])), 1)
        self.assertFalse([y for y in r.recs[tid] if y["branch"] == x["branch"]])           # nothing published
        return x

    def test_no_poetiq_hook(self):
        r = run33(_fail_all(["t000"]), n=1, hooks=("direct",))
        for a in (1, 2):
            self.assertEqual(chain_of(r, "t000", a), [f"A:{a}", f"B1:{a}", f"C-POETIQ1:{a}"])
            self._skipped(r, "t000", f"C-POETIQ1:{a}", "no_hook")
        self._duck_only(r)
        self.assertEqual(r.hooks, [])                                                       # Direct never reached
        st = r.e["e33"]["stages"]["C-POETIQ1:1"]
        self.assertEqual((st["taken"], st["started"], st["skipped"], st["skipped_for_time"]), (1, 0, {"no_hook": 1}, 0))
        self.assertEqual(r.e["e33"]["reached"]["t000"]["a1"]["end"], "skipped (no_hook) at C-POETIQ1:1")
        self.assertEqual(r.e["e33"]["hooks"], {"poetiq": False, "direct": True, "can_start": True})

    def test_no_direct_hook_and_no_can_start(self):
        r = run33(_fail_all(["t000"]), n=1, hooks=("poetiq",))
        self.assertEqual(chain_of(r, "t000", 1)[-1], "E-DIRECT:1")
        self._skipped(r, "t000", "E-DIRECT:1", "no_hook")
        r = run33(_fail_all(["t000"]), n=1, can=None)
        self._skipped(r, "t000", "C-POETIQ1:1", "no_can_start_hook")
        self.assertEqual(r.hooks, [])
        self._duck_only(r)

    def test_can_start_false_raise_tuple(self):
        r = run33(_fail_all(["t000"]), n=1, can=lambda st, dl: st != "E")
        self.assertEqual(chain_of(r, "t000", 1), ["A:1", "B1:1", "C-POETIQ1:1", "D-POETIQ2:1", "E-DIRECT:1"])
        self._skipped(r, "t000", "E-DIRECT:1", "late")
        self.assertEqual(r.e["e33"]["stages"]["E-DIRECT:1"]["skipped_for_time"], 1)

        def boom(st, dl):
            raise RuntimeError("bad admission hook")
        r = run33(_fail_all(["t000"]), n=1, can=boom)
        self._skipped(r, "t000", "C-POETIQ1:1", "hook_error RuntimeError")
        self.assertEqual(chain_of(r, "t000", 1), ["A:1", "B1:1", "C-POETIQ1:1"])
        r = run33(_fail_all(["t000"]), n=1, can=lambda st, dl: (st == "C", "why", 1.0))     # e33_lib.can_start's tuple
        self.assertEqual(chain_of(r, "t000", 1), ["A:1", "B1:1", "C-POETIQ1:1", "D-POETIQ2:1"])
        self._skipped(r, "t000", "D-POETIQ2:1", "late")

    def test_run_hook_raises_or_returns_nothing(self):
        for beh, label, err in (({("t000", "D"): {"raise": True}}, "D-POETIQ2:1", "RuntimeError: stub poetiq failed"),
                                ({("t000", "C"): {"ret": None}}, "C-POETIQ1:1", None),
                                ({("t000", "C"): {"ret": False}}, "C-POETIQ1:1", None),
                                ({("t000", "C"): {"ret": "x"}}, "C-POETIQ1:1", None),
                                ({("t000", "E"): {"raise": True}}, "E-DIRECT:1", "RuntimeError: stub direct failed")):
            r = run33(_fail_all(["t000"]), beh, n=1, attempt2=False)
            ch = chain_of(r, "t000", 1)
            self.assertEqual(ch[-1], label, beh)                                            # the chain ends there
            x = next(x for x in rows(r) if x["phase"] == label)
            self.assertEqual((x["status"], x["kind"], x["end_reason"], x["error"]),
                             ("no_record", "NO_RECORD", "error" if err else "no_record", err))
            self.assertFalse([y for y in r.recs["t000"] if y["branch"] == x["branch"]])
            self.assertEqual(r.rel.count(("t000", x["branch"])), 1)
            self.assertEqual(r.e["e33"]["stages"][label]["no_record"], 1)
            self.assertEqual(r.e["e33"]["reached"]["t000"]["a1"]["end"], f"NO_RECORD at {label}")
            self._duck_only(r)
            self.assertEqual(r.sl["worker_errors"], 0)

    def test_solver_not_admitted_is_a_skip(self):
        """e33_poetiq always returns a record: status "not_admitted" / "not_run", or poetiq.admitted False = the solver's own
        skip: not published, recorded as skipped after start, the chain ends (one start / done / release)."""
        for ret, why in (({"task_id": "t000", "status": "not_admitted", "accepted": False, "generated_tokens": 0,
                           "submissions": [], "poetiq": {"admitted": False, "reason": "late"}}, "not_admitted"),
                         ({"task_id": "t000", "status": "not_run", "accepted": False, "submissions": []}, "not_run"),
                         ({"task_id": "t000", "status": "done", "accepted": False, "submissions": [{"test_outputs": [g(4)]}],
                           "poetiq": {"admitted": False}}, "not_admitted")):
            for st in ("C", "D"):
                r = run33(_fail_all(["t000"]), {("t000", st): {"ret": ret}}, n=1, attempt2=False)
                label = "C-POETIQ1:1" if st == "C" else "D-POETIQ2:1"
                self.assertEqual(chain_of(r, "t000", 1)[-1], label)
                x = next(x for x in rows(r) if x["phase"] == label)
                self.assertEqual((x["status"], x["kind"], x["end_reason"], x["skipped_after_start"]), ("skipped", "SKIPPED", why, True))
                self.assertIsNotNone(x["start_sec"])
                self.assertFalse([y for y in r.recs["t000"] if y["branch"] == x["branch"]])
                self.assertEqual(r.rel.count(("t000", x["branch"])), 1)
                s = r.e["e33"]["stages"][label]
                self.assertEqual((s["started"], s["done"], s["skipped"], s["skipped_after_start"], s["ended"]), (1, 1, {why: 1}, 1, 0))
                self.assertEqual(s["skipped_for_time"], 1 if why == "not_admitted" else 0)
                self.assertEqual(r.e["e33"]["reached"]["t000"]["a1"]["end"], f"skipped ({why}) at {label}")


# ------------------------------------------------------------------ 5. the window end mid-chain
class TestWindow(unittest.TestCase):
    def test_window_closes_mid_chain(self):
        # deadline 9 s, min_admit 7 s: the admission window closes at 2 s (E28's live TIME check is deadline - 5 s = 4 s)
        beh = {(t, "C"): {"sleep": 1.0} for t in ("t000", "t001")}
        r = run33(_fail_all(["t000", "t001"]), beh, n=2, deadline_sec=9.0, min_admit=7.0)
        self.assertTrue(all((x.get("start_sec") or 0) <= 2.0 + 0.25 for x in rows(r)), [x.get("start_sec") for x in rows(r)])
        st = r.e["e33"]["stages"]
        self.assertEqual(st["C-POETIQ1:1"]["done"], 2)
        self.assertEqual(st["D-POETIQ2:1"]["left"], 2)                       # queued when C ended, never started
        self.assertEqual(st["D-POETIQ2:1"]["skipped_for_time"], 2)
        self.assertEqual(st["D-POETIQ2:1"]["started"], 0)
        self.assertEqual(r.e["e33"]["reached"]["t000"]["a1"]["open"], "queued D-POETIQ2:1")
        self.assertLess(r.el, 5.0)                                         # ends with the last running item
        self.assertEqual({k: v for k, v in r.e["started"].items() if v}, {k: v for k, v in r.e["done"].items() if v})
        self.assertEqual(r.sl["heap_left"], sum(s["left"] for s in st.values()))

    def test_late_admission_skips_and_ends_chain(self):
        beh = {(t, "C"): {"sleep": 1.0} for t in ("t000", "t001")}      # admissible until 2 s of the 9-s run (the C's)
        r = run33(_fail_all(["t000", "t001"]), beh, n=2, deadline_sec=9.0, min_admit=0.5,
                  can=lambda st, dl: time.monotonic() + 7.0 < dl, attempt2=False)
        st = r.e["e33"]["stages"]
        self.assertEqual(st["C-POETIQ1:1"]["started"], 2)
        self.assertEqual(st["D-POETIQ2:1"]["skipped"], {"late": 2})
        self.assertEqual(st["E-DIRECT:1"]["queued"], 0)                      # the chain ended at the late D
        self.assertEqual([c["st"] for c in r.hooks], ["C", "C"])


# ------------------------------------------------------------------ 6. the answer rule (synthetic records)
def drec(a, st, acc=None, part=None, name=None, nq=1, accs=None, seq=1):
    """A Duck record of stage A / B of attempt a: acc = an accepted grid value, part = (value, train_exact)."""
    label = f"{name or ('A' if st == 'A' else 'B1')}:{a}"
    subs = [{"accepted": True, "train_exact": [True, True], "test_outputs": [g(v)] * nq, "t": 10 - i, "branch": BR[(a, st)]}
            for i, v in enumerate(accs or ([acc] if acc is not None else []))]
    chk = [{"accepted": False, "train_exact": list(part[1]), "test_outputs": [g(part[0])] * nq, "t": 5}] if part else []
    return {"task_id": "x", "branch": BR[(a, st)], "submissions": subs, "checked": chk, "accepted": bool(subs),
            "publish_seq": seq, "e28": {"phase": label, "e33_attempt": a, "e33_stage": st}}


def prec(a, st, grids, tp=False, score=0.5, exact=(True, False), seq=1):
    return {"task_id": "x", "branch": BR[(a, st)], "accepted": tp, "publish_seq": seq,
            "submissions": [{"test_outputs": grids, "accepted": tp, "train_exact": [True, True] if tp else list(exact)}],
            "poetiq": {"train_pass": tp, "train_score": score},
            "e28": {"phase": f"{'C-POETIQ1' if st == 'C' else 'D-POETIQ2'}:{a}", "e33_attempt": a, "e33_stage": st}}


def erec(a, grids, seq=1):
    return {"task_id": "x", "branch": BR[(a, "E")], "accepted": False, "publish_seq": seq,
            "submissions": [{"test_outputs": grids, "accepted": False, "train_exact": []}],
            "e28": {"phase": f"E-DIRECT:{a}", "e33_attempt": a, "e33_stage": "E"}}


def srec(branch, acc):
    return {"task_id": "x", "branch": branch, "accepted": True, "checked": [], "e28": {"phase": "S1"},
            "submissions": [{"accepted": True, "train_exact": [True, True], "test_outputs": [g(acc)], "t": 3}]}


TASK1 = {"train": [{"input": [[0]], "output": [[0]]}] * 2, "test": [{"input": [[9]]}]}
TASK2 = {"train": [{"input": [[0]], "output": [[0]]}] * 2, "test": [{"input": [[9]]}, {"input": [[8]]}]}


class TestAnswerRule(unittest.TestCase):
    def sel(self, recs, task=TASK1):
        M = FIX["e33"]
        rows_ = []
        with Env33(on=True):
            out = M.select_attempts_e28(task, recs, rows_)                     # the dispatch (E33-tagged records)
            self.assertEqual(M.build_submission({"x": task}, {"x": recs})["x"], out)
        return [(o["attempt_1"][0][0], o["attempt_2"][0][0], r_["attempt_1"], r_["a1_step"], r_["attempt_2"], r_["a2_step"])
                for o, r_ in zip(out, rows_)]

    def test_attempt1_cases(self):
        P, N = (True, False), (False, False)
        # 1: the first train pass (Duck A; a Poetiq pass after non-pass Duck; the earlier of two passes)
        self.assertEqual(self.sel([drec(1, "A", acc=1)])[0][:4], (1, 9, "A:1", "1-pass"))
        self.assertEqual(self.sel([drec(1, "A", part=(5, P)), drec(1, "B", part=(6, N)), prec(1, "C", [g(3)], tp=True),
                                   prec(1, "D", [g(4)], tp=True)])[0][:4], (3, 4, "C-POETIQ1:1", "1-pass"))
        # 2: the best train match (Poetiq's train score when it has no train_exact); a tie keeps the earlier
        self.assertEqual(self.sel([drec(1, "A", part=(5, P)), prec(1, "C", [g(3)], score=0.8, exact=())])[0][:4],
                         (3, 5, "C-POETIQ1:1", "2-best"))
        self.assertEqual(self.sel([drec(1, "A", part=(5, P)), prec(1, "C", [g(3)], score=0.5, exact=())])[0][:4],
                         (5, 3, "A:1", "2-best"))
        self.assertEqual(self.sel([drec(1, "A", part=(5, N)), drec(1, "B", part=(6, P), name="B2")])[0][:4],
                         (6, 5, "B2:1", "2-best"))
        # 3: the attempt-1 Direct grid (no Duck / Poetiq candidate)
        self.assertEqual(self.sel([drec(1, "A"), erec(1, [g(7)])])[0][:4], (7, 9, "E-DIRECT:1", "3-direct"))
        # 4: the E28 fallbacks (select_attempts_r5 over the Duck records: here attempt 2's Duck grid)
        self.assertEqual(self.sel([drec(1, "A"), drec(2, "A", part=(8, P))])[0], (8, 9, "r5", "4-fallback", "input", "5-input"))
        # 5: the test input
        self.assertEqual(self.sel([drec(1, "A"), erec(1, [[[10]]])])[0], (9, 9, "input", "5-input", "same", "same"))

    def test_attempt2_cases(self):
        P = (True, False)
        # the attempt-2 chain's answer when it differs (Duck pass / Poetiq best / Direct only)
        self.assertEqual(self.sel([drec(1, "A", acc=1), drec(2, "A", acc=2)])[0][4:], ("A:2", "a2:1-pass"))
        self.assertEqual(self.sel([drec(1, "A", acc=1), drec(2, "A"), prec(2, "C", [g(3)])])[0][1:],
                         (3, "A:1", "1-pass", "C-POETIQ1:2", "a2:2-best"))
        self.assertEqual(self.sel([drec(1, "A", acc=1), erec(2, [g(6)])])[0][1:], (6, "A:1", "1-pass", "E-DIRECT:2", "a2:3-direct"))
        # equal to attempt_1 -> the next distinct attempt-1 candidate (its Direct grid; the second best)
        self.assertEqual(self.sel([drec(1, "A", acc=1), drec(2, "A", acc=1), erec(1, [g(7)])])[0][1:],
                         (7, "A:1", "1-pass", "E-DIRECT:1", "a1-next"))
        self.assertEqual(self.sel([drec(1, "A", part=(5, P)), prec(1, "C", [g(3)], score=0.3, exact=())])[0][:2], (5, 3))
        self.assertEqual(self.sel([drec(1, "A", part=(5, P)), prec(1, "C", [g(3)], score=0.3, exact=())])[0][4:],
                         ("C-POETIQ1:1", "a1-next"))
        # no attempt-1 candidate left -> the E28 fallbacks (an older accepted Duck grid), else the input
        self.assertEqual(self.sel([drec(1, "A", accs=[1, 4]), drec(2, "A", acc=1)])[0], (1, 4, "A:1", "1-pass", "r5", "4-fallback"))
        self.assertEqual(self.sel([drec(1, "A", acc=1)])[0][4:], ("input", "5-input"))
        self.assertEqual(self.sel([drec(1, "A"), drec(2, "A")])[0], (9, 9, "input", "5-input", "same", "same"))

    def test_supply_and_multi_test(self):
        P = (True, False)
        self.assertEqual(self.sel([drec(1, "A", part=(5, P)), srec(200, 6)])[0], (6, 5, "S200", "1-pass", "A:1", "a1-next"))
        # multi-test: grid K -> test K; an invalid grid at one test falls to the next candidate of THAT test
        r = self.sel([drec(1, "A"), prec(1, "C", [g(3), [[10]]], tp=True), erec(1, [g(7), g(8)]),
                      prec(2, "C", [g(5), g(6)])], task=TASK2)
        self.assertEqual(r[0], (3, 5, "C-POETIQ1:1", "1-pass", "C-POETIQ1:2", "a2:2-best"))
        self.assertEqual(r[1], (8, 6, "E-DIRECT:1", "3-direct", "C-POETIQ1:2", "a2:2-best"))

    def test_untagged_records_keep_the_e29_rule(self):
        M = FIX["e33"]
        recs = [ST.TestSlots.r(None, 0, "A", acc=1), ST.TestSlots.r(None, 1, "C", acc=3)]
        with Env33(on=True, supply=True):
            self.assertEqual(M.select_attempts_e28(TASK1, recs), FIX["e29"].select_attempts_e28(TASK1, recs))


# ------------------------------------------------------------------ 7. multi-test end to end
class TestMultiTestE2E(unittest.TestCase):
    def test_two_tests(self):
        sc = _fail_all(["t000", "t001"])
        beh = {("t000", 1, "C"): {"pass": True, "grids": [g(41), [[10]]]}, ("t000", 2, "C"): {"pass": True, "grids": [g(51), g(53)]},
               ("t001", "E"): {"grids": [g(71), g(72)]}}
        r = run33(sc, beh, n=2, nq=2)
        self.assertEqual(r.subs["t000"], [{"attempt_1": g(41), "attempt_2": g(51)}, {"attempt_1": [[9]], "attempt_2": g(53)}])
        self.assertEqual([s["a1_step"] for s in r.slots["t000"]], ["1-pass", "5-input"])
        self.assertEqual(r.subs["t001"], [{"attempt_1": g(41), "attempt_2": g(51)}, {"attempt_1": g(41), "attempt_2": g(51)}])
        self.assertEqual(chain_of(r, "t000", 1), ["A:1", "B1:1", "C-POETIQ1:1"])


# ------------------------------------------------------------------ 8. supply only when the chain queue is empty
class TestSupply(unittest.TestCase):
    def test_supply_after_chains(self):
        sc = _fail_all(["t000", "t002"])
        sc.update({("t001", "A"): {"kind": "pass", "at": 60000, "sleep": 0.002}, ("t000", "S"): {"kind": "partial", "at": 3000},
                   ("t002", "S"): {"kind": "partial", "at": 3000}})
        beh = {("t002", "C"): {"pass": True}}
        for workers in (1, 2):
            r = run33(sc, beh, n=3, workers=workers, supply=True, env={"E29_SUPPLY_MAX_ATTEMPTS": 14})
            srows = [x for x in rows(r) if str(x["phase"]).startswith("S")]
            self.assertEqual(sorted((x["task"], x["branch"]) for x in srows),
                             [("t000", 200), ("t000", 201), ("t000", 202), ("t000", 203)], workers)
            for x in srows:
                self.assertEqual(x["queues_at_start"], [0, 0, 0, 0], x)                # the chain queue was empty
            if workers == 1:
                last = max(x["start_seq"] for x in rows(r) if x["branch"] in ATT)
                self.assertTrue(all(x["start_seq"] > last for x in srows))
            sup = r.e["supply"]
            self.assertEqual(sup["attempts_by_task"]["t000"], 14)
            self.assertNotIn("t002", sup["by_task"])                                       # a Poetiq train pass stops supply
            self.assertNotIn("t001", sup["by_task"])
            self.assertIn("e33", r.e)


# ------------------------------------------------------------------ 9. the switches off = strict parity with the E29 post file
class TestOffParity(unittest.TestCase):
    def _run(self, name, scen, workers, strict, supply, env33=None, kit=None, **kw):
        tasks, order = scen()
        M = FIX[name]
        calls = []
        if name == "e33":
            M.E33_RUN_POETIQ = lambda *a: calls.append(a)
            M.E33_RUN_DIRECT = lambda *a: calls.append(a)
            M.E33_CAN_START = lambda *a: calls.append(a) or True
        k = kit(M) if kit else None
        try:
            with Env33(**(env33 or {"on": False})) if name == "e33" else ST.Env(supply=supply):
                if name == "e33":
                    os.environ.update({k_: v for k_, v in ST.Env(supply=supply).env.items()})
                    if not supply:
                        os.environ.pop("E29_SUPPLY", None)
                recs, el, sl = ST._solve(M, tasks, order, workers, **kw)
                sub = M.build_submission(tasks, recs)
        finally:
            if k:
                k.close()
            if name == "e33":
                M.E33_RUN_POETIQ = M.E33_RUN_DIRECT = M.E33_CAN_START = None
        self.assertEqual(calls, [])
        self.assertEqual(sl["e28"]["setup_failed"], [])
        e = copy.deepcopy(sl["e28"])
        (e.get("supply") or {}).pop("first_start_sec", None)
        self.assertNotIn("e33", e)
        return (ST._norm_recs(recs, strict), ST._norm_e28(e, strict), sub,
                {k_: v for k_, v in sl.items() if k_ not in ("durations", "gauges", "events", "e28")})

    def _cmp(self, scen, workers, strict, supply, env33=None, kit=None, **kw):
        a = self._run("e29", scen, workers, strict, supply, kit=kit, **kw)
        b = self._run("e33", scen, workers, strict, supply, env33=env33, kit=kit, **kw)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[2], b[2])
        if strict:
            self.assertEqual(a[3], b[3])
        return a, b

    def test_single_worker_strict(self):
        for supply in (False, True):
            for env33 in ({"on": False}, {"on": False, "poetiq": True, "direct": True, "attempt2": True},
                          {"on": True, "sched_value": "true"}, {"on": True, "sched_value": "0"}):
                a, b = self._cmp(ST._scen_single, 1, True, supply, env33=env33)
                self.assertEqual(b[1]["started"]["A"], 8)

    def test_multi_worker(self):
        for supply in (False, True):
            self._cmp(lambda: ST._scen_multi(24), 4, False, supply,
                      env33={"on": False, "poetiq": True, "direct": True, "attempt2": True})

    def test_kv_reservation(self):
        def scen():
            tasks = ST._tasks(6)
            ST.SCRIPTS.clear()
            ST.SCRIPTS[("t000", "A")] = {"kind": "cap", "sleep": 0.0}
            ST.SCRIPTS[("t000", "B")] = {"kind": "pass", "at": 20000, "value": 2}
            for t_ in list(tasks)[1:]:
                ST.SCRIPTS[(t_, "A")] = {"kind": "pass", "at": 10000, "sleep": 0.01}
                ST.SCRIPTS[(t_, "C")] = {"kind": "pass", "at": 5000, "sleep": 0.01, "value": 3}
            ST.SCRIPTS[("t001", "A")] = {"kind": "pass", "at": 100000, "sleep": 0.02}
            return tasks, list(tasks)
        self._cmp(scen, 3, False, False, kit=lambda M: ST.ResKit(M, cap=73))

    def test_no_admission_window(self):
        self._cmp(lambda: (ST._tasks(3), list(ST._tasks(3))), 2, True, True, deadline_sec=1.0, min_admit=5.0)

    def test_e28_off_is_original(self):
        """E28_SCHED unset (E33_SCHED=1 alone): the E33 file runs the original rev 6 scheduler (= the payload file)."""
        tasks = ST._tasks(6)
        ST.SCRIPTS.clear()
        for i, t in enumerate(tasks):
            ST.SCRIPTS[(t, "A")] = {"kind": "pass", "at": (i + 1) * 7000} if i % 3 else {"kind": "cap"}
            ST.SCRIPTS[(t, "C")] = {"kind": "pass", "at": 4000, "value": 3}
        out = []
        for name in ("orig", "e33"):
            with Env33(on=True, e28=False, supply=False):
                recs, el, sl = ST._solve(FIX[name], tasks, list(tasks), workers=1, deadline_sec=60.0, max_generated_tokens=50000)
                sub = FIX[name].build_submission(tasks, recs)
            out.append((ST._norm_recs(recs, True), {k: v for k, v in sl.items() if k not in ("durations", "gauges", "events")}, sub))
        self.assertEqual(out[0], out[1])
        self.assertNotIn("e28", out[1][1])

    def test_selection_on_real_records(self):
        if not FIX["eval"]:
            self.skipTest("evaluation challenges not found")
        n = 0
        for phase_dir in (ST.K100_PHASE, ST.W4_PHASE):
            raw = ST._raw_records(phase_dir)
            recs = {}
            for (t, b), r in raw.items():
                recs.setdefault(t, []).append(r)
            tasks = {t: FIX["eval"][t] for t in recs if t in FIX["eval"]}
            for env in ({"supply": False, "e28": False}, {"supply": False}, {"supply": True}):
                with Env33(on=True, **env):
                    self.assertEqual(FIX["e29"].build_submission(tasks, recs), FIX["e33"].build_submission(tasks, recs))
                    if env.get("e28", True):
                        self.assertEqual(FIX["e29"].e28_slot_table(tasks, recs), FIX["e33"].e28_slot_table(tasks, recs))
            n += len(tasks)
        if n == 0:
            self.skipTest("no K100 / W4 records found")


# ------------------------------------------------------------------ 10. the real Arc2ToolAgent on the E33 branches; LIVE
class TestAgentAndLive(unittest.TestCase):
    def test_agent_budget_window_on_e33_branches(self):
        if not FIX["eval"]:
            self.skipTest("evaluation challenges not found")
        M = FIX["e33"]
        task = FIX["eval"]["16b78196"]
        env = {"E12_ARC2_REV6": "1", "E12_B_MODE": "1", "E12_B_ATTEMPT_TOKENS": "104000", "E12_B_ATTEMPT_TOKENS_B1": "104000",
               "E12_DUCK_CONTEXT_WINDOW": "130560"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            mk = lambda b, view: M.make_agent_class()(task=task, task_id="16b78196", branch=b, view=view, model="fake", timeout=30,
                                                      base_url="http://127.0.0.1:9/v1", provider="vllm",
                                                      save_request_logs=False, dispatch_index=200001)
            with Env33(on=True):
                M.E28_PLANS.clear()
                M.E28_PLANS[("16b78196", 122)] = {"phase": "B2:2", "budget": 120000, "window": 130560, "branch": 122}
                M.E28_PLANS[("16b78196", 1)] = {"phase": "A:2", "budget": 104000, "window": None, "branch": 1}
                b2, a2 = mk(122, "grids"), mk(1, "grids")
                self.assertEqual((b2._arc2_b_budget, b2._arc2_window, b2._e28_plan["phase"]), (120000, 130560, "B2:2"))
                self.assertIn("about 120,000 generated tokens", b2._system_prompt)
                self.assertEqual((a2._arc2_b_budget, a2._arc2_window), (104000, 130560))
                M.E28_PLANS.clear()
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def test_live_lines_on_e33_records(self):
        sys.path.insert(0, str(E29B))
        try:
            import e29_live as LV
        finally:
            sys.path.remove(str(E29B))
        sc = _fail_all(["t000"])
        r = run33(sc, {("t000", "D"): {"pass": True}}, n=1)
        M = FIX["e33"]
        sol = {"t000": [g(42)]}
        with Env33(on=True):
            lv = LV.LiveLog(r.tasks, sol, select=lambda tid, task, recs: M.select_attempts_e28(task, recs), cand=M._e28_cand)
            lines = [lv.on_rec(x) for x in sorted(r.recs["t000"], key=lambda x: x["publish_seq"])]
        self.assertTrue(any("C-POETIQ1:1" in ln for ln in lines), lines)
        self.assertTrue(any("D-POETIQ2:2" in ln for ln in lines), lines)
        self.assertIn("queries 1/1 correct", lines[-1])


if __name__ == "__main__":
    prog = unittest.main(verbosity=2 if "-v" in sys.argv else 1, argv=[a for a in sys.argv if a != "-v"], exit=False)
    res = prog.result
    print(f"E33 SCHED tests{' (NEGATIVE CONTROL: the unpatched E29 post file)' if NEGCTL else ''}: {res.testsRun} run, "
          f"{len(res.failures)} failures, {len(res.errors)} errors, {len(res.skipped)} skipped")
    sys.exit(0 if res.wasSuccessful() else 1)
