"""E29 rev 2 DIRECT (the second branch = attempt_2, user 2026-11-02) local unit / e2e tests:  python e29d_direct_tests.py [-v]

Harness = e29_sched_tests.py's (the REAL arc2_duck.py from the Duck payload, the real BranchSession, the fake Duck agent) + the
DIRECT diff (e29d_sched_patch.py) + the REAL DIRECT solver (e29_direct.py = E31's e31_direct.py + N1 + task_index) against an
in-process fake streaming token-id /v1/completions server (the ids decide the text; a stub tokenizer whose stop / </think> ids
are the Flash-Next spec ids, so the solver's fail-closed check passes).
Covers: the patch (shas, apply / verify / revert, strict, only the E28 block changes); the switch off (E29_DIRECT unset, or no
hook) == the E29 post file (records, slots, SCHED_LAST["e28"]); DIRECT admitted in phase C's order with no Duck attempt 2; the
attempt_2 rule (valid + differs / equal to attempt_1 / no grid / no result; 1, 2, 3 test inputs; attempt_1 never changes);
the solver: phase-1 resend (kept ids), phase-2 resend with the kept phase-2 ids (N1), the drain inside the solve end; an e2e
run with Duck and DIRECT branches together (records, submission, report, transcripts).
rev 2d F1 / F1+ (the late phase-C no-op): the hook E29D_CAN_START(deadline) routes each phase-C item: True (31 min left with
the notebook formula now + 60 s + drain_min * 60 < deadline, drain 26) -> DIRECT; False (20 min / 1 min / the exact
boundary) / None / raising -> the ORIGINAL Duck C session; the switch OFF -> Duck C with record / answer parity with the E29
post file (hooks never called); a mixed early / late run; the real solver with a late window (the pre-fix no-op reproduced,
the fix -> Duck C answers). Every path: E28_RELEASE((task, 1)) exactly once, e28["done"]["C"] once per item, one start per
item (no double start).
Inputs: the Duck payload (E28_TEST_PAYLOAD), the E28 Build folder (E29_TEST_E28_DIR); e29d_sched_patch.py / e29_direct.py next
to this file (the kit copies them from the mounted E29 payload dataset).
"""
import copy
import hashlib
import http.server
import json
import os
import shutil
import socketserver
import sys
import threading
import time
import unittest
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e29_sched_tests as ST     # noqa: E402  the E29 scheduler harness (its fixtures are set up below)
import e29d_sched_patch as DS29  # noqa: E402
import e29_direct as D           # noqa: E402

E29S, E28S = ST.E29S, ST.E28S
FIX = ST.FIX
CLOSE, STOP = D.FN_THINK_CLOSE_ID, list(D.FN_STOP_IDS)
EOS = [248046, 248044]


# ------------------------------------------------------------------ the stub tokenizer (Flash-Next spec ids for the stop text)
def enc(s):
    if s == D.STOP_THINKING_TEXT:
        return list(STOP)
    if s == "</think>":
        return [CLOSE]
    return [1000 + ord(c) for c in s]


def dec(ids, skip):
    out = []
    for i in ids:
        i = int(i)
        if i == CLOSE:
            out.append("" if skip else "</think>")
        elif i == 271:
            out.append("\n\n")
        elif i >= 1000 and i < 200000:
            out.append(chr(i - 1000))
    return "".join(out)


TOK = {"render": lambda m: "<u>" + m[0]["content"] + "</u>", "encode": enc, "decode": dec, "eos_ids": EOS}


def gtext(g):
    return "\n".join(" ".join(str(v) for v in r) for r in g)


def answer_ids(grids, labeled):
    if not labeled:
        return enc("```grid\n" + gtext(grids[0]) + "\n```")
    return enc("\n".join("```grid %d\n%s\n```" % (k + 1, gtext(g)) for k, g in enumerate(grids)))


# ------------------------------------------------------------------ the fake streaming token-id server
FAKE = {"beh": {}, "idx": {}, "log": [], "calls": {}, "lock": threading.Lock()}


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def _chunk(self, ids, fr=None):
        d = {"choices": [{"token_ids": list(ids), "text": "", "finish_reason": fr}]}
        self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
        self.wfile.flush()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n))
        off = int(body["seed"]) - D.SEED
        idx, phase = off // 10, off % 10
        tid = FAKE["idx"][idx]
        beh = FAKE["beh"][tid]
        with FAKE["lock"]:
            k = (tid, phase)
            FAKE["calls"][k] = FAKE["calls"].get(k, 0) + 1
            nth = FAKE["calls"][k]
            FAKE["log"].append({"tid": tid, "phase": phase, "nth": nth, "prompt": list(body["prompt"]),
                                "max_tokens": body["max_tokens"], "seed": body["seed"], "t": time.monotonic()})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        kind, grids, lab = beh["kind"], beh["grids"], len(beh["grids"]) > 1
        ans = answer_ids(grids, lab)
        try:
            if phase == 1:
                if kind == "natural":
                    self._chunk([1000 + ord("x")] * 20)
                    self._chunk([CLOSE] + ans)
                    self._chunk([], "stop")
                elif kind == "forced":
                    self._chunk([1000 + ord("x")] * int(body["max_tokens"]))
                    self._chunk([], "length")
                elif kind == "p1err":
                    if nth == 1:
                        self._chunk([1000 + ord("y")] * 24)
                        return                              # dropped: the client sees no finish -> error
                    self._chunk([CLOSE] + ans)
                    self._chunk([], "stop")
                elif kind == "p2err":
                    self._chunk([1000 + ord("x")] * int(body["max_tokens"]))
                    self._chunk([], "length")
                elif kind == "slow":                        # thinks until the client aborts (the drain)
                    for _ in range(100000):
                        self._chunk([1000 + ord("z")])
                        time.sleep(0.01)
                return self._chunk([], "length") if kind == "slow" else None
            if kind == "p2err" and nth == 1:
                self._chunk(ans[:5])
                return                                      # dropped after 5 answer ids
            if kind == "p2err":
                self._chunk(ans[5:])
            else:
                self._chunk(ans)
            self._chunk(EOS[:1], "stop")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def setUpModule():
    ST.setUpModule()
    tmp = FIX["tmp"]
    with zipfile.ZipFile(ST.PAYLOAD) as z:
        z.extractall(tmp / "e29d")
    assert E29S.apply_e29_sched(tmp / "e29d")["ok"]
    r = DS29.apply_direct(tmp / "e29d")
    assert r["ok"] and r["applied"] == "applied", r
    FIX["e29d"] = ST._load(tmp / "e29d" / "arc2_duck.py", "arc2_duck_e29d")
    srv = TS(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FIX["srv"], FIX["url"] = srv, f"http://127.0.0.1:{srv.server_address[1]}"
    D.ADMIT_STOP_BEFORE_END_SEC, D.HARD_BEFORE_END_SEC = 2.0, 1.0      # second-scale solve windows (module constants)
    os.environ["E29_DIRECT_PHASE"] = "C"   # rev 2f: the rev 2e tests run scope C (every C item); TestC1Scope runs the C1 default


def tearDownModule():
    try:
        FIX["srv"].shutdown()
    except Exception:
        pass
    ST.tearDownModule()


def _tasks(spec):
    """spec: {tid: n_tests} -> ARC tasks (distinct test inputs per test)."""
    return {t: {"train": [{"input": [[0]], "output": [[0]]}] * 3, "test": [{"input": [[9, k]]} for k in range(n)]} for t, n in spec.items()}


def _solver(tasks, beh, deadline_sec=20.0, drain=0.05, think=400, answer=256, out=None):
    FAKE["beh"].update(beh)
    idx = {t: i for i, t in enumerate(sorted(tasks))}
    FAKE["idx"].update({i: t for t, i in idx.items()})
    out = out or (FIX["tmp"] / f"direct_{time.monotonic_ns()}")
    recs = {}
    for t in tasks:
        recs.update(D.solve_all_direct({t: tasks[t]}, [t], out_dir=out, deadline=time.monotonic() + deadline_sec, base_url=FIX["url"],
                                       model_id="fake", workers=1, cfg={"think": think, "answer": answer, "temperature": 0.6,
                                                                          "drain_min": drain, "workers": 1},
                                       tok=TOK, report_dir=out / "reports" / t, harness=True, task_index=idx, log=lambda *a, **k: None))
    return recs, out


def _hook(cfg=None, calls=None):
    """= the notebook's e29_direct_run on the fake server (the real solver, one task per call)."""
    def run(tid, task, plan, deadline, base_url, model_id, out_dir):
        if calls is not None:
            calls.append((tid, plan.get("c_priority"), plan.get("c_key_tokens"), time.monotonic()))
        c = dict({"think": 400, "answer": 256, "temperature": 0.6, "drain_min": 0.05, "workers": 1}, **(cfg or {}))
        out = D.solve_all_direct({tid: task}, [tid], out_dir=out_dir, deadline=deadline, base_url=FIX["url"], model_id=model_id,
                                 workers=1, cfg=c, tok=TOK, report_dir=Path(out_dir) / "reports" / tid, harness=True,
                                 task_index=FAKE["tindex"], log=lambda *a, **k: None)
        rs = out.get(tid) or []
        return dict(rs[-1], branch=1, e29_direct=True) if rs else None
    return run


def _stub_hook(calls, grid=7):
    def run(tid, task, plan, deadline, base_url, model_id, out_dir):
        calls.append((tid, plan.get("c_priority"), plan.get("c_key_tokens")))
        n = len(task["test"])
        return {"task_id": tid, "branch": 1, "status": "done", "accepted": False, "generated_tokens": 10, "e29_direct": True,
                "submissions": [{"test_outputs": [[[grid]]] * n, "accepted": False, "train_exact": [], "t": time.time()}],
                "e31": {"mode": "natural", "end_reason": "natural"}}
    return run


# ------------------------------------------------------------------ 1. the patch
class TestPatch(unittest.TestCase):
    def test_shas_states_strict(self):
        self.assertEqual(DS29.PRE, E29S.E29_POST)
        e29 = (FIX["tmp"] / "e29" / "arc2_duck.py").read_bytes().decode("utf-8")
        post = (FIX["tmp"] / "e29d" / "arc2_duck.py").read_bytes()
        self.assertEqual(hashlib.sha256(post).hexdigest(), DS29.POST)
        self.assertEqual((DS29.tree_state(e29), DS29.tree_state(post.decode())), ("E29_POST", "DIRECT_PATCHED"))
        self.assertEqual(DS29.revert_direct(post.decode())["text"], e29)
        self.assertEqual(DS29.apply_direct(post.decode())["applied"], "already")
        d = FIX["tmp"] / "t_dir_d"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        (d / "arc2_duck.py").write_bytes(e29.encode())
        r = DS29.apply_direct(d)
        self.assertEqual((r["applied"], r["after"], r["py_compile"]["rc"], r["ok"]), ("applied", "DIRECT_PATCHED", 0, True))
        self.assertTrue(DS29.verify_direct(d)[0])
        self.assertEqual(DS29.revert_direct(d)["after"], "E29_POST")
        self.assertEqual((d / "arc2_duck.py").read_bytes().decode(), e29)
        for bad in (ST._orig_text(), E28S.patched_text(ST._orig_text()), e29.replace("def _sec() -> float:", "def _sec( ) -> float:", 1)):
            with self.assertRaises(RuntimeError):
                DS29.apply_direct(bad)
        heads = [ln for ln in DS29.PATCH.splitlines() if ln.startswith("@@")]
        self.assertTrue(all(int(h.split()[1].split(",")[0][1:]) > 3700 for h in heads), heads)   # only the E28 / E29 block
        self.assertEqual(post.decode().count(DS29.MARKER), 1)
        self.assertNotIn("\r", DS29.PATCH)


class TestCell3(unittest.TestCase):
    def test_cell3_step(self):
        e29 = (FIX["tmp"] / "e29" / "arc2_duck.py").read_bytes()
        d = FIX["tmp"] / "t_cell3"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        (d / "arc2_duck.py").write_bytes(e29)
        os.environ.pop("E29_DIRECT", None)
        r = DS29.cell3(d, {"applied": True, "path": "E28"}, {"want": True, "ok": False})
        self.assertFalse(r["ok"])
        self.assertIn("not the E29 scheduler", r["error"])
        self.assertIsNone(os.environ.get("E29_DIRECT"))
        self.assertEqual((d / "arc2_duck.py").read_bytes(), e29)
        r = DS29.cell3(d, {"applied": True, "path": "E29"}, {"want": True, "ok": False})
        self.assertTrue(r["ok"] and r["after"] == "DIRECT_PATCHED")
        self.assertEqual(os.environ.get("E29_DIRECT"), "1")
        (d / "arc2_duck.py").write_bytes(e29.replace(b"def _sec() -> float:", b"def _sec( ) -> float:", 1))
        r = DS29.cell3(d, {"applied": True, "path": "E29"}, {"want": True, "ok": False})
        self.assertFalse(r["ok"])
        self.assertIsNone(os.environ.get("E29_DIRECT"))
        self.assertIn("revert_error", r)                                   # an unknown file is never written
        os.environ.pop("E29_DIRECT", None)


# ------------------------------------------------------------------ 2. the switch off == the E29 post file
class TestSwitchOff(unittest.TestCase):
    def _run(self, mod, env, scen, workers):
        tasks, order = scen()
        with ST.Env(supply=True, **env):
            recs, _, sl = ST._solve(mod, tasks, order, workers, deadline_sec=60.0)
        assert not sl["e28"]["setup_failed"], sl["e28"]["setup_failed"][:2]      # the Duck harness itself ran (e.g. requests importable)
        (sl["e28"].get("supply") or {}).pop("first_start_sec", None)                # a timing field (nested)
        return ST._norm_recs(recs, workers == 1), ST._norm_e28(sl["e28"], workers == 1)

    def test_off_identity(self):
        for workers, scen in ((1, ST._scen_single), (4, lambda: ST._scen_multi(24))):
            base = self._run(FIX["e29"], {}, scen, workers)
            FIX["e29d"].E29D_RUN = None
            self.assertEqual(self._run(FIX["e29d"], {}, scen, workers), base)               # env unset
            self.assertEqual(self._run(FIX["e29d"], {"E29_DIRECT": "1"}, scen, workers), base)   # env set, no hook
            calls, asked = [], []
            FIX["e29d"].E29D_RUN = _stub_hook(calls)
            FIX["e29d"].E29D_CAN_START = lambda d_: asked.append(d_) or True                  # rev 2d: both hooks set
            try:
                self.assertEqual(self._run(FIX["e29d"], {}, scen, workers), base)           # hooks, env unset
            finally:
                FIX["e29d"].E29D_RUN = FIX["e29d"].E29D_CAN_START = None
            self.assertEqual((calls, asked), ([], []))


# ------------------------------------------------------------------ 3. DIRECT in phase C's order, no Duck attempt 2
def _scen_c():
    tasks = ST._tasks(6)
    ST.SCRIPTS.clear()
    ST.SCRIPTS.update({("t000", "A"): {"kind": "pass", "at": 3000}, ("t001", "A"): {"kind": "pass", "at": 1000},
                       ("t002", "A"): {"kind": "partial", "at": 2000}, ("t002", "B"): {"kind": "partial", "at": 2000},
                       ("t003", "A"): {"kind": "pass", "at": 2000}, ("t004", "A"): {"kind": "fail"}, ("t004", "B"): {"kind": "fail"},
                       ("t005", "A"): {"kind": "pass", "at": 5000}})
    return tasks, list(tasks)


class TestCOrder(unittest.TestCase):
    def test_direct_admitted_in_c_order(self):
        tasks, order = _scen_c()
        with ST.Env(supply=True, E29_SUPPLY_MAX_ATTEMPTS="3"):
            kit = ST.RunLog(FIX["e29"])
            try:
                _, _, sl0 = ST._solve(FIX["e29"], tasks, order, 1, deadline_sec=60.0)
            finally:
                kit.close()
        self.assertFalse(sl0["e28"]["setup_failed"], "the Duck harness did not run (e.g. requests not importable)")
        c0 = [r["task"] for r in sorted(sl0["e28"]["attempts"], key=lambda r: r["start_seq"]) if r["phase"] == "C"]
        calls = []
        tasks, order = _scen_c()
        FIX["e29d"].E29D_RUN = _stub_hook(calls)
        FIX["e29d"].E29D_CAN_START = lambda d_: True                                       # rev 2d: the notebook sets both
        try:
            with ST.Env(supply=True, E29_DIRECT="1", E29_SUPPLY_MAX_ATTEMPTS="3"):
                kit = ST.RunLog(FIX["e29d"])
                try:
                    recs, _, sl = ST._solve(FIX["e29d"], tasks, order, 1, deadline_sec=60.0)
                finally:
                    kit.close()
                    runs = list(ST.RUNLOG)
        finally:
            FIX["e29d"].E29D_RUN = FIX["e29d"].E29D_CAN_START = None
        self.assertEqual([c[0] for c in calls], c0)                                       # = the E28 phase-C order
        self.assertEqual(c0, ["t002", "t004", "t001", "t003", "t000", "t005"])          # C1 (task order), then C2 by tokens
        self.assertEqual([c[1] for c in calls], [1, 1, 2, 2, 2, 2])
        self.assertFalse([r for r in runs if r[3] == "C"])                                # no Duck attempt 2 ran
        rows = [r for r in sl["e28"]["attempts"] if r["phase"] == "C"]
        self.assertTrue(rows and all(r["e29_direct"] and r["branch"] == 1 and r["kind"] == "DIRECT" for r in rows))
        d = sl["e28"]["direct"]
        self.assertEqual((d["started"], d["records"], d["no_record"], d["errors"]), (6, 6, 0, 0))
        self.assertTrue(all(any(r.get("e29_direct") and r["branch"] == 1 for r in recs[t]) for t in tasks))
        self.assertEqual(sl["e28"]["started"]["C"], 6)


# ------------------------------------------------------------------ 4. the attempt_2 rule
def _rec(branch, grids, accepted=False, exact=(), phase=None, direct=False, seq=1):
    r = {"task_id": "t", "branch": branch, "status": "done", "accepted": accepted, "publish_seq": seq,
         "submissions": [{"test_outputs": grids, "accepted": accepted, "train_exact": list(exact), "t": 1.0}], "checked": []}
    if phase:
        r["e28"] = {"phase": phase}
    if direct:
        r["e29_direct"] = True
    return r


class TestAttempt2(unittest.TestCase):
    def _sel(self, task, recs):
        with ST.Env(supply=True, E29_DIRECT="1"):
            M = FIX["e29d"]
            base = M.select_attempts_e28(task, [r for r in recs if not r.get("e29_direct")])
            rows = []
            out = M.select_attempts_e28(task, recs, rows)
        return base, out, rows

    def test_rule(self):
        for n in (1, 2, 3):
            task = {"train": [{"input": [[0]], "output": [[0]]}] * 2, "test": [{"input": [[9, k]]} for k in range(n)]}
            a = _rec(0, [[[1]]] * n, True, (True, True), "A")
            b = _rec(2, [[[2]]] * n, False, (True, False), "B1")
            for dg, want in (([[[5]]] * n, "D"), ([[[1]]] * n, "same"), ([None] * n, "none")):
                recs = [a, b, _rec(1, dg, phase="C", direct=True, seq=9)]
                base, out, rows = self._sel(task, recs)
                for qi in range(n):
                    self.assertEqual(out[qi]["attempt_1"], base[qi]["attempt_1"])              # attempt_1 never changes
                    self.assertEqual(out[qi]["attempt_2"], [[5]] if want == "D" else base[qi]["attempt_2"])
                    self.assertEqual(rows[qi]["e29_direct"], {"D": "attempt_2", "same": "same_as_attempt_1", "none": "no_grid"}[want])
            mixed = [[[5]], None, [[1]]][:n]                                                   # per test: grid K -> test K
            base, out, rows = self._sel(task, [a, b, _rec(1, mixed, phase="C", direct=True)])
            for qi in range(n):
                self.assertEqual(out[qi]["attempt_2"], [[5]] if qi == 0 else base[qi]["attempt_2"])
            no_d_base, no_d, _ = self._sel(task, [a, b])                                       # no DIRECT result = today's rule
            self.assertEqual(no_d, no_d_base)

    def test_no_duck_record(self):
        task = {"train": [{"input": [[0]], "output": [[0]]}] * 2, "test": [{"input": [[9]]}]}
        base, out, rows = self._sel(task, [_rec(1, [[[5]]], phase="C", direct=True)])
        self.assertEqual(out[0]["attempt_1"], base[0]["attempt_1"])
        self.assertEqual(out[0]["attempt_2"], [[5]] if base[0]["attempt_1"] != [[5]] else base[0]["attempt_2"])

    def test_report_slot_source(self):
        self.assertEqual(FIX["e29d"]._e29_src_branch({"A": 0, "C": 1, "B": 2}, "D"), 1)


# ------------------------------------------------------------------ 5. the solver: resends and the drain
class TestSolver(unittest.TestCase):
    def test_phase1_resend_keeps_ids(self):
        tasks = _tasks({"p1e": 1})
        FAKE["log"].clear()
        recs, _ = _solver(tasks, {"p1e": {"kind": "p1err", "grids": [[[3, 4]]]}})
        r = recs["p1e"][0]
        p1 = [x for x in FAKE["log"] if x["tid"] == "p1e" and x["phase"] == 1]
        self.assertEqual(len(p1), 2)
        self.assertEqual(p1[1]["prompt"], p1[0]["prompt"] + [1000 + ord("y")] * 24)          # prompt + the kept 24 ids
        self.assertEqual(p1[1]["max_tokens"], p1[0]["max_tokens"] - 24)
        self.assertEqual(p1[1]["seed"], p1[0]["seed"])
        self.assertEqual((r["e31"]["end_reason"], r["submissions"][0]["test_outputs"]), ("natural", [[[3, 4]]]))

    def test_phase2_resend_keeps_streamed_ids(self):
        tasks = _tasks({"p2e": 2})
        FAKE["log"].clear()
        g = [[[1, 2]], [[3, 4]]]
        recs, _ = _solver(tasks, {"p2e": {"kind": "p2err", "grids": g}}, answer=128)
        r = recs["p2e"][0]
        p2 = [x for x in FAKE["log"] if x["tid"] == "p2e" and x["phase"] == 2]
        self.assertEqual(len(p2), 2)
        ans = answer_ids(g, True)
        self.assertEqual(p2[0]["prompt"][-len(STOP):], STOP)                                  # forced close (token-exact)
        self.assertEqual(p2[1]["prompt"], p2[0]["prompt"] + ans[:5])                         # E29 N1: + the streamed ids
        self.assertEqual(p2[1]["max_tokens"], p2[0]["max_tokens"] - 5)                       # cap = the remaining tokens
        self.assertEqual(r["e31"]["p2_kept"], 5)
        self.assertEqual(r["submissions"][0]["test_outputs"], g)                              # grid K -> test K (labels)
        self.assertEqual(r["e31"]["mapping_source"], ["label", "label"])

    def test_drain_inside_solve_end(self):
        tasks = _tasks({"slw": 1})
        t0 = time.monotonic()
        recs, _ = _solver(tasks, {"slw": {"kind": "slow", "grids": [[[6]]]}}, deadline_sec=8.0, drain=4.0 / 60)
        r = recs["slw"][0]
        dt = time.monotonic() - t0
        self.assertLess(dt, 8.0)                                                              # returned inside the solve end
        self.assertGreater(dt, 3.0)                                                           # thought until the drain point
        self.assertEqual((r["e31"]["mode"], r["e31"]["p1_finish"]), ("forced_close", "drained"))
        self.assertEqual(r["submissions"][0]["test_outputs"], [[[6]]])

    def test_not_admitted_after_the_drain_point(self):
        tasks = _tasks({"late": 1})
        recs, _ = _solver(tasks, {"late": {"kind": "natural", "grids": [[[1]]]}}, deadline_sec=3.0, drain=4.0 / 60)
        self.assertEqual(recs["late"], [])                                                    # no record -> today's attempt_2


# ------------------------------------------------------------------ 6. e2e: Duck and DIRECT branches together
class TestE2E(unittest.TestCase):
    def test_duck_and_direct(self):
        tasks, order = _scen_c()
        tasks["t001"]["test"] = [{"input": [[9]]}, {"input": [[8]]}]                          # a two-test task
        FAKE["tindex"] = {t: i for i, t in enumerate(sorted(tasks))}
        FAKE["idx"].update({i: t for t, i in FAKE["tindex"].items()})
        g = {t: [[[int(t[-1]) + 4]]] * len(tasks[t]["test"]) for t in tasks}
        FAKE["beh"].update({t: {"kind": ("forced" if t in ("t002", "t005") else "natural"), "grids": g[t]} for t in tasks})
        calls = []
        FIX["e29d"].E29D_RUN = _hook(calls=calls)
        FIX["e29d"].E29D_CAN_START = lambda d_: time.monotonic() + 1.0 + 0.05 * 60.0 < d_   # the notebook formula (rev 2d)
        try:
            with ST.Env(supply=True, E29_DIRECT="1", E29_SUPPLY_MAX_ATTEMPTS="3"):
                recs, _, sl = ST._solve(FIX["e29d"], tasks, order, 3, deadline_sec=60.0)
                subs = FIX["e29d"].build_submission(tasks, recs)
        finally:
            FIX["e29d"].E29D_RUN = FIX["e29d"].E29D_CAN_START = None
        self.assertEqual(sorted(c[0] for c in calls), sorted(tasks))
        for t in tasks:
            d = [r for r in recs[t] if r.get("e29_direct")]
            duck = [r for r in recs[t] if not r.get("e29_direct")]
            self.assertTrue(d and duck, t)
            self.assertEqual((d[0]["branch"], d[0]["submissions"][0]["branch"], d[0]["accepted"], d[0]["submissions"][0]["train_exact"]),
                             (1, 1, False, []))
            self.assertEqual(d[0]["submissions"][0]["test_outputs"], g[t])
            with ST.Env(supply=True):
                base = FIX["e29d"].select_attempts_e28(tasks[t], duck)
            for qi, row in enumerate(subs[t]):
                self.assertEqual(row["attempt_1"], base[qi]["attempt_1"])
                self.assertEqual(row["attempt_2"], g[t][qi] if g[t][qi] != base[qi]["attempt_1"] else base[qi]["attempt_2"])
        d = sl["e28"]["direct"]
        self.assertEqual((d["started"], d["records"], d["errors"]), (6, 6, 0))
        self.assertEqual(d["by_task"]["t002"]["mode"], "forced_close")
        self.assertEqual(sl["e28"]["started"]["C"], 6)
        tr = list(FIX["tmp"].glob("sim_*/direct/*_b0/transcript.txt"))
        self.assertTrue(len(tr) >= 6, len(tr))                                               # full thinking saved per request

# ------------------------------------------------------------------ 7. F1 / F1+: late phase C -> the original Duck C (rev 2d)
DRAIN_MIN, MARGIN_SEC = 26.0, 60.0          # the production CELL 1 drain_min and the notebook's start margin


def _nb_can_start(left_min, drain_min=DRAIN_MIN, margin=MARGIN_SEC, seen=None):
    """= the notebook's E29D_CAN_START (time.monotonic() + margin + drain_min * 60 < deadline) on a fake clock that puts
    'now' left_min minutes before the scheduler's deadline."""
    def can(deadline):
        if seen is not None:
            seen.append(deadline)
        now = deadline - float(left_min) * 60.0
        return now + margin + float(drain_min) * 60.0 < deadline
    return can


def _run_c(can_start, run_hook="stub", env_on=True, workers=1, deadline_sec=60.0):
    """_scen_c on the DIRECT post file with these hooks; E28_RELEASE counted; Duck runs logged. -> dict of the results."""
    M = FIX["e29d"]
    tasks, order = _scen_c()
    calls, rel = [], []
    M.E29D_RUN = _stub_hook(calls) if run_hook == "stub" else run_hook(calls)          # run_hook: (calls) -> the hook
    M.E29D_CAN_START = can_start
    M.E28_RELEASE = lambda key: rel.append(tuple(key))
    env = dict(E29_SUPPLY_MAX_ATTEMPTS="3", **({"E29_DIRECT": "1"} if env_on else {}))
    try:
        with ST.Env(supply=True, **env):
            kit = ST.RunLog(M)
            try:
                recs, _, sl = ST._solve(M, tasks, order, workers, deadline_sec=deadline_sec)
            finally:
                kit.close()
                runs = list(ST.RUNLOG)
            subs = M.build_submission(tasks, recs)
    finally:
        M.E29D_RUN = M.E29D_CAN_START = M.E28_RELEASE = None
    return {"tasks": tasks, "recs": recs, "sl": sl, "runs": runs, "calls": calls, "rel": rel, "subs": subs}


def _run_base(workers=1):
    """The same scenario on the E29 post file (no DIRECT code at all) = today's answers."""
    M = FIX["e29"]
    tasks, order = _scen_c()
    with ST.Env(supply=True, E29_SUPPLY_MAX_ATTEMPTS="3"):
        recs, _, sl = ST._solve(M, tasks, order, workers, deadline_sec=60.0)
        subs = M.build_submission(tasks, recs)
    return {"tasks": tasks, "recs": recs, "sl": sl, "subs": subs}


def _e28n(e):
    """SCHED_LAST["e28"] minus timing fields (as TestSwitchOff: + the nested supply first_start_sec), strict (1 worker)."""
    e = copy.deepcopy(e)
    (e.get("supply") or {}).pop("first_start_sec", None)
    return ST._norm_e28(e, True)


class TestF1LateC(unittest.TestCase):
    def _per_item(self, r, direct_tasks, duck_tasks):
        """The invariants of every path: one start per C item, one C completion, one release of (task, 1)."""
        e = r["sl"]["e28"]
        self.assertFalse(e["setup_failed"], e["setup_failed"][:2])
        tasks = sorted(r["tasks"])
        self.assertEqual(sorted(direct_tasks + duck_tasks), tasks)
        duck_c = sorted(x[1] for x in r["runs"] if x[0] == "start" and x[3] == "C")
        self.assertEqual(duck_c, sorted(duck_tasks))                                        # Duck C ran exactly there
        self.assertEqual(sorted(c[0] for c in r["calls"]), sorted(direct_tasks))            # DIRECT ran exactly there
        for t_ in tasks:                                                                    # no double start
            self.assertEqual(duck_c.count(t_) + [c[0] for c in r["calls"]].count(t_), 1, t_)
            self.assertEqual(r["rel"].count((t_, 1)), 1, (t_, r["rel"]))                    # E28_RELEASE exactly once
            b1 = [x for x in r["recs"][t_] if int(x.get("branch", -1)) == 1]
            self.assertEqual(len(b1), 1, t_)
            self.assertEqual(bool(b1[0].get("e29_direct")), t_ in direct_tasks, t_)
            c_rows = [x for x in e["attempts"] if x["task"] == t_ and x["phase"] == "C"]
            self.assertEqual(len(c_rows), 1, t_)
            self.assertEqual(c_rows[0]["kind"] == "DIRECT", t_ in direct_tasks, (t_, c_rows[0]["kind"]))
        self.assertEqual((e["started"]["C"], e["done"]["C"]), (len(tasks), len(tasks)))   # done.C once per item
        d = e.get("direct") or {}
        if direct_tasks or d:
            self.assertEqual((d.get("started"), d.get("records"), d.get("no_record")), (len(direct_tasks), len(direct_tasks), 0))
            self.assertEqual(d.get("duck_c"), len(duck_tasks))
            self.assertEqual(sorted(d.get("duck_c_tasks") or {}), sorted(duck_tasks))
        return e

    def _duck_parity(self, r, base, tasks_):
        """Duck-routed tasks: records and answers == the E29 post file (today)."""
        nr, nb = ST._norm_recs(r["recs"], True), ST._norm_recs(base["recs"], True)
        for t_ in tasks_:
            self.assertEqual(nr[t_], nb[t_], t_)
            self.assertEqual(r["subs"][t_], base["subs"][t_], t_)

    def test_notebook_formula_and_boundary(self):
        """The hook formula (drain 26, margin 60 s) through the patched module's _e29d_can_start."""
        M = FIX["e29d"]
        dl = time.monotonic() + 3600.0
        try:
            for left_min, want in ((31, True), (27.5, True), (20, False), (1, False), (0, False), (-3, False),
                                   ((MARGIN_SEC + DRAIN_MIN * 60) / 60.0, False),          # the exact boundary: strict <
                                   ((MARGIN_SEC + DRAIN_MIN * 60 + 1) / 60.0, True)):
                M.E29D_CAN_START = _nb_can_start(left_min)
                self.assertEqual(M._e29d_can_start(dl), (want, "ok" if want else "late"), left_min)
            M.E29D_CAN_START = None
            self.assertEqual(M._e29d_can_start(dl), (False, "no_hook"))
            M.E29D_CAN_START = "not callable"
            self.assertEqual(M._e29d_can_start(dl), (False, "no_hook"))

            def boom(_):
                raise RuntimeError("hook failed")
            M.E29D_CAN_START = boom
            self.assertEqual(M._e29d_can_start(dl), (False, "hook_error RuntimeError"))
            M.E29D_CAN_START = lambda d_: time.monotonic() + MARGIN_SEC + DRAIN_MIN * 60 < d_   # the real-clock lambda
            self.assertEqual(M._e29d_can_start(time.monotonic() + 31 * 60)[0], True)
            self.assertEqual(M._e29d_can_start(time.monotonic() + 20 * 60)[0], False)
            self.assertEqual(M._e29d_can_start(time.monotonic() + 60)[0], False)
        finally:
            M.E29D_CAN_START = None

    def test_hook_true_31min_direct(self):
        seen = []
        r = _run_c(_nb_can_start(31, seen=seen))
        tasks = sorted(r["tasks"])
        self._per_item(r, tasks, [])
        self.assertEqual(len(seen), len(tasks))                                             # asked once per C item
        self.assertEqual(len(set(seen)), 1)                                                 # with the scheduler's deadline
        self.assertTrue(0 < seen[0] - time.monotonic() < 60.0)
        for t_ in tasks:                                                                    # attempt_2 = the DIRECT grid
            self.assertTrue(all(q["attempt_2"] == [[7]] for q in r["subs"][t_]) or
                            all(q["attempt_1"] == [[7]] for q in r["subs"][t_]), t_)

    def test_hook_false_20min_and_1min_duck_c(self):
        base = _run_base()
        for left_min in (20, 1):
            r = _run_c(_nb_can_start(left_min))
            tasks = sorted(r["tasks"])
            e = self._per_item(r, [], tasks)
            self.assertEqual(e["direct"]["duck_c_why"], {"late": len(tasks)})
            self._duck_parity(r, base, tasks)                                              # a valid attempt_2 = today's
            e_ = dict(e)
            e_.pop("direct")
            self.assertEqual(_e28n(e_), _e28n(base["sl"]["e28"]))

    def test_hook_none_or_raising_duck_c(self):
        base = _run_base()

        def boom(_):
            raise ValueError("bad hook")
        for hook, why in ((None, "no_hook"), (boom, "hook_error ValueError")):
            r = _run_c(hook)
            tasks = sorted(r["tasks"])
            e = self._per_item(r, [], tasks)
            self.assertEqual(e["direct"]["duck_c_why"], {why: len(tasks)})
            self._duck_parity(r, base, tasks)

    def test_switch_off_parity(self):
        base = _run_base()
        asked = []
        r = _run_c(lambda d_: asked.append(d_) or True, env_on=False)                     # hooks set, E29_DIRECT unset
        tasks = sorted(r["tasks"])
        e = self._per_item(r, [], tasks)
        self.assertEqual((asked, r["calls"]), ([], []))                                    # the hooks are never called
        self.assertNotIn("direct", e)
        self._duck_parity(r, base, tasks)
        self.assertEqual(_e28n(e), _e28n(base["sl"]["e28"]))
        self.assertEqual(r["subs"], base["subs"])

    def test_mixed_early_direct_late_duck(self):
        """Time passes during phase C: the first 3 C items (in C's order) still fit -> DIRECT, the rest -> Duck C."""
        base = _run_base()
        n = {"k": 0}

        def can(d_):
            n["k"] += 1
            return n["k"] <= 3
        r = _run_c(can)
        c0 = ["t002", "t004", "t001", "t003", "t000", "t005"]                              # = the E28 phase-C order
        self._per_item(r, c0[:3], c0[3:])
        self.assertEqual([c[0] for c in r["calls"]], c0[:3])
        self._duck_parity(r, base, c0[3:])
        for t_ in c0[:3]:                                                                   # attempt_1 never changes
            duck = [x for x in r["recs"][t_] if not x.get("e29_direct")]
            with ST.Env(supply=True):
                qd = FIX["e29d"].select_attempts_e28(r["tasks"][t_], duck)
            for q, qb in zip(r["subs"][t_], qd):
                self.assertEqual(q["attempt_1"], qb["attempt_1"])
                self.assertEqual(q["attempt_2"], [[7]] if qb["attempt_1"] != [[7]] else qb["attempt_2"])

    def test_multi_worker_mixed(self):
        """4 workers: the per-item invariants hold with concurrent C items (DIRECT / Duck split by the hook)."""
        lock, n = threading.Lock(), {"k": 0}

        def can(d_):
            with lock:
                n["k"] += 1
                return n["k"] % 2 == 1
        r = _run_c(can, workers=4)
        direct = sorted(c[0] for c in r["calls"])
        duck = sorted(set(r["tasks"]) - set(direct))
        self.assertEqual((len(direct), len(duck)), (3, 3))
        self._per_item(r, direct, duck)

    def test_late_c_real_solver(self):
        """The real solver with a solve window shorter than its drain: the pre-fix route (always DIRECT) is a no-op for
        every C item (no record, no Duck C); the fixed route (the notebook formula) runs Duck C and gets its attempt_2."""
        FAKE["tindex"] = {t_: i for i, t_ in enumerate(sorted(_scen_c()[0]))}
        FAKE["idx"].update({i: t_ for t_, i in FAKE["tindex"].items()})
        FAKE["beh"].update({t_: {"kind": "natural", "grids": [[[8]]]} for t_ in FAKE["tindex"]})
        drain = 1.5                                                                        # 90 s > the 60-s window
        old = _run_c(lambda d_: True, run_hook=lambda c_: _hook(cfg={"drain_min": drain}, calls=c_))   # = rev 2c (no guard)
        e = old["sl"]["e28"]
        self.assertEqual((e["direct"]["started"], e["direct"]["records"], e["direct"]["no_record"]), (6, 0, 6))
        self.assertEqual(len(old["calls"]), 6)                                             # the solver was asked, admitted 0
        self.assertFalse([x for x in old["runs"] if x[0] == "start" and x[3] == "C"])     # the Duck C attempts were lost
        base = _run_base()
        new = _run_c(lambda d_: time.monotonic() + 1.0 + drain * 60.0 < d_,
                     run_hook=lambda c_: _hook(cfg={"drain_min": drain}, calls=c_))
        tasks = sorted(new["tasks"])
        self._per_item(new, [], tasks)
        self._duck_parity(new, base, tasks)
        early = _run_c(lambda d_: time.monotonic() + 1.0 + 0.05 * 60.0 < d_,
                       run_hook=lambda c_: _hook(cfg={"drain_min": 0.05}, calls=c_))
        self._per_item(early, tasks, [])                                                   # early C -> DIRECT records
        self.assertTrue(all(any(x.get("e29_direct") and x["submissions"][0]["test_outputs"] == [[[8]]] for x in early["recs"][t_])
                            for t_ in tasks))


# ------------------------------------------------------------------ rev 2f: the DIRECT scope (E29_DIRECT2['phase'], env E29_DIRECT_PHASE)
# The tests above run with scope "C" (setUpModule: every C item may go to DIRECT = rev 2e). These run the production default
# "C1": DIRECT only on train-fail C items (c_priority 1: t002 partial, t004 fail in _scen_c); the C2 items (t001 / t003 / t000 /
# t005, full train match) run the original Duck C session through the F1 path (why "c2_duck").
class _Scope:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = os.environ.get("E29_DIRECT_PHASE")
        if self.value is None:
            os.environ.pop("E29_DIRECT_PHASE", None)
        else:
            os.environ["E29_DIRECT_PHASE"] = self.value
        return self

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("E29_DIRECT_PHASE", None)
        else:
            os.environ["E29_DIRECT_PHASE"] = self.old


C1_TASKS, C2_TASKS = ["t002", "t004"], ["t001", "t003", "t000", "t005"]     # = the E28 phase-C order of _scen_c


class TestC1Scope(unittest.TestCase):
    _per_item, _duck_parity = TestF1LateC._per_item, TestF1LateC._duck_parity   # the F1 invariants (not its tests)

    def _duck_parity_order_free(self, r, base, tasks_):
        """rev 2g: an early DIRECT item changes the queue order, so the queue snapshot / start-order fields of the C2 Duck
        records differ from today; everything else (attempt content, answers) must be identical."""
        nr, nb = ST._norm_recs(r["recs"], False), ST._norm_recs(base["recs"], False)
        for t_ in tasks_:
            self.assertEqual(nr[t_], nb[t_], t_)
            self.assertEqual(r["subs"][t_], base["subs"][t_], t_)

    def test_can_start_scope(self):
        M = FIX["e29d"]
        dl = time.monotonic() + 3600.0
        c1, c2 = {"phase": "C", "c_priority": 1}, {"phase": "C", "c_priority": 2}
        try:
            M.E29D_CAN_START = lambda d_: True
            for scope in (None, "", "C1", "c", "C2", " x "):                                # unset / unknown = the C1 default
                with _Scope(scope):
                    self.assertEqual(M._e29d_scope(), "C1", scope)
                    self.assertEqual(M._e29d_can_start(dl, c1), (True, "ok"), scope)
                    self.assertEqual(M._e29d_can_start(dl, c2), (False, "c2_duck"), scope)
                    self.assertEqual(M._e29d_can_start(dl, None), (False, "c2_duck"), scope)
                    self.assertEqual(M._e29d_can_start(dl), (False, "c2_duck"), scope)
            for scope in ("C", " C "):
                with _Scope(scope):
                    self.assertEqual(M._e29d_scope(), "C", scope)
                    self.assertEqual(M._e29d_can_start(dl, c1), (True, "ok"))
                    self.assertEqual(M._e29d_can_start(dl, c2), (True, "ok"))
            with _Scope(None):
                M.E29D_CAN_START = lambda d_: False
                self.assertEqual(M._e29d_can_start(dl, c1), (False, "late"))
                self.assertEqual(M._e29d_can_start(dl, c2), (False, "c2_duck"))
                M.E29D_CAN_START = None
                self.assertEqual(M._e29d_can_start(dl, c1), (False, "no_hook"))
                self.assertEqual(M._e29d_can_start(dl, c2), (False, "no_hook"))
        finally:
            M.E29D_CAN_START = None

    def test_c1_default_direct_on_train_fail_only(self):
        base = _run_base()
        with _Scope(None):
            r = _run_c(lambda d_: True)
        e = self._per_item(r, C1_TASKS, C2_TASKS)
        self.assertEqual([c[0] for c in r["calls"]], C1_TASKS)                              # DIRECT asked only for C1, in C's order
        self.assertEqual({c[1] for c in r["calls"]}, {1})
        self.assertEqual(e["direct"]["duck_c_why"], {"c2_duck": len(C2_TASKS)})
        self._duck_parity_order_free(r, base, C2_TASKS)                                     # C2 = today's Duck C, same answers
        self.assertEqual((e["direct"]["early_queued"], e["direct"]["early_b_end_no_c"]), (len(C1_TASKS), len(C1_TASKS)))   # rev 2g
        for t_ in C1_TASKS:                                                                 # attempt_1 never changes
            duck = [x for x in r["recs"][t_] if not x.get("e29_direct")]
            with ST.Env(supply=True):
                qd = FIX["e29d"].select_attempts_e28(r["tasks"][t_], duck)
            for q, qb in zip(r["subs"][t_], qd):
                self.assertEqual(q["attempt_1"], qb["attempt_1"])
                self.assertEqual(q["attempt_2"], [[7]] if qb["attempt_1"] != [[7]] else qb["attempt_2"])

    def test_c1_late_all_duck(self):
        base = _run_base()
        with _Scope("C1"):
            r = _run_c(_nb_can_start(20))
        e = self._per_item(r, [], C1_TASKS + C2_TASKS)
        self.assertEqual(e["direct"]["duck_c_why"], {"late": len(C1_TASKS), "c2_duck": len(C2_TASKS)})
        self._duck_parity(r, base, C1_TASKS + C2_TASKS)
        self.assertEqual(r["subs"], base["subs"])

    def test_scope_c_is_rev2e(self):
        with _Scope("C"):
            r = _run_c(lambda d_: True)
        self._per_item(r, C1_TASKS + C2_TASKS, [])
        self.assertEqual([c[0] for c in r["calls"]], C1_TASKS + C2_TASKS)

    def test_c1_multi_worker(self):
        with _Scope(None):
            r = _run_c(lambda d_: True, workers=4)
        self._per_item(r, C1_TASKS, C2_TASKS)
        self.assertEqual(sorted(c[0] for c in r["calls"]), sorted(C1_TASKS))

    def test_c1_switch_off_parity(self):
        base = _run_base()
        with _Scope("C1"):
            r = _run_c(lambda d_: True, env_on=False)
        self.assertEqual(r["calls"], [])
        self.assertNotIn("direct", r["sl"]["e28"])
        self.assertEqual(r["subs"], base["subs"])

    def test_c1_real_solver(self):
        """The real solver on the fake server: C1 tasks get the DIRECT grid as attempt_2, C2 tasks keep today's answers."""
        base = _run_base()
        FAKE["tindex"] = {t_: i for i, t_ in enumerate(sorted(_scen_c()[0]))}
        FAKE["idx"].update({i: t_ for t_, i in FAKE["tindex"].items()})
        FAKE["beh"].update({t_: {"kind": "natural", "grids": [[[8]]]} for t_ in FAKE["tindex"]})
        with _Scope(None):
            r = _run_c(lambda d_: time.monotonic() + 1.0 + 0.05 * 60.0 < d_,
                       run_hook=lambda c_: _hook(cfg={"drain_min": 0.05}, calls=c_))
        self._per_item(r, C1_TASKS, C2_TASKS)
        self._duck_parity_order_free(r, base, C2_TASKS)
        for t_ in C1_TASKS:
            self.assertTrue(any(x.get("e29_direct") and x["submissions"][0]["test_outputs"] == [[[8]]] for x in r["recs"][t_]), t_)
            self.assertTrue(all(q["attempt_2"] == [[8]] or q["attempt_1"] == [[8]] for q in r["subs"][t_]), t_)


# ------------------------------------------------------------------ rev 2g: DIRECT early on train-fail tasks (scope C1)
def _timed_stub(calls, grid=7):
    """_stub_hook + the call time (the order vs the Duck B / C starts)."""
    def run(tid, task, plan, deadline, base_url, model_id, out_dir):
        calls.append((tid, plan.get("c_priority"), plan.get("c_from"), time.monotonic()))
        n = len(task["test"])
        return {"task_id": tid, "branch": 1, "status": "done", "accepted": False, "generated_tokens": 10, "e29_direct": True,
                "submissions": [{"test_outputs": [[[grid]]] * n, "accepted": False, "train_exact": [], "t": time.time()}],
                "e31": {"mode": "natural", "end_reason": "natural"}}
    return run


class TestEarlyDirect(unittest.TestCase):
    _per_item = TestF1LateC._per_item

    def test_early_before_b_one_worker(self):
        """1 worker: every A first, then the early DIRECT items (t002, t004: train-fail, c_from A_DIRECT_EARLY), then their B
        re-runs, then the C2 Duck items; the B end of an early task queues no second C item."""
        with _Scope(None):
            r = _run_c(lambda d_: True, run_hook=lambda c_: _timed_stub(c_))
        e = self._per_item(r, C1_TASKS, C2_TASKS)
        self.assertEqual([c[0] for c in r["calls"]], C1_TASKS)
        self.assertEqual({c[2] for c in r["calls"]}, {"A_DIRECT_EARLY"})
        starts = [(x[1], x[3], x[4]) for x in r["runs"] if x[0] == "start"]
        last_a = max(t_ for _, ph, t_ in starts if ph == "A")
        first_b = min(t_ for _, ph, t_ in starts if ph in ("B1", "B2"))
        self.assertTrue(all(last_a < c[3] < first_b for c in r["calls"]), (last_a, [c[3] for c in r["calls"]], first_b))
        b_tasks = sorted(x for x, ph, _ in starts if ph in ("B1", "B2"))
        self.assertEqual(b_tasks, sorted(C1_TASKS))                                        # B runs once per train-fail task
        self.assertEqual((e["direct"]["early_queued"], e["direct"]["early_b_end_no_c"]), (2, 2))
        self.assertEqual(e["queued"]["C1"], 2)

    def test_early_attempts(self):
        """attempt_1 of a C1 task = the Duck rule over A + B (unchanged); attempt_2 = the DIRECT grid when it differs."""
        with _Scope(None):
            r = _run_c(lambda d_: True, run_hook=lambda c_: _timed_stub(c_))
        for t_ in C1_TASKS:
            duck = [x for x in r["recs"][t_] if not x.get("e29_direct")]
            self.assertTrue(any(int(x.get("branch", -1)) == 2 for x in duck), t_)          # the B re-run is there
            with ST.Env(supply=True):
                qd = FIX["e29d"].select_attempts_e28(r["tasks"][t_], duck)
            for q, qb in zip(r["subs"][t_], qd):
                self.assertEqual(q["attempt_1"], qb["attempt_1"])
                self.assertEqual(q["attempt_2"], [[7]] if qb["attempt_1"] != [[7]] else qb["attempt_2"])

    def test_no_early_when_not_admissible(self):
        """The hook False at the A end: no early item; the E28 path exactly (B, then the C1 item -> Duck C as late)."""
        base = _run_base()
        with _Scope(None):
            r = _run_c(lambda d_: False)
        e = self._per_item(r, [], C1_TASKS + C2_TASKS)
        self.assertEqual(e["direct"]["early_queued"], 0)
        self.assertEqual(e["direct"]["duck_c_why"], {"late": len(C1_TASKS), "c2_duck": len(C2_TASKS)})
        self._duck_parity = TestF1LateC._duck_parity.__get__(self)
        self._duck_parity(r, base, C1_TASKS + C2_TASKS)
        self.assertEqual(r["subs"], base["subs"])

    def test_scope_c_no_early(self):
        with _Scope("C"):
            r = _run_c(lambda d_: True, run_hook=lambda c_: _timed_stub(c_))
        self._per_item(r, C1_TASKS + C2_TASKS, [])
        self.assertEqual({c[2] for c in r["calls"]} & {"A_DIRECT_EARLY"}, set())
        self.assertEqual(r["sl"]["e28"]["direct"]["early_queued"], 0)

    def test_early_multi_worker(self):
        with _Scope(None):
            r = _run_c(lambda d_: True, workers=4, run_hook=lambda c_: _timed_stub(c_))
        e = self._per_item(r, C1_TASKS, C2_TASKS)
        self.assertEqual(sorted(c[0] for c in r["calls"]), sorted(C1_TASKS))
        self.assertEqual(e["direct"]["early_queued"], 2)

    def test_switch_off_no_early(self):
        base = _run_base()
        with _Scope(None):
            r = _run_c(lambda d_: True, env_on=False)
        self.assertEqual(r["calls"], [])
        self.assertNotIn("direct", r["sl"]["e28"])
        self.assertEqual(r["subs"], base["subs"])


class TestDirect2Cfg(unittest.TestCase):
    def test_phase_key(self):
        sys.path.insert(0, str(HERE))
        import e29_lib as L
        base = {"on": True, "think": 104000, "answer": 8192, "temperature": 1.0, "drain_min": 26}
        self.assertEqual(L.direct2_cfg(base)["phase"], "C1")                               # absent = the C1 default
        self.assertEqual(L.direct2_cfg(dict(base, phase="C1"))["phase"], "C1")
        self.assertEqual(L.direct2_cfg(dict(base, phase="C"))["phase"], "C")
        for bad in ("C2", "c1", 1, None):
            c = L.direct2_cfg(dict(base, phase=bad))
            self.assertFalse(c["on"], bad)
            self.assertIn("phase", c["invalid"], bad)
        self.assertEqual(L.direct2_cfg(dict(base, phase="C1"), '{"phase": "C"}')["phase"], "C")   # kit override
        self.assertTrue(L.direct2_cfg(dict(base, phase="C1"))["on"])


class _Result(unittest.TextTestResult):
    pass


if __name__ == "__main__":
    r = unittest.main(exit=False, verbosity=2 if "-v" in sys.argv else 1, argv=[sys.argv[0]]).result
    print(f"E29 DIRECT tests: {r.testsRun} run, {len(r.failures)} failures, {len(r.errors)} errors, {len(r.skipped)} skipped", flush=True)
    sys.exit(0 if r.wasSuccessful() else 1)
