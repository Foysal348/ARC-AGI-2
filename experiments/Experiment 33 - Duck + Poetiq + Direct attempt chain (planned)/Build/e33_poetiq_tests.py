"""E33 Poetiq solver (Build/e33_poetiq.py) local unit tests:  python e33_poetiq_tests.py [-v]   (stdlib unittest + numpy)

Harness = the pattern of Experiment 29 Build/e29d_direct_tests.py: an in-process fake streaming token-id /v1/completions server
(the request seed decides the behaviour: seed = SEED + task_index * 1000 + expert * 100 + iteration0 * 2 + phase, E18's
e10_seed), a stub tokenizer whose stop / </think> ids are the Flash-Next spec ids (the fail-closed check passes) and whose
text ids are 1000 + ord(char) (prompts decode back to text).
Sandbox: on Linux as root with libseccomp (the Kaggle CPU kit) the REAL E18 runner (UID drop + rlimits + seccomp); elsewhere
(Windows) a plain subprocess runner with E18's input / output / timeout contract (no UID drop) for BOTH the module and the
exec'd E18 functions, so feedback texts are comparable byte for byte.
E18 reference: the E18 R2 notebook's Poetiq core cell (index 5) exec'd into its own namespace (skipped when the notebook is
not on disk).
"""
import http.server
import json
import os
import shutil
import socketserver
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
import e33_poetiq as P       # noqa: E402

try:
    import e33_poetiq_build as PB   # noqa: E402
except Exception:                   # the kit may ship the module without the builder
    PB = None

CLOSE, STOP = P.FN_THINK_CLOSE_ID, list(P.FN_STOP_IDS)
EOS = [248046, 248044]
X = 1000 + ord("x")


# ------------------------------------------------------------------ the stub tokenizer
def enc(s):
    if s == P.STOP_THINKING_TEXT:
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
        elif 1000 <= i < 200000:
            out.append(chr(i - 1000))
    return "".join(out)


def render(m):
    return "<u>" + m[0]["content"] + "</u>"


TOK = {"render": render, "encode": enc, "decode": dec, "eos_ids": EOS}


# ------------------------------------------------------------------ sandbox (real E18 runner when possible)
def _real_sandbox_available():
    if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        return False
    try:
        import ctypes
        ctypes.CDLL("libseccomp.so.2")
        return True
    except Exception:
        return False


REAL_SANDBOX = _real_sandbox_available()
_RUNNER = r'''
import sys, json
import numpy as np
raw = json.load(sys.stdin)
ns = {"__name__": "poetiq_candidate", "np": np}
try:
    exec(compile(raw["code"], "<poetiq_candidate>", "exec"), ns)
    r = ns["transform"](np.array(raw["input"], dtype=int))
    if hasattr(r, "tolist"):
        r = r.tolist()
    p = {"ok": True, "result": r}
except BaseException as ex:
    p = {"ok": False, "error": (type(ex).__name__ + ": " + str(ex))[:1200]}
print("__E10_PROGRAM_RESULT__" + json.dumps(p, separators=(",", ":")), flush=True)
'''


def fake_sandbox_run(code, grid, timeout_s=5.0):
    """E18 poetiq_sandbox_run's contract (returns (ok, json result | error text), 'sandbox timeout') without the UID drop
    (no -I: a local numpy may live in the user site)."""
    timeout_s = max(.1, float(timeout_s))
    proc = subprocess.Popen([sys.executable, "-B", "-c", _RUNNER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    try:
        sout, serr = proc.communicate(json.dumps({"code": code, "input": grid}).encode(), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=5)
        return False, "sandbox timeout"
    sout, serr = sout.decode("utf-8", "replace"), serr.decode("utf-8", "replace")
    if proc.returncode:
        return False, ("sandbox process failure " + str(proc.returncode) + ": " + (serr or sout))[-1500:]
    marker = "__E10_PROGRAM_RESULT__"
    pos = sout.rfind(marker)
    if pos < 0:
        return False, "sandbox missing result: " + sout[-800:]
    payload = json.loads(sout[pos + len(marker):].strip())
    if not payload.get("ok"):
        return False, str(payload.get("error", "program error"))[:1200]
    return True, json.dumps(payload.get("result"))


# ------------------------------------------------------------------ the E18 reference (notebook cell index 5, exec'd)
E18_NB = PB.E18_NB if PB is not None else Path("/nonexistent")
E18 = None


def _load_e18():
    global E18
    if E18 is not None or not E18_NB.is_file():
        return E18
    nb = json.loads(E18_NB.read_bytes())
    core = "".join(nb["cells"][5]["source"])
    ns = {"__name__": "e18_core"}
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(core, "e18_cell5", "exec"), ns)     # its import-time self-test fails closed off Kaggle (caught)
    if not REAL_SANDBOX:
        ns["poetiq_sandbox_run"] = fake_sandbox_run       # the same runner as the module under test
    E18 = ns
    return ns


# ------------------------------------------------------------------ the fake streaming token-id server
FAKE = {"beh": {}, "log": [], "calls": {}, "lock": threading.Lock(), "health_ok": True}


def decode_seed(seed):
    off = int(seed) - P.SEED
    tix, rem = off // 1000, off % 1000
    expert, v = rem // 100, rem % 100
    phase = 1 if v % 2 else 2
    return tix, expert, (v - phase) // 2, phase


def ans_ids(text):
    return enc(text)


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        ok = FAKE["health_ok"]
        self.send_response(200 if ok else 503)
        self.end_headers()
        self.wfile.write(b"ok" if ok else b"down")

    def _chunk(self, ids, fr=None):
        d = {"choices": [{"token_ids": list(ids), "text": "", "finish_reason": fr}]}
        self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
        self.wfile.flush()

    def _xs(self, n, tok=X):
        for k in range(0, n, 100):
            self._chunk([tok] * min(100, n - k))

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n))
        tix, expert, it, phase = decode_seed(body["seed"])
        key = (tix, expert, it)
        beh = FAKE["beh"].get(key) or {"kind": "natural", "answer": "no code here"}
        with FAKE["lock"]:
            ck = key + (phase,)
            FAKE["calls"][ck] = FAKE["calls"].get(ck, 0) + 1
            nth = FAKE["calls"][ck]
            FAKE["log"].append({"key": key, "phase": phase, "nth": nth, "prompt": list(body["prompt"]), "path": self.path,
                                "max_tokens": body["max_tokens"], "seed": body["seed"], "temperature": body["temperature"],
                                "top_p": body["top_p"], "top_k": body["top_k"], "model": body["model"]})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        kind, ans = beh["kind"], ans_ids(beh["answer"])
        try:
            if phase == 1:
                if kind == "natural":
                    self._xs(20)
                    self._chunk([CLOSE] + ans)
                    return self._chunk([], "stop")
                if kind in ("forced", "p2err"):
                    self._xs(int(body["max_tokens"]))
                    return self._chunk([], "length")
                if kind == "forced_eos":                 # an EOS before </think>: stripped before the stop ids
                    self._xs(30)
                    return self._chunk(EOS[:1], "stop")
                if kind == "continue":                   # </think> closed, the answer cut by the cap
                    self._xs(10)
                    self._chunk([CLOSE] + ans[:7])
                    return self._chunk([], "length")
                if kind in ("p1err", "p1err_down"):
                    if nth == 1:
                        self._xs(24, 1000 + ord("y"))
                        if kind == "p1err_down":
                            FAKE["health_ok"] = False
                            threading.Timer(0.8, lambda: FAKE.__setitem__("health_ok", True)).start()
                        return                           # dropped: the client sees no finish -> "error"
                    self._chunk([CLOSE] + ans)
                    return self._chunk([], "stop")
                if kind == "slow":                       # thinks until the client aborts (drain / hard stop)
                    for _ in range(100000):
                        self._chunk([1000 + ord("z")])
                        time.sleep(0.01)
                    return self._chunk([], "length")
                return self._chunk([], "stop")
            if kind == "p2err" and nth == 1:
                self._chunk(ans[:5])
                return                                   # dropped after 5 answer ids
            if kind == "p2err":
                self._chunk(ans[5:])
            elif kind == "continue":
                self._chunk(ans[7:])
            else:
                self._chunk(ans)
            self._chunk(EOS[:1], "stop")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass


class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


FIX = {}
CFG = {"think": 400, "answer": 256, "drain_answer": 256, "drain_min": 0.05, "tok_s": 1000.0, "admit_margin_sec": 0.1,
       "sandbox_timeout": 2.0}


def setUpModule():
    FIX["tmp"] = Path(tempfile.mkdtemp(prefix="e33_poetiq_tests_"))
    srv = TS(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FIX["srv"], FIX["url"] = srv, f"http://127.0.0.1:{srv.server_address[1]}"
    FIX["saved"] = (P.ADMIT_STOP_BEFORE_END_SEC, P.HARD_BEFORE_END_SEC, P.HEALTH_EVERY_SEC, P._E18_POETIQ_SANDBOX_RUN,
                    P._SANDBOX["selftest"])
    P.ADMIT_STOP_BEFORE_END_SEC, P.HARD_BEFORE_END_SEC, P.HEALTH_EVERY_SEC = 2.0, 1.0, 0.2   # second-scale windows
    if REAL_SANDBOX:
        assert P.sandbox_ready(force=True)["ok"], P._SANDBOX["selftest"]
    else:
        P._E18_POETIQ_SANDBOX_RUN = fake_sandbox_run
        P._SANDBOX["selftest"] = {"ok": True, "mode": "test_fake_runner"}
    _load_e18()


def tearDownModule():
    try:
        FIX["srv"].shutdown()
    except Exception:
        pass
    (P.ADMIT_STOP_BEFORE_END_SEC, P.HARD_BEFORE_END_SEC, P.HEALTH_EVERY_SEC, P._E18_POETIQ_SANDBOX_RUN,
     P._SANDBOX["selftest"]) = FIX["saved"]
    shutil.rmtree(FIX["tmp"], ignore_errors=True)


# ------------------------------------------------------------------ tasks and programs
def flip(g):
    return [list(reversed(r)) for r in reversed(g)]


def _task(tests=1):
    ins = [[[1, 2], [3, 4]], [[5, 0, 1], [2, 2, 3]], [[7, 8, 9], [0, 1, 2], [3, 4, 5]]]
    tins = [[[4, 3], [2, 1]], [[1, 1, 2], [3, 5, 8]], [[0, 0], [0, 6], [6, 6]]][:tests]
    return {"train": [{"input": g, "output": flip(g)} for g in ins], "test": [{"input": g} for g in tins]}


GOOD = "import numpy as np\n\ndef transform(grid: np.ndarray) -> np.ndarray:\n    return np.flip(grid)"
WRONG = "import numpy as np\n\ndef transform(grid: np.ndarray) -> np.ndarray:\n    return grid[::-1]"
BOOM3 = ("import numpy as np\n\ndef transform(grid: np.ndarray) -> np.ndarray:\n    if grid.shape == (2, 3) and grid[0, 0] == 1:\n"
         "        raise ValueError('boom on test 2')\n    return np.flip(grid)")
LOOP = "def transform(grid):\n    while True:\n        pass"
RAISE = "def transform(grid):\n    raise ValueError('bad rule')"


def answer(code, extra=""):
    return f"The rule flips the grid.{extra}\n```python\n{code}\n```\n"


def _run(tix, beh, iteration=1, prior=None, task=None, cfg=None, deadline_sec=30.0, attempt=None, tok=TOK, harness=True,
         **kw):
    expert = (attempt or 1) - 1
    FAKE["beh"][(tix, expert, iteration - 1)] = beh
    tid = f"t{tix:03d}"
    out = FIX["tmp"] / f"run_{tix}_{time.monotonic_ns()}"
    rec = P.poetiq_run(tid, task or _task(), iteration=iteration, prior=prior, deadline=time.monotonic() + deadline_sec,
                       base_url=FIX["url"] + "/v1", model_id="fake", out_dir=out, tok=tok, cfg=dict(CFG, **(cfg or {})),
                       task_index={tid: tix}, harness=harness, log=lambda *a, **k: None, attempt=attempt, **kw)
    return rec, out / f"{tid}_a{attempt or 1}_i{iteration}", out


def _reqs(tix, expert=0, it=0, phase=None):
    return [r for r in FAKE["log"] if r["key"] == (tix, expert, it) and (phase is None or r["phase"] == phase)]


def _text(ids):
    return dec(ids, False)


# ------------------------------------------------------------------ 1. iteration 1
class TestIteration1(unittest.TestCase):
    def test_correct_program_train_pass(self):
        task = _task(1)
        rec, bdir, _ = _run(1, {"kind": "natural", "answer": answer(GOOD)}, task=task)
        self.assertEqual((rec["status"], rec["accepted"]), ("done", True))
        p = rec["poetiq"]
        self.assertEqual((p["train_pass"], p["score"], p["parse_status"], p["mode"], p["end_reason"]),
                         (True, 1.0, "ok", "natural", "natural"))
        self.assertEqual(p["program"], GOOD)
        s = rec["submissions"][0]
        self.assertEqual(s["test_outputs"], [flip(task["test"][0]["input"])])
        self.assertEqual((s["accepted"], s["train_exact"]), (True, [True, True, True]))
        self.assertEqual(p["e18_two_attempts"][0]["attempt_1"], flip(task["test"][0]["input"]))
        self.assertEqual(rec["poetiq_state"]["status"], "train_verified")
        self.assertEqual(len(rec["poetiq_state"]["history"]), 1)
        self.assertEqual(p["think_tokens"], 20)
        r1 = _reqs(1, phase=1)
        self.assertEqual(len(r1), 1)
        self.assertEqual((r1[0]["path"], r1[0]["temperature"], r1[0]["top_p"], r1[0]["top_k"], r1[0]["max_tokens"]),
                         ("/v1/completions", 1.0, 0.95, 20, 400))
        self.assertEqual(_reqs(1, phase=2), [])
        self.assertEqual(rec["generated_tokens"], p["p1_tokens"])

    def test_wrong_program_feedback(self):
        task = _task(1)
        rec, _, _ = _run(2, {"kind": "natural", "answer": answer(WRONG)}, task=task)
        p = rec["poetiq"]
        self.assertFalse(rec["accepted"])
        self.assertLess(p["score"], 1.0)
        self.assertIn("Solves Example #1 incorrectly. Wrong cells appear as prediction/correct:", p["feedback"])
        self.assertEqual(rec["submissions"][0]["train_exact"], [False, False, False])
        self.assertEqual(rec["submissions"][0]["test_outputs"], [task["test"][0]["input"][::-1]])
        h = rec["poetiq_state"]["history"]
        self.assertEqual([x["code"] for x in h], [WRONG])
        self.assertEqual((h[0]["feedback"], h[0]["score"]), (p["feedback"], p["score"]))
        if E18 is not None:
            ref = E18["poetiq_verify_code"](WRONG, task, 2.0)
            self.assertEqual((ref["feedback"], ref["score"]), (p["feedback"], p["score"]))


# ------------------------------------------------------------------ 2. iteration 2 = E18's feedback prompt, byte for byte
class TestIteration2(unittest.TestCase):
    def test_feedback_prompt_byte_identical(self):
        task = _task(1)
        tix = 3
        rec1, _, _ = _run(tix, {"kind": "natural", "answer": answer(WRONG)}, task=task)
        self.assertFalse(rec1["accepted"])
        rec2, _, _ = _run(tix, {"kind": "natural", "answer": answer(GOOD)}, iteration=2, prior=rec1, task=task)
        self.assertTrue(rec2["accepted"])
        self.assertEqual(rec2["poetiq"]["history_in"], 1)
        p1 = _text(_reqs(tix, it=0, phase=1)[0]["prompt"])
        p2 = _text(_reqs(tix, it=1, phase=1)[0]["prompt"])
        self.assertIn("**EXISTING PARTIAL/INCORRECT SOLUTIONS:**", p2)
        self.assertIn(rec1["poetiq"]["feedback"], p2)
        self.assertIn("<solution_score>\n%.2f\n</solution_score>" % rec1["poetiq"]["score"], p2)
        self.assertNotIn("EXISTING PARTIAL", p1)
        st2 = rec2["poetiq_state"]
        self.assertEqual((len(st2["history"]), len(st2["candidate_history"]), st2["iteration"], st2["best_index"]), (2, 2, 2, 1))
        if E18 is None:
            self.skipTest("the E18 notebook is not on disk")
        ref_v = E18["poetiq_verify_code"](WRONG, task, 2.0)
        hist = [{"code": WRONG, "feedback": ref_v["feedback"], "score": ref_v["score"]}]
        seed = E18["POETIQ_SEED"] if "POETIQ_SEED" in E18 else 0
        kw = dict(variant=1, selection_probability=1.0, max_solutions=5, improving_order=True, shuffle=True, seed=seed + tix * 1000)
        self.assertEqual(p1, render([{"role": "user", "content": E18["poetiq_build_message"](task, 0, 0, [], **kw)}]))
        self.assertEqual(p2, render([{"role": "user", "content": E18["poetiq_build_message"](task, 1, 0, hist, **kw)}]))
        # the same prompt when prior = the embedded poetiq_state
        tix2 = 4
        FAKE["beh"][(tix2, 0, 1)] = {"kind": "natural", "answer": answer(GOOD)}
        st = dict(rec1["poetiq_state"], task_index=tix2)
        _run(tix2, {"kind": "natural", "answer": answer(GOOD)}, iteration=2, prior=st, task=task)
        p2b = _text(_reqs(tix2, it=1, phase=1)[0]["prompt"])
        kw["seed"] = seed + tix2 * 1000
        self.assertEqual(p2b, render([{"role": "user", "content": E18["poetiq_build_message"](task, 1, 0, hist, **kw)}]))

    def test_iteration2_without_code_has_no_feedback(self):
        task = _task(1)
        rec1, _, _ = _run(5, {"kind": "natural", "answer": "I think the rule is a flip, no code."}, task=task)
        self.assertEqual(rec1["poetiq"]["parse_status"], "no_code_block")
        rec2, _, _ = _run(5, {"kind": "natural", "answer": answer(GOOD)}, iteration=2, prior=rec1, task=task)
        self.assertNotIn("EXISTING PARTIAL", _text(_reqs(5, it=1, phase=1)[0]["prompt"]))
        self.assertTrue(rec2["accepted"])


# ------------------------------------------------------------------ 3. parse failure, sandbox timeout / exception, multi-test
class TestFailures(unittest.TestCase):
    def test_parse_failure(self):
        rec, bdir, _ = _run(6, {"kind": "natural", "answer": "Rule: flip. ```grid\n1 2\n```"})
        p = rec["poetiq"]
        self.assertEqual((rec["status"], rec["accepted"], p["parse_status"], p["program"], p["score"], p["train_score"]),
                         ("done", False, "no_code_block", None, None, 0.0))
        self.assertEqual(rec["submissions"][0]["test_outputs"], [None])
        self.assertEqual(rec["submissions"][0]["train_exact"], [False, False, False])
        self.assertEqual(rec["poetiq_state"]["history"], [])
        self.assertEqual(rec["poetiq_state"]["last_error"], "missing code block")
        self.assertFalse((bdir / "program.py").exists())

    def test_sandbox_timeout(self):
        rec, _, _ = _run(7, {"kind": "natural", "answer": answer(LOOP)}, cfg={"sandbox_timeout": 1.0})
        p = rec["poetiq"]
        self.assertFalse(rec["accepted"])
        self.assertEqual(p["score"], 0.0)
        self.assertIn("Invalid or unavailable grid. Error: sandbox timeout", p["feedback"])
        self.assertIn("TEST EXECUTION / FORMAT ERRORS", p["feedback"])
        self.assertEqual(rec["submissions"][0]["test_outputs"], [None])

    def test_sandbox_exception(self):
        rec, _, _ = _run(8, {"kind": "natural", "answer": answer(RAISE)})
        p = rec["poetiq"]
        self.assertFalse(rec["accepted"])
        self.assertIn("ValueError: bad rule", p["feedback"])
        self.assertEqual(rec["submissions"][0]["test_outputs"], [None])
        self.assertEqual(len(rec["poetiq_state"]["history"]), 1)
        if E18 is not None:
            self.assertEqual(E18["poetiq_verify_code"](RAISE, _task(1), 2.0)["feedback"], p["feedback"])

    def test_multi_test(self):
        task = _task(3)
        rec, _, _ = _run(9, {"kind": "natural", "answer": answer(GOOD)}, task=task)
        self.assertTrue(rec["accepted"])
        self.assertEqual(rec["submissions"][0]["test_outputs"], [flip(q["input"]) for q in task["test"]])
        task2 = _task(3)
        rec2, _, _ = _run(10, {"kind": "natural", "answer": answer(BOOM3)}, task=task2)
        outs = rec2["submissions"][0]["test_outputs"]
        self.assertTrue(rec2["accepted"])                         # train pass; one test input fails -> None for that test only
        self.assertEqual((outs[0], outs[1], outs[2]), (flip(task2["test"][0]["input"]), None, flip(task2["test"][2]["input"])))
        self.assertFalse(rec2["poetiq"]["test_usable"])
        self.assertIn("boom on test 2", rec2["poetiq"]["test_errors"][1])
        self.assertEqual(rec2["poetiq_state"]["status"], "train_pass_test_unusable")


# ------------------------------------------------------------------ 4. phases: forced close, continue, EOS, drain, hard stop
class TestPhases(unittest.TestCase):
    def test_forced_close_then_answer(self):
        rec, bdir, _ = _run(11, {"kind": "forced", "answer": answer(GOOD)})
        p = rec["poetiq"]
        self.assertEqual((p["mode"], p["end_reason"], p["p1_finish"], p["p1_tokens"], p["think_closed"]),
                         ("forced_close", "forced_close", "length", 400, True))
        self.assertTrue(rec["accepted"])
        r1, r2 = _reqs(11, phase=1)[0], _reqs(11, phase=2)[0]
        self.assertEqual(r2["prompt"], r1["prompt"] + [X] * 400 + STOP)       # token-exact, never re-tokenized
        self.assertEqual((r2["max_tokens"], r2["seed"]), (256, P.SEED + 11 * 1000 + 2))
        self.assertEqual(p["think_tokens"], 400 + STOP.index(CLOSE))
        self.assertEqual(rec["usage"][1]["prompt"], len(r2["prompt"]))

    def test_eos_stripped_before_stop_ids(self):
        rec, _, _ = _run(12, {"kind": "forced_eos", "answer": answer(GOOD)})
        r1, r2 = _reqs(12, phase=1)[0], _reqs(12, phase=2)[0]
        self.assertEqual(r2["prompt"], r1["prompt"] + [X] * 30 + STOP)
        self.assertEqual((rec["poetiq"]["mode"], rec["poetiq"]["p1_finish"]), ("forced_close", "stop"))
        self.assertTrue(rec["accepted"])

    def test_continue_mode(self):
        rec, _, _ = _run(13, {"kind": "continue", "answer": answer(GOOD)})
        r1, r2 = _reqs(13, phase=1)[0], _reqs(13, phase=2)[0]
        self.assertEqual(r2["prompt"], r1["prompt"] + [X] * 10 + [CLOSE] + ans_ids(answer(GOOD))[:7])
        self.assertEqual(rec["poetiq"]["mode"], "continue")
        self.assertTrue(rec["accepted"])

    def test_drain_forced_close(self):
        rec, _, _ = _run(14, {"kind": "slow", "answer": answer(GOOD)}, deadline_sec=6.0, cfg={"drain_min": 0.05})
        p = rec["poetiq"]
        self.assertEqual((p["drained"], p["p1_finish"], p["mode"], p["end_reason"]), (True, "drained", "forced_close", "forced_close"))
        self.assertTrue(rec["accepted"])
        r1, r2 = _reqs(14, phase=1)[0], _reqs(14, phase=2)[0]
        kept = json.loads((FIX["tmp"].glob("run_14_*").__next__() / "t014_a1_i1" / "token_ids.json").read_text())["p1"]
        self.assertGreater(len(kept), 0)
        self.assertEqual(r2["prompt"], r1["prompt"] + kept + STOP)

    def test_hard_stop(self):
        rec, _, _ = _run(15, {"kind": "slow", "answer": answer(GOOD)}, deadline_sec=4.0, cfg={"drain_min": 0.0})
        p = rec["poetiq"]
        self.assertEqual((p["end_reason"], p["parse_status"], rec["accepted"]), ("hard_stop", "no_think_close", False))
        self.assertEqual(rec["submissions"][0]["test_outputs"], [None])
        self.assertGreater(p["p1_tokens"], 0)


# ------------------------------------------------------------------ 5. request errors, resends, server down
class TestErrors(unittest.TestCase):
    def test_phase1_error_resend_keeps_ids(self):
        rec, _, _ = _run(16, {"kind": "p1err", "answer": answer(GOOD)})
        a, b = _reqs(16, phase=1)
        self.assertEqual(b["prompt"], a["prompt"] + [1000 + ord("y")] * 24)
        self.assertEqual((b["max_tokens"], b["seed"]), (400 - 24, a["seed"]))
        p = rec["poetiq"]
        self.assertEqual((len(p["request_errors"]), p["error_retries"], p["mode"], rec["accepted"]), (1, 1, "natural", True))
        self.assertEqual(p["think_tokens"], 24)

    def test_phase2_error_resend_keeps_ids(self):
        rec, _, _ = _run(17, {"kind": "p2err", "answer": answer(GOOD)})
        a, b = _reqs(17, phase=2)
        ans = ans_ids(answer(GOOD))
        self.assertEqual(b["prompt"], a["prompt"] + ans[:5])
        self.assertEqual(b["max_tokens"], 256 - 5)
        p = rec["poetiq"]
        self.assertEqual((p["p2_kept"], p["mode"], rec["accepted"], p["program"]), (5, "forced_close", True, GOOD))

    def test_error_retries_exhausted(self):
        beh = {"kind": "p1err", "answer": answer(GOOD)}
        rec, _, _ = _run(18, beh, cfg={"max_error_retries": 0})
        self.assertEqual((rec["poetiq"]["end_reason"], rec["accepted"]), ("error", False))
        self.assertEqual(len(_reqs(18, phase=1)), 1)

    def test_server_down_mid_run_waits(self):
        rec, _, _ = _run(19, {"kind": "p1err_down", "answer": answer(GOOD)}, health=lambda: FAKE["health_ok"])
        p = rec["poetiq"]
        self.assertGreaterEqual(p["health_fail_polls"], 1)
        self.assertTrue(rec["accepted"])
        a, b = _reqs(19, phase=1)
        self.assertEqual(b["prompt"], a["prompt"] + [1000 + ord("y")] * 24)
        FAKE["health_ok"] = True

    def test_server_down_event_then_up(self):
        ev = threading.Event()
        ev.set()
        threading.Timer(0.7, ev.clear).start()
        rec, _, _ = _run(20, {"kind": "natural", "answer": answer(GOOD)}, server_down=ev)
        self.assertEqual(rec["status"], "done")
        self.assertGreater(rec["poetiq"]["server_down_polls"], 0)

    def test_server_down_until_late(self):
        ev = threading.Event()
        ev.set()
        rec, _, _ = _run(21, {"kind": "natural", "answer": answer(GOOD)}, server_down=ev, deadline_sec=5.0)
        self.assertEqual((rec["status"], rec["poetiq"]["reason"], rec["submissions"]), ("not_admitted", "server_down_until_late", []))
        self.assertEqual(_reqs(21), [])


# ------------------------------------------------------------------ 6. admission (the 60,000 floor), fail-closed, sandbox off
class TestAdmission(unittest.TestCase):
    def test_not_admitted_time(self):
        rec, _, _ = _run(22, {"kind": "natural", "answer": answer(GOOD)}, cfg={"tok_s": 10.0})   # 0.1 + 400 / 10 = 40 s > 27 s
        self.assertEqual((rec["status"], rec["poetiq"]["reason"], rec["accepted"], rec["generated_tokens"]),
                         ("not_admitted", "time", False, 0))
        self.assertEqual(_reqs(22), [])
        self.assertAlmostEqual(rec["poetiq"]["admission"]["need_sec"], 40.1, places=1)

    def test_not_admitted_context(self):
        task = _task(1)
        n = len(enc(render([{"role": "user", "content": P.poetiq_message(task, 0, 0, [], 23)}])))
        rec, _, _ = _run(23, {"kind": "natural", "answer": answer(GOOD)}, task=task, cfg={"max_model_len": n + 256 + 256 + 399})
        self.assertEqual((rec["status"], rec["poetiq"]["reason"], rec["poetiq"]["p1_cap"]), ("not_admitted", "insufficient_context", 399))
        self.assertEqual(_reqs(23), [])
        rec, _, _ = _run(24, {"kind": "natural", "answer": answer(GOOD)}, task=task, cfg={"max_model_len": n + 256 + 256 + 400})
        self.assertEqual((rec["status"], rec["poetiq"]["p1_cap"]), ("done", 400))

    def test_production_floor(self):
        rec, _, _ = _run(25, {"kind": "natural", "answer": answer(GOOD)}, harness=False)          # think 400 < 60,000
        self.assertEqual((rec["status"], rec["poetiq"]["reason"]), ("not_run", "cfg_think_below_floor"))
        self.assertEqual(_reqs(25), [])
        now = 1000.0
        cfg = {"tok_s": 18.0}
        need = 60.0 + 60000 / 18.0
        drain = 9.0 * 60
        ok, info = P.can_admit(now + need + drain + 0.5, cfg, now=now)
        self.assertTrue(ok)
        self.assertEqual((info["floor"], info["rate_tok_s"], info["need_sec"]), (60000, 18.0, round(need, 1)))
        self.assertFalse(P.can_admit(now + need + drain - 0.5, cfg, now=now)[0])
        self.assertFalse(P.can_admit(now + need + drain + 0.5, dict(cfg, tok_s_measured=17.0), now=now)[0])
        self.assertFalse(P.can_admit(now + 6 * 3600, dict(cfg, harness_floor=10), now=now + 6 * 3600 - 1)[0])
        self.assertEqual(P.can_admit(now + 4 * 3600, dict(cfg, harness_floor=10), now=now)[1]["floor"], 60000)   # not harness

    def test_measured_rate_lowers_admission(self):
        saved = list(P._RATES)
        try:
            P._RATES.clear()
            P._RATES.extend([30.0, 12.0, 25.0])
            self.assertEqual(P.measured_tok_s(), 25.0)
            self.assertEqual(P.admission_rate({"tok_s": 18.0}), 18.0)
            P._RATES.extend([10.0, 11.0])
            self.assertEqual(P.measured_tok_s(), 12.0)
            self.assertEqual(P.admission_rate({"tok_s": 18.0}), 12.0)
        finally:
            P._RATES.clear()
            P._RATES.extend(saved)

    def test_fail_closed_token_ids(self):
        bad = dict(TOK, encode=lambda s: [1, 2, 3] if s == P.STOP_THINKING_TEXT else enc(s))
        rec, _, _ = _run(26, {"kind": "natural", "answer": answer(GOOD)}, tok=bad)
        self.assertEqual((rec["status"], rec["poetiq"]["reason"]), ("not_run", "FAIL_CLOSED_TOKEN_IDS"))
        rec, _, _ = _run(26, {"kind": "natural", "answer": answer(GOOD)}, tok=dict(TOK, eos_ids=[]))
        self.assertEqual(rec["poetiq"]["reason"], "FAIL_CLOSED_TOKEN_IDS")
        self.assertEqual(_reqs(26), [])

    def test_sandbox_unavailable(self):
        saved = P._SANDBOX["selftest"]
        P._SANDBOX["selftest"] = {"ok": False, "mode": "unavailable", "error": "test"}
        try:
            rec, _, _ = _run(27, {"kind": "natural", "answer": answer(GOOD)})
        finally:
            P._SANDBOX["selftest"] = saved
        self.assertEqual((rec["status"], rec["poetiq"]["reason"]), ("not_run", "sandbox_unavailable"))
        self.assertEqual(_reqs(27), [])

    def test_sandbox_concurrency_bound(self):
        P.set_sandbox_workers(2)
        P._SANDBOX["max_live"] = 0
        code = "import time\ndef transform(grid):\n    time.sleep(0.4)\n    return grid"
        th = [threading.Thread(target=P.poetiq_sandbox_run, args=(code, [[1]], 5.0)) for _ in range(6)]
        [t.start() for t in th]
        [t.join() for t in th]
        self.assertEqual(P._SANDBOX["max_live"], 2)
        self.assertEqual(P._SANDBOX["live"], 0)


# ------------------------------------------------------------------ 7. seeds, attempt 2, traces
class TestSeedsTraces(unittest.TestCase):
    def test_seeds_deterministic_per_task_index(self):
        task = _task(1)
        _run(28, {"kind": "forced", "answer": answer(GOOD)}, task=task)
        _run(28, {"kind": "forced", "answer": answer(GOOD)}, task=task)
        a, b = _reqs(28, phase=1)
        self.assertEqual((a["seed"], a["prompt"]), (b["seed"], b["prompt"]))
        self.assertEqual(a["seed"], P.SEED + 28 * 1000 + 1)
        self.assertEqual([r["seed"] for r in _reqs(28, phase=2)], [P.SEED + 28 * 1000 + 2] * 2)
        _run(29, {"kind": "natural", "answer": answer(GOOD)}, task=task)
        c = _reqs(29, phase=1)[0]
        self.assertEqual(c["seed"], P.SEED + 29 * 1000 + 1)
        self.assertEqual(_text(c["prompt"]), render([{"role": "user", "content": P.poetiq_build_message(
            task, 0, 0, [], variant=1, selection_probability=1.0, max_solutions=5, improving_order=True, shuffle=True,
            seed=29 * 1000)}]))
        rec2, _, _ = _run(28, {"kind": "natural", "answer": answer(GOOD)}, task=task, attempt=2)   # attempt 2 = expert 1
        d = _reqs(28, expert=1, phase=1)[0]
        self.assertEqual(d["seed"], P.SEED + 28 * 1000 + 100 + 1)
        self.assertEqual((rec2["poetiq"]["expert"], rec2["attempt"]), (1, 2))
        self.assertEqual(_text(d["prompt"]), render([{"role": "user", "content": P.poetiq_build_message(
            task, 0, 1, [], variant=1, selection_probability=1.0, max_solutions=5, improving_order=True, shuffle=True,
            seed=28 * 1000 + 100)}]))
        self.assertEqual((P.poetiq_seed(28, 0, 1, 1), P.poetiq_seed(28, 0, 1, 2)), (P.SEED + 28003, P.SEED + 28004))

    def test_trace_files(self):
        rec, bdir, out = _run(30, {"kind": "forced", "answer": answer(GOOD)})
        tr = (bdir / "transcript.txt").read_text(encoding="utf-8")
        for s in ("### PROMPT", "### GENERATED", "### ANSWER (after the first </think>)", "</think>", GOOD, "mode=forced_close"):
            self.assertIn(s, tr)
        self.assertEqual(tr.count("x" * 400), 1)                   # the full thinking is kept
        ti = json.loads((bdir / "token_ids.json").read_text())
        self.assertEqual((len(ti["p1"]), ti["p2_prefix"]), (400, STOP))
        self.assertEqual((bdir / "program.py").read_text(encoding="utf-8"), GOOD)
        self.assertTrue(json.loads((bdir / "verification.json").read_text())["passed"])
        br = json.loads((bdir / "branch_result.json").read_text())
        self.assertEqual((br["task_id"], br["accepted"], br["poetiq"]["iteration"]), ("t030", True, 1))
        rows = [json.loads(l) for l in (out / "e33_poetiq_records.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        t_ = time.monotonic()
        while time.monotonic() - t_ < 3.0:                         # the stream thread appends its line after the finish event
            reqs = [json.loads(l) for l in (out / "e33_poetiq_requests.jsonl").read_text().splitlines()]
            if len(reqs) >= 2:
                break
            time.sleep(0.05)
        self.assertEqual(sorted(r["rid"].split("|")[1] for r in reqs), ["1", "2"])
        rec2, _, out2 = _run(31, {"kind": "natural", "answer": answer(GOOD)}, cfg={"tok_s": 0.01})   # not admitted: a record too
        self.assertEqual(rec2["status"], "not_admitted")
        self.assertEqual(json.loads((out2 / "t031_a1_i1" / "branch_result.json").read_text())["status"], "not_admitted")

    def test_record_shape(self):
        rec, _, _ = _run(32, {"kind": "natural", "answer": answer(WRONG)})
        for k in ("task_id", "status", "accepted", "generated_tokens", "submissions", "poetiq", "poetiq_state", "usage", "t0", "t1"):
            self.assertIn(k, rec)
        s = rec["submissions"][0]
        self.assertEqual(sorted(s), sorted(["test_outputs", "accepted", "train_exact", "t", "branch", "origin", "name"]))
        for k in ("iteration", "score", "train_score", "program", "parse_status", "feedback", "mode", "think_tokens",
                  "answer_tokens", "train_pass"):
            self.assertIn(k, rec["poetiq"])
        json.dumps(rec)                                            # JSON-serializable (the scheduler publishes it)


# ------------------------------------------------------------------ 8. the build: verbatim regions, reproducible, LF
class TestBuild(unittest.TestCase):
    def test_lf_only(self):
        for n in ("e33_poetiq.py", "e33_poetiq_build.py", "e33_poetiq_tests.py"):
            f = HERE / n
            if f.is_file():
                self.assertNotIn(b"\r", f.read_bytes(), n)

    def test_rebuild_and_verbatim(self):
        if PB is None or not PB.E18_NB.is_file() or not PB.E29_DIRECT.is_file():
            self.skipTest("the sources are not on disk")
        text = PB.make_module()
        self.assertEqual(text.encode("utf-8"), (HERE / "e33_poetiq.py").read_bytes())
        for name, region in PB.verbatim_regions().items():
            self.assertEqual(text.count(region), 1, name)
        self.assertEqual(text.count(PB.CORE_SELFTEST_NEW), 1)
        for fn in ("poetiq_format_problem", "poetiq_build_message", "poetiq_parse_code", "poetiq_sandbox_run", "poetiq_verify_code",
                   "poetiq_array_diff", "poetiq_create_examples", "poetiq_vote", "poetiq_two_attempts", "poetiq_sandbox_selftest"):
            self.assertIn("\ndef " + fn + "(", PB.verbatim_regions()["e18_core_before_selftest"])

    def test_e18_functions_behave_identically(self):
        if E18 is None:
            self.skipTest("the E18 notebook is not on disk")
        task = _task(3)
        for it, hist in ((0, []), (1, [{"code": WRONG, "feedback": "fb", "score": 0.5}]),
                         (2, [{"code": "a", "feedback": "f1", "score": 0.2}, {"code": "b", "feedback": "f2", "score": 0.7}])):
            for seed in (0, 7000, 12100):
                self.assertEqual(P.poetiq_build_message(task, it, 0, hist, seed=seed), E18["poetiq_build_message"](task, it, 0, hist, seed=seed))
        for resp in ("```python\nx=1\n```\n```python\ndef transform(g):\n return g\n```", "none", "```python\ndef f(:\n```"):
            self.assertEqual(P.poetiq_parse_code(resp), E18["poetiq_parse_code"](resp))
        self.assertEqual(P.POETIQ_SOLVER_PROMPTS, E18["POETIQ_SOLVER_PROMPTS"])
        self.assertEqual(P.POETIQ_FEEDBACK_PROMPT, E18["POETIQ_FEEDBACK_PROMPT"])


if __name__ == "__main__":
    unittest.main(verbosity=2 if "-v" in sys.argv else 1, argv=[sys.argv[0]])
