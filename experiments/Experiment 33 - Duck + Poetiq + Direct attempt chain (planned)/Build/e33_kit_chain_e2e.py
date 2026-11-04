"""E33 kit: the attempt-chain e2e (Kaggle CPU chain kit; mocked GPU):  python e33_kit_chain_e2e.py [-v]   (exit 0 = all pass)

A scripted run of 6 tasks through A -> B -> C -> D -> E and attempt 2 (DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md sections 3 / 4):
  - the REAL arc2_duck.py of the Duck payload + the REAL E29 scheduler diff (e29_sched_patch) + the REAL E33 chain diff
    (e33_sched_patch.apply_e33), the real BranchSession / scheduler, the scripted fake Duck agent of e29_sched_tests.py;
  - the notebook's OWN hooks: e33_nbx.py (the payload-dataset file) executed into a notebook-like namespace -> e33_setup()
    installs E12A.E33_RUN_POETIQ / E33_RUN_DIRECT / E33_CAN_START (the switch parsed by e33_lib.check_chain with the kit
    override = the CELL 1 harness path), e33_live_setup() wraps solve_all_r6 (one LIVE line per finished stage);
  - the REAL e33_poetiq.py (prompt, transport, parse, the E18 sandbox verify, feedback, iteration-2 history) and the REAL
    e29_direct.py, both against an in-process fake streaming token-id /v1/completions server (the Flash-Next stop / think-close
    ids; a stub tokenizer: 1000 + ord(c)).
Asserted: the start order on 1 worker (every attempt-1 item before any attempt-2 item; A, B, C, D, E; group then task order;
attempt-2 group 1 = no attempt-1 A train pass in task order, group 2 = the rest); the stop at the first train pass of every
stage kind; B1 / B2 by the A kind; the iteration-2 prompt carries iteration 1's program + feedback; per-stage counts / train
passes / skips of SCHED_LAST["e28"]["e33"]; the answer slots (attempt_1 / attempt_2 + their source label and step) of every
task; one start / one row / one release per (task, branch); 4 workers: the same answers, never two stages of one (task,
attempt) at once; the late-window skip (E33_CAN_START False -> skipped, chain ends, nothing published); LIVE lines.
The Poetiq sandbox is the REAL one (root + libseccomp: the Kaggle image). E33KT_LOCAL_SANDBOX=1 (local debugging only, never
in the kit) replaces the sandbox child by an in-process exec; the kit's verdict requires it unset.
Inputs: the Duck payload (E28_TEST_PAYLOAD), e28_sched_patch.py (E29_TEST_E28_DIR), next to this file: e29_sched_tests.py,
e29_sched_patch.py, e33_sched_patch.py, e33_poetiq.py, e29_direct.py, e33_lib.py, e33_nbx.py (the kit stages the last five
from the mounted E33 payload dataset by sha).
"""
import io
import contextlib
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import types
import unittest
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e29_sched_tests as ST     # noqa: E402  the scripted-Duck harness (fixtures set up in setUpModule)
import e33_sched_patch as S33    # noqa: E402
import e33_poetiq as P           # noqa: E402
import e29_direct as D           # noqa: E402
import e33_lib as L33            # noqa: E402

E29S = ST.E29S
FIX = ST.FIX
CLOSE, STOP = D.FN_THINK_CLOSE_ID, list(D.FN_STOP_IDS)
EOS = [248046, 248044]
LOCAL_SANDBOX = os.environ.get("E33KT_LOCAL_SANDBOX") == "1"


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
        elif i in STOP:
            continue
        elif 1000 <= i < 200000:
            out.append(chr(i - 1000))
    return "".join(out)


TOK = {"render": lambda m: "<u>" + m[0]["content"] + "</u>", "encode": enc, "decode": dec, "eos_ids": EOS}

# ------------------------------------------------------------------ the tasks: train [[k]] -> [[k + 1]], test [[5]] (gold [[6]])
TIDS = [f"t{i:03d}" for i in range(6)]
TASKS = {t: {"train": [{"input": [[1]], "output": [[2]]}, {"input": [[3]], "output": [[4]]}], "test": [{"input": [[5]]}]} for t in TIDS}
GOLD = {t: [[[6]]] for t in TIDS}
PROG = {"pass": "def transform(grid):\n    return [[(int(v) + 1) % 10 for v in row] for row in grid]\n",
        "half": "def transform(grid):\n    return [[2]]\n",                 # train 1 / 2 exact, test grid [[2]]
        "wrong": "def transform(grid):\n    return [[int(v) for v in row] for row in grid]\n"}   # native JSON integers, same E18 sandbox contract
# Duck scripts (e29_sched_tests fake agent; key (task, "b<branch>")): a1 A = b0, a1 B = b2, a2 A = b1, a2 B = b122
DUCK = {("t000", "b0"): {"kind": "pass", "at": 2000, "value": 1}, ("t000", "b1"): {"kind": "pass", "at": 2000, "value": 2},
        ("t001", "b0"): {"kind": "fail"}, ("t001", "b2"): {"kind": "pass", "at": 2000, "value": 3},
        ("t001", "b1"): {"kind": "fail"}, ("t001", "b122"): {"kind": "fail"},
        ("t002", "b0"): {"kind": "cap"}, ("t002", "b2"): {"kind": "fail"}, ("t002", "b1"): {"kind": "pass", "at": 2000, "value": 4},
        ("t003", "b0"): {"kind": "fail"}, ("t003", "b2"): {"kind": "fail"}, ("t003", "b1"): {"kind": "pass", "at": 2000, "value": 5},
        ("t004", "b0"): {"kind": "fail"}, ("t004", "b2"): {"kind": "fail"}, ("t004", "b1"): {"kind": "fail"}, ("t004", "b122"): {"kind": "fail"},
        ("t005", "b0"): {"kind": "fail"}, ("t005", "b2"): {"kind": "fail"}, ("t005", "b1"): {"kind": "pass", "at": 2000, "value": 9}}
# Poetiq answers per (task, attempt, iteration); Direct grid per (task, attempt) (None = an answer without a grid)
POETIQ = {("t002", 1, 1): "pass", ("t003", 1, 1): "wrong", ("t003", 1, 2): "pass", ("t004", 1, 1): "half", ("t004", 1, 2): "wrong",
          ("t005", 1, 1): "nocode", ("t005", 1, 2): "nocode", ("t001", 2, 1): "pass", ("t004", 2, 1): "nocode", ("t004", 2, 2): "nocode"}
DIRECT = {("t004", 1): [[7]], ("t005", 1): [[8]], ("t004", 2): None}

# the expected chain (1 worker): the start order (phase labels) and the answers (grid, source label, step)
ORDER_1W = ([("A:1", t) for t in TIDS] + [("B1:1", "t001"), ("B2:1", "t002"), ("B1:1", "t003"), ("B1:1", "t004"), ("B1:1", "t005")]
            + [("C-POETIQ1:1", t) for t in ("t002", "t003", "t004", "t005")] + [("D-POETIQ2:1", t) for t in ("t003", "t004", "t005")]
            + [("E-DIRECT:1", t) for t in ("t004", "t005")]
            + [("A:2", t) for t in ("t001", "t002", "t003", "t004", "t005", "t000")] + [("B1:2", "t001"), ("B1:2", "t004")]
            + [("C-POETIQ1:2", "t001"), ("C-POETIQ1:2", "t004"), ("D-POETIQ2:2", "t004"), ("E-DIRECT:2", "t004")])
ANSWERS = {"t000": ([[1]], "A:1", "1-pass", [[2]], "A:2", "a2:1-pass"),
           "t001": ([[3]], "B1:1", "1-pass", [[6]], "C-POETIQ1:2", "a2:1-pass"),
           "t002": ([[6]], "C-POETIQ1:1", "1-pass", [[4]], "A:2", "a2:1-pass"),
           "t003": ([[6]], "D-POETIQ2:1", "1-pass", [[5]], "A:2", "a2:1-pass"),
           "t004": ([[2]], "C-POETIQ1:1", "2-best", [[5]], "D-POETIQ2:1", "a1-next"),
           "t005": ([[8]], "E-DIRECT:1", "3-direct", [[9]], "A:2", "a2:1-pass")}
STAGE_COUNTS = {"A:1": (6, 1), "B1:1": (4, 1), "B2:1": (1, 0), "C-POETIQ1:1": (4, 1), "D-POETIQ2:1": (3, 1), "E-DIRECT:1": (2, 0),
                "A:2": (6, 4), "B1:2": (2, 0), "B2:2": (0, 0), "C-POETIQ1:2": (2, 1), "D-POETIQ2:2": (1, 0), "E-DIRECT:2": (1, 0)}


# ------------------------------------------------------------------ the fake streaming token-id server
FAKE = {"log": [], "lock": threading.Lock(), "index": {t: i for i, t in enumerate(sorted(TIDS))}}


def _who(body):
    """(solver, task, attempt, iteration, phase) of one request: Poetiq prompts carry Poetiq's solver prompt ('def transform');
    the seeds name the task (Poetiq: SEED + tix * 1000 + expert * 100 + it * 2 + phase; Direct: SEED + 10 * index + phase)."""
    text = dec(body["prompt"], False)
    off = int(body["seed"]) - D.SEED
    m_ = str(body.get("model") or "").split("|")             # "fake|<task>|a<attempt>" (the kit's hook wrapper, see run_chain)
    tid = m_[1] if len(m_) == 3 else None
    if "def transform" in text:
        r = off % 1000
        ex, r = divmod(r, 100)
        it, ph = (r - 1) // 2, (r - 1) % 2 + 1
        return "poetiq", tid, ex + 1, it + 1, ph, text
    ph = 1 if THINK_CLOSE_SEEN not in text else 2              # Direct: phase 1 = the prompt only (phase 2 adds the kept ids)
    return "direct", tid, None, None, ph, text


THINK_CLOSE_SEEN = "\x00never\x00"


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
        solver, tid, att, it, ph, text = _who(body)
        m_ = str(body.get("model") or "").split("|")             # "fake|<task>|a<attempt>" (the kit's hook wrapper, see run_chain)
        if solver == "direct":
            att = int(m_[2][1:]) if len(m_) == 3 else None        # the Direct seed carries no attempt (ORDER2's seed per task)
        assert len(m_) == 3 and m_[1] == tid and int(m_[2][1:]) == att, (m_, tid, att)
        with FAKE["lock"]:
            FAKE["log"].append({"solver": solver, "tid": tid, "attempt": att, "iteration": it, "phase": ph, "max_tokens": body["max_tokens"],
                                "feedback": "EXISTING PARTIAL/INCORRECT SOLUTIONS" in text, "prev_code": PROG["wrong"].strip() in text
                                or PROG["half"].strip() in text, "model": body.get("model"), "t": time.monotonic()})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if solver == "poetiq":
            kind = POETIQ.get((tid, att, it), "nocode")
            ans = "I could not find it." if kind == "nocode" else "```python\n" + PROG[kind] + "```"
        else:
            g = DIRECT.get((tid, att))
            ans = "no grid here" if g is None else "```grid\n" + "\n".join(" ".join(str(v) for v in r) for r in g) + "\n```"
        try:
            if ph == 1:
                self._chunk(enc("thinking " * 3))
                self._chunk([CLOSE] + enc(ans))
                self._chunk(EOS[:1], "stop")
            else:                                   # a forced-close phase 2 (not used by these scripts): the answer
                self._chunk(enc(ans))
                self._chunk(EOS[:1], "stop")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ------------------------------------------------------------------ the notebook-like namespace with the notebook's own hooks
NB = {}


def _local_sandbox_run(code, grid, timeout_s=5.0):
    """E33KT_LOCAL_SANDBOX=1 only (local debugging, never in the kit): E18's runner contract (ok, json result) in-process."""
    try:
        g = {}
        exec(code, g)
        return True, json.dumps(g["transform"](grid))
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def nb_namespace(url, chain_override, workdir):
    nbx_p = HERE / "e33_nbx.py"
    files = json.loads(os.environ["E33KT_FILES"]) if os.environ.get("E33KT_FILES") else None

    def q27x_payload(name, sha_):
        import hashlib
        b = (HERE / name).read_bytes()
        assert hashlib.sha256(b).hexdigest() == sha_, (name, "sha mismatch")
        return b
    lit = dict(L33.DEFAULT_CHAIN)
    if os.environ.get("E33KT_SWITCH"):
        lit = json.loads(os.environ["E33KT_SWITCH"])          # the notebook's CELL 1 literal (the kit passes it)
    chain = L33.check_chain(lit, json.dumps(chain_override))    # = CELL 1 with E11_HARNESS=1 + E33_CHAIN_JSON
    chain["a1_order"] = "e28"  # mirror the released notebook's explicit answer order
    log_lines = []
    g = {"__name__": "e33_nbx_kit", "__builtins__": __builtins__, "os": os, "json": json, "time": time, "threading": threading,
         "Path": Path, "E33_CHAIN": chain, "E33L": L33, "E12A": FIX["e33"], "E11_SERVER": types.SimpleNamespace(url=url),
         "SGL_URL": url, "E11_HARNESS": True, "NOTEBOOK_T0": time.perf_counter(), "TMP": Path(workdir), "q27x_payload": q27x_payload,
         "E33D_FILES": files or {"direct": ["e29_direct.py", _sha(HERE / "e29_direct.py")],
                                 "poetiq": ["e33_poetiq.py", _sha(HERE / "e33_poetiq.py")]},
         "E33_SCHED_REC": {"ok": True, "env_names": list(S33.ENV_NAMES)}, "E29_RT": {}, "TASK_IDS": sorted(TIDS),
         "log": lambda *a, **k: log_lines.append(" ".join(str(x) for x in a)), "e21_prom": lambda: [],
         "RERUN": False, "SOLUTIONS": GOLD, "SUBMISSION_PATH": str(Path(workdir) / "submission.json"), "test_challenges": TASKS}
    import e29_lib as E29L
    g["E29L"] = E29L
    exec(compile(nbx_p.read_bytes().decode("utf-8"), "e33_nbx.py", "exec"), g)
    g["e29d_tok"] = lambda: TOK                                   # the stub tokenizer (the real one: the real kit)
    g["_log_lines"] = log_lines
    return g


def _sha(p):
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setUpModule():
    ST.setUpModule()
    tmp = FIX["tmp"]
    with zipfile.ZipFile(ST.PAYLOAD) as z:
        z.extractall(tmp / "e33")
    assert E29S.apply_e29_sched(tmp / "e33")["ok"]
    r = S33.apply_e33(tmp / "e33")
    assert r["ok"] and r["after"] == "E33_PATCHED", r
    FIX["e33"] = ST._load(tmp / "e33" / "arc2_duck.py", "arc2_duck_e33kit")
    srv = TS(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    FIX["srv"], FIX["url"] = srv, f"http://127.0.0.1:{srv.server_address[1]}"
    P.ADMIT_STOP_BEFORE_END_SEC, P.HARD_BEFORE_END_SEC = 2.0, 1.0       # second-scale solve windows (module constants)
    D.ADMIT_STOP_BEFORE_END_SEC, D.HARD_BEFORE_END_SEC = 2.0, 1.0
    L33.ADMIT_STOP_BEFORE_END_SEC = 2.0
    P.CFG_DEFAULT["drain_min"] = 0.5                                      # Poetiq's own drain (9 min) -> 30 s
    if LOCAL_SANDBOX:
        P._E18_POETIQ_SANDBOX_RUN = _local_sandbox_run
        P._SANDBOX["selftest"] = {"ok": True, "mode": "LOCAL_IN_PROCESS (E33KT_LOCAL_SANDBOX=1, never the kit)"}
    FIX["sandbox"] = P.sandbox_ready()


def tearDownModule():
    try:
        FIX["srv"].shutdown()
    except Exception:
        pass
    ST.tearDownModule()


OVERRIDE = {"poetiq_think": 400, "poetiq_answer": 256, "direct_think": 400, "direct_answer": 256, "drain_min": 1}


def run_chain(workers, deadline_sec=300.0, override=None, setup=True):
    """The scripted run through the notebook's hooks: solve_all_r6 on the E33 post module (base_url = the fake server)."""
    M = FIX["e33"]
    ST.SCRIPTS.clear()
    ST.SCRIPTS.update(DUCK)
    with FAKE["lock"]:
        FAKE["log"].clear()
    work = FIX["tmp"] / f"chain_{workers}_{time.monotonic_ns()}"
    work.mkdir(parents=True)
    g = nb_namespace(FIX["url"], dict(OVERRIDE, **(override or {})), work)
    env = dict(S33.server_env(g["E33_CHAIN"]))
    out = io.StringIO()
    try:
        with ST.Env(supply=False, **env):      # E29 supply off here (S's tests cover supply with the chain)
            if setup:
                g["e33_setup"]()
                g["e33_live_setup"]()
            for h_ in ("E33_RUN_POETIQ", "E33_RUN_DIRECT"):   # the notebook's hook, called with a model id naming (task, attempt)
                f_ = getattr(M, h_, None)
                if callable(f_):
                    setattr(M, h_, (lambda f0: lambda tid, task, plan, deadline, base_url, model_id, out_dir, *a: f0(
                        tid, task, plan, deadline, base_url, f"{model_id}|{tid}|a{int(plan['e33_attempt'])}", out_dir, *a))(f_))
            kit = ST.RunLog(M)
            try:
                with contextlib.redirect_stdout(out):
                    recs = M.solve_all_r6(TASKS, list(TIDS), out_dir=work / "out", deadline=time.monotonic() + deadline_sec,
                                          base_url=FIX["url"] + "/v1", model_id="fake", branches=2, workers=workers,
                                          agent_cls=ST.make_fake(M), first_exposure_tokens=None, min_admit_sec=0.2,
                                          capacity_probe=None, kv_poll_sec=0.05, fair_share=False, join_grace_sec=2.0,
                                          analyzer_timeout=30.0)
            finally:
                kit.close()
        sl = json.loads(json.dumps(M.SCHED_LAST, default=str))
    finally:
        try:
            (g.get("E33_RT") or {}).get("_rate_stop") and g["E33_RT"]["_rate_stop"].set()
        except Exception:
            pass
        if getattr(M, "_e33_live_orig", None) is not None:          # unwrap the LIVE wrapper (the next run installs it again)
            M.solve_all_r6 = M._e33_live_orig
            M._e33_live_orig = None
        M.E33_RUN_POETIQ = M.E33_RUN_DIRECT = M.E33_CAN_START = None
    if Path('/kaggle/working').is_dir():
        Path(f'/kaggle/working/e33_chain_debug_{workers}.json').write_text(json.dumps(
            {'records': recs, 'scheduler': sl, 'fake': FAKE['log'], 'logs': g['_log_lines'], 'stdout': out.getvalue()}, default=str))
    return recs, sl, g, out.getvalue(), list(ST.RUNLOG), list(FAKE["log"])


def answers(recs):
    M = FIX["e33"]
    res = {}
    for t in TIDS:
        slots = []
        with ST.Env(supply=False, E33_A1_ORDER="e28"):
            att = M.select_attempts_e33(TASKS[t], list(recs.get(t) or []), slots)
        res[t] = (att[0]["attempt_1"], slots[0]["attempt_1"], slots[0]["a1_step"], att[0]["attempt_2"], slots[0]["attempt_2"],
                  slots[0]["a2_step"])
    return res


class TestChainE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r1 = run_chain(1)

    def test_00_sandbox_and_hooks(self):
        sb = FIX["sandbox"]
        self.assertTrue(sb.get("ok"), sb)
        recs, sl, g, out, runlog, fake = self.r1
        rt = g["E33_RT"]
        self.assertEqual(rt["state"], "ON", rt)
        self.assertEqual(sorted(rt["hooks"]), ["E33_CAN_START", "E33_RUN_DIRECT", "E33_RUN_POETIQ"])
        self.assertTrue(str(rt["items"]["poetiq"]).startswith("ON (e33_poetiq") and str(rt["items"]["direct"]).startswith("ON (e29_direct"),
                        rt["items"])
        e33 = sl["e28"]["e33"]
        self.assertTrue(e33["enabled"] and all(e33["hooks"].values()) and all(e33["switches"].values()), e33.get("hooks"))
        self.assertEqual(e33["branch_map"], dict(FIX["e33"].E33_SRC))

    def test_01_order_one_worker(self):
        recs, sl, g, out, runlog, fake = self.r1
        rows = sorted((r for r in sl["e28"]["attempts"] if r.get("start_seq") is not None and r.get("status") != "skipped"),
                      key=lambda r: r["start_seq"])
        got = [(r["phase"], r["task"]) for r in rows]
        self.assertEqual(got, ORDER_1W)
        first_a2 = next(i for i, (ph, _) in enumerate(got) if ph.endswith(":2"))
        self.assertTrue(all(ph.endswith(":1") for ph, _ in got[:first_a2]) and all(ph.endswith(":2") for ph, _ in got[first_a2:]))

    def test_02_stage_counts(self):
        recs, sl, g, out, runlog, fake = self.r1
        st = sl["e28"]["e33"]["stages"]
        got = {ph: (st[ph]["done"], st[ph]["train_pass"]) for ph in STAGE_COUNTS}
        self.assertEqual(got, STAGE_COUNTS)
        self.assertTrue(all(not st[ph]["skipped"] and st[ph]["no_record"] == 0 and st[ph]["errors"] == 0 and st[ph]["left"] == 0
                            for ph in STAGE_COUNTS), {ph: (st[ph]["skipped"], st[ph]["no_record"], st[ph]["errors"]) for ph in st})
        for ph, (n, _) in STAGE_COUNTS.items():
            self.assertEqual(st[ph]["queued"], n, ph)

    def test_03_answers_and_slots(self):
        recs, sl, g, out, runlog, fake = self.r1
        self.assertEqual(answers(recs), ANSWERS)
        with ST.Env(supply=False, **S33.server_env(g["E33_CHAIN"])):   # the run's env (E28_SCHED=1 + E33_SCHED=1: CELL 3)
            sub = FIX["e33"].build_submission(TASKS, recs)                  # = the notebook writer's rule (E12A.build_submission)
        for t in TIDS:
            self.assertEqual((sub[t][0]["attempt_1"], sub[t][0]["attempt_2"]), (ANSWERS[t][0], ANSWERS[t][3]), t)

    def test_04_poetiq_iteration2_feedback_and_requests(self):
        recs, sl, g, out, runlog, fake = self.r1
        pq = [x for x in fake if x["solver"] == "poetiq" and x["phase"] == 1]
        self.assertEqual(sorted((x["tid"], x["attempt"], x["iteration"]) for x in pq), sorted(POETIQ))
        for x in pq:                                                         # iteration 2 = Poetiq's feedback prompt (when
            hist = x["iteration"] == 2 and POETIQ[(x["tid"], x["attempt"], 1)] != "nocode"   # iteration 1 left a program)
            self.assertEqual(x["feedback"], hist, x)
        it2 = [x for x in pq if x["iteration"] == 2 and (x["tid"], x["attempt"], 1) in POETIQ and POETIQ[(x["tid"], x["attempt"], 1)] != "nocode"]
        self.assertTrue(it2 and all(x["prev_code"] for x in it2), it2)      # with iteration 1's program in the history
        dr = [x for x in fake if x["solver"] == "direct" and x["phase"] == 1]
        self.assertEqual(sorted((x["tid"], x["attempt"]) for x in dr), sorted(DIRECT))
        self.assertTrue(all(x["max_tokens"] <= 400 for x in pq + dr))        # the switch budgets (kit override) reach the solvers
        by = {(r.get("task_id"), int(r.get("branch"))): r for v in recs.values() for r in v}
        r = by[("t003", 114)]
        self.assertEqual((r["poetiq"]["iteration"], r["poetiq"]["train_pass"], r["poetiq"]["history_in"]), (2, True, 1), r["poetiq"])
        self.assertEqual((by[("t004", 115)]["submissions"][0]["test_outputs"], by[("t004", 115)]["e28"]["phase"]), ([[[7]]], "E-DIRECT:1"))

    def test_05_one_start_row_release_per_branch(self):
        recs, sl, g, out, runlog, fake = self.r1
        starts = [(t, b) for ev, t, b, ph, _ in runlog if ev == "start"]
        self.assertEqual(len(starts), len(set(starts)))                     # Duck sessions: one start per (task, branch)
        rows = sl["e28"]["attempts"]
        self.assertEqual(len(rows), sum(n for n, _ in STAGE_COUNTS.values()))
        pub = [(r.get("task_id"), int(r.get("branch"))) for v in recs.values() for r in v]
        self.assertEqual(len(pub), len(set(pub)))
        self.assertEqual(sorted(pub), sorted((r["task"], int(r["branch"])) for r in rows))
        bm = dict(FIX["e33"].E33_SRC)
        self.assertTrue(all(int(r["branch"]) == bm[r["phase"]] for r in rows), [(r["phase"], r["branch"]) for r in rows])

    def test_06_live_lines(self):
        recs, sl, g, out, runlog, fake = self.r1
        lines = [l for l in out.splitlines() if l.startswith("[LIVE ") and not l.startswith(("[LIVE start]", "[LIVE 15-min"))]
        firsts = [l for l in lines if "(again)" not in l]
        self.assertEqual(len(firsts), len(sl["e28"]["attempts"]), lines[:5])
        self.assertTrue(any("[LIVE start] algo" in l for l in out.splitlines()))
        want = sorted(f"{r['task']} a{r['phase'].split(':')[1]}:{r['phase'].split(':')[0]}" for r in sl["e28"]["attempts"])   # E33Live: a<attempt>:<stage name>
        got = sorted(" ".join(l.split("]", 1)[1].split()[:2]) for l in firsts)
        self.assertEqual(got, want)
        self.assertEqual(g["E33_RT"]["live"]["errors"], 0, g["E33_RT"]["live"])

    def test_07_four_workers_same_answers_no_overlap(self):
        recs, sl, g, out, runlog, fake = run_chain(4)
        self.assertEqual(answers(recs), ANSWERS)
        st = sl["e28"]["e33"]["stages"]
        self.assertEqual({ph: (st[ph]["done"], st[ph]["train_pass"]) for ph in STAGE_COUNTS}, STAGE_COUNTS)
        iv = {}
        for r in sl["e28"]["attempts"]:
            a = r["phase"].split(":")[1]
            iv.setdefault((r["task"], a), []).append((float(r["start_sec"]), float(r["end_sec"]), r["phase"]))
        for k, v in iv.items():
            v.sort()
            for (s0, e0, p0), (s1, e1, p1) in zip(v, v[1:]):
                self.assertLessEqual(e0, s1 + 1e-6, (k, p0, p1))            # strictly sequential per (task, attempt)
        rows = sorted(sl["e28"]["attempts"], key=lambda r: r["start_seq"])
        seen_a2 = False
        for r in rows:                                                       # attempt 2 never ahead of a queued attempt-1 item
            if r["phase"].endswith(":2"):
                seen_a2 = True
                self.assertEqual(int(r.get("e33_q1_left") or 0), 0, r)
        self.assertTrue(seen_a2)

    def test_08_late_window_skips_and_ends_chain(self):
        # Poetiq / Direct with the REAL 60,000 floor in the admission (no kit floor override): every C item is late -> skipped,
        # recorded, the chain ends there; nothing published for it; Duck stages unaffected
        M = FIX["e33"]
        recs, sl, g, out, runlog, fake = run_chain(1, override={"poetiq_think": 60000, "direct_think": 60000})
        st = sl["e28"]["e33"]["stages"]
        for ph in ("C-POETIQ1:1", "C-POETIQ1:2"):
            n_c = sum(1 for t in TIDS for a in (1, 2) if ph.endswith(f":{a}") and (t, a, 1) in POETIQ)
            self.assertEqual((st[ph]["done"], sum(st[ph]["skipped"].values())), (0, n_c), (ph, st[ph]))
            self.assertEqual(set(st[ph]["skipped"]), {"late"}, st[ph]["skipped"])
        for ph in ("D-POETIQ2:1", "D-POETIQ2:2", "E-DIRECT:1", "E-DIRECT:2"):
            self.assertEqual((st[ph]["queued"], st[ph]["done"]), (0, 0), ph)
        self.assertFalse([x for x in fake], fake[:3])
        self.assertTrue(all(int(r["branch"]) not in (113, 114, 115, 123, 124, 125) for v in recs.values() for r in v))
        self.assertEqual(st["A:1"]["done"], 6)
        self.assertTrue(g["E33_RT"]["can_start"]["C"]["no"] >= 1 and g["E33_RT"]["can_start"]["C"]["go"] == 0, g["E33_RT"]["can_start"])
        self.assertIs(M.E33_RUN_POETIQ, None)


if __name__ == "__main__":
    prog = unittest.main(exit=False, verbosity=2 if "-v" in sys.argv else 1, argv=[sys.argv[0]])
    r = prog.result
    print(f"E33 CHAIN e2e: {r.testsRun} run, {len(r.failures)} failures, {len(r.errors)} errors, {len(r.skipped)} skipped, "
          f"sandbox {FIX.get('sandbox', {}).get('mode')}, local_sandbox {LOCAL_SANDBOX}", flush=True)
    sys.exit(0 if r.wasSuccessful() else 1)
