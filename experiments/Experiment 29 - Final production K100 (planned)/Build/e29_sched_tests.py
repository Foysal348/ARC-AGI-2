"""E29 SCHED (attempts supply, scheduler S-1) local unit tests: python e29_sched_tests.py [-v]   (exit 0 = all pass)

Covers: the patch (apply / verify / revert from ORIGINAL and from E28_PATCHED, INVALID_MIXED and CRLF refusal, text
mode, py_compile, strict diff, hunks); E29_SUPPLY unset == the E28 post file on several scheduler scenarios (records,
slots, SCHED_LAST["e28"] minus timing fields; and E28 off == the original rev 6 scheduler); the supply rules (S1 when
the queues are empty, S2 only after S1 is exhausted, A / B / C priority over supply, the per-task cap, the stop after a
TRAIN_PASS, the deadline guard, no idle worker while supply exists, concurrent supply attempts of one task, no KeyError
on the phase counters, S2 = the B2 budget / window); the E29 answer rule (synthetic cases + no S records == the E28
rule on the real K100Q / W4 records and on fake-run records); the real Arc2ToolAgent budget / window for branches >= 3.

Harness = e28_sched_tests.py's (the real arc2_duck.py from the Duck payload, the real BranchSession, a fake agent, the
real NB5 Reservation for the KV scenario). Inputs (read only): the Duck payload, the E28 Build folder (e28_sched_patch.py,
imported by e29_sched_patch.py), the recorded K100Q / W4 run folders, the ARC eval challenges, the E25 code exports.
Overrides: E29_TEST_E28_DIR / E28_TEST_PAYLOAD / E28_TEST_K100 / E28_TEST_W4 / E28_TEST_EVAL / E28_TEST_EXPORTS.
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
import threading
import time
import unittest
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:/ARC AGI 2")
E28_DIR = Path(os.environ.get("E29_TEST_E28_DIR", ROOT / "experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)"
                                                          "/Experiment Notebooks/Build"))
sys.path.insert(0, str(HERE))
if not (HERE / "e28_sched_patch.py").is_file():
    sys.path.insert(1, str(E28_DIR))
import e28_sched_patch as E28S  # noqa: E402
import e29_sched_patch as E29S  # noqa: E402

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
ALL_KEYS = list(E29S.ENV_NAMES)


# ------------------------------------------------------------------ fixtures (one extraction for the whole run)
def _tmp_root():
    for c in (os.environ.get("E28_TEST_TMP"), str(ROOT / "_scratch")):
        if c and Path(c).is_dir():
            return tempfile.mkdtemp(prefix="e29_sched_tests_", dir=c)
    return tempfile.mkdtemp(prefix="e29_sched_tests_")


FIX = {}


def setUpModule():
    data = PAYLOAD.read_bytes()
    assert hashlib.sha256(data).hexdigest() == PAYLOAD_SHA, "Duck payload sha mismatch"
    tmp = Path(_tmp_root())
    FIX["tmp"] = tmp
    for name in ("orig", "e28", "e29"):
        with zipfile.ZipFile(PAYLOAD) as z:
            z.extractall(tmp / name)
    rec = E28S.apply_e28_sched(tmp / "e28")
    assert rec["ok"] and rec["applied"] == "applied", rec
    rec = E29S.apply_e29_sched(tmp / "e29")
    assert rec["ok"] and rec["applied"] == "applied" and rec["e28_step"] == "applied", rec
    for k in ALL_KEYS + ["ARC3_MAX_ACTIVE_STREAMS"]:
        os.environ.pop(k, None)
    os.environ.update({"E12_ARC2_REV5": "1"})
    sys.path[:0] = [str(tmp / "orig" / "vendor")]          # the vendor tree is byte-identical in all copies
    FIX["orig"] = _load(tmp / "orig" / "arc2_duck.py", "arc2_duck_orig29")
    FIX["e28"] = _load(tmp / "e28" / "arc2_duck.py", "arc2_duck_e28_29")
    FIX["e29"] = _load(tmp / "e29" / "arc2_duck.py", "arc2_duck_e29")
    FIX["eval"] = json.loads(EVAL.read_bytes().decode("utf-8")) if EVAL.is_file() else {}


def tearDownModule():
    for k in ALL_KEYS:
        os.environ.pop(k, None)
    shutil.rmtree(FIX.get("tmp", ""), ignore_errors=True)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class Env:
    """Set exactly these env names for a with-block (every E28 / E29 name not given is removed); restores after."""
    def __init__(self, supply=True, e28=True, **extra):
        self.env = dict(E29S.server_env(supply=supply) if e28 else ({"E29_SUPPLY": "1"} if supply else {}),
                        **{k: str(v) for k, v in extra.items()})

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in set(ALL_KEYS) | set(self.env)}
        for k in ALL_KEYS:
            os.environ.pop(k, None)
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


def _orig_text():
    return (FIX["tmp"] / "orig" / "arc2_duck.py").read_bytes().decode("utf-8")


# ------------------------------------------------------------------ 1. the patch
class TestPatch(unittest.TestCase):
    def test_shas_and_constants(self):
        s = E29S.E29_SCHED_SHA
        self.assertEqual(s["pre"], "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128")
        self.assertEqual(s["e28_post"], "35b8d8577ab05be60e4e66f80400e57ccec3005dfad9b5290d9fa6f53adee57f")
        self.assertEqual(s["post"], E29S.E29_POST)
        self.assertEqual(s["patch"], hashlib.sha256(E29S.E29_PATCH.encode()).hexdigest())
        self.assertEqual(s["e28_patch"], E28S.E28_SCHED_SHA["patch"])
        self.assertEqual(s["applier"], E28S.E28_SCHED_SHA["applier"])
        self.assertEqual(s["arc2_duck.py"], {"pre": s["pre"], "e28_post": s["e28_post"], "post": s["post"]})
        self.assertEqual(E29S.ENV_NAMES, E28S.ENV_NAMES + ("E29_SUPPLY", "E29_SUPPLY_MAX_ATTEMPTS"))
        self.assertEqual(E29S.server_env(supply=True), dict(E28S.server_env(), E29_SUPPLY="1"))
        self.assertEqual(E29S.server_env(supply=False), E28S.server_env())
        self.assertEqual(E29S.server_env(), dict(E28S.server_env(), E29_SUPPLY="1"))
        self.assertEqual((E29S.T1, E29S.B2_TOKENS, E29S.MAX_MODEL_LEN, E29S.B2_WINDOW), (104000, 120000, 131072, 130560))
        self.assertIs(E29S.NOTEBOOK_EDITS, E28S.NOTEBOOK_EDITS)
        self.assertIs(E29S.apply_notebook_edits, E28S.apply_notebook_edits)
        self.assertEqual(E29S.SUPPLY_MAX_ATTEMPTS, 16)
        self.assertNotIn("\r", E29S.E29_PATCH)
        self.assertNotIn(b"\r", (HERE / "e29_sched_patch.py").read_bytes())
        self.assertNotIn(b"\r", (HERE / "e29_sched_tests.py").read_bytes())

    def _fresh(self, name):
        d = FIX["tmp"] / name
        shutil.rmtree(d, ignore_errors=True)
        with zipfile.ZipFile(PAYLOAD) as z:
            z.extract("arc2_duck.py", d)
        return d

    def test_apply_verify_revert_from_original(self):
        d = self._fresh("t_dir")
        orig = (d / "arc2_duck.py").read_bytes()
        self.assertEqual(E29S.tree_state(d), "ORIGINAL")
        self.assertFalse(E29S.verify_e29_sched(d)[0])
        r = E29S.apply_e29_sched(d)
        self.assertEqual((r["applied"], r["e28_step"], r["before"], r["after"], r["py_compile"]["rc"], r["ok"]),
                         ("applied", "applied", "ORIGINAL", "FULLY_PATCHED", 0, True))
        self.assertEqual(set(r) - {"e29_sched_sha", "e28_step"},
                         {"mode", "file", "before", "sha_before", "applied", "after", "sha_after", "py_compile", "ok",
                          "e28_sched_sha"})
        self.assertEqual(hashlib.sha256((d / "arc2_duck.py").read_bytes()).hexdigest(), E29S.E29_POST)
        ok, why = E29S.verify_e29_sched(d)
        self.assertTrue(ok, why)
        self.assertEqual(E29S.apply_e29_sched(d / "arc2_duck.py")["applied"], "already_applied")   # file path form
        self.assertEqual(list(d.glob("*tmp")), [])
        self.assertFalse(E28S.verify_e28_sched(d)[0])              # E28 alone does not take the E29 file
        rv = E29S.revert_e29_sched(d)
        self.assertEqual((rv["reverted"], rv["after"], rv["ok"]), ("reverted", "ORIGINAL", True))
        self.assertEqual((d / "arc2_duck.py").read_bytes(), orig)
        self.assertEqual(E29S.revert_e29_sched(d)["reverted"], "not_applied")

    def test_from_e28_patched(self):
        d = self._fresh("t_e28")
        orig = (d / "arc2_duck.py").read_bytes()
        E28S.apply_e28_sched(d)
        e28b = (d / "arc2_duck.py").read_bytes()
        self.assertEqual(E29S.tree_state(d), "E28_PATCHED")
        self.assertFalse(E29S.verify_e29_sched(d)[0])
        r = E29S.apply_e29_sched(d)
        self.assertEqual((r["applied"], r["e28_step"], r["before"], r["after"]),
                         ("applied", "already_applied", "E28_PATCHED", "FULLY_PATCHED"))
        self.assertTrue(E29S.verify_e29_sched(d)[0])
        self.assertEqual(E29S.revert_e29_sched(d)["reverted"], "reverted")
        self.assertEqual((d / "arc2_duck.py").read_bytes(), orig)          # all the way back to ORIGINAL
        (d / "arc2_duck.py").write_bytes(e28b)                             # revert straight from E28_PATCHED
        rv = E29S.revert_e29_sched(d)
        self.assertEqual((rv["before"], rv["after"]), ("E28_PATCHED", "ORIGINAL"))
        self.assertEqual((d / "arc2_duck.py").read_bytes(), orig)

    def test_invalid_mixed_refused(self):
        d = self._fresh("t_bad")
        f = d / "arc2_duck.py"
        bad = f.read_bytes().replace(b"def _rev6()", b"def _rev6( )", 1)
        f.write_bytes(bad)
        self.assertEqual(E29S.tree_state(d), "INVALID_MIXED")
        with self.assertRaisesRegex(RuntimeError, "E29_SCHED_PRE_SHA_MISMATCH"):
            E29S.apply_e29_sched(d)
        with self.assertRaisesRegex(RuntimeError, "E29_SCHED_REVERT_SHA_MISMATCH"):
            E29S.revert_e29_sched(d)
        self.assertEqual(f.read_bytes(), bad)                         # nothing written
        self.assertEqual(E29S.tree_state(FIX["tmp"] / "no_such_dir_e29"), "INVALID_MIXED")
        post = (FIX["tmp"] / "e29" / "arc2_duck.py").read_bytes()
        f.write_bytes(post.replace(E29S.E29_MARKER.encode(), b"# x", 1))   # E29 text without its marker: refused
        self.assertEqual(E29S.tree_state(d), "INVALID_MIXED")
        with self.assertRaises(RuntimeError):
            E29S.apply_e29_sched(d)
        e28t = E28S.patched_text(_orig_text())
        mixed = e28t.replace("    sup = None\n", "", 1) + "\n# E29 SUPPLY partial\n"
        self.assertEqual(E29S.tree_state(mixed), "INVALID_MIXED")
        with self.assertRaises(RuntimeError):
            E29S.apply_e29_sched(mixed)

    def test_crlf_refused(self):
        for t in (_orig_text(), E28S.patched_text(_orig_text()),
                  (FIX["tmp"] / "e29" / "arc2_duck.py").read_bytes().decode("utf-8")):
            crlf = t.replace("\n", "\r\n")
            self.assertEqual(E29S.tree_state(crlf), "INVALID_MIXED")
            with self.assertRaises(RuntimeError):
                E29S.apply_e29_sched(crlf)
            with self.assertRaises(RuntimeError):
                E29S.revert_e29_sched(crlf)
        d = self._fresh("t_crlf")
        f = d / "arc2_duck.py"
        b = f.read_bytes().replace(b"\n", b"\r\n")
        f.write_bytes(b)
        with self.assertRaises(RuntimeError):
            E29S.apply_e29_sched(d)
        self.assertEqual(f.read_bytes(), b)

    def test_text_mode_and_strict_diff(self):
        t = _orig_text()
        e29_file = (FIX["tmp"] / "e29" / "arc2_duck.py").read_bytes().decode("utf-8")
        r = E29S.apply_e29_sched(t)
        self.assertEqual((r["mode"], r["applied"], r["e28_step"], r["after"], r["ok"]),
                         ("text", "applied", "applied", "FULLY_PATCHED", True))
        self.assertEqual(r["text"], e29_file)
        self.assertEqual(E29S.patched_text(t), e29_file)
        r2 = E29S.apply_e29_sched(E28S.patched_text(t))
        self.assertEqual((r2["e28_step"], r2["text"]), ("already_applied", e29_file))
        self.assertEqual(E29S.apply_e29_sched(e29_file)["applied"], "already_applied")
        self.assertEqual(E29S.revert_e29_sched(e29_file)["text"], t)
        self.assertEqual(E29S.revert_e29_sched(E28S.patched_text(t))["text"], t)
        self.assertEqual(E29S.revert_e29_sched(t)["reverted"], "not_applied")
        self.assertTrue(E29S.verify_e29_sched(e29_file)[0])
        e28t = E28S.patched_text(t)
        self.assertEqual(E29S.apply_unified(e28t, E29S.E29_PATCH), e29_file)
        self.assertEqual(E29S.apply_unified(e29_file, E29S.reverse_unified(E29S.E29_PATCH)), e28t)
        lines = e28t.splitlines(True)                                   # strict: no fuzz, no offset
        i = next(i for i, ln in enumerate(lines) if "def _sec() -> float:" in ln and i > 3900)
        lines[i] = lines[i].rstrip("\n") + "  \n"
        with self.assertRaises(ValueError):
            E29S.apply_unified("".join(lines), E29S.E29_PATCH)
        with self.assertRaises(ValueError):
            E29S.apply_unified("\n" + e28t, E29S.E29_PATCH)

    def test_compile_and_hunks(self):
        e29_file = (FIX["tmp"] / "e29" / "arc2_duck.py").read_bytes().decode("utf-8")
        compile(e29_file, "arc2_duck.py", "exec")
        self.assertIn(E28S.DUCK_MARKER, e29_file)                        # the E28 block stays, E29 adds its own
        self.assertEqual(e29_file.count(E29S.E29_MARKER), 1)
        heads = [ln for ln in E29S.E29_PATCH.splitlines() if ln.startswith("@@")]
        self.assertEqual(len(heads), 12, heads)
        body = E29S.E29_PATCH.splitlines()
        rem = [ln for ln in body if ln.startswith("-") and not ln.startswith("---")]
        add = [ln for ln in body if ln.startswith("+") and not ln.startswith("+++")]
        self.assertEqual((len(rem), len(add)), (11, 214))
        self.assertTrue(all(int(h.split()[1].split(",")[0][1:]) > 3800 for h in heads))   # only the E28 block changes
        for name in ("e29_sched_patch.py", "e29_sched_tests.py"):
            compile((HERE / name).read_bytes().decode("utf-8"), name, "exec")


# ------------------------------------------------------------------ fake Duck agent (= e28_sched_tests.py + "partial")
class FakeResult:
    retryable_failure = False
    yielded_control = False
    step_executed = True


SCRIPTS = {}            # (task, "b<branch>") or (task, phase) or (task, phase[:1]) -> script
RUNLOG = []             # (event, task, branch, phase, monotonic) from the BranchSession.run wrapper


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
            self.script = (SCRIPTS.get((task_id, f"b{branch}")) or SCRIPTS.get((task_id, ph)) or SCRIPTS.get((task_id, ph[:1]))
                           or {"kind": "pass", "at": 5000})
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
            if sc["kind"] == "partial" and self.generated_tokens >= sc["at"]:
                return None                               # ended on its own after a partial check (-> FAIL, not B2)
            time.sleep(sc.get("sleep", sleep))
            self.generated_tokens += step
            self.total_tokens += step
            self.arc2_usage.append({"prompt": 1000, "completion": step, "cached": 0})
            if sc["kind"] == "pass" and self.generated_tokens >= sc["at"]:
                c = self._cand(True, sc.get("value", 1), [True, True])
                self.arc2_submissions.append(c)
                self.arc2_accepted = c
                self.arc2_progress["best"] = {"name": "transform", "k": 2, "n": 2, "acc": 1.0}
            elif sc["kind"] in ("cap", "partial") and not self.arc2_checked:
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


class RunLog:
    """Wrap BranchSession.run: logs start / end (task, branch, phase, time) and the peak concurrency per phase."""
    def __init__(self, M):
        self.M, self.run0, self.lock = M, M.BranchSession.run, threading.Lock()
        self.active, self.peak = {}, {}
        RUNLOG.clear()
        kit = self

        def wrapped(s, pause_at_tokens=None, tag=None):
            ph = (getattr(s, "e28_plan", None) or {}).get("phase")
            with kit.lock:
                kit.active[ph] = kit.active.get(ph, 0) + 1
                kit.peak[ph] = max(kit.peak.get(ph, 0), kit.active[ph])
                RUNLOG.append(("start", s.task_id, s.branch, ph, time.monotonic()))
            try:
                return kit.run0(s, pause_at_tokens=pause_at_tokens, tag=tag)
            finally:
                with kit.lock:
                    kit.active[ph] -= 1
                    RUNLOG.append(("end", s.task_id, s.branch, ph, time.monotonic()))
        M.BranchSession.run = wrapped

    def close(self):
        self.M.BranchSession.run = self.run0


def _nb5_lib():
    sys.path.insert(0, str(EXPORTS))
    try:
        import nb5_lib
    finally:
        sys.path.remove(str(EXPORTS))
    return nb5_lib


class ResKit:
    """= e28_sched_tests.ResKit: the real NB5L.Reservation + the N4 hooks + a BranchSession.run wrapper."""

    def __init__(self, M, cap, mode="projected", f=6, block=4096):
        L = _nb5_lib()
        self.M, self.L, self.res = M, L, L.Reservation(cap)
        lab, old, new = E29S.NOTEBOOK_EDITS[3]
        self.ns = {"NB5_RES": self.res, "NB5_ADMIT_MODE": mode, "NB5L": L, "MAX_LEN": 131072, "NB5_PROMPT_ALLOW": 8192,
                   "NB5_TOOL_ALLOW": 8192, "NB5_BLOCKS": {"b0": 40, "b1": 40, "start": 8, "f": f, "block": block},
                   "E12A": M, "time": time}
        exec(new[len(old):], self.ns)
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


# ------------------------------------------------------------------ normalisation for the identity checks
TIME_KEYS = {"t0", "t1", "sec", "publish_seq", "t"}
ROW_TIME = {"start_sec", "end_sec", "queued_sec", "first_tried_sec", "wait_sec"}
ROW_ORDER = {"worker_slot", "start_seq", "queues_at_start", "fit_fails", "backfill_starts_while_waiting"}
TOP_TIME = {"first_start_sec", "last_start_sec", "queue_empty_sec", "idle_worker_seconds", "idle_sec_by_reason",
            "backfill_log"}
TOP_ORDER = {"admission_checks", "fit_checks", "fit_fails", "backfill_starts", "forced_admits"}


def _clean(o, drop):
    if isinstance(o, dict):
        return {k: _clean(v, drop) for k, v in o.items() if k not in drop}
    if isinstance(o, list):
        return [_clean(v, drop) for v in o]
    return o


def _norm_recs(recs, strict):
    drop = TIME_KEYS | ROW_TIME | (set() if strict else ROW_ORDER)
    return {t: sorted((_clean(r, drop) for r in rs), key=lambda r: r.get("branch", 0)) for t, rs in recs.items()}


def _norm_e28(e, strict):
    drop_top = TOP_TIME | (set() if strict else TOP_ORDER)
    out = {k: v for k, v in e.items() if k not in drop_top and k != "attempts"}
    rows = [_clean(r, ROW_TIME | (set() if strict else ROW_ORDER)) for r in e["attempts"]]
    out["attempts"] = rows if strict else sorted(rows, key=lambda r: (r["task"], r["branch"]))
    return out


# ------------------------------------------------------------------ 2. E29_SUPPLY unset == the E28 post file
def _scen_single(tasks_n=8):
    tasks = _tasks(tasks_n)
    order = list(tasks)[::-1]
    SCRIPTS.clear()
    SCRIPTS.update({("t007", "A"): {"kind": "pass", "at": 50000}, ("t006", "A"): {"kind": "cap"},
                    ("t006", "B"): {"kind": "pass", "at": 30000, "value": 2},
                    ("t005", "A"): {"kind": "pass", "at": 20000}, ("t004", "A"): {"kind": "fail"},
                    ("t004", "B"): {"kind": "pass", "at": 40000, "value": 2},
                    ("t003", "A"): {"kind": "cap"}, ("t003", "B"): {"kind": "cap"},
                    ("t002", "A"): {"kind": "partial", "at": 8000}, ("t002", "B"): {"kind": "partial", "at": 3000},
                    ("t001", "A"): {"kind": "pass", "at": 10000}, ("t000", "A"): {"kind": "pass", "at": 20000}})
    for t in tasks:
        SCRIPTS.setdefault((t, "C"), {"kind": "pass", "at": 5000, "value": 3})
    return tasks, order


def _scen_multi(n=40, seed=7):
    rnd = random.Random(seed)
    tasks = _tasks(n)
    SCRIPTS.clear()
    for t_ in tasks:
        u = rnd.random()
        SCRIPTS[(t_, "A")] = ({"kind": "cap"} if u < 0.2 else {"kind": "fail"} if u < 0.3 else
                              {"kind": "partial", "at": 6000} if u < 0.4 else {"kind": "pass", "at": rnd.randrange(3, 60) * 1000})
        SCRIPTS[(t_, "B")] = {"kind": "pass", "at": 30000, "value": 2} if rnd.random() < 0.5 else {"kind": "partial", "at": 4000}
        SCRIPTS[(t_, "C")] = {"kind": "pass", "at": rnd.randrange(3, 40) * 1000, "value": 3}
    return tasks, list(tasks)


class TestSupplyOffIdentity(unittest.TestCase):
    def _both(self, scen, workers, strict, supply_env=None, kit=None, **kw):
        out = []
        for name in ("e28", "e29"):
            tasks, order = scen()
            k = kit(FIX[name]) if kit else None
            try:
                with Env(supply=False, **(supply_env or {})):
                    recs, el, sl = _solve(FIX[name], tasks, order, workers=workers, **kw)
                    sub = FIX[name].build_submission(tasks, recs)
            finally:
                if k:
                    k.close()
            out.append((_norm_recs(recs, strict), _norm_e28(sl["e28"], strict), sub,
                        {k_: v for k_, v in sl.items() if k_ not in ("durations", "gauges", "events", "e28")}))
        return out

    def _cmp(self, out, strict):
        (r0, e0, s0, l0), (r1, e1, s1, l1) = out
        self.assertEqual(r0, r1)
        self.assertEqual(e0, e1)
        self.assertEqual(s0, s1)
        if strict:
            self.assertEqual(l0, l1)
        self.assertNotIn("supply", e1)
        self.assertNotIn("S1", e1["started"])

    def test_single_worker_strict(self):
        out = self._both(_scen_single, 1, True)
        self._cmp(out, True)
        self.assertEqual(out[1][1]["started"], {"A": 8, "B1": 2, "B2": 2, "C": 8})

    def test_multi_worker(self):
        out = self._both(_scen_multi, 6, False)
        self._cmp(out, False)
        self.assertEqual(out[1][1]["left"], {"A": 0, "B": 0, "C": 0})

    def test_kv_reservation_backfill(self):
        def scen():
            tasks = _tasks(6)
            SCRIPTS.clear()
            SCRIPTS[("t000", "A")] = {"kind": "cap", "sleep": 0.0}
            SCRIPTS[("t000", "B")] = {"kind": "pass", "at": 20000, "value": 2}
            for t_ in list(tasks)[1:]:
                SCRIPTS[(t_, "A")] = {"kind": "pass", "at": 10000, "sleep": 0.01}
                SCRIPTS[(t_, "C")] = {"kind": "pass", "at": 5000, "sleep": 0.01, "value": 3}
            SCRIPTS[("t001", "A")] = {"kind": "pass", "at": 100000, "sleep": 0.02}
            return tasks, list(tasks)
        out = self._both(scen, 3, False, kit=lambda M: ResKit(M, cap=73))
        (r0, e0, s0, _), (r1, e1, s1, _) = out
        self.assertEqual((r0, s0, e0["started"], e0["slots"]), (r1, s1, e1["started"], e1["slots"]))
        self.assertEqual(e1["started"], {"A": 6, "B1": 0, "B2": 1, "C": 6})

    def test_time_cut_and_admission_window(self):
        def scen():
            tasks = _tasks(5)
            SCRIPTS.clear()
            SCRIPTS[("t002", "A")] = {"kind": "hang"}
            return tasks, list(tasks)
        out = self._both(scen, 3, False, deadline_sec=3.0, min_admit=0.5)
        # the hang's tokens and how many C attempts fit before the 3-s window closes are time-dependent: compare the
        # deterministic part (every A starts, the time-cut task gets no B / C, the other A's end TRAIN_PASS)
        (r0, e0, s0, _), (r1, e1, s1, _) = out
        for e_ in (e0, e1):
            self.assertEqual((e_["started"]["A"], e_["started"]["B1"], e_["started"]["B2"]), (5, 0, 0))
            self.assertEqual({t: c["kind"] for t, c in e_["a_class"].items()},
                             {"t000": "TRAIN_PASS", "t001": "TRAIN_PASS", "t002": "TIME", "t003": "TRAIN_PASS",
                              "t004": "TRAIN_PASS"})
            self.assertFalse(any(r["task"] == "t002" and r["phase"] != "A" for r in e_["attempts"]))
            self.assertNotIn("supply", e_)
        out = self._both(lambda: (_tasks(3), list(_tasks(3))), 2, True, deadline_sec=1.0, min_admit=5.0)
        self._cmp(out, True)
        self.assertEqual(out[1][1]["started"]["A"], 0)

    def test_e28_off_is_original(self):
        """E28_SCHED unset (E29_SUPPLY=1 alone, or nothing): the E29 file runs the original rev 6 scheduler."""
        tasks = _tasks(10)
        SCRIPTS.clear()
        for i, t in enumerate(tasks):
            SCRIPTS[(t, "A")] = {"kind": "pass", "at": (i + 1) * 7000} if i % 3 else {"kind": "cap"}
            SCRIPTS[(t, "C")] = {"kind": "pass", "at": 4000, "value": 3}
        for env in ({"supply": False, "e28": False}, {"supply": True, "e28": False}):
            out = []
            for name in ("orig", "e29"):
                with Env(**env):
                    recs, el, sl = _solve(FIX[name], tasks, list(tasks), workers=1, deadline_sec=60.0,
                                          max_generated_tokens=50000)
                    sub = FIX[name].build_submission(tasks, recs)
                out.append((_norm_recs(recs, True), {k: v for k, v in sl.items() if k not in ("durations", "gauges", "events")},
                            sub))
            self.assertEqual(out[0], out[1])
            self.assertNotIn("e28", out[1][1])

    def test_selection_identical_on_real_records(self):
        """E29_SUPPLY unset: build_submission / select_attempts_e28 of the E29 file == the E28 file (E28 on) and == the
        original (E28 off) on the real K100Q / W4 records. E29_SUPPLY=1: no S records -> the same answers and slots."""
        if not FIX["eval"]:
            self.skipTest("evaluation challenges not found")
        for phase_dir in (K100_PHASE, W4_PHASE):
            raw = _raw_records(phase_dir)
            recs = {}
            for (t, b), r in raw.items():
                recs.setdefault(t, []).append(r)
            tasks = {t: FIX["eval"][t] for t in recs}
            with Env(supply=False, e28=False):
                self.assertEqual(FIX["orig"].build_submission(tasks, recs), FIX["e29"].build_submission(tasks, recs))
            with Env(supply=False):
                a = FIX["e28"].build_submission(tasks, recs)
                ta = FIX["e28"].e28_slot_table(tasks, recs)
                self.assertEqual(a, FIX["e29"].build_submission(tasks, recs))
                self.assertEqual(ta, FIX["e29"].e28_slot_table(tasks, recs))
            with Env(supply=True):
                self.assertEqual(a, FIX["e29"].build_submission(tasks, recs))
                self.assertEqual(ta, FIX["e29"].e28_slot_table(tasks, recs))
                for t in tasks:
                    self.assertEqual(FIX["e29"].select_attempts_e29(tasks[t], recs[t]),
                                     FIX["e28"].select_attempts_e28(tasks[t], recs[t]))


# ------------------------------------------------------------------ 3. the supply rules (E29_SUPPLY=1)
def _rows(sl):
    return sorted(sl["e28"]["attempts"], key=lambda r: r["start_seq"])


class TestSupply(unittest.TestCase):
    def setUp(self):
        self.M = FIX["e29"]

    def _common(self, sl):
        e = sl["e28"]
        for k in ("started", "done", "queued"):
            self.assertIn("S1", e[k])
            self.assertIn("S2", e[k])
        self.assertEqual((sl["worker_errors"], sl["setup_failed"], sl.get("stage2_errors", 0)), (0, 0, 0))
        self.assertIn("no_supply", e["idle_sec_by_reason"])
        self.assertEqual(e["idle_sec_by_reason"]["no_runnable"], 0.0)
        sup = e["supply"]
        self.assertEqual(set(sup), {"enabled", "max_attempts", "started", "done", "first_start_sec", "by_task",
                                    "attempts_by_task", "train_pass_from_supply", "slots_from_supply", "s_class"})
        self.assertTrue(sup["enabled"])
        for r in e["attempts"]:
            if r["phase"] in ("S1", "S2"):
                self.assertEqual(r["queues_at_start"], [0, 0, 0], r)     # supply only with A / B / C all empty
                self.assertGreaterEqual(r["branch"], 3)
        return e

    def test_s1_when_queues_empty_single_worker(self):
        """1 worker, exact order: t001 never passes (A / B1 / C partial) -> S1 branches 3, 4 (cap 5 attempts), view
        b1_view for s_index 1, "facts" for 2; t000 passes at A -> no supply."""
        tasks = _tasks(2)
        SCRIPTS.clear()
        SCRIPTS.update({("t001", "A"): {"kind": "partial", "at": 4000}, ("t001", "B"): {"kind": "partial", "at": 4000},
                        ("t001", "C"): {"kind": "partial", "at": 4000}, ("t001", "S"): {"kind": "partial", "at": 6000,
                                                                                         "value": 7}})
        with Env(E29_SUPPLY_MAX_ATTEMPTS=5):
            recs, el, sl = _solve(self.M, tasks, list(tasks), workers=1, b1_view="indexed")
        e = self._common(sl)
        seq = [(r["phase"], r["task"], r["branch"]) for r in _rows(sl)]
        self.assertEqual(seq, [("A", "t000", 0), ("A", "t001", 0), ("B1", "t001", 2), ("C", "t001", 1), ("C", "t000", 1),
                               ("S1", "t001", 3), ("S1", "t001", 4)])          # C1 (no train pass) before C2
        by = {r["branch"]: r for r in e["attempts"] if r["task"] == "t001"}
        self.assertEqual([(by[b]["s_index"], by[b]["view_plan"], by[b]["s_reason"]) for b in (3, 4)],
                         [(1, "indexed", "no_train_pass"), (2, "facts", "no_train_pass")])
        self.assertEqual({b: recs["t001"][[r["branch"] for r in recs["t001"]].index(b)]["view"] for b in (3, 4)},
                         {3: "indexed", 4: "facts"})
        for b in (3, 4):
            self.assertEqual((by[b]["budget"], by[b]["window_plan"], by[b]["agent_budget"], by[b]["session_cap"],
                              by[b]["request_timeout"], by[b]["kind"]), (104000, None, 104000, 104000, 30.0, "FAIL"))
        self.assertEqual(e["started"], {"A": 2, "B1": 1, "B2": 0, "C": 2, "S1": 2, "S2": 0})
        self.assertEqual(e["done"], e["started"])
        self.assertEqual(e["queued"]["S1"], 2)
        self.assertEqual(e["supply"]["by_task"], {"t001": 2})
        self.assertEqual(e["supply"]["attempts_by_task"], {"t000": 2, "t001": 5})
        self.assertEqual(e["supply"]["max_attempts"], 5)
        self.assertIsNotNone(e["supply"]["first_start_sec"])
        self.assertEqual(e["supply"]["train_pass_from_supply"], [])

    def test_s2_only_after_s1_exhausted(self):
        """t000: every attempt cap-cut (A / B2 / C) -> S2; t001: A / B1 / C partial -> S1. One worker: every S1 start
        comes before any S2 start; S2 = 120,000 tokens, window 130,560, the B2 request timeout."""
        tasks = _tasks(2)
        SCRIPTS.clear()
        SCRIPTS.update({("t000", "A"): {"kind": "cap", "sleep": 0.0}, ("t000", "B"): {"kind": "cap", "sleep": 0.0},
                        ("t000", "C"): {"kind": "cap", "sleep": 0.0}, ("t000", "S"): {"kind": "cap", "sleep": 0.0},
                        ("t001", "A"): {"kind": "partial", "at": 3000}, ("t001", "B"): {"kind": "partial", "at": 3000},
                        ("t001", "C"): {"kind": "partial", "at": 3000}, ("t001", "S"): {"kind": "partial", "at": 3000}})
        with Env(E29_SUPPLY_MAX_ATTEMPTS=6):
            recs, el, sl = _solve(self.M, tasks, list(tasks), workers=1, analyzer_timeout=7200.0)
        e = self._common(sl)
        seq = [(r["phase"], r["task"], r["branch"]) for r in _rows(sl)]
        self.assertEqual(seq, [("A", "t000", 0), ("A", "t001", 0), ("B2", "t000", 2), ("B1", "t001", 2),
                               ("C", "t000", 1), ("C", "t001", 1),
                               ("S1", "t001", 3), ("S1", "t001", 4), ("S1", "t001", 5),
                               ("S2", "t000", 3), ("S2", "t000", 4), ("S2", "t000", 5)])
        self.assertEqual(e["started"], {"A": 2, "B1": 1, "B2": 1, "C": 2, "S1": 3, "S2": 3})
        for r in e["attempts"]:
            if r["phase"] == "S2":
                self.assertEqual((r["budget"], r["window_plan"], r["window"], r["agent_budget"], r["session_cap"],
                                  r["tokens"], r["kind"], r["s_reason"]),
                                 (120000, 130560, 130560, 120000, 120000, 120000, "B2", "all_cap_cut"))
                self.assertAlmostEqual(r["request_timeout"], 7200.0 * 120000 / 104000, places=3)
            if r["phase"] == "S1":
                self.assertEqual((r["budget"], r["window_plan"], r["window"]), (104000, None, 130560))
        self.assertEqual(e["supply"]["started"], {"S1": 3, "S2": 3})

    def test_abc_priority_over_supply(self):
        """2 workers: t000 runs out of A / B / C fast and feeds S1 attempts (~0.2 s each) while t001's long attempt 1
        runs; when it ends its C item is queued and taken at the next admission check, before any further supply."""
        tasks = _tasks(2)
        SCRIPTS.clear()
        SCRIPTS.update({("t000", "A"): {"kind": "fail"}, ("t000", "B"): {"kind": "fail"}, ("t000", "C"): {"kind": "fail"},
                        ("t000", "S"): {"kind": "partial", "at": 20000, "sleep": 0.01},
                        ("t001", "A"): {"kind": "pass", "at": 100000, "sleep": 0.012},
                        ("t001", "C"): {"kind": "pass", "at": 5000, "value": 3}})
        M = self.M
        takes, push = [], []                  # both logged under the scheduler lock: an exact order
        take0, ckey0 = M._e29_take, M.e28_c_key

        def take_logged(*a, **k):
            takes.append(time.monotonic())
            return take0(*a, **k)

        def ckey_logged(full, tokens, idx):
            if idx == 1:                      # t001's C item is queued (push_c in route)
                push.append(time.monotonic())
            return ckey0(full, tokens, idx)
        M._e29_take, M.e28_c_key = take_logged, ckey_logged
        try:
            with Env(E29_SUPPLY_MAX_ATTEMPTS=14):
                recs, el, sl = _solve(M, tasks, list(tasks), workers=2)
        finally:
            M._e29_take, M.e28_c_key = take0, ckey0
        e = self._common(sl)
        self.assertEqual(len(push), 1)
        k = sum(1 for x in takes if x < push[0])          # supply items taken before t001's C was queued
        s_rows = [r for r in _rows(sl) if r["phase"] == "S1"]
        crow = next(r for r in e["attempts"] if r["task"] == "t001" and r["phase"] == "C")
        self.assertGreater(k, 1)                              # supply really ran while t001's A ran
        self.assertGreater(len(s_rows), k)                    # and continued after the C item
        self.assertTrue(all(r["start_seq"] < crow["start_seq"] for r in s_rows[:k]))
        self.assertTrue(all(r["start_seq"] > crow["start_seq"] for r in s_rows[k:]))   # C first at the next admission
        self.assertEqual(crow["queues_at_start"], [0, 0, 0])
        self.assertLess(crow["wait_sec"], 1.0)
        self.assertEqual(e["started"]["S1"], 14 - 3)
        self.assertEqual(e["supply"]["attempts_by_task"]["t000"], 14)

    def test_per_task_cap_default_and_env(self):
        tasks = _tasks(1)
        SCRIPTS.clear()
        SCRIPTS.update({("t000", "A"): {"kind": "fail"}, ("t000", "B"): {"kind": "fail"}, ("t000", "C"): {"kind": "fail"},
                        ("t000", "S"): {"kind": "fail"}})
        with Env():
            recs, el, sl = _solve(self.M, tasks, list(tasks), workers=3)
        e = self._common(sl)
        self.assertEqual(e["supply"]["max_attempts"], 16)
        self.assertEqual(len(e["attempts"]), 16)
        self.assertEqual(sorted(r["branch"] for r in e["attempts"]), list(range(16)))
        self.assertEqual(e["started"]["S1"], 13)
        with Env(E29_SUPPLY_MAX_ATTEMPTS=4):
            recs, el, sl = _solve(self.M, tasks, list(tasks), workers=3)
        e = self._common(sl)
        self.assertEqual(sorted(r["branch"] for r in e["attempts"]), [0, 1, 2, 3])
        self.assertEqual(len(recs["t000"]), 4)

    def test_stop_after_train_pass(self):
        """S1 branch 3 fails, branch 4 passes -> no further supply; attempt_1 = the S4 answer (full beats non-full)."""
        tasks = _tasks(2)
        SCRIPTS.clear()
        SCRIPTS.update({("t000", "A"): {"kind": "partial", "at": 3000}, ("t000", "B"): {"kind": "fail"},
                        ("t000", "C"): {"kind": "partial", "at": 3000, "value": 3, "exact": [False, False]},
                        ("t000", "b3"): {"kind": "fail"}, ("t000", "b4"): {"kind": "pass", "at": 4000, "value": 8},
                        ("t000", "S"): {"kind": "fail"}})
        with Env():
            recs, el, sl = _solve(self.M, tasks, list(tasks), workers=1)
            sub = self.M.build_submission(tasks, recs)
        e = self._common(sl)
        rows = [r for r in e["attempts"] if r["task"] == "t000"]
        self.assertEqual(sorted(r["branch"] for r in rows), [0, 1, 2, 3, 4])
        self.assertEqual(e["supply"]["train_pass_from_supply"], ["t000"])
        self.assertEqual(e["slots"]["t000"][0]["attempt_1"], "S4")
        self.assertEqual(e["slots"]["t000"][0]["attempt_2"], "C")             # C partial vs displaced A partial: C
        self.assertEqual(sub["t000"][0], {"attempt_1": _grid(8), "attempt_2": _grid(3)})
        by = {r["branch"]: r for r in rows}
        self.assertEqual((by[4]["answer_slots"], by[1]["answer_slots"], by[0]["answer_slots"], by[3]["answer_slots"]),
                         (["attempt_1"], ["attempt_2"], [None], [None]))
        self.assertEqual(e["supply"]["slots_from_supply"], 1)
        self.assertEqual(e["supply"]["s_class"]["t000"][1]["kind"], "TRAIN_PASS")
        self.assertNotIn("t001", e["supply"]["by_task"])                      # t001 passed at A

    def test_deadline_guard(self):
        tasks = _tasks(2)
        SCRIPTS.clear()
        for t in tasks:
            SCRIPTS.update({(t, "A"): {"kind": "fail"}, (t, "B"): {"kind": "fail"}, (t, "C"): {"kind": "fail"},
                            (t, "S"): {"kind": "partial", "at": 10000, "sleep": 0.01}})
        with Env(E29_SUPPLY_MAX_ATTEMPTS=100000):
            # deadline 8 s, min_admit 6.5 s: the window closes at 1.5 s (the live TIME check is deadline - 5 s)
            recs, el, sl = _solve(self.M, tasks, list(tasks), workers=2, deadline_sec=8.0, min_admit=6.5)
        e = self._common(sl)
        self.assertGreater(e["started"]["S1"], 2)
        self.assertTrue(all(r["start_sec"] <= 8.0 - 6.5 + 0.15 for r in e["attempts"]), [r["start_sec"] for r in e["attempts"]])
        self.assertLess(el, 6.0)                       # infinite supply, yet no start after the window: ends early
        with Env():
            recs, el, sl = _solve(self.M, tasks, list(tasks), workers=2, deadline_sec=1.0, min_admit=5.0)
        self.assertEqual(sum(sl["e28"]["started"].values()), 0)              # less than min_admit_sec left: nothing

    def test_no_idle_while_supply_exists(self):
        """6 tasks, 6 workers, supply never runs out before the admission window (6 s) closes: no worker waits."""
        tasks = _tasks(6)
        SCRIPTS.clear()
        for t in tasks:
            SCRIPTS.update({(t, "A"): {"kind": "partial", "at": 20000, "sleep": 0.01},
                            (t, "B"): {"kind": "partial", "at": 3000, "sleep": 0.01},
                            (t, "C"): {"kind": "partial", "at": 3000, "sleep": 0.01},
                            (t, "S"): {"kind": "partial", "at": 4000, "sleep": 0.01}})
        log = RunLog(self.M)
        try:
            with Env(E29_SUPPLY_MAX_ATTEMPTS=100000):
                recs, el, sl = _solve(self.M, tasks, list(tasks), workers=6, deadline_sec=20.0, min_admit=14.0)
        finally:
            log.close()
        e = self._common(sl)
        diag = json.dumps({"el": round(el, 2), "started": e["started"], "idle": e["idle_sec_by_reason"], "a": e["a_class"],
                           "rows": [(r["phase"], r["task"], r["start_sec"], r["end_sec"], r["kind"]) for r in _rows(sl)][:14]})
        self.assertGreater(e["started"]["S1"], 6, diag)
        idle = e["idle_sec_by_reason"]
        self.assertEqual((idle["nothing_fits"], idle["kv_gate"], idle["no_runnable"]), (0.0, 0.0, 0.0))
        self.assertLess(idle["no_supply"], 0.05, idle)
        self.assertEqual(e["idle_worker_seconds"], 0.0)
        # busy fraction while the window was open: 6 sessions running at almost every moment
        starts = sorted(t for ev, *_x, t in RUNLOG if ev == "start")
        self.assertEqual(log.peak.get("S1"), 6)                  # all 6 workers busy on supply at once
        self.assertGreater(len(starts), 6 * 3 + 6)

    def test_concurrent_supply_one_task(self):
        """1 task, 4 workers: while its B1 runs, the three other workers run S1 attempts of the same task at once."""
        tasks = _tasks(1)
        SCRIPTS.clear()
        SCRIPTS.update({("t000", "A"): {"kind": "partial", "at": 3000, "sleep": 0.01},
                        ("t000", "B"): {"kind": "partial", "at": 30000, "sleep": 0.01},
                        ("t000", "C"): {"kind": "partial", "at": 30000, "sleep": 0.01},
                        ("t000", "S"): {"kind": "partial", "at": 20000, "sleep": 0.01}})
        log = RunLog(self.M)
        try:
            with Env(E29_SUPPLY_MAX_ATTEMPTS=8):
                recs, el, sl = _solve(self.M, tasks, list(tasks), workers=4)
        finally:
            log.close()
        e = self._common(sl)
        self.assertEqual(log.peak.get("S1"), 3)
        self.assertEqual(log.peak.get("B1"), 1)
        self.assertEqual(sorted(r["branch"] for r in e["attempts"]), list(range(8)))
        self.assertEqual(e["started"], {"A": 1, "B1": 1, "B2": 0, "C": 1, "S1": 5, "S2": 0})
        self.assertGreater(e["idle_sec_by_reason"]["no_supply"], 0.0)      # waits while attempt 1 ran (nothing eligible)

    def test_fewest_attempts_first(self):
        """Two eligible tasks, one worker: supply alternates by attempts started (then task order)."""
        tasks = _tasks(2)
        SCRIPTS.clear()
        for t in tasks:
            SCRIPTS.update({(t, "A"): {"kind": "fail"}, (t, "B"): {"kind": "fail"}, (t, "C"): {"kind": "fail"},
                            (t, "S"): {"kind": "fail"}})
        SCRIPTS[("t001", "A")] = {"kind": "pass", "at": 2000}
        SCRIPTS[("t001", "C")] = {"kind": "fail"}
        tasks2 = dict(tasks, t002=tasks["t000"])
        SCRIPTS.update({("t002", "A"): {"kind": "fail"}, ("t002", "B"): {"kind": "fail"}, ("t002", "C"): {"kind": "fail"},
                        ("t002", "S"): {"kind": "fail"}})
        with Env(E29_SUPPLY_MAX_ATTEMPTS=6):
            recs, el, sl = _solve(self.M, tasks2, list(tasks2), workers=1)
        e = self._common(sl)
        s = [(r["task"], r["branch"]) for r in _rows(sl) if r["phase"] == "S1"]
        self.assertEqual(s, [("t000", 3), ("t002", 3), ("t000", 4), ("t002", 4), ("t000", 5), ("t002", 5)])
        self.assertNotIn("t001", e["supply"]["by_task"])

    def test_kv_reservation_with_supply(self):
        """The real Reservation (projected, 73 blocks): supply goes through the same fit hook; nothing leaks."""
        tasks = _tasks(3)
        SCRIPTS.clear()
        for t in tasks:
            SCRIPTS.update({(t, "A"): {"kind": "cap", "sleep": 0.0}, (t, "B"): {"kind": "cap", "sleep": 0.0},
                            (t, "C"): {"kind": "partial", "at": 3000}, (t, "S"): {"kind": "partial", "at": 5000}})
        kit = ResKit(self.M, cap=73)
        try:
            with Env(E29_SUPPLY_MAX_ATTEMPTS=6):
                recs, el, sl = _solve(self.M, tasks, list(tasks), workers=3)
        finally:
            kit.close()
        e = self._common(sl)
        self.assertEqual(e["started"]["S1"] + e["started"]["S2"], 9)     # S2 while C is still to come, S1 after C
        self.assertEqual(e["started"]["B2"], 3)
        self.assertEqual((kit.res.used, kit.res.held), (0, {}))
        self.assertEqual(kit.res.stats["e28_try_grants"] + e["forced_admits"], sum(e["started"].values()))

    def test_slots_mode_no_idle(self):
        """Slots mode (the notebooks'): Reservation capacity = workers, the KV gate = the reservation fraction."""
        tasks = _tasks(8)
        SCRIPTS.clear()
        for t in tasks:
            SCRIPTS.update({(t, "A"): {"kind": "partial", "at": 4000, "sleep": 0.005},
                            (t, "B"): {"kind": "partial", "at": 4000, "sleep": 0.005},
                            (t, "C"): {"kind": "partial", "at": 4000, "sleep": 0.005},
                            (t, "S"): {"kind": "partial", "at": 4000, "sleep": 0.005}})
        W = 4
        kit = ResKit(self.M, cap=W, mode="slots")
        probe = lambda: {"kv_usage": kit.res.frac()}
        try:
            with Env(E29_SUPPLY_MAX_ATTEMPTS=6):
                recs, el, sl = _solve(self.M, tasks, list(tasks), workers=W, probe=probe, kv_poll=5.0, kv_admit_limit=1.0)
        finally:
            kit.close()
        e = self._common(sl)
        self.assertEqual(e["started"]["S1"], 8 * 3)
        self.assertEqual((e["fit_fails"], e["idle_sec_by_reason"]["nothing_fits"]), (0, 0.0))
        self.assertEqual((kit.res.used, kit.res.held), (0, {}))


# ------------------------------------------------------------------ 4. the E29 answer rule (synthetic records)
class TestSlots(unittest.TestCase):
    def setUp(self):
        self.M = FIX["e29"]
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
        out = self.M.select_attempts_e29(self.task, recs, _slots_out=rows)
        return (out[0]["attempt_1"][0][0], out[0]["attempt_2"][0][0], rows[0]["attempt_1"], rows[0]["attempt_2"])

    def test_rules(self):
        r = self.r
        P, N = [True, False], [False, False]
        self.assertEqual(self.sel([r(0, "A", part=(5, P)), r(2, "B1", part=(6, N)), r(3, "S1", acc=7), r(1, "C", acc=3)]),
                         (7, 3, "S3", "C"))                                          # S full beats non-full A / B
        self.assertEqual(self.sel([r(0, "A", part=(5, P)), r(3, "S1", part=(6, P))]), (5, 6, "A", "S3"))   # tie keeps A
        self.assertEqual(self.sel([r(0, "A", part=(5, N)), r(3, "S1", part=(6, P))]), (6, 5, "S3", "A"))   # higher frac
        self.assertEqual(self.sel([r(0, "A", part=(5, P)), r(2, "B2", part=(6, P)), r(3, "S1", part=(7, P)),
                                   r(4, "S2", part=(8, [True, True]))]), (8, 5, "S4", "A"))   # 1.0 non-full? see below
        # the displaced = the best of the rest by (full, frac, earlier): A full stays, S3 full displaced, C absent
        self.assertEqual(self.sel([r(0, "A", acc=1), r(3, "S1", acc=7)]), (1, 7, "A", "S3"))
        # C not full and the displaced full -> the displaced first; C full -> C first; C not full, displaced not full -> C
        self.assertEqual(self.sel([r(0, "A", acc=1), r(3, "S1", acc=7), r(1, "C", part=(3, P))]), (1, 7, "A", "S3"))
        self.assertEqual(self.sel([r(0, "A", acc=1), r(3, "S1", acc=7), r(1, "C", acc=3)]), (1, 3, "A", "C"))
        self.assertEqual(self.sel([r(0, "A", part=(5, P)), r(3, "S1", part=(7, N)), r(1, "C", part=(3, N))]), (5, 3, "A", "C"))
        # two full S: the earlier branch wins attempt_1; the later full one is displaced and beats a non-full C
        self.assertEqual(self.sel([r(0, "A", part=(5, N)), r(4, "S1", acc=8), r(3, "S1", acc=7), r(1, "C", part=(3, P))]),
                         (7, 8, "S3", "S4"))
        # the displaced best of several: (full, frac, earlier)
        self.assertEqual(self.sel([r(0, "A", acc=1), r(3, "S1", part=(6, P)), r(4, "S2", part=(7, P)), r(5, "S1", acc=9)]),
                         (1, 9, "A", "S5"))
        self.assertEqual(self.sel([r(0, "A", acc=1), r(3, "S1", part=(6, N)), r(4, "S2", part=(7, P))]), (1, 7, "A", "S4"))
        self.assertEqual(self.sel([r(0, "A", acc=1), r(3, "S1", part=(6, P)), r(4, "S2", part=(7, P))]), (1, 6, "A", "S3"))
        # displaced equal to attempt_1 -> the next distinct candidate (C)
        self.assertEqual(self.sel([r(0, "A", acc=1), r(3, "S1", acc=1), r(1, "C", part=(3, P))]), (1, 3, "A", "C"))
        # records without e28 fields: branch >= 3 = S
        self.assertEqual(self.sel([r(0, part=(5, P)), r(3, acc=7)]), (7, 5, "S3", "A"))
        self.assertEqual(self.sel([]), (9, 9, "r5", "same"))

    def test_partial_one_point_zero_is_not_full(self):
        """A non-accepted candidate with every train pair exact (frac 1.0, not host-accepted) is not 'full'."""
        rows = []
        self.M.select_attempts_e29(self.task, [self.r(0, "A", part=(5, [True, False])),
                                               self.r(4, "S2", part=(8, [True, True]))], _slots_out=rows)
        self.assertEqual((rows[0]["attempt_1"], rows[0]["a1_train"]["full"], rows[0]["s_full"]), ("S4", False, {"S4": False}))

    def test_no_s_records_equals_e28_rule(self):
        """Every E28 synthetic case (e28_sched_tests TestSlots) and the C-not-full / displaced-full exception, which is
        inert on E28-scheduled records (a full A never gets a B; a full B displaces a non-full A)."""
        r = self.r
        P, N = [True, False], [False, False]
        cases = [[r(0, acc=1), r(1, acc=3)], [r(0, acc=1), r(1, acc=1)],
                 [r(0, part=(5, P)), r(2, "B2", acc=2), r(1, acc=3)], [r(0, part=(5, P)), r(2, "B2", acc=2)],
                 [r(0, part=(5, P)), r(2, "B1", part=(6, N))], [r(0, part=(5, N)), r(2, "B1", part=(6, P))],
                 [r(0, part=(5, P)), r(2, "B1", part=(6, P))], [r(0, part=(5, P)), r(1, part=(7, N))],
                 [r(1, acc=3)], [], [r(0, part=(5, P)), r(2, acc=2), r(1, acc=3)],
                 [r(0, part=(5, P)), r(2, "B2", acc=2), r(1, part=(3, P))],
                 [r(0, acc=1), r(1, part=(3, P))], [r(0, part=(5, N)), r(2, "B1", part=(6, P)), r(1, part=(3, N))]]
        for recs in cases:
            a, b = [], []
            with Env(supply=False):
                ea = self.M.select_attempts_e28(self.task, recs, _slots_out=a)
            with Env(supply=True):
                eb = self.M.select_attempts_e28(self.task, recs, _slots_out=b)
            self.assertEqual((ea, a), (eb, b), recs)
            self.assertEqual(ea, FIX["e28"].select_attempts_e28(self.task, recs))

    def test_fake_run_records_equal_e28(self):
        """Records of a fake E28 run (no S attempts): the E29 rule == the E28 rule for every task."""
        tasks, order = _scen_single()
        with Env(supply=False):
            recs, el, sl = _solve(FIX["e29"], tasks, order, workers=1)
        for t in tasks:
            a, b = [], []
            ea = FIX["e28"].select_attempts_e28(tasks[t], recs[t], _slots_out=a)
            eb = self.M.select_attempts_e29(tasks[t], recs[t], _slots_out=b)
            self.assertEqual((ea, a), (eb, b), t)


# ------------------------------------------------------------------ 5. the real Arc2ToolAgent for branches >= 3
class TestAgentHook(unittest.TestCase):
    def test_branches_ge3_budget_window_prompt(self):
        if not FIX["eval"]:
            self.skipTest("evaluation challenges not found")
        task = FIX["eval"]["16b78196"]
        env = {"E12_ARC2_REV6": "1", "E12_B_MODE": "1", "E12_B_ATTEMPT_TOKENS": "104000", "E12_B_ATTEMPT_TOKENS_B1": "104000",
               "E12_DUCK_CONTEXT_WINDOW": "130560"}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            mk = lambda mod, b, view="facts": mod.make_agent_class()(task=task, task_id="16b78196", branch=b, view=view,
                                                                     model="fake", timeout=30,
                                                                     base_url="http://127.0.0.1:9/v1", provider="vllm",
                                                                     save_request_logs=False, dispatch_index=300001)
            M = FIX["e29"]
            for b in (0, 1, 2, 3, 4):
                with Env(supply=False, e28=False):
                    a0, a1 = mk(FIX["orig"], b), mk(M, b)       # E28 off: identical to the original
                self.assertEqual((a0._system_prompt, a0._arc2_b_budget, a0._arc2_window),
                                 (a1._system_prompt, a1._arc2_b_budget, a1._arc2_window))
            with Env():
                M.E28_PLANS.clear()
                M.E28_PLANS[("16b78196", 3)] = {"phase": "S1", "budget": 104000, "window": None, "branch": 3}
                M.E28_PLANS[("16b78196", 4)] = {"phase": "S2", "budget": 120000, "window": 130560, "branch": 4}
                M.E28_PLANS[("16b78196", 5)] = {"phase": "S1", "budget": 104000, "window": None, "branch": 5}
                s1, s2, s1i = mk(M, 3), mk(M, 4), mk(M, 5, view="indexed")
                self.assertEqual((s1._arc2_b_budget, s1._arc2_window), (104000, 130560))
                self.assertIn("about 104,000 generated tokens", s1._system_prompt)
                self.assertEqual((s2._arc2_b_budget, s2._arc2_window), (120000, 130560))
                self.assertIn("about 120,000 generated tokens", s2._system_prompt)
                self.assertEqual(s2._arc2_apply_cap(), 120000 - 4096)
                self.assertEqual((s1._e28_plan["phase"], s2._e28_plan["phase"]), ("S1", "S2"))
                self.assertEqual((s1.arc2_view, s1i.arc2_view, s1i._arc2_b_budget), ("facts", "indexed", 104000))
                M.E28_PLANS.clear()
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


if __name__ == "__main__":
    prog = unittest.main(verbosity=2 if "-v" in sys.argv else 1, argv=[a for a in sys.argv if a != "-v"], exit=False)
    res = prog.result
    print(f"E29 SCHED tests: {res.testsRun} run, {len(res.failures)} failures, {len(res.errors)} errors, "
          f"{len(res.skipped)} skipped")
    sys.exit(0 if res.wasSuccessful() else 1)
