"""CPU tests for e14_measure decision and deadline code (GPT audits 17-19, Notes/20). A tiny stub HTTP server stands in
for SGLang: /generate can hang forever, /abort_request can fail, /metrics reports the hung requests as running."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import e14_measure as M


class _Stub:
    def __init__(self, abort_ok=False):
        self.hits = {"/generate": 0, "/abort_request": 0, "/flush_cache": 0}
        self.hung = 0
        self.abort_ok = abort_ok
        self.lock = threading.Lock()          # ~28 handler threads update the counters (review 22)
        self.flush_busy = True
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body, code=200, ctype="application/json"):
                b = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/metrics":
                    return self._send(f'sglang:num_running_reqs{{tp_rank="0"}} {stub.hung}\n'
                                      'sglang:num_queue_reqs 0\n', ctype="text/plain")
                self._send("{}", 404)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                self.rfile.read(n)
                with stub.lock:
                    stub.hits[self.path] = stub.hits.get(self.path, 0) + 1
                if self.path == "/generate":           # hangs: ignores aborts, like a stuck server
                    with stub.lock:
                        stub.hung += 1
                    time.sleep(120)
                    return
                if self.path.startswith("/flush_cache"):   # 0.5.21: 400 + plain text while requests run
                    if stub.flush_busy:
                        return self._send("Flush cache failed.\n", 400, "text/plain")
                    return self._send("Cache flushed.\n", 200, "text/plain")
                if self.path == "/abort_request":
                    if stub.abort_ok:
                        return self._send("{}")
                    time.sleep(30)                     # an abort that never answers in time
                    return self._send("{}", 500)
                self._send("{}", 404)

        ThreadingHTTPServer.request_queue_size = 64          # the default backlog of 5 dropped connects under load
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def stub(tmp_path):
    s = _Stub()
    yield s
    s.close()


def _ctx(stub, tmp_path, left):
    return M.Ctx(stub.url, lambda: "", tmp_path, deadline=time.perf_counter() + left, log=None)


# ------------------------------------------------------------------------------------------- deadlines and cleanup
def test_post_json_past_deadline_sends_nothing(stub, tmp_path):
    ctx = _ctx(stub, tmp_path, -2.0)
    with pytest.raises(TimeoutError):
        M.post_json(ctx, "/generate", {"x": 1})
    time.sleep(0.2)
    assert stub.hits["/generate"] == 0             # the old 5 s floor would have sent it


def test_post_json_timeout_never_exceeds_the_deadline(stub, tmp_path):
    ctx = _ctx(stub, tmp_path, 1.0)
    t0 = time.monotonic()
    with pytest.raises(Exception):
        M.post_json(ctx, "/generate", {"x": 1}, timeout=600)
    # 1 s left -> read timeout 1 s (the old rule gave max(5, left + 5) = 6 s); 4 s leaves room for this host's
    # occasional connect delay (measured 1.03 s outside pytest, once 2.8 s inside)
    assert time.monotonic() - t0 < 4.0


def test_cleanup_of_14_hung_requests_with_failing_aborts_is_bounded(stub, tmp_path):
    """Audit 19 P0-B: 14 generators that never return + aborts that never answer: everything ends inside ONE bound
    (the old code: 14 x join(60) = 840 s), and the server is reported NOT idle so the next arm is not measured on it."""
    ctx = _ctx(stub, tmp_path, 600.0)
    gens = [M.Gen(ctx, f"r{i}", [1, 2, 3], 16) for i in range(14)]
    for g in gens:
        g.start()
    end = time.monotonic() + 10
    while stub.hung < 14 and time.monotonic() < end:     # all 14 really hang before the cleanup starts
        time.sleep(0.05)
    assert stub.hung == 14
    t0 = time.monotonic()
    r = M.cleanup(ctx, gens, budget=3.0)
    dt = time.monotonic() - t0
    assert dt < 4.5, dt
    assert r["aborted"] == 14 and r["alive_threads"] == 14 and r["server_idle"] is False
    assert stub.hits["/abort_request"] == 14          # aborts sent at once, not one after another


def test_join_all_shares_one_deadline():
    ths = [threading.Thread(target=time.sleep, args=(30,), daemon=True) for _ in range(14)]
    for t in ths:
        t.start()
    t0 = time.monotonic()
    assert M.join_all(ths, 1.0) == 14
    assert time.monotonic() - t0 < 1.5


def test_join_all_skips_threads_never_started():
    ths = [threading.Thread(target=time.sleep, args=(0.01,), daemon=True) for _ in range(3)]
    ths[0].start()
    assert M.join_all(ths, 1.0) == 0


# ------------------------------------------------------------------------------------------- correctness gate
def _row(ids, margins=None, error=None):
    return {"ids": ids, "margins": margins if margins is not None else [2.4] * len(ids), "error": error}


def _rows(n=12, length=20, flip_at=None, flip_margin=2.4, error_at=None, drop_margins=False, seed=0):
    out = []
    for i in range(n):
        ids = [(seed * 0 + i * 7 + k) % 90 for k in range(length)]
        mg = [2.4] * length
        if flip_at is not None and i in flip_at:
            ids = ids[:5] + [99] + ids[6:]
            mg = mg[:5] + [flip_margin] + mg[6:]
        if drop_margins:
            mg = []
        out.append(_row(ids, mg, "boom" if error_at is not None and i in error_at else None))
    return out


def test_gate_stable_aa_identical_arm_passes():
    base, b2, arm = _rows(), _rows(), _rows()
    v = M.parity_ok(M.compare_passes(base, arm, "arm"), M.compare_passes(base, b2, "aa"), identity="require")
    assert v["ok"] and v["verdict"] == "PASS"


def test_gate_large_margin_flip_fails():
    base, b2, arm = _rows(), _rows(), _rows(flip_at={3})
    v = M.parity_ok(M.compare_passes(base, arm, "arm"), M.compare_passes(base, b2, "aa"))
    assert not v["ok"] and v["verdict"] == "FAIL" and v["large_margin_flips"] == 1


def test_gate_low_margin_benign_divergence_passes_with_identity_ignored():
    base, b2 = _rows(), _rows()
    arm = _rows(flip_at={1, 2, 3, 4}, flip_margin=0.2)
    for r in arm:                                        # both runs near a tie at the divergence
        r["margins"][5] = 0.2
    for r in base:
        r["margins"][5] = 0.2
    v = M.parity_ok(M.compare_passes(base, arm, "arm"), M.compare_passes(base, b2, "aa"), identity="ignore")
    assert v["ok"], v


def test_gate_unstable_aa_zero_identical_is_inconclusive_when_identity_required():
    base = _rows(flip_at=set(range(12)), flip_margin=0.3)
    for r in base:
        r["margins"][5] = 0.3
    b2, arm = _rows(), _rows()
    for r in b2 + arm:
        r["margins"][5] = 0.3
    aa = M.compare_passes(base, b2, "aa")
    assert aa["identical"] == 0
    v = M.parity_ok(M.compare_passes(base, arm, "arm"), aa, identity="require")
    assert not v["ok"] and v["verdict"] == "INCONCLUSIVE"


def test_gate_missing_margins_fail_closed():
    base, b2 = _rows(), _rows()
    arm = _rows(flip_at={2}, drop_margins=True)
    v = M.parity_ok(M.compare_passes(base, arm, "arm"), M.compare_passes(base, b2, "aa"))
    assert not v["ok"] and v["verdict"] == "INCOMPLETE" and "margin" in v["why"]


def test_gate_missing_requests_fail_closed():
    base, b2 = _rows(), _rows()
    for arm in (_rows()[:1], _rows(error_at={4})):       # 1 of 12 compared; one request errored
        v = M.parity_ok(M.compare_passes(base, arm, "arm"), M.compare_passes(base, b2, "aa"), allowance=2)
        assert not v["ok"] and v["verdict"] == "INCOMPLETE", v


def test_gate_counterexamples_from_the_audits_no_longer_pass():
    """Audit 17 A: arm 0/12 identical, A/A 0/12, no flips -> was PASS. Audit 18/19: compared=1 -> was PASS."""
    assert not M.parity_ok({"expected": 12, "compared": 12, "identical": 0, "large_margin_flips": 0,
                            "missing_margins": 0},
                           {"expected": 12, "compared": 12, "identical": 0, "large_margin_flips": 0,
                            "missing_margins": 0}, allowance=2, identity="require")["ok"]
    assert not M.parity_ok({"expected": 12, "compared": 1, "identical": 1, "large_margin_flips": 0},
                           {"expected": 12, "compared": 12, "identical": 3, "large_margin_flips": 0},
                           allowance=2)["ok"]


def test_compare_passes_truncates_to_n_tokens():
    long_, short = _rows(length=40), _rows(length=20)
    c = M.compare_passes(long_, short, "n", n=20)
    assert c["identical"] == 12 and c["compared"] == 12


# ------------------------------------------------------------------------------------------- speed and occupancy
def _w(tok, level=12, lines=20, ctx=4000.0, idle=True):
    return {"tok_s_p50": tok, "measured_level": level, "lines": lines, "ctx_per_req_p50": ctx,
            "cleanup": {"server_idle": idle}}


@pytest.mark.parametrize("base,arm,why", [
    (_w(370), _w(380, lines=2), "decode lines"),
    (_w(370, ctx=40000), _w(380, ctx=8000), "context differs"),
    (_w(350, level=11), _w(354, level=12), "running level differs"),
    (_w(370), _w(380, idle=False), "not idle"),
    (None, _w(380), "missing"),
])
def test_adopt_speed_invalid_windows_are_inconclusive(base, arm, why):
    v = M.adopt_speed(base, arm)
    assert not v["adopt"] and v["verdict"] == "INCONCLUSIVE" and why in v["why"], v


def test_adopt_speed_occupancy_comparison_is_labelled():
    v = M.adopt_speed(_w(350, level=11), _w(380, level=12), occupancy=True)
    assert v["adopt"] and "occupancy" in v["label"]


def test_latch_gain_11_queued_to_12_is_a_gain_and_a_tie_is_not():
    ok = {"decodes_reached": True, "samples": 20}
    w1 = dict(ok, share_running_ge_level=0.0, share_stuck_below_queued=0.9, retracted_delta=0)
    nd = dict(ok, share_running_ge_level=0.95, share_stuck_below_queued=0.0, retracted_delta=0)
    assert M.latch_gain(w1, nd)["gain"]
    assert not M.latch_gain(w1, dict(w1))["gain"]
    assert not M.latch_gain(w1, dict(nd, retracted_delta=3))["gain"]       # more retractions: no
    assert not M.latch_gain(w1, {})["gain"]


# ------------------------------------------------------------------------------------------- Duck reserve (fake clock)
@pytest.mark.parametrize("now,expect", [
    (600.0, 900.0),        # early: the full cap
    (8000.0, 900.0),       # medium: still the full cap
    (9000.0, 400.0),       # late: cut to what is left above the reserve
    (9400.0, None),        # too late: skipped, the reserve is kept
])
def test_phase_budget_keeps_the_duck_reserve(now, expect):
    hard_stop, reserve = 12000.0, 2120.0
    b = M.phase_budget(now, hard_stop, 900.0, reserve=reserve, margin=480.0)
    assert b == expect
    if b is not None:
        assert now + b <= hard_stop - 480.0 - reserve + 1e-9


def test_launch_allowed_respects_reserve_unless_forced():
    hs, rs = 12000.0, 2980.0
    assert M.launch_allowed(1000.0, hs, reserve=rs)
    assert not M.launch_allowed(hs - rs - 500.0, hs, reserve=rs)          # would eat the reserve
    assert M.launch_allowed(hs - rs - 500.0, hs, reserve=rs, force=True)  # the Duck launch itself
    assert not M.launch_allowed(hs - 600.0, hs, reserve=0.0, force=True)  # never in the last 15 min


def test_log_errors_ignores_benign_signs():
    assert M.log_errors({"Exception": 3, "Retract": 1}) == {}
    assert M.log_errors({"Not enough space": 2, "Traceback": 0}) == {"Not enough space": 2}


# ------------------------------------------------------------------------------------------- review round 2
def test_flush_cache_classifies_a_refused_flush(stub, tmp_path):
    """A refused flush (HTTP 400, plain text) is ok=False, never assumed; a later success is ok=True."""
    ctx = _ctx(stub, tmp_path, 60.0)
    r = M.flush_cache(ctx, wait=0.5)
    assert r["ok"] is False and r["idle"] is True
    stub.flush_busy = False
    assert M.flush_cache(ctx, wait=0.5)["ok"] is True


def test_flush_cache_works_after_the_phase_deadline(stub, tmp_path):
    stub.flush_busy = False
    assert M.flush_cache(_ctx(stub, tmp_path, -5.0), wait=0.5)["ok"] is True     # a cleanup call, not a measurement


@pytest.mark.parametrize("bad", [
    {"decodes_reached": False, "samples": 20},
    {"decodes_reached": True, "samples": 0},
    {"decodes_reached": True, "samples": 20, "cleanup": {"server_idle": False}},
])
def test_latch_gain_rejects_an_invalid_latch(bad):
    w1 = {"share_running_ge_level": 0.0, "share_stuck_below_queued": 0.9, "decodes_reached": True, "samples": 20}
    nd = dict({"share_running_ge_level": 0.95, "share_stuck_below_queued": 0.0}, **bad)
    g = M.latch_gain(w1, nd)
    assert not g["gain"] and "invalid" in g["why"]


def test_adopt_speed_per_step_removes_accept_noise():
    """DFlash2 accept length swings per boot (#38009): a boot with a lower accept is not a slower arm."""
    base = dict(_w(370), accept_p50=4.4)
    arm = dict(_w(330), accept_p50=3.9)          # same steps/s: 370/4.4 = 84.1, 330/3.9 = 84.6
    assert not M.adopt_speed(base, arm)["adopt"]                       # raw tok/s would reject it
    v = M.adopt_speed(base, arm, per_step=True)
    assert v["adopt"] and v["basis"].startswith("steps/s")


# E14A SLIM run1 (Kaggle, Python 3.12): Sampler kept its stop Event in `self._stop`, which shadowed Thread._stop();
# Thread.join() calls self._stop() on Python <= 3.12, so every throughput window crashed ("'Event' object is not
# callable"). The local harness ran on Python 3.14, whose join() no longer calls it.
_THREAD_INTERNALS_312 = {"_stop", "_started", "_is_stopped", "_tstate_lock", "_target", "_args", "_kwargs", "_name",
                         "_daemonic", "_ident", "_native_id", "_initialized", "_stderr", "_invoke_excepthook",
                         "_wait_for_tstate_lock", "_bootstrap", "_bootstrap_inner", "_delete", "_reset_internal_locks",
                         "_set_ident", "_set_native_id", "_set_tstate_lock", "_handle", "_os_thread_handle"}


def test_thread_subclasses_never_shadow_thread_internals():
    import ast
    from pathlib import Path
    taken = _THREAD_INTERNALS_312 | {n for n in dir(threading.Thread) if n.startswith("_") or
                                     callable(getattr(threading.Thread, n, None))} \
        | set(vars(threading.Thread(target=lambda: None)))
    bad, seen = [], []
    for f in sorted(Path(__file__).resolve().parent.glob("e14_*.py")):
        for cls in (n for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))) if isinstance(n, ast.ClassDef)):
            if not any(getattr(b, "attr", getattr(b, "id", None)) == "Thread" for b in cls.bases):
                continue
            seen.append(cls.name)
            for n in ast.walk(cls):
                if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name) \
                        and n.value.id == "self" and n.attr in taken and not n.attr.startswith("__"):
                    bad.append(f"{f.name}: {cls.name}.{n.attr} (line {n.lineno})")
    assert {"Sampler", "Gen", "Staging"} <= set(seen), seen
    assert not bad, bad


def test_sampler_stop_joins_and_returns_rows(stub, tmp_path):
    s = M.Sampler(_ctx(stub, tmp_path, 30), every=0.05)
    s.start()
    time.sleep(0.2)
    rows = s.stop()                       # crashed on Python 3.12 before the fix
    assert not s.is_alive() and isinstance(rows, list) and rows
