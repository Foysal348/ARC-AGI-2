"""E15P builder (user 2026-10-25: "load Poetiq on the Duck engine"): the S1-60K Poetiq solver (E11-05 S60 cells, LB 34.58
method) on the E15B engine (16 streams, mrr 17, CUDA graphs 1..17, mamba 68, DFlash width 6, the min-free-slots delayer
off, /dev/shm staging, the 12-stream / W1 fallback launches). The E14 SGLang backports 44401 + 41872 are NOT applied: in
R3K1 they ran only in the E14 measurement cell (rev 1 docs claimed otherwise; agent A LOW-1). rev 2 = the fixes of
Exp 15 Notes/19 (F1-F15), marked "E15P rev 2" in the cells.

Sources (read only):
  engine side = the R3K1 notebook (pushed, sha 6baa9361...): CELL 0, CELL 1 (config), CELL 1b (E14 staging), CELL 2
                (install), the task-order cell, the engine cell, CELL 7b (E14 report + bundle)
  Poetiq side = the E11-05 S60 EVAL120 6H notebook (3af4660a...): the Poetiq donor cell, the scheduler, the report
Dropped: CELL 3 (Duck package), CELL 5b / 6b (E14 measurements), CELL 6 / 7 (Duck solver and report). CELL 1 keeps
everything up to the Poetiq block and replaces the Duck block with the S60 Poetiq values. rev 2 adds S60's vLLM FP8
fallback cell (F1) after the engine cell.

usage: python build_e15p.py      -> ../Q27-XHIGH-E15P_EVAL120_S60_S15_POETIQ_SGLANG.ipynb"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
E11 = NBDIR.parents[1] / "Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)" / "Experiment Notebooks"
SRC_ENGINE = NBDIR / "Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3K1_DUCK_SGLANG.ipynb"
SRC_POETIQ = E11 / "Q27-XHIGH-v1_4_4_E11_05_S60_SGLANG_INT4_DFLASH2_EVAL120_6H.ipynb"
SHA_ENGINE = "6baa9361"
SHA_POETIQ = "3af4660a"
OUT = NBDIR / "Q27-XHIGH-E15P_EVAL120_S60_S15_POETIQ_SGLANG.ipynb"   # rev 2: 15 streams (F6)
PUSHED: set = {"56a22f7f1db6b6e26c6afe37a22e0ed4711dc6403ca4d6491a70e2c2b688510c"}   # rev 2, esfersami50 2026-10-25; a rebuild never overwrites a pushed one


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_cells(p):
    return ["".join(c["source"]) for c in json.load(open(p, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def rep(text, a, b, n=1):
    assert text.count(a) == n, (text.count(a), a[:100])
    return text.replace(a, b)


assert sha(SRC_ENGINE).startswith(SHA_ENGINE), sha(SRC_ENGINE)
assert sha(SRC_POETIQ).startswith(SHA_POETIQ), sha(SRC_POETIQ)
E = code_cells(SRC_ENGINE)
S = code_cells(SRC_POETIQ)
assert len(E) == 12 and len(S) == 9
assert "CELL 1b" in E[2] and "CELL 3 — E12: Duck package" in E[4] and "CELL 5b — E14-A1" in E[7] \
    and "CELL 6 — E12: Duck REPL agent" in E[8] and "CELL 7 — E12: report" in E[9] and "CELL 7b — E14" in E[11]
assert "Poetiq donor adaptation" in S[4] and "Poetiq iterative program-synthesis solver" in S[7] \
    and "Poetiq program synthesis, train-verification" in S[8]
assert E[5] == S[3], "the task-order cell must be the same in both sources"

# ---- CELL 1: the E15B config without the Duck block, + the S60 Poetiq values and the E15P profile
c1 = E[1]
cut = c1.index("\n\n# ---- E12: Duck REPL agent")
tail = 'print("E12:", json.dumps(CONFIG["e12"]))'
assert c1.rstrip().endswith(tail)
c1 = c1[:cut]
c1 = rep(c1, 'os.environ.get("Q27X_PROFILE", "E15BEVAL120R3I")', 'os.environ.get("Q27X_PROFILE", "E15PEVAL120")')
c1 = rep(c1, '''    "E14SEVAL120G4B": {''',
         '''    # E15P (user 2026-10-25: Poetiq on the Duck engine): all 120 evaluation tasks, a 6-h notebook like the E11-05 S60
    # EVAL120 6H run (3af4660a) and the Duck R3K EVAL120 runs; the solve runs to the drain (12 min before the hard stop)
    "E15PEVAL120": {"source": "eval", "n_tasks": 120, "time_budget_sec": 6.0 * 3600, "admit_frac": 0.90,
                    "drain_frac": 0.94, "hard_margin_sec": 2 * 60, "verify_margin_sec": 1 * 60,
                    "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60},
    "E14SEVAL120G4B": {''')
c1 = rep(c1, "SGL_DRAIN_MARGIN_SEC = 8 * 60      # E11-06: back to 8 min (EVAL30 drain <= 83 s)",
         "SGL_DRAIN_MARGIN_SEC = 12 * 60     # E15P: = S60 (S1-60K); E11-06 used 8 min")
c1 = rep(c1, "SGL_KV_PER_REQUEST = 50_000         # E15B:",
         "SGL_KV_PER_REQUEST = 56_000         # E15P rev 2 (F6, agent B H1): 852,152 // 56k = 15 Poetiq streams. Poetiq units "
         "reach the 60k cap as a cohort: the E11-06 decode-line peak 0.73 of 866,744 = 51.9k per request x 16 = 0.97 of the "
         "E15B pool; simulated SGLang retractions at 16: 2 / 8 six-hour and 2 / 4 twelve-hour runs, at 15: 0 / 28 (-1 % tokens). "
         "The client KV gate cannot stop running units from growing. A smaller pool adapts (784k -> 14, 700k -> 12). "
         "Server mrr / graphs stay 17. E15B was 50k:")
c1 = rep(c1, "POETIQ_MAX_SOLUTIONS = int(os.environ.get('E10_POETIQ_MAX_SOLUTIONS','2'))",
         "POETIQ_MAX_SOLUTIONS = int(os.environ.get('E10_POETIQ_MAX_SOLUTIONS','5'))   # E15P: = S60 (S1-60K)")
c1 = rep(c1, "POETIQ_SHUFFLE_EXAMPLES = os.environ.get('E10_POETIQ_SHUFFLE','0')=='1'",
         "POETIQ_SHUFFLE_EXAMPLES = os.environ.get('E10_POETIQ_SHUFFLE','1')=='1'   # E15P: = S60 (S1-60K)")
c1 += '''


# ---- E15P (user 2026-10-25): the Poetiq solver of E11-05 S60 (S1-60K, LB 34.58) on the E15B engine. No Duck: the Duck
# package, solver and report cells are not in this notebook. The engine cell reads one Duck name (the mamba-track test
# flag); it stays off as in every E15B run.
E12_MAMBA_TRACK_64 = False
CONFIG["e15p"] = {"solver": "poetiq-s60", "streams": SGL_MAX_RUNNING, "server_max_running": SGL_SERVER_MAX_RUNNING,
                  "kv_per_request": SGL_KV_PER_REQUEST, "drain_margin_sec": SGL_DRAIN_MARGIN_SEC,
                  "engine_source": "''' + SHA_ENGINE + '''", "poetiq_source": "''' + SHA_POETIQ + '''"}
print("E15P:", json.dumps(CONFIG["e15p"]), "| poetiq", json.dumps(CONFIG["poetiq"]))
'''

# ---- CELL 1b: no Duck payload; the build record names E15P
c1b = E[2]
c1b = rep(c1b, '''E14_RESULTS = {"build": {"variant": "E15B", "arc2_duck_sha": "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128", "modules": _E14_MOD_SHA,''',
          '''E14_RESULTS = {"build": {"variant": "E15P", "modules": _E14_MOD_SHA,''')
c1b = rep(c1b, ''',
                                     "q27x_e15b_duck_9e4b89285136.bin": "9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566"}},''',
          '''}},''')

# ==================================================================== rev 2 fixes (Exp 15 Notes/19: agent A, GPT audits 1-2)
# F10: no E14 measurement in E15P (the E14A profile row stays in the table but cannot switch the measurement cells on)
c1b = rep(c1b, 'E14_DO_MEASURE = bool(P.get("e14")) and not RERUN',
          'E14_DO_MEASURE = False   # E15P rev 2 (F10): no E14 measurement cells in this notebook')
c1b = rep(c1b, 'os.environ["E12_RAW_REASONING_HISTORY"] = "1"          # E14-P (Notes/12): Duck keeps the server\'s raw reasoning\n', '')
c1b = rep(c1b, '"modules": _E14_MOD_SHA, "arc2_duck_sha": "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128"}',
          '"modules": _E14_MOD_SHA}')
c1b = rep(c1b, '"staging": E14_STAGING is not None, "raw_reasoning": True,', '"staging": E14_STAGING is not None,')
# F10: the EVAL120 row = S60's 6-h value (it was R3K1's 12 h - 3 min)
c1 = rep(c1, '"EVAL120":   {"source": "eval", "n_tasks": 120, "time_budget_sec": 12.0 * 3600 - 3 * 60,',
         '"EVAL120":   {"source": "eval", "n_tasks": 120, "time_budget_sec": 6.0 * 3600,   # E15P rev 2: = S60 6H',)
c1 = rep(c1, 'E12_MAMBA_TRACK_64 = False\n', '''E12_MAMBA_TRACK_64 = False
# rev 2 (Notes/19): F2 mid-run server restarts (12-stream config, only with >= 45 min before the admission deadline),
# F5 sandbox self-test attempts (it runs while the model copy reads with 24 threads)
E15P_MAX_RESTARTS = int(os.environ.get("E15P_MAX_RESTARTS", 2))
E15P_RESTART_MIN_LEFT_SEC = float(os.environ.get("E15P_RESTART_MIN_LEFT_SEC", 45 * 60))
E15P_SELFTEST_TRIES = int(os.environ.get("E15P_SELFTEST_TRIES", 3))
''')
c1 = rep(c1, '''"kv_per_request": SGL_KV_PER_REQUEST, "drain_margin_sec": SGL_DRAIN_MARGIN_SEC,''',
         '''"kv_per_request": SGL_KV_PER_REQUEST, "drain_margin_sec": SGL_DRAIN_MARGIN_SEC, "rev": 2,
                  "max_restarts": E15P_MAX_RESTARTS, "restart_min_left_sec": E15P_RESTART_MIN_LEFT_SEC,''')

# F5: the sandbox self-test, up to E15P_SELFTEST_TRIES attempts (a slow first numpy import under the model copy must not
# switch the engine off for the whole run); the attack probes stay strict
don = S[4]
don = rep(don, '''try:
    E10_SANDBOX_SELFTEST=poetiq_sandbox_selftest()
except BaseException as _e10_sandbox_error:
    E10_SANDBOX_SELFTEST={'ok':False,'mode':'unavailable',
                         'error':type(_e10_sandbox_error).__name__+': '+str(_e10_sandbox_error)[:1200]}''',
          '''_E15P_ST_ERRORS=[]
for _e15p_try in range(max(1,E15P_SELFTEST_TRIES)):   # E15P rev 2 (F5): retried; the attack probes stay strict
    try:
        E10_SANDBOX_SELFTEST=poetiq_sandbox_selftest()
        E10_SANDBOX_SELFTEST['tries']=_e15p_try+1
        if _E15P_ST_ERRORS:E10_SANDBOX_SELFTEST['earlier_errors']=_E15P_ST_ERRORS
        break
    except BaseException as _e10_sandbox_error:
        _E15P_ST_ERRORS.append(type(_e10_sandbox_error).__name__+': '+str(_e10_sandbox_error)[:600])
        E10_SANDBOX_SELFTEST={'ok':False,'mode':'unavailable','tries':_e15p_try+1,
                             'error':type(_e10_sandbox_error).__name__+': '+str(_e10_sandbox_error)[:1200],
                             'errors':list(_E15P_ST_ERRORS)}
        if _e15p_try+1<max(1,E15P_SELFTEST_TRIES):time.sleep(10)''')

# ---- the engine cell
eng = rep(E[6], "# E12: = E11-06 CELL 5 unchanged except the Duck serving flags in e11_flags().",
          "# E15P: the E15B engine cell (Poetiq uses /generate through the shim; the chat parsers of the Duck flags do not touch\n"
          "# /generate). rev 2 (Notes/19): F3 pending deltas for the drain, F4 admission usage = max(KV, mamba), F15 a smoke\n"
          "# failure relaunches the 12-stream config once, F2 e15p_relaunch_s12() for a mid-run restart, F12 pool maxima.\n"
          "# E12: = E11-06 CELL 5 unchanged except the Duck serving flags in e11_flags().")
# F3: step() serves held items first, so take_pending() can pull the deltas of aborted requests without reordering others
eng = rep(eng, "        self.server, self.q, self.aborted, self.done = server, _queue.Queue(), set(), {}",
          "        self.server, self.q, self.aborted, self.done = server, _queue.Queue(), set(), {}\n"
          "        self.held = []                      # E15P rev 2 (F3): items of other requests set aside by take_pending()")
eng = rep(eng, '''        items = []
        try:
            items.append(self.q.get(timeout=0.05))
        except _queue.Empty:
            return []''', '''        items, self.held = list(self.held), []
        if not items:
            try:
                items.append(self.q.get(timeout=0.05))
            except _queue.Empty:
                return []''')
eng = rep(eng, '''    def abort_request(self, ids, wait_sec=30.0):''', '''    def take_pending(self, rids):
        """E15P rev 2 (F3, GPT audit 1 HIGH-2): after abort_request() the streams of rids have ended, but their last deltas
        may still sit in the queue (step() moves them only on its next call). Returns rid -> (token ids in order, final
        reason or None) for rids; every other item is kept, in order, for the next step()."""
        rids, out, rest = set(rids), {}, []
        items, self.held = list(self.held), []
        while True:
            try:
                items.append(self.q.get_nowait())
            except _queue.Empty:
                break
        for rid, ids, reason in items:
            if rid in rids:
                o = out.setdefault(rid, [[], None])
                if o[1] is None:
                    o[0].extend(ids)
                    o[1] = reason
            else:
                rest.append((rid, ids, reason))
        self.held = rest
        return {k: (v[0], v[1]) for k, v in out.items()}

    def abort_request(self, ids, wait_sec=30.0):''')
# F4: the admission metric = sglang:token_usage = max(full KV, mamba) (pool_stats_observer.py:69-73, 128); the row keeps a
# name with 'kv_cache_usage' (the scheduler reads that); missing -> no row (the scheduler then holds new admissions)
eng = rep(eng, '''        full = [r.value for r in rows if r.name == "sglang:full_token_usage"]
        anyu = [r.value for r in rows if r.name == "sglang:token_usage"]
        kv = max(full) if full else (max(anyu) if anyu else None)
        if kv is not None:
            rows.append(_types.SimpleNamespace(name="e11:kv_cache_usage(sglang:full_token_usage)", labels=None, value=kv))
        return rows''', '''        # E15P rev 2 (F4, GPT audits 1-2): the bottleneck pool, not the KV pool alone. SGLang 0.5.21 token_usage =
        # max(full, swa, mamba) (pool_stats_observer.py:69-73, 128); mamba_usage counts active slots only (:272-276)
        anyu = [r.value for r in rows if r.name == "sglang:token_usage"]
        pools = [r.value for r in rows if r.name in ("sglang:full_token_usage", "sglang:mamba_usage")]
        kv = max(anyu) if anyu else (max(pools) if pools else None)
        if kv is not None and kv == kv and 0.0 <= kv <= 1.5:
            rows.append(_types.SimpleNamespace(name="e11:kv_cache_usage(sglang:token_usage=max(full,mamba))", labels=None,
                                               value=kv))
        return rows''')
eng = rep(eng, '''                  "num_requests_total")''', '''                  "num_requests_total", "mamba_usage")   # E15P rev 2 (F12): + mamba_usage''')
# F12: pool maxima from every server log of the run (a restart starts a new log)
eng = rep(eng, '''    if E11_SERVER is None:
        return None
    rows = []
    for line in E11_SERVER.log_text().splitlines():''', '''    if E11_SERVER is None:
        return None
    rows = []
    _texts = []
    for _p in list(globals().get("E15P_OLD_LOGS") or []):     # E15P rev 2: servers replaced by a restart
        try:
            _texts.append(Path(_p).read_text(encoding="utf-8", errors="replace"))
        except Exception:
            pass
    _texts.append(E11_SERVER.log_text())
    for line in "\\n".join(_texts).splitlines():''')
eng = rep(eng, '''           "retract_lines": sum(1 for l in E11_SERVER.log_text().splitlines() if "retract" in l.lower() and "retraction_policy" not in l and "server_args" not in l.lower()),''',
          '''           "retract_lines": sum(1 for l in "\\n".join(_texts).splitlines() if "retract" in l.lower() and "retraction_policy" not in l and "server_args" not in l.lower()),
           # E15P rev 2 (F12): pool maxima seen in the decode lines
           "full_token_usage_max": max((r["full token usage"] for r in rows if "full token usage" in r), default=None),
           "mamba_usage_max": max((r["mamba usage"] for r in rows if "mamba usage" in r), default=None),
           "running_max": max((r.get("#running-req", 0) for r in rows), default=None),''')
# F15: a 16-stream server that is READY and passes the capacity gate but fails the smoke -> one 12-stream relaunch
eng = rep(eng, '''    if not E11_SGL["smoke"]["ok"]:
        raise E11Fail("smoke failed: " + json.dumps(E11_SGL["smoke"])[:600])
    return eng, caps''', '''    if not E11_SGL["smoke"]["ok"]:
        if not cfg.get("streams12") and SGL_GIVE_UP_AT - time.perf_counter() > 120:   # E15P rev 2 (F15, GPT audit 2)
            E11_SGL["smoke16"] = dict(E11_SGL["smoke"])
            print("E15P: 16-stream smoke failed -> 12-stream relaunch:", json.dumps(E11_SGL["smoke"])[:600], flush=True)
            return e15p_relaunch_s12("smoke", min(SGL_RELAUNCH_TIMEOUT_SEC, max(60.0, SGL_GIVE_UP_AT - time.perf_counter())))
        raise E11Fail("smoke failed: " + json.dumps(E11_SGL["smoke"])[:600])
    return eng, caps


E15P_OLD_LOGS = []          # server logs of servers replaced by a relaunch (F12 decode summary)
E15P_RELAUNCHES = []


def e15p_relaunch_s12(reason, timeout_sec):
    """E15P rev 2 (F2 / F15): stop the current server and start the 12-stream configuration (mrr 13, graphs 1..13, mamba
    56; the W1 flags stay W1) with the same venv, env and staged model (no reinstall); then the capacity gate and the
    smoke. Returns (engine, caps); raises E11Fail. Sets SGL_MAX_RUNNING 12 so the gate gives concurrency 12."""
    global E11_SERVER, SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING
    rec = {"reason": reason, "t": round(time.perf_counter() - NOTEBOOK_T0), "timeout_sec": round(timeout_sec)}
    E15P_RELAUNCHES.append(rec)
    old = E11_SERVER
    env = getattr(old, "env", None)
    if old is not None:
        try:
            E15P_OLD_LOGS.append(str(old.log_path))
            old.kill()
        except Exception as e:
            rec["kill_error"] = repr(e)[:200]
    rec["gpu_free"] = _e11_gpu_free()
    if env is None or not E14_VENV.get("launch"):
        raise E11Fail("relaunch: no launch command / env recorded")
    cfg0 = dict(E14_VENV.get("cfg") or W1)
    cfg = dict(cfg0) if (cfg0.get("w1") or cfg0.get("streams12")) else dict(cfg0, streams12=True, mamba_cache=56)
    SGL_MAX_RUNNING, SGL_SERVER_MAX_RUNNING = 12, 13
    srv = E11Server(list(E14_VENV["launch"]) + e11_flags(cfg), env, f"relaunch{len(E15P_RELAUNCHES)}_{reason}")
    srv.start()
    ok, why = srv.wait_ready(timeout_sec)
    rec.update(ready=ok, why=why, cfg=dict(cfg), sec=round(time.perf_counter() - srv.t0, 1))
    E11_SGL.setdefault("attempts", []).append(dict(rec, **{"try": "relaunch_" + reason}))
    if not ok:
        srv.kill()
        raise E11Fail(f"12-stream relaunch ({reason}) not ready: {why}")
    E11_SERVER = srv
    E11_SGL["launch_flags"] = e11_flags(cfg)
    E14_VENV["cfg"] = dict(cfg)
    caps = e11_caps(srv.log_text())
    caps["gate"], caps["concurrency"] = e11_capacity_gate(caps, cfg)
    rec["gate"] = caps["gate"]
    if not caps["gate"]["pass"]:
        raise E11Fail(f"12-stream relaunch ({reason}) capacity gate failed: " + json.dumps(caps["gate"]))
    eng_ = E11SglangEngine(srv)
    E11_SGL["smoke"] = e11_smoke(eng_)
    rec["smoke_ok"] = bool(E11_SGL["smoke"]["ok"])
    if not E11_SGL["smoke"]["ok"]:
        raise E11Fail(f"12-stream relaunch ({reason}) smoke failed: " + json.dumps(E11_SGL["smoke"])[:400])
    E11_SGL["caps"] = caps
    print("E15P relaunch:", json.dumps(rec, default=str)[:800], flush=True)
    return eng_, caps''')

# ---- F1: S60's vLLM FP8 fallback cell, unchanged (SGLang never ready or the tokenizer check failed -> Poetiq on vLLM)
fallback = S[6]
assert fallback.startswith("# ==================================================================== CELL 5b — vLLM FP8 fallback")

# ---- the Poetiq scheduler
sch = S[7]
# F4: a missing / unreadable metric no longer means "empty" (fail-closed above the 10-running floor)
sch = rep(sch, '''_E10_KV={'t':-1e9,'v':0.0}
def e10_kv_usage():
    # S1: vLLM KV usage (0..1), read at most every 5 s; 0.0 if the metric is unavailable (gate then stays open)
    now=time.perf_counter()
    if now-_E10_KV['t']>=5.0:
        _E10_KV['t']=now
        try:
            vals=[float(getattr(m,'value',0) or 0) for m in llm.get_metrics() if 'kv_cache_usage' in getattr(m,'name','')]
            _E10_KV['v']=max(vals) if vals else 0.0
        except Exception:
            _E10_KV['v']=0.0
    return _E10_KV['v']

def e10_kv_gate_open():
    if len(E10_ACTIVE)<E10_KV_GATE_MIN_ACTIVE:return True
    kv=e10_kv_usage()
    if kv<=E10_KV_ADMIT_MAX:return True''', '''_E10_KV={'t':-1e9,'v':None}
def e10_kv_usage():
    # S1: KV usage (0..1), read at most every 5 s. E15P rev 2 (F4, GPT audits 1-2): on SGLang the shim gives
    # max(KV, mamba); a missing or unreadable metric is None (the gate then holds new admissions above the floor)
    now=time.perf_counter()
    if now-_E10_KV['t']>=5.0:
        _E10_KV['t']=now
        try:
            vals=[float(getattr(m,'value',0) or 0) for m in llm.get_metrics() if 'kv_cache_usage' in getattr(m,'name','')]
            vals=[v for v in vals if v==v and 0.0<=v<=1.5]
            _E10_KV['v']=max(vals) if vals else None
        except Exception:
            _E10_KV['v']=None
    return _E10_KV['v']

def e10_kv_gate_open():
    # the floor stays (S1 design): below 10 running a new unit is always admitted (9 x 76k max context < the pool)
    if len(E10_ACTIVE)<E10_KV_GATE_MIN_ACTIVE:return True
    kv=e10_kv_usage()
    if kv is None:
        E10_SCHED['kv_metric_missing']=E10_SCHED.get('kv_metric_missing',0)+1
        return False
    if kv<=E10_KV_ADMIT_MAX:return True''')
# F7 + F2: one verification pool for the whole run (a restart keeps the queued verifications); leaving the with-block
# never waits for running sandbox jobs; on an error exit the pool stays open for a restart
sch = rep(sch, '''def e10_run_solver():
    global E10_TOTAL_GEN
    workers=max(1,POETIQ_SANDBOX_WORKERS)
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:''', '''E15P_POOL=[]
class _E15PPool:
    """E15P rev 2 (F7, GPT audit 1 HIGH-3; F2): the with-block of a ThreadPoolExecutor waits for its running jobs on exit.
    One pool serves the whole run (a restart keeps the queued verifications); a normal exit shuts it without waiting
    (queued jobs cancelled, as S60's explicit shutdown), an error exit leaves it open for a restart."""
    def __init__(self,n):
        if not E15P_POOL:E15P_POOL.append(futures.ThreadPoolExecutor(max_workers=n))
        self.p=E15P_POOL[0]
    def __enter__(self):return self.p
    def __exit__(self,et,ev,tb):
        if et is None:self.p.shutdown(wait=False,cancel_futures=True)
        return False
def e15p_pool_close():
    for _p in E15P_POOL:
        try:_p.shutdown(wait=False,cancel_futures=True)
        except Exception:pass

def e10_run_solver():
    global E10_TOTAL_GEN
    workers=max(1,POETIQ_SANDBOX_WORKERS)
    with _E15PPool(workers) as pool:''')
# F3: the drain keeps the deltas that arrived before the abort ended the stream; an unconfirmed abort starts no phase 2;
# a phase 1 that had closed its thinking naturally is finished as natural
sch = rep(sch, '''                p1=[r for r in E10_ACTIVE.values() if r['phase']==1]
                if p1:engine.abort_request([e10_req(E10_STATES[r['key']],1) for r in p1])
                for req in p1:
                    req['p1_finish']='drained'
                    if not e10_start_phase2(req,DRAIN_ANSWER_MAX_TOKENS):e10_after_decode(req,'drain_no_room',pool)''',
          '''                p1=[r for r in E10_ACTIVE.values() if r['phase']==1]
                _p1_rids=[e10_req(E10_STATES[r['key']],1) for r in p1]
                _ab=(engine.abort_request(_p1_rids) or {}) if p1 else {}
                _pend=engine.take_pending(_p1_rids) if (p1 and hasattr(engine,'take_pending')) else {}   # E15P rev 2 (F3)
                for req,_rid in zip(p1,_p1_rids):
                    _tail,_fin=_pend.get(_rid,([],None))
                    if _tail:
                        req['p1_ids'].extend(_tail);E10_TOTAL_GEN+=len(_tail)
                        E10_SCHED['drain_tail_tokens']=E10_SCHED.get('drain_tail_tokens',0)+len(_tail)
                    if _fin=='stop' and THINK_CLOSE_ID in req['p1_ids']:
                        req['p1_finish']='stop';e10_after_decode(req,'natural',pool);continue
                    if isinstance(_ab,dict) and _ab.get(_rid) not in (None,'abort','stop','length'):
                        E10_SCHED['abort_unconfirmed']=E10_SCHED.get('abort_unconfirmed',0)+1
                        req['p1_finish']='abort_unconfirmed';e10_after_decode(req,'abort_unconfirmed',pool);continue
                    req['p1_finish']='drained'
                    if not e10_start_phase2(req,DRAIN_ANSWER_MAX_TOKENS):e10_after_decode(req,'drain_no_room',pool)''')
# F2: a dead SGLang server mid-run -> the running units go back to the front of the queue (iteration not advanced, their
# thinking saved as .restart.partial.txt), one 12-stream relaunch, the solver continues (at most E15P_MAX_RESTARTS)
# F2b: a stream that ends with 'error' while the server is dying, or a call that finds the server gone, must not close the
# unit ('unit_error' / a consumed iteration): it goes to the restart path with the unit still active
sch = rep(sch, '''                reason=comp.finish_reason if comp is not None else None
                if phase==1:''', '''                reason=comp.finish_reason if comp is not None else None
                if reason=='error' and e15p_server_dead(wait=5.0):   # E15P rev 2 (F2b)
                    raise RuntimeError('SGLang server exited (stream error on '+o.request_id+')')
                if phase==1:''')
sch = rep(sch, '''              except Exception as _uerr:''', '''              except Exception as _uerr:
                if isinstance(_uerr,RuntimeError) and 'SGLang server' in str(_uerr):raise   # E15P rev 2 (F2b): restart path''')
sch = rep(sch, '''def e10_run_solver():
    global E10_TOTAL_GEN''', '''def e15p_server_dead(wait=0.0):
    """E15P rev 2 (F2b): True when the SGLang server process has exited (polled up to wait seconds: a stream can fail just
    before the process is gone). Always False on the vLLM fallback."""
    _srv=globals().get('E11_SERVER')
    if _srv is None or not hasattr(_srv,'alive'):return False
    _t=time.perf_counter()
    while True:
        if not _srv.alive():return True
        if time.perf_counter()-_t>=wait:return False
        time.sleep(0.2)

def e10_run_solver():
    global E10_TOTAL_GEN''')
sch = rep(sch, '''try:
    if ENGINE_READY:
        e10_run_solver()''', '''E15P_RESTARTS=[]
def e15p_run_solver_with_restart():
    global engine,CONCURRENCY,E10_TOTAL_GEN
    while True:
        try:
            e10_run_solver();return
        except RuntimeError as _exc:
            _srv=globals().get('E11_SERVER')
            _dead=_srv is not None and hasattr(_srv,'alive') and not _srv.alive()
            _left=ADMIT_DEADLINE-time.perf_counter()
            _rec={'t':round(time.perf_counter()-NOTEBOOK_T0),'error':repr(_exc)[:300],'server_dead':_dead,
                  'left_to_admit_sec':round(_left),'active':len(E10_ACTIVE)}
            E15P_RESTARTS.append(_rec)
            if (not _dead or len([r for r in E15P_RESTARTS if r.get('relaunched')])>=E15P_MAX_RESTARTS
                    or _left<E15P_RESTART_MIN_LEFT_SEC or not str(ENGINE_INFO.get('backend','')).startswith('sglang')):
                _rec['action']='none';print('[E15P] solver stopped, no restart:',json.dumps(_rec),flush=True)
                raise
            print('[E15P] SGLang died mid-run -> requeue the running units and relaunch at 12 streams:',json.dumps(_rec),flush=True)
            _keys=list(E10_ACTIVE)
            for _k in _keys:
                _r=E10_ACTIVE[_k];_st=E10_STATES[_k]
                try:
                    (E10_THINKING_DIR/f"{_st['task_id']}_e{_st['expert']}_i{_st['iteration']+1}.restart{len(E15P_RESTARTS)}.partial.txt").write_text(
                        f"# INCOMPLETE (server restart) task={_st['task_id']} phase={_r['phase']} p1_tokens={len(_r['p1_ids'])}\\n"
                        +tokenizer.decode(list(_r['p1_ids'])+list(_r['p2_prefix'])+list(_r['p2_ids']),skip_special_tokens=False),encoding='utf-8')
                except Exception:pass
                _st['status']='queued'
            E10_ACTIVE.clear()
            for _k in reversed(_keys):E10_QUEUE.appendleft(_k)
            _rec['requeued']=len(_keys)
            try:
                engine,_caps=e15p_relaunch_s12('death',min(SGL_RELAUNCH_TIMEOUT_SEC,max(60.0,_left-E15P_RESTART_MIN_LEFT_SEC/2)))
            except Exception as _re:
                _rec.update(relaunched=False,relaunch_error=repr(_re)[:300]);raise
            CONCURRENCY=int(_caps['concurrency']);_E10_KV.update(t=-1e9,v=None)
            _rec.update(relaunched=True,concurrency=CONCURRENCY)
            ENGINE_INFO['restarts']=list(E15P_RESTARTS);ENGINE_INFO['concurrency']=CONCURRENCY

try:
    if ENGINE_READY:
        e15p_run_solver_with_restart()   # E15P rev 2 (F2)''')
sch = rep(sch, '''finally:
    for _tid in TASK_IDS:
        e10_update_task_submission(_tid)''', '''finally:
    e15p_pool_close()   # E15P rev 2: no new verification after the solver; queued ones cancelled
    E10_SCHED['restarts']=E15P_RESTARTS
    for _tid in TASK_IDS:
        e10_update_task_submission(_tid)''')

# ---- the report: F12 coverage and memory evidence
rpt = rep(S[8], '''    write_json_atomic(OUT/'poetiq_run_report.json',report)''', '''    try:   # E15P rev 2 (F12, GPT audits 1-2, agent A): coverage and memory evidence of the run
        _it1=[r for r in E10_RECORDS if r.get('iteration')==1]
        _hist=lambda rs,k:{str(v):sum(1 for r in rs if r.get(k)==v) for v in sorted({str(r.get(k)) for r in rs})}
        _m=[x for s in E10_METRICS for x in (s.get('metrics') or []) if isinstance(x,dict) and 'name' in x]
        _pk=lambda n:max([x['value'] for x in _m if x['name']==n] or [None],key=lambda v:-1 if v is None else v)
        _ds=(report.get('e11') or {}).get('decode_summary') or {}
        report['e15p']={'rev':2,
            'iteration1_started_tasks':len({k[0] for k,s in E10_STATES.items() if s['iteration']>=1 or s['status'] in ('running','verifying')}),
            'iteration1_completed_tasks':len({r['task_id'] for r in _it1}),
            'iteration1_train_verified_tasks':len({r['task_id'] for r in _it1 if r.get('train_passed')}),
            'not_started':E10_SCHED.get('not_started'),'tasks':len(TASK_IDS),
            'iteration1_mode':_hist(_it1,'mode'),'iteration1_finish':_hist(_it1,'finish'),
            'all_mode':_hist(E10_RECORDS,'mode'),'all_finish':_hist(E10_RECORDS,'finish'),
            'phase1_tokens':sum(int(r.get('p1_tokens') or 0) for r in E10_RECORDS),
            'phase2_tokens':sum(int(r.get('p2_tokens') or 0) for r in E10_RECORDS),
            'iterations_finished':E10_SCHED.get('iterations_finished'),'tok_per_sec':E10_SCHED.get('tok_per_sec'),
            'peaks_sampled_120s':{n:_pk(n) for n in ('sglang:token_usage','sglang:full_token_usage','sglang:mamba_usage',
                                                     'sglang:num_running_reqs','sglang:num_queue_reqs')},
            'decode_log':{k:_ds.get(k) for k in ('full_token_usage_max','mamba_usage_max','running_max','queue_max',
                                                 'retract_lines','gen_tok_s_median_at_max','accept_len_median','running_hist')},
            'kv_gate_blocks':E10_SCHED.get('kv_gate_blocks'),'kv_metric_missing':E10_SCHED.get('kv_metric_missing',0),
            'drain_tail_tokens':E10_SCHED.get('drain_tail_tokens',0),'abort_unconfirmed':E10_SCHED.get('abort_unconfirmed',0),
            'restarts':globals().get('E15P_RESTARTS'),'relaunches':globals().get('E15P_RELAUNCHES'),
            'concurrency_final':CONCURRENCY,'backend':ENGINE_INFO.get('backend')}
        print('E15P:',json.dumps(report['e15p'],default=str)[:3000],flush=True)
    except Exception as _e15pe:
        report['e15p']={'error':repr(_e15pe)[:300]}
    write_json_atomic(OUT/'poetiq_run_report.json',report)''')
# the E11 section must be read before the e15p block uses it: move the e11 try-block above (it is right after scoring)
assert rpt.index("report['e11']={'backend'") < rpt.index("report['e15p']={'rev':2"), "e11 section must come first"

c7b = rep(E[11], '_pack = WORKING / "Q27X_E14A_BUNDLE.zip"', '_pack = WORKING / "Q27X_E15P_ENGINE_BUNDLE.zip"')

cells = [E[0], c1, c1b, E[3], E[5], don, eng, fallback, sch, rpt, c7b]
md = """# E15P rev 2 — Poetiq S1-60K on the E15B Duck engine (SGLang 0.5.21 + RedHat INT4 + DFlash2 W6, 15 streams)

**Solver:** the E11-05 S60 Poetiq cells (S1-60K method, LB 34.58): one expert, up to 10 iterations, 60,000 thinking tokens
per iteration, 5 feedback solutions, shuffled examples, ORDER2 + cells order for iteration 1, train verification in the
sandbox, voting, the KV admission gate. rev 2 changes (Exp 15 Notes/19): the gate reads max(KV, mamba) and holds new work
when the metric is missing; the drain keeps the last streamed tokens; one shared verification pool that never blocks the
hard stop; a mid-run server death requeues the running units and relaunches SGLang at 12 streams (at most 2 times);
coverage and pool maxima in the report.

**Engine:** the E15B notebook's engine cell (server mrr 17, CUDA graphs 1..17, mamba 68), 15 client streams (rev 2: KV per
request 56k; the E15B Duck ran 16), DFlash width 6, the
min-free-slots delayer off, /dev/shm staging; 12-stream and W1 fallback launches; rev 2: a smoke failure relaunches at 12
streams. The E14 SGLang backports (44401, 41872) are NOT applied (as in every E15B production run). If SGLang never comes up
or the tokenizer check fails, S60's vLLM FP8 fallback cell runs Poetiq (as in S60).

**Commit:** all 120 evaluation tasks, a 6-h notebook, scored. **Rerun:** every hidden task with the same solver.

**Inputs:** competition data; FP8 `qwen3-8-27b-fp8-repacked-v1` (tokenizer, vLLM fallback); RedHat INT4; DFlash2 bf16;
SGLang 0.5.21 wheelhouse; vLLM wheelhouse; the payload dataset `esfersami50/q27-e15b-payload` with
`q27x_e14_modules_55c1eb7911c6.bin` at its root (without it CELL 1b stops the notebook).
"""
nb = json.load(open(SRC_ENGINE, encoding="utf-8"))
nb["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)}] + [
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(keepends=True)}
    for c in cells]
for i, c in enumerate(cells):
    compile(c, f"<cell{i}>", "exec")
out = OUT
if out.exists() and sha(out) in PUSHED:
    out = out.with_name(out.stem + "_REBUILD.ipynb")
data = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
out.write_bytes(data.encode("utf-8"))
print(out.name, sha(out)[:8], f"{len(data.encode('utf-8')) / 1024:.0f} kB", len(cells), "code cells")
