"""Build Q27-AGENT v1.8.2: the Tufa-style REPL tool agent (python + submit, verified on the demos) from the public
madarshbb/arc-qwen-3-8-27b-agent notebook, on our Qwen3.8-27B-FP8, with our dual-mode rule:

  * commit / interactive run  -> REPL agent on the 12 EVAL12 evaluation tasks, scored, report written
  * competition rerun         -> REPL agent on every hidden test task, submission.json rewritten as attempts finish

The agent package (grids, sandbox, backend, agent, repl, client, repl_agent, run_direct, run_submit, submit_main)
is copied verbatim from the source notebook; only submit_main gets a 3-line dispatch to the new run_eval.py.

usage: python make_q27_agent.py [out.ipynb]
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "LB_compare" / "madarshbb__arc-qwen-3-8-27b-agent" / "arc-qwen-3-8-27b-agent.ipynb"
if not SRC.exists():                           # audit-bundle layout: build/ next to source_notebook/
    SRC = HERE.parent / "source_notebook" / "arc-qwen-3-8-27b-agent.ipynb"
DST = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path(r"D:\ARC AGI 2\Research Base Experiment\Q27-AGENT-v1_8_2.ipynb") if (HERE.parent / "LB_compare").exists()
    else HERE.parent / "notebook" / "Q27-AGENT-v1_8_2_REBUILT.ipynb")

src = json.load(open(SRC, encoding="utf-8"))
cells = ["".join(c["source"]) for c in src["cells"]]
assert len(cells) == 14, len(cells)
assert cells[8].startswith("%%writefile /kaggle/working/agent/repl_agent.py")
assert cells[11].startswith("%%writefile /kaggle/working/agent/submit_main.py")


def patch(i, old, new):
    """v1.1: exact, single-occurrence replacement in source cell i (documented in provenance)."""
    assert cells[i].count(old) == 1, (i, cells[i].count(old), old[:80])
    cells[i] = cells[i].replace(old, new)


# ---------------------------------------------------------------- v1.1 (GPT audit R1, every point reproduced first)
# client.py: an unclosed final <parameter> (ended by </function>) keeps its code; non-dict tool arguments normalised
patch(7, 'PARAM = re.compile(r"<parameter=([\\w\\-]+)>\\n?(.*?)\\n?</parameter>", re.S)',
      # v1.1: TOOL_BLOCK hands PARAM the body WITHOUT </function>, so an unclosed final parameter ends at the body end
      'PARAM = re.compile(r"<parameter=([\\w\\-]+)>\\n?(.*?)\\n?(?:</parameter>|\\Z)", re.S)   # v1.1')
patch(7, '''        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments")}''', '''        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments")}
        if not isinstance(args, dict):                 # v1.1: a list/str/number would crash .get() later
            args = {"_invalid_arguments": repr(args)[:300]}''')
# v1.5 client.py: the budget-drop retry fired on ANY 400 mentioning "reasoning"/"thinking" (run 1: "Unexpected
# reasoning effort high") and silently dropped the budget; now only an error about the budget itself counts
patch(7, '''any(w in detail.lower() for w in ("thinking", "budget", "reasoning"))''',
      '''any(w in detail.lower() for w in ("thinking_token_budget", "thinking budget", "token budget"))''')
# v1.3 client.py: ONE absolute deadline per logical request (the budget-rejection retry used a fresh timeout), and a
# visible record when the client drops thinking_token_budget at runtime
patch(7, "import json\nimport re\n", "import json\nimport re\nimport time\n")
patch(7, '''    def chat(self, messages, tools=None, max_tokens=None, seed=None, timeout=None,
             tool_choice=None, template_kwargs=None):''', '''    def chat(self, messages, tools=None, max_tokens=None, seed=None, timeout=None,
             tool_choice=None, template_kwargs=None, _deadline=None):
        if _deadline is None:                          # v1.3: absolute deadline shared with the retry below
            _deadline = time.monotonic() + (timeout or self.timeout)
        _remaining = _deadline - time.monotonic()
        if _remaining <= 0:
            return {"error": "request deadline exhausted", "context_overflow": False, "timeout": True}''')
patch(7, '''            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:''',
      '''            with urllib.request.urlopen(req, timeout=_remaining) as r:''')
patch(7, '''                self.thinking_token_budget = None
                return self.chat(messages, tools, max_tokens, seed, timeout, tool_choice, template_kwargs)''',
      '''                self.thinking_token_budget = None
                self.budget_dropped = True             # v1.3: reported by run_eval / smoke, printed once
                print("WARNING: server rejected thinking_token_budget -> dropped for all later calls", flush=True)
                return self.chat(messages, tools, max_tokens, seed, timeout, tool_choice, template_kwargs,
                                 _deadline=_deadline)''')
# repl_agent.py: every request is capped by the attempt's remaining token budget
patch(8, '''        r = client.chat(trim(messages, context_budget, keep_recent, trim_drain, trim_state),
                        tools=tools, seed=seed + att.turns,
                        timeout=req_timeout, max_tokens=turn_max_tokens,''', '''        # v1.1: never ask for more than the attempt has left (the last turn could overshoot by 12k before)
        request_cap = min(turn_max_tokens or client.max_tokens, max_gen_tokens - att.gen_tokens)
        if request_cap < 256:
            att.status, att.note = "failed", "token budget"
            break
        r = client.chat(trim(messages, context_budget, keep_recent, trim_drain, trim_state),
                        tools=tools, seed=seed + att.turns,
                        timeout=req_timeout, max_tokens=request_cap,''')
# repl_agent.py: a tool call without a usable `code` string gets targeted feedback instead of crashing/running ""
patch(8, '''            code = (c["arguments"] or {}).get("code") or ""
            if c["name"] == "python":''', '''            _args = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}   # v1.1
            code = _args.get("code") if isinstance(_args.get("code"), str) else ""
            if c["name"] in ("python", "submit") and not code.strip():
                out = ("[harness] That tool call had no valid 'code' string. Call the tool again with the "
                       "complete Python source in the `code` argument.")
            elif c["name"] == "python":''')
# repl_agent.py: attempt k+1 of a task is queued only after attempt k ended without a verified program
patch(8, '''    # attempt-major order: every task gets its first attempt before any gets a second
    atts = [Attempt(tid, k) for k in range(attempts) for tid in tasks]
    t0 = time.time()''', '''    # v1.1: attempt 0 of every task is queued first; attempt k+1 of a task is queued (at the back) only when its
    # attempt k ended unverified -- two attempts of one task never run at the same time
    atts = []
    t0 = time.time()''')
patch(8, '''    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(work, atts))''', '''    from concurrent.futures import wait, FIRST_COMPLETED
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        live = set()
        for tid in tasks:
            a0 = Attempt(tid, 0)
            atts.append(a0)
            live.add(ex.submit(work, a0))
        while live:
            finished, live = wait(live, return_when=FIRST_COMPLETED)
            for f in finished:
                a = f.result()
                if (a.idx + 1 < attempts and time.time() < deadline
                        and shared.verified.get(a.task_id, 0) < stop_after):
                    nxt = Attempt(a.task_id, a.idx + 1)
                    atts.append(nxt)
                    live.add(ex.submit(work, nxt))''')
# run_submit.py: per-attempt cap = token supply / tasks (no x1.3: a second attempt may follow)
patch(10, 'supply / max(n, 1) * 1.3)))', 'supply / max(n, 1) * cfg.get("cap_factor", 1.0))))   # v1.1: was x1.3')
# v1.6 run_submit.py: HYBRID -- the agent until agent_frac of the time, then DIRECT answers (rescue.py) for every task
# without a verified program; the circuit breaker on every finished attempt
patch(10, '''from repl_agent import solve_repl, AGENT_OPTS      # noqa: E402''',
      '''from repl_agent import solve_repl, AGENT_OPTS, breaker   # noqa: E402''')
patch(10, '''    lock = threading.Lock()
    done = {}

    def on_done(rec):
        with lock:
            done.setdefault(rec["task_id"], []).append(rec)
            task = tasks[rec["task_id"]]
            for j, cands in enumerate(ranked_answers(done[rec["task_id"]], task)):
                inp = task["test"][j]["input"]
                a1 = cands[0] if cands else inp
                a2 = cands[1] if len(cands) > 1 else inp
                sub[rec["task_id"]][j] = {"attempt_1": a1, "attempt_2": a2}
            write_atomic(sub, sub_path)

    _, report = solve_repl(
        tasks, client, attempts=cfg.get("attempts", 1), concurrency=cfg.get("concurrency", 24),
        deadline=deadline, stop_after=1,''', '''    from rescue import run_rescue, merge, direct_grid      # v1.6
    lock = threading.Lock()
    done = {}
    direct_recs = {}

    def tagged(tid):
        """ranked_answers with the verified flag (rescue.merge puts the direct answer after verified grids only)"""
        task, recs = tasks[tid], done.get(tid, [])
        ntr = len(task["train"])
        out = []
        for j, cands in enumerate(ranked_answers(recs, task)):
            ver = {json.dumps(p["test"][j]) for r in recs for p in r["programs"]
                   if p["n_ok"] == ntr and j < len(p["test"]) and p["test"][j] is not None}
            out.append([(g, "verified" if json.dumps(g) in ver else "unverified") for g in cands])
        return out

    def refresh(tid):                                  # caller holds `lock`
        task = tasks[tid]
        for j, ranked in enumerate(tagged(tid)):
            sub[tid][j] = merge(ranked, direct_grid(direct_recs.get(tid), j), task["test"][j]["input"])
        write_atomic(sub, sub_path)

    def on_done(rec):
        breaker(rec)                                   # v1.6: stop early if the first attempts all die on HTTP errors
        with lock:
            done.setdefault(rec["task_id"], []).append(rec)
            refresh(rec["task_id"])

    rescue_log = cfg.get("rescue_log", "/kaggle/working/rescue_attempts.jsonl")

    def log_rescue(tid, rec):                          # v1.8.2: one line per finished test input, flushed
        try:
            with open(rescue_log, "a") as f:
                f.write(json.dumps({"task_id": tid, "status": rec.get("status"), "gen_tokens": rec.get("gen_tokens"),
                                    "answered": sum(1 for g in rec.get("grids") or [] if g is not None)},
                                   default=str) + "\\n")
        except Exception as e:
            print("rescue log failed:", e, flush=True)

    if cfg.get("agent_disabled"):                      # v1.8.2: the agent failed its end-to-end probe
        print("AGENT disabled by the e2e gate -> DIRECT rescue on every task for the whole time", flush=True)

        def on_result0(tid, rec):                      # run_rescue calls this under its own lock
            with lock:
                direct_recs[tid] = rec
                refresh(tid)
                log_rescue(tid, rec)

        recs0, _ = run_rescue(list(tasks), tasks, cfg, deadline, log=lambda m: print(m, flush=True),
                              on_result=on_result0)
        print(f"direct-only done: {sum(1 for r in recs0.values() for g in r['grids'] if g is not None)} test inputs "
              f"answered, {time.time() - t0:.0f}s", flush=True)
        return

    agent_deadline = (t0 + (deadline - t0) * float(cfg.get("agent_frac", 0.5))
                      if cfg.get("rescue") else deadline)
    if cfg.get("rerun_attempt_caps"):                  # v1.7: 240 tasks need a smaller scout than EVAL12 (GPT audit)
        cfg["attempt_caps"] = list(cfg["rerun_attempt_caps"])
    print(f"per-attempt caps {cfg.get('attempt_caps')}", flush=True)
    print(f"agent until {(agent_deadline - time.time()) / 3600:.2f} h from now, then direct rescue until "
          f"{(deadline - time.time()) / 3600:.2f} h", flush=True)
    _, report = solve_repl(
        tasks, client, attempts=cfg.get("attempts", 1), concurrency=cfg.get("concurrency", 24),
        deadline=agent_deadline, stop_after=1,''')
patch(10, '''    ver = sum(1 for a in report["attempts"] if a["status"] == "verified")
    print(f"done: {ver}/{n} tasks verified, {sum(a['gen_tokens'] for a in report['attempts'])} "
          f"tokens, {time.time() - t0:.0f}s", flush=True)''', '''    ver = sum(1 for a in report["attempts"] if a["status"] == "verified")
    print(f"agent done: {ver}/{n} tasks verified, {sum(a['gen_tokens'] for a in report['attempts'])} "
          f"tokens, {time.time() - t0:.0f}s", flush=True)
    if cfg.get("rescue"):
        verified = {a["task_id"] for a in report["attempts"] if a["status"] == "verified"}
        # v1.7: tasks the agent never finished an attempt on come first (nothing at all to submit), then the
        # attempted-but-unverified ones; cheapest prompts first inside each group
        todo = ([t for t in tasks if t not in verified and t not in done]
                + [t for t in tasks if t not in verified and t in done])

        def on_result(tid, rec):                       # run_rescue calls this under its own lock
            with lock:
                direct_recs[tid] = rec
                refresh(tid)
                log_rescue(tid, rec)

        print(f"rescue (direct answers): {len(todo)} unverified tasks, {(deadline - time.time()) / 3600:.2f} h left",
              flush=True)
        recs, _ = run_rescue(todo, tasks, cfg, deadline, log=lambda m: print(m, flush=True), on_result=on_result)
        n_q = sum(len(r["grids"]) for r in recs.values())
        n_g = sum(1 for r in recs.values() for g in r["grids"] if g is not None)
        print(f"rescue done: {n_g}/{n_q} test inputs answered ({len(todo)} tasks), {time.time() - t0:.0f}s",
              flush=True)''')
# last cell: ARC grid limits 1..30, ints only (bool excluded), atomic rewrite
patch(13, '''    def ok_grid(g):
        return (isinstance(g, list) and g and all(isinstance(r, list) and r and len(r) == len(g[0]) for r in g)
                and all(isinstance(v, int) and 0 <= v <= 9 for r in g for v in r))''', '''    def ok_grid(g):   # v1.1: 1..30 on each side, plain ints 0-9 (bool is not a colour)
        return (isinstance(g, list) and 1 <= len(g) <= 30
                and all(isinstance(r, list) and 1 <= len(r) <= 30 and len(r) == len(g[0]) for r in g)
                and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))''')
patch(13, '''        json.dump(sub, open("/kaggle/working/submission.json", "w"))''', '''        with open("/kaggle/working/submission.json.tmp", "w") as _f:   # v1.1: atomic rewrite
            json.dump(sub, _f)
        os.replace("/kaggle/working/submission.json.tmp", "/kaggle/working/submission.json")''')
patch(13, "import json, glob", "import json, glob, os")

# ---------------------------------------------------------------- v1.4 (GPT re-audit: preexec_fn in threads)
# Python docs: preexec_fn "is not safe in the presence of threads ... the child process could deadlock before exec".
# The sandbox and the python tool are called from 16 worker threads -> set RLIMIT_AS inside the child instead.
_CHILD_LIMIT = """try:                                    # v1.4: memory cap set by the child itself (no preexec_fn)
    import os as _q27_os, resource as _q27_res
    _q27_lim = int(_q27_os.environ.get("Q27_CHILD_MAX_AS", "0"))
    if _q27_lim > 0:
        _q27_res.setrlimit(_q27_res.RLIMIT_AS, (_q27_lim, _q27_lim))
except Exception:
    pass
# v1.6 guard: a PEP 578 audit hook (cannot be removed by the program). v1.7 (GPT final audit): installation failure or
# an inactive hook now ABORTS the child (it was fail-open), native routes are narrowed (ctypes symbols for file /
# process functions, cffi, _posixsubprocess.fork_exec), sqlite3/ctypes.dlopen paths are checked. This is a best-effort
# guard for model-written code, NOT a security sandbox (Python documents audit hooks as unsuitable for that).
try:
    import sys as _q27_sys, os as _q27_os2, re as _q27_re
    _q27_deny = tuple(p for p in _q27_os2.environ.get("Q27_DENY_PATHS", "/kaggle/input|solution").lower().split("|") if p)
    _q27_busy = []
    _q27_native = _q27_re.compile(
        r"^_*(open|fopen|freopen|fdopen|wopen|wfopen|creat|read|pread|fread|fgets|fgetc|getline|getdelim|fscanf|mmap|"
        r"syscall|system|popen|wpopen|exec|fexecve|fork|vfork|clone|posix_spawn|dlopen|dlmopen|dlsym|sendfile|splice|"
        r"copy_file_range|opendir|fdopendir|scandir|readlink|createfile|createprocess|winexec|shellexecute|"
        r"loadlibrary|ntcreatefile|ntopenfile|ntreadfile)", _q27_re.I)
    def _q27_hook(event, args, _deny=_q27_deny, _busy=_q27_busy, _native=_q27_native):
        if event in ("ctypes.dlsym", "ctypes.dlsym/handle"):
            _n = str(args[1]) if len(args) > 1 else ""
            if _native.match(_n):
                raise PermissionError("[sandbox] not allowed: native " + _n)
            return
        if event == "import" and args and str(args[0]).split(".")[0] in ("cffi", "_cffi_backend"):
            raise ImportError("[sandbox] not allowed: " + str(args[0]))
        if event in ("open", "os.listdir", "os.scandir", "os.chdir", "os.rename", "os.remove", "shutil.copyfile",
                     "shutil.copytree", "glob.glob", "os.walk", "os.symlink", "os.link", "sqlite3.connect",
                     "ctypes.dlopen"):
            if _busy:
                return
            _busy.append(1)
            try:                        # the path as given AND resolved (relative paths after chdir, symlinks)
                _ps = []
                for _a in args[:2]:
                    if isinstance(_a, (str, bytes)) or hasattr(_a, "__fspath__"):
                        _ps.append(str(_a))
                        try:
                            _ps.append(str(_q27_os2.path.realpath(_a)))
                        except Exception:
                            pass
            finally:
                _busy.pop()
            for _p in _ps:
                if any(d in _p.lower().replace("\\\\", "/") for d in _deny):
                    raise PermissionError("[sandbox] access denied: " + _p)
        elif event in ("subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
                       "socket.connect"):
            raise PermissionError("[sandbox] not allowed: " + event)
    _q27_sys.addaudithook(_q27_hook)
    try:                                # scipy imports subprocess, so the module stays; its raw fork_exec does not
        import _posixsubprocess as _q27_ps
        def _q27_no_fork(*a, **k):
            raise PermissionError("[sandbox] not allowed: fork_exec")
        _q27_ps.fork_exec = _q27_no_fork
    except ImportError:
        pass
    try:                                # canary: the hook must really deny a forbidden path, else stop here
        open("/" + _q27_deny[0].strip("/") + "/__q27_canary__")
        _q27_ok = False
    except PermissionError:
        _q27_ok = True
    except OSError:
        _q27_ok = False
    if not _q27_ok:
        raise RuntimeError("audit hook inactive")
except BaseException as _q27_e:
    import sys as _q27_sys2
    _q27_sys2.stderr.write("[sandbox] guard failed, refusing to run: %r\\n" % (_q27_e,))
    _q27_sys2.stderr.flush()
    _q27_sys2.exit(3)
"""
patch(3, "RUNNER = r'''\nimport sys, json, io, contextlib, copy, traceback\n",
      "RUNNER = r'''\n" + _CHILD_LIMIT + "import sys, json, io, contextlib, copy, traceback\n")
patch(3, """            preexec_fn=_limits(mem_bytes),
            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                     MKL_NUM_THREADS="1", PYTHONHASHSEED="0"))""",
      """            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                     MKL_NUM_THREADS="1", PYTHONHASHSEED="0", Q27_CHILD_MAX_AS=str(mem_bytes)))   # v1.4""")
patch(6, "RUNNER = r'''\nimport sys, json, io, contextlib, traceback\n",
      "RUNNER = r'''\n" + _CHILD_LIMIT + "import sys, json, io, contextlib, traceback\n")
patch(6, """                           capture_output=True, text=True, timeout=timeout, preexec_fn=pre,
                           env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))""",
      """                           capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                    Q27_CHILD_MAX_AS=str(mem_bytes)))   # v1.4: no preexec_fn""")

# ---------------------------------------------------------------- v1.6 (GPT deep audit): agent loop
# circuit breaker: when the first attempts all die on HTTP errors with zero tokens (run 1), stop the whole run early
patch(8, '''def verify(code, task, timeout=10.0):
    """Run a submitted program; return (n_ok, test_outputs, feedback_text)."""
    tr = task["train"]''', '''ABORT = {"flag": False, "why": ""}                 # v1.6: shared stop switch (circuit breaker)
_BREAKER = {"n": 0, "fail": 0, "recent": []}
_BREAKER_LOCK = threading.Lock()


def _request_failure(rec):
    """an attempt that died on a request error (HTTP status, refused connection, ...) without one generated token"""
    return (rec.get("status") == "failed" and rec.get("gen_tokens", 0) == 0
            and str(rec.get("note", "")) not in ("", "token budget", "turn budget"))


def breaker(rec, first_n=6, window=12):
    """v1.6: call once per finished attempt; trips ABORT if the first `first_n` attempts all died on request errors
    with no generated tokens (a request-shape or server problem -- more attempts would only burn the GPU budget).
    v1.7 (GPT final audit): also if the last `window` finished attempts all did (the server died mid-run)."""
    with _BREAKER_LOCK:
        if ABORT["flag"]:
            return
        bad = _request_failure(rec)
        _BREAKER["recent"] = (_BREAKER["recent"] + [bad])[-window:]
        if _BREAKER["n"] < first_n:
            _BREAKER["n"] += 1
            _BREAKER["fail"] += int(bad)
            if _BREAKER["n"] == first_n and _BREAKER["fail"] == first_n:
                ABORT.update(flag=True, why=f"first {first_n} attempts failed on request errors: "
                                            f"{str(rec.get('note', ''))[:200]}")
        if not ABORT["flag"] and len(_BREAKER["recent"]) == window and all(_BREAKER["recent"]):
            ABORT.update(flag=True, why=f"last {window} attempts failed on request errors: "
                                        f"{str(rec.get('note', ''))[:200]}")
        if ABORT["flag"]:
            print("CIRCUIT BREAKER:", ABORT["why"], flush=True)


def _nonportable(code):
    """v1.6: names that exist only inside the python tool; a transform using them fails in the submit sandbox.
    v1.7: scope-aware (repl.tool_only_globals) and includes the preloaded `train`."""
    from repl import tool_only_globals
    return tool_only_globals(code)


def verify(code, task, timeout=10.0):
    """Run a submitted program; return (n_ok, test_outputs, feedback_text)."""
    tr = task["train"]
    _bad = sorted(_nonportable(code))
    if _bad:                                           # v1.6: fail fast with a precise instruction
        return 0, [None] * len(task["test"]), (
            "The program uses " + ", ".join(_bad) + ", which exist only inside the python tool. transform(grid) must "
            "compute everything from its `grid` argument (constants and its own helpers are fine). Rewrite and submit.")''')
patch(8, '''    while att.turns < max_turns:
        with shared.lock:''', '''    while att.turns < max_turns:
        if ABORT["flag"]:                              # v1.6: circuit breaker tripped
            att.status, att.note = "stopped", "aborted: " + ABORT["why"][:150]
            break
        with shared.lock:''')
# per-attempt token caps: a short scout attempt for every task first, a deeper second attempt only where it failed
patch(8, '''            over = dict(kw, **(task_kw or {}).get(att.task_id, {}))
            arm_client = over.pop("client", client)''', '''            over = dict(kw, **(task_kw or {}).get(att.task_id, {}))
            _caps = over.pop("attempt_caps", None)     # v1.6: e.g. [24000, 36000] = scout, then deep
            if _caps:
                over["max_gen_tokens"] = int(_caps[min(att.idx, len(_caps) - 1)])
            arm_client = over.pop("client", client)''')
patch(8, '''              "degenerate_guard", "timeout_as_cut", "snippet_timeout")''',
      '''              "degenerate_guard", "timeout_as_cut", "snippet_timeout", "attempt_caps")   # v1.6''')

# v1.6 repl.py: never KEEP a function that reads the python tool's own globals (GPT audit #10, reproduced: a kept
# transform using train_inputs passed the keep check, then failed in the submit sandbox with NameError)
patch(6, '''    notes = []
    for name, src in kept["defs"].items():
        if store["defs"].get(name) != src:''', '''    notes = []
    for name, src in list(kept["defs"].items()):
        _bad = sorted(tool_only_globals(src))
        if _bad:                                     # v1.6: not portable to submit -> not kept (older copy dropped)
            kept["defs"].pop(name)
            store["defs"].pop(name, None)
            notes.append(f"not kept: {name} (uses {', '.join(_bad)}, which exist only in this tool -- "
                         "compute from the function's own arguments)")
    for name, src in kept["defs"].items():
        if store["defs"].get(name) != src:''')

patch(6, '''def new_store():''', '''TOOL_ONLY = ("train_inputs", "train_outputs", "test_inputs", "train")


def tool_only_globals(src):
    """v1.7 (GPT final audit: `train` is preloaded too): names only the python tool preloads that `src` uses as
    GLOBALS. Scope-aware via symtable: a parameter, a function-local or a program-defined module-level `train` is
    fine; anything that would raise NameError in the submit sandbox is reported."""
    import symtable
    try:
        top = symtable.symtable(src, "<src>", "exec")
    except SyntaxError:
        return set()
    own = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    bad, todo = set(), [top]
    while todo:
        t = todo.pop()
        for s in t.get_symbols():
            n = s.get_name()
            if n in TOOL_ONLY and n not in own and s.is_referenced() and s.is_global():
                bad.add(n)
        todo.extend(t.get_children())
    return bad


def new_store():''')

# v1.8.2 repl.py: the retention doc told the model a kept function "may use ... the preloaded names" -- but submit
# rejects the tool-only ones (GPT audit). Say exactly what is portable.
patch(6, '''A kept function may use builtins, the preloaded names, imports and "
    "other kept functions; pass anything else in as an argument.")''', '''A kept function may use builtins, `np`, the helpers show / diff / objects, imports "
    "and other kept functions. It must NOT read train_inputs, train_outputs, test_inputs or train: "
    "those exist only inside `python`, so a function that uses them is not kept and cannot be "
    "submitted -- compute everything from the function's own arguments.")''')

EVAL12 = ["136b0064", "20a9e565", "21897d95", "247ef758", "2b83f449", "4c3d4a41", "4c7dc4dd", "6e4f6532",
          "88bcf3b4", "898e7135", "8b7bacbf", "b6f77b65"]   # Q27-XHIGH EVAL12 (seed 20261005): direct = 3/22 (run 7)

HEAD = r'''# Q27-AGENT v1.8.2 — Qwen3.8-27B REPL tool agent + DIRECT rescue (HYBRID), 4×L4

**Why.** Our direct-answer pipeline (Q27-XHIGH) scored LB 18.75: the model reasons up to 49k tokens and writes the
grid. The Kaggle discussions attribute the October jump to 60+ to Qwen3.8-27B *agents*; the public
`madarshbb/arc-qwen-3-8-27b-agent` notebook contains a REPL agent modelled on the Tufa Labs duck harness (Tufa Labs is
#1 at 88). This notebook runs that agent on our model.

**How the agent works** (code copied verbatim from the source notebook, cells 2–11):
* two tools: `python(code)` explores the task (grids preloaded as numpy arrays + helpers `show`, `diff`, `objects`);
  `submit(code)` runs a `transform(grid)` program on every train input and returns which examples it reproduced plus a
  cell diff. Reproducing all of them verifies the program; its test outputs become the answer.
* many short turns instead of one 49k-token monologue; per-turn thinking cap (`thinking_token_budget`), early-submit
  nudges, forced submit after 10 python calls, reasoning-effort ladder after a cut turn, function retention, context
  trimming with hysteresis (keeps the vLLM prefix cache valid), repetition-loop guard.
* answers: verified programs vote; otherwise the best unverified programs (most train examples right).
* serving: vLLM OpenAI server from our wheelhouse + the `taaf-duck-qwen38-serving-v1` transformers overlay,
  tool parser `qwen3_coder`, reasoning parser `qwen3`, prefix caching, fp8 KV, MTP (ladder falls back to plain).

**v1.8.2 (GPT deep pre-GPU audit of v1.8.1; each finding reproduced in code first):** GPU run 3 (v1.8) had a live
server that was never detected (`import urllib.request` inside start_server made `urllib` local -> the readiness poll
raised UnboundLocalError, swallowed) -- v1.8.1 removed it, the poll re-raises code errors, and a local test runs the
REAL GPU path (only install / libcuda / the vLLM command substituted). v1.8.2: an agent that fails the end-to-end
probe on every effort stops the commit run (AGENT_E2E_NO_GO) and, in the hidden run, gives the whole time to the
DIRECT method; the DIRECT rescue's thinking and 49k budget no longer follow the agent's thinking fallback; a rejected
rescue probe only disables a LIVE rescue (recorded EVAL answers stay); the retention doc no longer invites tool-only
names; the hidden path logs every rescue answer and the dry-run check verifies it; the report gives the net gain of
HYBRID over DIRECT after the DIRECT hits the agent's tokens displace (the split metric).

**v1.8 (2026-10-12: run 2 crashed; token budget re-checked against our own run records; lessons of every earlier
experiment on this base applied):**
* fix: the smoke's rescue probe crashed on `cfg["base_url"]` (run 2, after ~14 min) -- defaults now; a local
  end-to-end harness runs THIS notebook's cells with the launcher's real cfg (no hand-made cfg any more).
* DIRECT rescue = the LB-18.75 method exactly: one request per test input, v1.4's prompt and grid parser verbatim,
  xhigh, 49,152 thinking + 8,192 answer tokens (v1.7: one 32k request per task -- the EVAL12 records show the correct
  answers needed 26-37k thinking tokens, so it would have lost 3 of 5).
* the commit run spends no GPU on DIRECT: the agent is merged with the LB-18.75 pipeline's recorded EVAL12 answers
  (runs 7 and 4, parsed with the verbatim parser; correctness agrees with the run records 22/22), so the agent keeps
  the whole eval time (default 1.5 h). `eval_direct: "live"` would regenerate them.
* merge: only ONE train-verified grid ranks above the direct answer (R6 code loop: 3 of 4 train-verified programs
  were wrong on test).
* 4,096 tokens of action room after a forced </think> (PIV v2.4G: a 768-token room truncated tool calls).
* DRYRUN: a commit run without a GPU (CPU session, no GPU quota) drives the whole real path against a fake OpenAI
  server (`fake_vllm.py`) on Kaggle's exact image and ends with a check cell (`dryrun_report.json`): smoke, e2e,
  agent loop, live rescue incl. the commit call, eval report fields, the hidden-test path on 20 tasks, and the
  guard on Linux (the solutions file, ctypes/libc, fork_exec, subprocess, an inactive guard).

**v1.7 (GPT final pre-GPU audit: NO-GO for v1.6; every finding checked in code first):**
* the EVAL score is now the DEPLOYED policy (DIRECT only where the agent did not verify -- what the hidden run does);
  the all-tasks-rescued merge is reported beside it (`hybrid_all_rescued_pct`), with agent-only, direct-only, the
  oracle union and how often the merge lost a correct agent / direct answer.
* `verified_but_wrong_tasks` / `_queries` come from the AGENT-only submission (a rescue no longer hides them).
* throughput per stage (agent, rescue, combined); DIRECT answers checkpointed to `rescue_attempts.jsonl` plus a
  deployed snapshot after every answer.
* guard: an install failure or an inactive hook (canary) now stops the child (v1.6 was fail-open); ctypes symbols for
  file / process functions, cffi and `_posixsubprocess.fork_exec` are blocked; sqlite3 / ctypes.dlopen paths checked.
  Still a best-effort guard for model-written code, not a security sandbox; denials are listed in the report.
* `train` (preloaded by the python tool) is now a tool-only name too; the check is scope-aware (symtable): a
  parameter, local or program-defined `train` is fine.
* agent end-to-end probe after the smoke: the agent's exact request (effort, budget, 8,192 max tokens, tools) does
  python -> tool result -> submit that verifies; if the configured effort fails, the first ladder effort that passes
  is used. A rejected rescue request shape disables the rescue (the agent keeps the time).
* circuit breaker also trips if the last 12 attempts all died on request errors (server lost mid-run).
* rerun: scout / deep caps 16k / 32k (EVAL keeps 24k / 36k and reports which verifications fit 16k); the rescue
  starts with tasks the agent never finished an attempt on.

**v1.6 (GPT deep audit DIRECT vs AGENT; each claim checked first):**
* HYBRID: the agent runs until `agent_frac` of the time (rerun 0.5, eval 0.6); then every task without a verified
  program gets a DIRECT answer (`rescue.py` → `run_direct.answer`, xhigh, 32,768 thinking + 4,096 answer tokens, one
  no-thinking commit call if cut). Merge per test input: verified grids → direct grid → unverified grids → test input.
  The commit run rescues EVERY task and reports agent-only / direct-only / hybrid official scores and their overlap.
* isolation: a PEP 578 audit hook in every child (sandbox + python tool) denies opening / listing any path containing
  `/kaggle/input` or `solution` and blocks subprocess / os.system / socket connect (the generated code cannot remove
  it; v1.7 made its failure stop the child); the regex scan of executed code stays as a second, report-only signal.
* two-tier budget `attempt_caps` [24k scout, 36k deep] instead of one 60k cap per attempt.
* circuit breaker: if the first 6 finished attempts all failed on HTTP errors with 0 tokens (run 1's failure mode),
  the agent stops and the time goes to the rescue.
* a submitted `transform` that uses the python-tool-only names `train_inputs` / `train_outputs` / `test_inputs` is
  rejected with feedback instead of failing silently in the sandbox.

**v1.5 (commit run 1 on esfersami50, 2026-10-11):** the infrastructure passed (libcuda probe PASS, rung 0 = MTP + fp8 KV
+ reasoning-config up in 490 s, every smoke stage incl. the roundtrip and the 6,144 budget OK in 64 s), but every
agent request failed: Qwen3.8's chat template rejects `reasoning_effort: "high"` (only xhigh / medium / low). Now
effort `xhigh` (template default; LB 18.75 and ARC Prize 42.4 % used it) with ladder [`medium`, `low`]; the smoke also sends the agent's EXACT request (its effort, sampling, budget,
tools; 32 output tokens) for its own effort and every ladder step, and a rejection stops the ladder (AGENT_PARAMS: a
config error, the same on every rung); the client no longer treats any 400 mentioning "reasoning" as a budget
rejection (run 1 dropped the budget because of that message).

**v1.4 (external re-audit; verified first):** the sandbox and the python tool no longer use `preexec_fn` (Python
documents a deadlock risk with threads; both run from 16 worker threads) -- the child sets its own RLIMIT_AS from
`Q27_CHILD_MAX_AS`; `pip install` has a timeout (wheelhouse 30 min, overlay 10 min); the GPU-drain wait after killing a
server is bounded (240 s, each nvidia-smi call 20 s).

**v1.3 (external audit R3; each point verified first):** the tool smoke adds a ROUNDTRIP on the mode the agent
will use (python call -> tool result sent back as role "tool" -> second request -> a `submit` that verifies, without
tool_choice), so a server that cannot take tool results is rejected; the thinking test sends the agent's real
`thinking_token_budget` (6,144; effort low to stay short); the python check uses a fresh random grid (a hard-coded
print cannot pass); the client keeps ONE absolute deadline per logical request including its budget-rejection retry,
and a runtime budget drop is printed and reported (`thinking_budget_dropped_at_runtime`); a rung that fails to start
is classified from the server log (CUDA_LINKER / UNSUPPORTED_FLAGS / OOM / UNKNOWN) and a CUDA-linker failure stops
the ladder (no rung changes the attention backend, so it would repeat); the file-access detector is wider (still
detection for review, not a sandbox).

**v1.2 (external audit R2; each point reproduced first):** the tool smoke now runs the returned code (a submit
must verify on a tiny identity task and return the right test output; a python call must run and print the shape), the
thinking test fails if the client had to drop `thinking_token_budget` (then the agent runs with thinking off and no
budget, explicitly), the whole smoke has one deadline (<= 180 s, each request <= 75 s) inside the agent reserve, a
failed `-lcuda` probe is reported loudly (not fatal: only FlashInfer JIT needs it); install/start/smoke times and the
time left for the agent are in `agent_server_report.json`; `progress.json` is written atomically; the eval report adds
`verified_but_wrong_tasks` and `file_access_suspicious_tasks` (generated code that touched files/paths, since the
evaluation solutions file exists on disk); the last cell is rewritten to repair bad ROWS (non-dict file, null/str rows,
ragged or out-of-range grids, unknown keys); the notebook is written with LF newlines so its SHA-256 is reproducible.

**v1.1 (external audit R1; every point reproduced or checked in code before the change):**
* server: libcuda shim + `-lcuda` link probe (Q27-XHIGH run 2 failed without it), `--max-num-batched-tokens 4096` and
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` (run-2 OOM fix); each ladder rung must pass a real tool-calling smoke
  test (forced `submit`, automatic `python`, automatic with thinking) before the agent starts -- a failed forced test
  turns forced submits off, a failed thinking test runs the agent without thinking; the ladder stops when less than
  `min_agent_s` (eval 30 min, rerun 60 min) would remain; server PID file + SIGTERM handler, and the launcher kills a
  leftover vLLM process group after `submit_main` ends or is killed; `agent_server_report.json`.
* agent: every request is capped by the attempt's remaining tokens; attempt 2 of a task is queued only after attempt 1
  ended unverified (never both at once); a tool call without a usable `code` string gets targeted feedback; markup
  recovery keeps a final `<parameter>` closed only by `</function>`; non-dict tool arguments are normalised.
* budget: per-attempt cap = token supply / tasks (was x1.3); per turn 6,144 thinking / 8,192 total, 30 turns, nudge
  every 4 python calls, forced submit after 5 (were 8,192 / 12,288 / 40 / 6 / 10: seven full turns filled the 60k cap
  before a forced submit could happen).
* eval report: `official_score_pct` = correct test outputs / all test outputs (Kaggle's metric; the old task average
  is kept as `task_macro_score_pct`); fresh `attempts.jsonl` per run; `progress.json` after every attempt.
* last cell: grids must be 1..30 on each side with plain ints 0-9; the repair is written atomically.

**Modes** (same notebook):
* commit / interactive run: the agent on the 12 EVAL12 evaluation tasks (the same tasks as Q27-XHIGH run 7: direct =
  3/22 queries), budget 2.5 h from the first cell; report in `/kaggle/working/agent_eval/`.
* competition rerun: every hidden test task; `submission.json` is rewritten as each attempt finishes; the agent stops
  at 11 h 35 min and the process is cut at 11 h 50 min (10 min before the 12 h limit). A valid fallback submission is
  written in the first cell, and the last cell repairs any malformed row.
'''

C1 = cells[1]                                  # fallback submission + start epoch (verbatim)

RESCUE = r'''%%writefile /kaggle/working/agent/rescue.py
"""HYBRID rescue: the LB-18.75 DIRECT method for every test input whose task the agent did not verify.

v1.8 (run-record check, 2026-10-12): v1.6/v1.7 asked one 32k-token answer per TASK with another prompt. The EVAL12
records of our DIRECT pipeline (run 4 = v1.4, LB 18.75; run 7 = v1.4.4, LB 26.94; same prompt/parser) show every correct answer needed 26-37k thinking tokens (one more came from
a forced close at 49k), so 32k would have lost 3 of its 5 hits. Now exactly as Q27-XHIGH v1.4 (LB 18.75): one request
per TEST INPUT, its prompt and grid parser copied verbatim, xhigh, temperature 0.6 / top_p 0.95 / top_k 20, thinking
capped at 49,152 tokens and up to 8,192 answer tokens. Difference: the server forces </think> at the budget (v1.4
appended Qwen's stop sentence); 1 of 37 forced closes was correct there, the natural closes carry the method.
A reply without a parseable grid gets ONE no-thinking commit call with the tail of its notes.

Candidate order per test input (merge): the top verified program grid -> the direct grid -> further verified grids ->
unverified grids -> the test input; the first two distinct grids are the attempts. v1.8: only ONE verified grid ranks
above the direct answer (R6 code loop: 3 of 4 train-verified programs were wrong on test).
"""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from client import ChatClient


# ---------------------------------------------------------------- verbatim from Q27-XHIGH v1.4 (LB 18.75), cells 4-5
def render_grid(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)


def build_user_message(train, test_input):
    parts = ["You are solving an abstract reasoning puzzle. Each example has an "
             "INPUT grid and its OUTPUT grid (digits 0-9 are colors; 0 is usually "
             "background). Infer the ONE rule that maps every input to its output, "
             "then apply it to the FINAL INPUT.\n"]
    for i, p in enumerate(train):
        parts.append("Example %d INPUT:\n%s\nExample %d OUTPUT:\n%s\n" %
                     (i + 1, render_grid(p["input"]), i + 1, render_grid(p["output"])))
    parts.append("FINAL INPUT:\n%s\n" % render_grid(test_input))
    parts.append("Reason step by step however you like. Then give ONLY the final "
                 "output grid as rows of space-separated digits inside a fenced "
                 "block:\n```grid\n<rows>\n```\nOutput nothing after the block.")
    return "\n".join(parts)


def valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))


_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
_SEP_ROW = re.compile(r"\[?\s*\d(?:\s*[, ]\s*\d)*\s*\]?,?")
_JSON_ROW = re.compile(r"\[\s*\d(?:\s*,\s*\d)*\s*\]")


def _parse_row(s):
    s = s.strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):                                  # contiguous digits, one per cell ("102")
        return [int(c) for c in s]
    if _SEP_ROW.fullmatch(s):
        return [int(c) for c in re.findall(r"\d", s)]
    return None


def _first_row_run(lines):
    rows = []
    for line in lines:
        r = _parse_row(line)
        if r is None:
            if rows:
                break
            continue
        rows.append(r)
    return rows if valid_grid(rows) else None


def _grid_from_obj(obj):
    if isinstance(obj, dict):
        for k in ("output", "grid", "test_output", "answer", "test_outputs"):
            if k in obj:
                return _grid_from_obj(obj[k])
        return None
    if valid_grid(obj):
        return obj
    if isinstance(obj, list) and obj and valid_grid(obj[-1]):     # a list of grids: the last one
        return obj[-1]
    return None


def _json_rows_grid(b):
    grids, cur = [], []
    for m in _JSON_ROW.finditer(b):
        cur.append(json.loads(m.group(0)))
        if b[m.end():].lstrip().startswith("]"):
            grids.append(cur); cur = []
    if cur:
        grids.append(cur)
    for g in reversed(grids):
        if valid_grid(g):
            return g
    return None


def _grid_from_block(body):
    b = body.strip()
    if b[:1] in ("[", "{"):
        try:
            g = _grid_from_obj(json.loads(b))
        except Exception:
            g = None
        if g is None:
            g = _json_rows_grid(b)
        if g is not None:
            return g
    return _first_row_run(b.splitlines())


def parse_grid(text):
    """answer text after </think> -> grid or None: the last fenced block (any label) that holds a valid grid; else the
    last run of digit rows in the text (unfenced answer or a fence that was never closed)"""
    for _lab, body in reversed(_FENCE_RE.findall(text or "")):
        g = _grid_from_block(body)
        if g is not None:
            return g
    runs, cur = [], []
    for line in (text or "").splitlines():
        r = _parse_row(line)
        if r is None:
            if cur:
                runs.append(cur); cur = []
        else:
            cur.append(r)
    if cur:
        runs.append(cur)
    for run in reversed(runs):
        if valid_grid(run):
            return run
    return None
# ---------------------------------------------------------------- end of the verbatim block


COMMIT = """{prompt}

## Your notes so far (they were cut off)
{notes}

Stop analysing now. Apply the rule you were converging on and give ONLY the final output grid as rows of space-separated digits inside a fenced block:
```grid
<rows>
```"""


def direct_client(cfg):
    """v1.8: base_url / model default to the local server (v1.7 crashed: the smoke passed a cfg without base_url)"""
    think = int(cfg.get("rescue_think", 49152))
    use_think = bool(cfg.get("rescue_thinking", True))   # v1.8.2: NOT the agent's cfg["thinking"] fallback
    return ChatClient(cfg.get("base_url", "http://127.0.0.1:1234/v1"), cfg.get("model", "qwen"),
                      max_tokens=think + int(cfg.get("rescue_answer", 8192)),
                      temperature=cfg.get("rescue_temperature", 0.6), top_p=0.95, top_k=20, thinking=use_think,
                      thinking_token_budget=(think if use_think else None),   # v1.8.2: not the agent's budget
                      template_kwargs={"reasoning_effort": cfg.get("rescue_effort", "xhigh")})


def direct_answer(client, task, j, seed, deadline, commit_tokens=8192, notes_chars=16000):
    """one LB-18.75 request for test input j; returns a record with "grid" (or None)"""
    prompt = build_user_message(task["train"], task["test"][j]["input"])

    def timeout():
        return max(60.0, min(4 * 3600, deadline - time.time() + 60))

    mt = client.max_tokens
    for _ in range(3):
        r = client.chat([{"role": "user", "content": prompt}], seed=seed, max_tokens=mt, timeout=timeout())
        if not (r.get("context_overflow") and mt > 16384):
            break
        mt = int(mt * 0.8)                     # a very long prompt: shrink the reply, never give up
    if "error" in r:
        return {"status": "error: " + str(r["error"])[:200], "grid": None, "gen_tokens": 0}
    rec = {"finish": r.get("finish_reason"), "gen_tokens": int(r.get("completion_tokens") or 0), "max_tokens": mt,
           "reasoning_chars": len(r.get("reasoning") or ""), "answer_tail": (r.get("content") or "")[-1200:],
           "reasoning_tail": (r.get("reasoning") or "")[-2000:]}
    grid = parse_grid(r.get("content") or "")
    if grid is not None:
        return dict(rec, status="answered", grid=grid)
    notes = ((r.get("reasoning") or "") + "\n" + (r.get("content") or "")).strip()
    if notes and deadline - time.time() > 120:
        r2 = client.chat([{"role": "user", "content": COMMIT.format(prompt=prompt, notes=notes[-notes_chars:])}],
                         seed=seed + 1, max_tokens=commit_tokens, timeout=timeout(),
                         template_kwargs={"enable_thinking": False, "reasoning_effort": "low"})
        if "error" in r2:
            rec["commit_error"] = str(r2["error"])[:200]
        else:
            rec["gen_tokens"] += int(r2.get("completion_tokens") or 0)
            rec["commit_finish"] = r2.get("finish_reason")
            grid = parse_grid(r2.get("content") or "") or parse_grid(r2.get("reasoning") or "")
            if grid is not None:
                return dict(rec, status="committed", grid=grid)
    return dict(rec, status="no grid", grid=None)


def run_rescue(task_ids, tasks, cfg, deadline, log=print, on_result=None):
    """direct answers for every test input of task_ids (in that order), concurrently.
    Returns ({tid: {"grids": [grid or None per test input], "units": {j: rec}, "gen_tokens", "status"}}, client)."""
    client = direct_client(cfg)
    out, lock = {}, threading.Lock()
    for tid in task_ids:
        out[tid] = {"grids": [None] * len(tasks[tid]["test"]), "units": {}, "gen_tokens": 0, "status": "pending"}
    units = [(tid, j) for tid in task_ids for j in range(len(tasks[tid]["test"]))]

    def work(unit):
        tid, j = unit
        if deadline - time.time() < float(cfg.get("rescue_min_start_s", 300)):
            rec = {"status": "skipped: deadline", "grid": None, "gen_tokens": 0}
        else:
            t = time.time()
            try:
                rec = direct_answer(client, tasks[tid], j, 7 + 1000 * j, deadline,
                                    commit_tokens=int(cfg.get("rescue_commit_tokens", 8192)))
            except Exception as e:                       # one unit must never sink the stage
                rec = {"status": f"crashed: {type(e).__name__}: {e}"[:200], "grid": None, "gen_tokens": 0}
            rec["wall"] = round(time.time() - t, 1)
        with lock:
            tr = out[tid]
            tr["units"][j] = rec
            tr["grids"][j] = rec.get("grid")
            tr["gen_tokens"] += int(rec.get("gen_tokens") or 0)
            sts = [str(u.get("status")).split(":")[0] for u in tr["units"].values()]
            tr["status"] = sts[0] if len(set(sts)) == 1 else "mixed"
            if on_result:
                try:
                    on_result(tid, tr)
                except Exception as e:
                    log(f"[rescue] on_result failed for {tid}: {e}")
        log(f"[rescue] {tid}#{j} {str(rec.get('status'))[:30]} tok {rec.get('gen_tokens', 0)} "
            f"finish {rec.get('finish')}")

    with ThreadPoolExecutor(max_workers=int(cfg.get("rescue_concurrency", cfg.get("concurrency", 16)))) as ex:
        list(ex.map(work, units))
    return out, client


def direct_grid(rec, j):
    grids = (rec or {}).get("grids")
    return grids[j] if isinstance(grids, list) and j < len(grids) else None


def merge(agent_ranked, direct, inp):
    """agent_ranked: [(grid, source)] best first, source 'verified ...' or 'unverified ...'; direct: grid or None"""
    ver = [g for g, s in agent_ranked if str(s).startswith("verified")]
    unv = [g for g, s in agent_ranked if not str(s).startswith("verified")]
    order = (ver[:1] + [direct] + ver[1:] + unv) if direct is not None else (ver + unv)
    out = []
    for g in order + [inp]:
        if g is not None and g not in out:
            out.append(g)
        if len(out) == 2:
            break
    while len(out) < 2:
        out.append(inp)
    return {"attempt_1": out[0], "attempt_2": out[1]}


def hybrid_submission(answers, rescue_recs, tasks):
    return {t: [merge(answers.get(t, [[]] * len(task["test"]))[j], direct_grid(rescue_recs.get(t), j), x["input"])
                for j, x in enumerate(task["test"])] for t, task in tasks.items()}


def direct_submission(rescue_recs, tasks):
    sub = {}
    for t, task in tasks.items():
        rows = []
        for j, x in enumerate(task["test"]):
            g = direct_grid(rescue_recs.get(t), j)
            rows.append({"attempt_1": g if g is not None else x["input"], "attempt_2": x["input"]})
        sub[t] = rows
    return sub


def hits(sub, sol, ids):
    """per-query correctness (Kaggle metric: either attempt exact)"""
    return {(t, j): int(s in (sub[t][j]["attempt_1"], sub[t][j]["attempt_2"]))
            for t in ids for j, s in enumerate(sol[t])}


def load_direct_ref(path, run="run7"):
    """v1.8: the LB-18.75 pipeline's own answers on EVAL12 (runs 4 and 7, parsed with the verbatim parser) in the
    rescue-record shape, so the commit run can merge the agent with DIRECT exactly without spending GPU on DIRECT"""
    ref = json.load(open(path))
    return {t: {"grids": [q.get("grid") for q in qs], "gen_tokens": sum(int(q.get("think_tokens") or 0) for q in qs),
                "status": "reference", "units": {j: q for j, q in enumerate(qs)}}
            for t, qs in ref[run].items()}
'''

# v1.8: the LB-18.75 pipeline's answers on EVAL12 (runs 4 + 7; parsed with the verbatim v1.4 parser; correctness agrees
# with the run records 22/22 for both runs) -> the commit run merges the agent with DIRECT exactly, no GPU for DIRECT
DIRECT_REF = "%%writefile /kaggle/working/agent/direct_ref_eval12.json\n" + (HERE / "direct_ref_eval12.json").read_text()

RUN_EVAL = r'''%%writefile /kaggle/working/agent/run_eval.py
"""Commit-run evaluation: the REPL agent on a fixed set of evaluation tasks, scored against the public solutions.

    python run_eval.py '<json config>'   (normally called by submit_main once the server is up)

Writes /kaggle/working/agent_eval/attempts.jsonl (one line per finished attempt, with transcript) and
eval_report.json (score, per task, tokens, turns, statuses). Never touches /kaggle/working/submission.json.
"""
import glob
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import ChatClient                      # noqa: E402
from repl_agent import solve_repl, AGENT_OPTS, breaker, ABORT   # noqa: E402
from agent import to_submission, score             # noqa: E402
from rescue import run_rescue, hybrid_submission, direct_submission, hits, load_direct_ref   # noqa: E402  (v1.6/1.8)


def find(name, root="/kaggle/input"):
    found = glob.glob(f"{root}/**/{name}", recursive=True)
    if not found:
        raise FileNotFoundError(name)
    return found[0]


def main(cfg):
    t0 = time.time()
    root = cfg.get("data_dir", "/kaggle/input")
    ch = json.load(open(find("arc-agi_evaluation_challenges.json", root)))
    sol = json.load(open(find("arc-agi_evaluation_solutions.json", root)))
    ids = [t for t in cfg["eval_task_ids"] if t in ch]
    tasks = {t: ch[t] for t in ids}
    out = Path(cfg.get("eval_out", "/kaggle/working/agent_eval"))
    out.mkdir(parents=True, exist_ok=True)
    client = ChatClient(cfg["base_url"], cfg["model"], max_tokens=cfg.get("max_tokens", 16384),
                        temperature=cfg.get("temperature", 0.6), top_p=cfg.get("top_p", 0.95),
                        top_k=cfg.get("top_k", 20), thinking=cfg.get("thinking", True),
                        reasoning_effort=cfg.get("reasoning_effort"),
                        thinking_token_budget=cfg.get("thinking_token_budget"),
                        template_kwargs=cfg.get("template_kwargs"))
    lock = threading.Lock()
    jsonl = open(out / "attempts.jsonl", "w", encoding="utf-8")     # v1.1: fresh log per run
    progress = {"attempts_done": 0, "verified_tasks": set(), "gen_tokens": 0, "statuses": Counter()}

    def on_done(rec):
        breaker(rec)                                   # v1.6: stop early if the first attempts all die on HTTP errors
        with lock:
            jsonl.write(json.dumps(rec, default=str) + "\n")
            jsonl.flush()
            # v1.1: a progress snapshot after every attempt survives a deadline cut or a killed process
            progress["attempts_done"] += 1
            progress["gen_tokens"] += rec.get("gen_tokens", 0)
            progress["statuses"][rec.get("status")] += 1
            if rec.get("status") == "verified":
                progress["verified_tasks"].add(rec["task_id"])
            (out / "progress.json.tmp").write_text(json.dumps(
                {"attempts_done": progress["attempts_done"], "verified_tasks": sorted(progress["verified_tasks"]),
                 "gen_tokens": progress["gen_tokens"], "statuses": dict(progress["statuses"]),
                 "elapsed_s": round(time.time() - t0)}, indent=1))
            os.replace(out / "progress.json.tmp", out / "progress.json")     # v1.2: atomic

    # v1.6: the agent gets agent_frac of the time; the rest is for DIRECT rescue answers.
    # v1.8: eval_direct "ref" (default) merges with the LB-18.75 pipeline's recorded EVAL12 answers instead of
    # re-generating them (~1.1M tokens, ~70 GPU-min) -> the agent keeps the whole eval time; "live" re-runs DIRECT
    direct_mode = cfg.get("eval_direct", "ref") if cfg.get("rescue") else "off"
    agent_deadline = (t0 + (cfg["deadline_epoch"] - t0) * float(cfg.get("agent_frac", 0.6))
                      if direct_mode == "live" else cfg["deadline_epoch"])
    print(f"eval: {len(tasks)} tasks, agent deadline in {(agent_deadline - time.time()) / 60:.0f} min, "
          f"final deadline in {(cfg['deadline_epoch'] - time.time()) / 60:.0f} min", flush=True)
    answers, report = solve_repl(
        tasks, client, attempts=cfg.get("attempts", 2), concurrency=cfg.get("concurrency", 16),
        deadline=agent_deadline, stop_after=1, max_turns=cfg.get("max_turns", 40),
        max_gen_tokens=cfg.get("eval_cap", 60000), context_budget=cfg.get("context_budget", 28000),
        nudge_every=cfg.get("nudge_every", 0), turn_max_tokens=cfg.get("turn_max_tokens"),
        in_python=cfg.get("in_python", False), log=lambda m: print(m, flush=True), on_done=on_done,
        **{k: cfg[k] for k in AGENT_OPTS if k in cfg and k not in ("turn_max_tokens", "in_python")})
    jsonl.close()
    # v1.6 HYBRID: DIRECT answers for tasks the agent did not verify (rescue_all: every task, to measure overlap)
    atts0 = report["attempts"]
    verified0 = {a["task_id"] for a in atts0 if a["status"] == "verified"}
    rescue_recs, rescue_client = {}, None
    rescue_t0 = rescue_t1 = time.time()
    ref_path = Path(__file__).with_name("direct_ref_eval12.json")
    ref_run4 = {}
    if direct_mode == "ref":
        ref = load_direct_ref(ref_path, cfg.get("eval_direct_run", "run7"))
        rescue_recs = {t: ref[t] for t in ids if t in ref and (cfg.get("rescue_all") or t not in verified0)}
        ref_run4 = {t: r for t, r in load_direct_ref(ref_path, "run4").items() if t in ids}
        print(f"DIRECT from the LB-18.75 run records ({cfg.get('eval_direct_run', 'run7')}): {len(rescue_recs)} tasks",
              flush=True)
    elif direct_mode == "live":
        ids_r = ids if cfg.get("rescue_all") else [t for t in ids if t not in verified0]
        print(f"rescue (direct answers): {len(ids_r)} tasks, {(cfg['deadline_epoch'] - time.time()) / 60:.0f} min left",
              flush=True)
        rjsonl = open(out / "rescue_attempts.jsonl", "w", encoding="utf-8")
        done_r = {}

        def on_rescue(tid, rec):                       # v1.7: durable record + deployed snapshot after every answer
            rjsonl.write(json.dumps(dict(rec, task_id=tid), default=str) + "\n")
            rjsonl.flush()
            done_r[tid] = rec
            snap = hybrid_submission(answers, {t: r for t, r in done_r.items() if t not in verified0}, tasks)
            (out / "eval_submission.partial.json.tmp").write_text(json.dumps(snap))
            os.replace(out / "eval_submission.partial.json.tmp", out / "eval_submission.partial.json")

        rescue_t0 = time.time()
        rescue_recs, rescue_client = run_rescue(ids_r, tasks, cfg, cfg["deadline_epoch"],
                                                log=lambda m: print(m, flush=True), on_result=on_rescue)
        rescue_t1 = time.time()
        rjsonl.close()
    sol_ids = {t: sol[t] for t in ids}
    agent_sub = to_submission(answers, tasks)
    direct_sub = direct_submission(rescue_recs, tasks)
    # v1.7 (GPT final audit): the DEPLOYED policy rescues only tasks the agent did not verify -- that is the score that
    # predicts the hidden run; the all-tasks-rescued merge is a counterfactual reported beside it
    deployed_rescue = {t: r for t, r in rescue_recs.items() if t not in verified0}
    sub = hybrid_submission(answers, deployed_rescue, tasks)
    all_sub = hybrid_submission(answers, rescue_recs, tasks)
    h_agent, h_direct, h_hybrid, h_all = (hits(x, sol_ids, ids) for x in (agent_sub, direct_sub, sub, all_sub))
    rescued_q = [(t, j) for t in rescue_recs for j in range(len(sol[t]))]
    r_status = Counter(str(u.get("status")).split(":")[0] for r in rescue_recs.values()
                       for u in (r.get("units") or {}).values())             # v1.8: per test input
    complementarity = {
        "queries": len(h_agent),
        "agent_correct": sum(h_agent.values()), "direct_correct": sum(h_direct.values()),
        "hybrid_correct": sum(h_hybrid.values()), "hybrid_all_rescued_correct": sum(h_all.values()),
        "agent_only": sum(1 for k in h_agent if h_agent[k] and not h_direct[k]),
        "direct_only": sum(1 for k in h_agent if h_direct[k] and not h_agent[k]),
        "both": sum(1 for k in h_agent if h_agent[k] and h_direct[k]),
        "oracle_union_hits": sum(1 for k in h_agent if h_agent[k] or h_direct[k]),   # retrospective bound, not a policy
        "hybrid_lost_vs_agent": sum(1 for k in h_agent if h_agent[k] and not h_hybrid[k]),
        "hybrid_lost_vs_direct": sum(1 for k in h_agent if h_direct[k] and not h_hybrid[k]),
        "all_rescued_lost_vs_agent": sum(1 for k in h_agent if h_agent[k] and not h_all[k]),
        "all_rescued_lost_vs_direct": sum(1 for k in h_agent if h_direct[k] and not h_all[k]),
        "direct_correct_on_unverified_queries": sum(h_direct[k] for k in h_direct if k[0] not in verified0),
        "direct_correct_on_rescued_queries": sum(h_direct[k] for k in rescued_q),
        "rescued_queries": len(rescued_q),
        "rescue_statuses": dict(r_status),
        "rescue_gen_tokens": sum(r.get("gen_tokens", 0) for r in rescue_recs.values()),
        "rescue_budget_dropped": bool(getattr(rescue_client, "budget_dropped", False)),
        "rescue_smoke_ok": cfg.get("rescue_smoke_ok"),
        "direct_source": {"ref": {"run7": "recorded: Q27-XHIGH v1.4.4 (LB 26.94), run 7",
                                  "run4": "recorded: Q27-XHIGH v1.4 (LB 18.75), run 4"}.get(
                                      str(cfg.get("eval_direct_run", "run7")), str(cfg.get("eval_direct_run"))),
                          "live": "generated in this run", "off": "none"}[direct_mode],
        # v1.7: the three-way comparison is only complete if DIRECT really answered (or tried) every task
        "three_way_complete": bool(direct_mode != "off" and cfg.get("rescue_all")
                                   and (direct_mode == "ref" or cfg.get("rescue_smoke_ok", True))
                                   and len(rescue_recs) == len(ids)
                                   and not any(s in r_status for s in ("skipped", "crashed", "error"))),
    }
    # v1.8.2 (GPT audit): the decision metric for the hidden-run split -- net exact-output gain of the deployed HYBRID
    # over DIRECT alone, minus the DIRECT hits the agent's tokens would have displaced (DIRECT tokens per query and
    # hit rate from the same reference)
    _units = [u for r in rescue_recs.values() for u in (r.get("units") or {}).values()]
    _dtok = [int(u.get("think_tokens") or u.get("gen_tokens") or 0) for u in _units]
    _dtok_q = (sum(_dtok) / len(_dtok)) if _dtok and sum(_dtok) else None
    _agent_tok = sum(a["gen_tokens"] for a in report["attempts"])
    _q_n = max(len(h_agent), 1)
    _disp_q = (_agent_tok / _dtok_q) if _dtok_q else None
    _disp_hits = (_disp_q * complementarity["direct_correct"] / _q_n) if _disp_q is not None else None
    complementarity["net_gain"] = {
        "hybrid_minus_direct_hits": complementarity["hybrid_correct"] - complementarity["direct_correct"],
        "agent_tokens": _agent_tok, "direct_tokens_per_query": round(_dtok_q) if _dtok_q else None,
        "displaced_direct_queries_equiv": round(_disp_q, 2) if _disp_q is not None else None,
        "expected_displaced_direct_hits": round(_disp_hits, 3) if _disp_hits is not None else None,
        "net_after_displacement": (round(complementarity["hybrid_correct"] - complementarity["direct_correct"]
                                         - _disp_hits, 3) if _disp_hits is not None else None)}
    if ref_run4:                                       # v1.8: the same merge against the other recorded DIRECT run
        h4 = hits(direct_submission(ref_run4, tasks), sol_ids, ids)
        h4h = hits(hybrid_submission(answers, {t: r for t, r in ref_run4.items() if t not in verified0}, tasks),
                   sol_ids, ids)
        complementarity["reference_run4"] = {"direct_correct": sum(h4.values()), "hybrid_correct": sum(h4h.values()),
                                             "agent_only_vs_run4": sum(1 for k in h_agent if h_agent[k] and not h4[k])}
    total, pct, per = score(sub, sol_ids)
    _, _, agent_per = score(agent_sub, sol_ids)
    q_tot = len(h_hybrid)
    q_hit = sum(h_hybrid.values())
    atts = report["attempts"]
    agent_gen = sum(a["gen_tokens"] for a in atts)
    agent_wall = float(report.get("wall_s") or (rescue_t0 - t0))
    rescue_gen = complementarity["rescue_gen_tokens"]
    rescue_wall = max(0.0, rescue_t1 - rescue_t0) if direct_mode == "live" else 0.0
    if direct_mode != "live":
        rescue_gen = 0                                 # reference tokens were spent in runs 4 / 7, not here
    wall = time.time() - t0
    official_pct = 100.0 * q_hit / q_tot if q_tot else 0.0
    verified_tasks = sorted({a["task_id"] for a in atts if a["status"] == "verified"})
    scout_cap = (cfg.get("rerun_attempt_caps") or cfg.get("attempt_caps") or [0])[0]
    # v1.2: generated code runs with the real filesystem, where the evaluation SOLUTIONS file exists. Flag every task
    # whose executed code touched files/paths so a score cannot silently come from reading answers.
    # v1.3: wider net (audit R3 read a file via builtins) -- this is DETECTION for manual review, not a sandbox
    leak_re = re.compile(r"solution|/kaggle/|open\s*\(|listdir|scandir|glob|pathlib|Path\s*\(|walk\s*\(|builtins|"
                         r"__import__|importlib|\bio\b|\bos\b|\bsys\b|subprocess|shutil|socket|eval\s*\(|exec\s*\(|"
                         r"compile\s*\(|getattr\s*\(|globals\s*\(|locals\s*\(|vars\s*\(|ctypes|cffi|mmap", re.I)
    suspicious = sorted({a["task_id"] for a in atts for x in a.get("transcript", [])
                         if isinstance(x, dict) and isinstance(x.get("code"), str) and leak_re.search(x["code"])})
    tdump = {id(a): json.dumps(a.get("transcript", []), default=str) for a in atts}
    summary = {
        # v1.7 (GPT final audit): from the AGENT-only submission -- a DIRECT rescue must not hide a wrong verified program
        "verified_but_wrong_tasks": [t for t in verified_tasks if agent_per.get(t, 0) < 1],   # train-fit != test-correct
        "verified_but_wrong_queries": [f"{t}#{j}" for t in verified_tasks for j in range(len(sol[t]))
                                       if not h_agent[(t, j)]],
        "file_access_suspicious_tasks": suspicious,
        # v1.7: what the guard actually stopped, and children that refused to run because the guard failed
        "isolation_denied_tasks": sorted({a["task_id"] for a in atts if "[sandbox] access denied" in tdump[id(a)]
                                          or "[sandbox] not allowed" in tdump[id(a)]}),
        "isolation_guard_failed_tasks": sorted({a["task_id"] for a in atts if "[sandbox] guard failed" in tdump[id(a)]}),
        "thinking_budget_dropped_at_runtime": bool(getattr(client, "budget_dropped", False)),   # v1.3
        # v1.1: Kaggle's metric = correct test outputs / all test outputs (micro); the task average is kept beside it
        # v1.7: official = the DEPLOYED hybrid (rescue only where the agent did not verify)
        "official_score_pct": round(official_pct, 2), "task_macro_score_pct": round(pct, 2),
        "hybrid_all_rescued_pct": round(100.0 * complementarity["hybrid_all_rescued_correct"] / max(q_tot, 1), 2),
        "agent_only_official_pct": round(100.0 * complementarity["agent_correct"] / max(q_tot, 1), 2),
        "direct_only_official_pct": round(100.0 * complementarity["direct_correct"] / max(q_tot, 1), 2),
        "complementarity": complementarity,
        "circuit_breaker": dict(ABORT),
        "agent_deadline_epoch": agent_deadline,
        "score_tasks": round(total, 3), "score_pct": round(pct, 2), "n_tasks": len(ids),
        "query_hits": q_hit, "queries": q_tot, "per_task": per, "per_task_agent_only": agent_per,
        "tasks_with_verified_program": len(verified_tasks),
        # v1.7: how many verifications the hidden run's smaller scout cap would have kept (counterfactual)
        "verified_first_attempt_within_rerun_scout_cap": {"cap": scout_cap, "tasks": sorted(
            {a["task_id"] for a in atts if a["status"] == "verified" and a["idx"] == 0
             and a["gen_tokens"] <= scout_cap})},
        "status_counts": dict(Counter(a["status"] for a in atts)),
        "notes": dict(Counter(a["note"] for a in atts if a["note"])),
        # v1.7 (GPT final audit): throughput per stage (the old figure divided agent tokens by agent + rescue time)
        "agent_generated_tokens": agent_gen, "agent_wall_s": round(agent_wall, 1),
        "agent_gen_tok_per_s": round(agent_gen / max(agent_wall, 1e-6), 1),
        "rescue_generated_tokens": rescue_gen, "rescue_wall_s": round(rescue_wall, 1),
        "rescue_gen_tok_per_s": round(rescue_gen / max(rescue_wall, 1e-6), 1),
        "combined_gen_tok_per_s": round((agent_gen + rescue_gen) / max(wall, 1e-6), 1),
        "gen_tokens_total": agent_gen + rescue_gen, "gen_tokens_mean_per_attempt": round(agent_gen / max(len(atts), 1)),
        "turns_mean": round(sum(a["turns"] for a in atts) / max(len(atts), 1), 1),
        "cuts_total": sum(a["cuts"] for a in atts), "submits_total": sum(a["submits"] for a in atts),
        "python_calls_total": sum(a["python_calls"] for a in atts),
        "wall_s": round(wall),
        "config": {k: v for k, v in cfg.items() if k not in ("base_url",)},
        "per_attempt": [{k: a[k] for k in ("task_id", "idx", "status", "note", "turns", "python_calls", "submits",
                                           "cuts", "forced", "gen_tokens", "prompt_tokens", "wall", "best_ok",
                                           "n_train", "effort_steps", "evictions")} for a in atts]}
    (out / "eval_report.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "eval_submission.json").write_text(json.dumps(sub))                     # the deployed policy
    (out / "eval_all_rescued_submission.json").write_text(json.dumps(all_sub))     # counterfactual
    print(f"\nscores (official = deployed hybrid): hybrid {official_pct:.1f}% | all-rescued "
          f"{summary['hybrid_all_rescued_pct']}% | agent {summary['agent_only_official_pct']}% | direct "
          f"{summary['direct_only_official_pct']}% | {complementarity}", flush=True)
    print(f"\nEVAL OFFICIAL {official_pct:.1f}% = queries {q_hit}/{q_tot} | task average {total:.2f} / {len(ids)} "
          f"({pct:.1f}%) | verified tasks {summary['tasks_with_verified_program']} "
          f"(wrong on test: {summary['verified_but_wrong_tasks']}) | agent {agent_gen} tok "
          f"({summary['agent_gen_tok_per_s']} tok/s) | rescue {rescue_gen} tok ({summary['rescue_gen_tok_per_s']} tok/s) "
          f"| wall {wall / 60:.0f} min | {summary['status_counts']}", flush=True)
    print("per task:", {t: round(v, 2) for t, v in per.items()}, flush=True)

if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
'''

C11 = cells[11]
_old = '''        sys.path.insert(0, HERE)
        import run_submit
        run_submit.main(dict(cfg, base_url="http://127.0.0.1:1234/v1", model="qwen"))'''
assert C11.count(_old) == 1
C11 = C11.replace(_old, '''        sys.path.insert(0, HERE)
        if cfg.get("eval"):                  # Q27-AGENT: commit run = evaluation subset, scored
            import run_eval
            run_eval.main(dict(cfg, base_url="http://127.0.0.1:1234/v1", model="qwen"))
        else:
            import run_submit
            run_submit.main(dict(cfg, base_url="http://127.0.0.1:1234/v1", model="qwen"))''')


def patch11(old, new):
    global C11
    assert C11.count(old) == 1, (C11.count(old), old[:80])
    C11 = C11.replace(old, new)


# v1.1 submit_main.py: libcuda shim + link probe, memory-safe flags, deadline-bounded ladder, real tool smoke gate,
# server PID file + SIGTERM cleanup, server report
patch11('''HERE = os.path.dirname(os.path.abspath(__file__))''', '''HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_PID = Path("/kaggle/working/vllm_server.pid")     # v1.1: lets the launcher kill a leftover server group
REPORT_PATH = Path("/kaggle/working/agent_server_report.json")
LINK_DIR = Path("/tmp/q27a_cuda_link")
CUDA_LINK = {}
CURRENT = {"proc": None}


def ensure_libcuda():
    """v1.1 (Q27-XHIGH run 2: FlashInfer JIT failed with `ld: cannot find -lcuda`; Kaggle ships only libcuda.so.1).
    Make -lcuda linkable through a private symlink directory and prove it with a tiny link."""
    import shutil as _sh
    info = {"needed": True}
    cands = []
    for d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/lib64", "/usr/lib64"):
        cands += sorted(glob.glob(f"{d}/libcuda.so.1")) + sorted(glob.glob(f"{d}/libcuda.so.*[0-9]"))
    real = next((Path(c) for c in cands if Path(c).is_file()), None)
    if real is None:
        info.update(probe="SKIP", reason="libcuda.so.1 not found")
    else:
        LINK_DIR.mkdir(parents=True, exist_ok=True)
        link = LINK_DIR / "libcuda.so"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(real.resolve())
        info.update(real=str(real), shim=str(link))
        cxx = _sh.which("c++") or _sh.which("g++") or _sh.which("gcc")
        if cxx:
            src = Path("/tmp/q27a_cuda_probe.c")
            src.write_text("int cuInit(unsigned int);\\nint main(void){return 0;}\\nvoid *p = (void *)cuInit;\\n")
            env = dict(os.environ, LIBRARY_PATH=f"{LINK_DIR}:" + os.environ.get("LIBRARY_PATH", ""))
            r = subprocess.run([cxx, "-x", "c", str(src), "-o", "/tmp/q27a_cuda_probe", "-lcuda"],
                               capture_output=True, text=True, env=env, timeout=120)
            info.update(probe="PASS" if r.returncode == 0 else "FAIL", probe_out=(r.stdout + r.stderr)[-600:])
        else:
            info.update(probe="SKIP", reason="no C compiler")
    CUDA_LINK.clear()
    CUDA_LINK.update(info)
    print("libcuda link:", {k: v for k, v in info.items() if k != "probe_out"}, flush=True)
    return info


def _on_sigterm(signum, frame):
    """v1.1: a SIGTERM to this launcher also ends the vLLM server's process group (it runs in its own session)."""
    if CURRENT["proc"] is not None:
        try:
            os.killpg(CURRENT["proc"].pid, 9)
        except Exception:
            pass
    sys.exit(0)


def classify_startup_failure(path=None):
    """v1.3: why did a rung fail to come up? (read from the tail of the server log)"""
    try:
        tail = Path(path or LOG).read_text(errors="replace")[-20000:].lower()
    except Exception:
        return "NO_LOG"
    if "cannot find -lcuda" in tail or "cannot find libcuda" in tail:
        return "CUDA_LINKER"
    if "unrecognized arguments" in tail or "error: argument" in tail:
        return "UNSUPPORTED_FLAGS"
    if "out of memory" in tail or "outofmemory" in tail:
        return "OOM"
    return "UNKNOWN"


def _smoke_task():
    """v1.3: a fresh random identity task per smoke, so a hard-coded print cannot pass the python check."""
    import random
    h, w = random.choice([(2, 3), (3, 2), (2, 4), (4, 3), (3, 5), (5, 2)])
    grid = [[random.randint(1, 9) for _ in range(w)] for _ in range(h)]
    test = [[random.randint(1, 9) for _ in range(w)] for _ in range(h)]
    return {"train": [{"input": grid, "output": grid}], "test": [{"input": test}]}, f"({h}, {w})"


def smoke(cfg, smoke_deadline):
    """The server answering /v1/models does not prove tool calling. v1.3 checks, all inside ONE deadline:
      forced    a named `submit` call (thinking off)
      auto      an automatic `python` call (thinking off)
      thinking  an automatic `python` call with thinking on and the agent's REAL thinking_token_budget
      roundtrip the agent's real loop: python call -> tool result sent back (role "tool") -> second request ->
                a `submit` that verifies (no tool_choice, so the automatic submit path is tested too)
    A call only counts if its code really runs on a fresh random task (python must print the right shape, submit
    must verify and return the test grid). The thinking test fails if the client had to drop the budget.
    ok = (auto or thinking) and roundtrip."""
    sys.path.insert(0, HERE)
    from client import ChatClient
    from repl_agent import make_tools, verify
    from repl import run_snippet
    task, shape = _smoke_task()
    tools = make_tools(cfg.get("retain_functions", False))
    out = {"forced": [], "auto": [], "thinking": [], "roundtrip": [], "deadline_hit": False}

    def left():
        return smoke_deadline - time.time()

    def semantic(name, code):
        try:
            compile(code, "<smoke>", "exec")
        except Exception as e:
            return False, f"does not compile: {e}"[:200], None
        if left() <= 2:
            return False, "smoke deadline", None
        if name == "submit":
            n_ok, test_out, fb = verify(code, task, timeout=min(8.0, left()))
            return (n_ok == 1 and test_out == [task["test"][0]["input"]]), fb[:200], None
        o = run_snippet(code, task, timeout=min(8.0, left()))
        return (shape in o and "Traceback" not in o and "[snippet crashed]" not in o), o[:200], o

    def check(r, name):
        calls = r.get("tool_calls") or []
        if "error" in r or not calls or calls[0].get("name") != name:
            return False, "no tool call, or the wrong tool", None
        args = calls[0].get("arguments")
        code = args.get("code") if isinstance(args, dict) else None
        if not isinstance(code, str) or not code.strip():
            return False, "no code string", None
        return semantic(name, code)

    def rec(r, ok, why, t):
        return {"ok": ok, "why": why, "sec": round(time.time() - t, 1), "error": r.get("error"),
                "finish": r.get("finish_reason"), "completion_tokens": r.get("completion_tokens"),
                "calls": [(c.get("name"), str(c.get("arguments"))[:200]) for c in r.get("tool_calls") or []],
                "content": (r.get("content") or "")[:200]}

    def ask(cl, msgs, choice):
        rem = left()
        if rem <= 5:
            out["deadline_hit"] = True
            return None
        return cl.chat(msgs, tools=tools, tool_choice=choice, timeout=min(75.0, rem))

    sysmsg = {"role": "system", "content": "You are testing tool calling. Answer only with the requested tool call."}
    ask_submit = {"role": "user", "content": "Call the `submit` tool with a Python program that defines "
                                             "transform(grid) and returns grid unchanged."}
    ask_python = {"role": "user", "content": "Call the `python` tool with code that prints train_inputs[0].shape."}
    budget = cfg.get("thinking_token_budget") or 6144           # v1.3: the agent's real budget, not 1024
    plain = ChatClient("http://127.0.0.1:1234/v1", "qwen", temperature=0.0, max_tokens=1024, thinking=False)
    # effort "low" keeps the smoke short; what is tested is that the server ACCEPTS this budget value with tools
    think = ChatClient("http://127.0.0.1:1234/v1", "qwen", temperature=0.6,
                       max_tokens=max(int(cfg.get("turn_max_tokens") or 8192), budget + 1024), thinking=True,
                       thinking_token_budget=budget, template_kwargs={"reasoning_effort": "low"})
    first_python = {}
    for key, cl, msgs, name, choice in (("forced", plain, [sysmsg, ask_submit], "submit", "submit"),
                                        ("auto", plain, [sysmsg, ask_python], "python", None),
                                        ("thinking", think, [sysmsg, ask_python], "python", None)):
        for _ in range(2):
            t = time.time()
            r = ask(cl, msgs, choice)
            if r is None:
                break
            ok, why, tool_out = check(r, name)
            out[key].append(rec(r, ok, why, t))
            if ok:
                if name == "python":
                    first_python.setdefault(key, (cl, msgs, r, tool_out))
                break
    out["forced_ok"] = any(x["ok"] for x in out["forced"])
    out["auto_ok"] = any(x["ok"] for x in out["auto"])
    out["thinking_budget_requested"] = budget
    out["thinking_budget_retained"] = think.thinking_token_budget == budget
    out["thinking_ok"] = any(x["ok"] for x in out["thinking"]) and out["thinking_budget_retained"]
    # v1.5: the EXACT request shape the agent will send (its template_kwargs, sampling, budget, max_tokens, tools),
    # cut to 32 output tokens -- the server only has to ACCEPT it (run 1: the template rejected effort "high")
    agent_cl = ChatClient("http://127.0.0.1:1234/v1", "qwen", temperature=cfg.get("temperature", 0.6),
                          top_p=cfg.get("top_p", 0.95), top_k=cfg.get("top_k", 20), max_tokens=32,
                          thinking=cfg.get("thinking", True) and out["thinking_ok"],
                          thinking_token_budget=cfg.get("thinking_token_budget") if out["thinking_ok"] else None,
                          template_kwargs=cfg.get("template_kwargs"))   # same fallback the ladder will apply
    out["agent_params"] = []
    rungs_eff = [None] + [{"reasoning_effort": e} for e in (cfg.get("effort_ladder") or [])]
    for extra_kw in rungs_eff:                      # the agent's own effort and every effort-ladder step
        if left() <= 5:
            out["deadline_hit"] = True
            break
        t = time.time()
        r = agent_cl.chat([sysmsg, ask_python], tools=tools, timeout=min(75.0, left()), template_kwargs=extra_kw)
        okp = "error" not in r and not getattr(agent_cl, "budget_dropped", False)
        out["agent_params"].append({"ok": okp, "effort": (extra_kw or cfg.get("template_kwargs") or {}),
                                    "sec": round(time.time() - t, 1), "error": (r.get("error") or "")[:300],
                                    "finish": r.get("finish_reason")})
    out["agent_params_ok"] = bool(out["agent_params"]) and all(x["ok"] for x in out["agent_params"])
    # v1.6: the DIRECT rescue's two request shapes (its effort + 32k budget; the no-thinking commit call), 32 output
    # tokens each. Informational: a rejection is reported, it does not fail the smoke (the agent can still run)
    out["rescue_params"] = []
    if cfg.get("rescue"):
        from rescue import direct_client
        rcl = direct_client(cfg)
        for extra_kw in (None, {"enable_thinking": False, "reasoning_effort": "low"}):
            if left() <= 5:
                out["deadline_hit"] = True
                break
            t = time.time()
            r = rcl.chat([sysmsg, {"role": "user", "content": "Reply with the digit 7."}], max_tokens=32,
                         timeout=min(75.0, left()), template_kwargs=extra_kw)
            out["rescue_params"].append({"ok": "error" not in r and not getattr(rcl, "budget_dropped", False),
                                         "commit_call": extra_kw is not None, "sec": round(time.time() - t, 1),
                                         "error": (r.get("error") or "")[:300], "finish": r.get("finish_reason")})
    out["rescue_params_ok"] = (all(x["ok"] for x in out["rescue_params"]) if out["rescue_params"] else None)
    # roundtrip on the mode the agent will actually use (thinking if it passed, else plain)
    mode = "thinking" if out["thinking_ok"] and "thinking" in first_python else "auto"
    out["roundtrip_mode"] = mode
    if mode in first_python:
        cl, msgs, r1, tool_out = first_python[mode]
        c = r1["tool_calls"][0]
        follow = msgs + [
            {"role": "assistant", "content": r1.get("content") or "",
             "tool_calls": [{"id": c["id"], "type": "function",
                             "function": {"name": "python", "arguments": json.dumps(c["arguments"])}}]},
            {"role": "tool", "tool_call_id": c["id"], "content": tool_out},
            {"role": "user", "content": "Good. Now call the `submit` tool with a Python program that defines "
                                        "transform(grid) and returns grid unchanged."}]
        for _ in range(2):
            t = time.time()
            r2 = ask(cl, follow, None)
            if r2 is None:
                break
            ok, why, _ = check(r2, "submit")
            out["roundtrip"].append(rec(r2, ok, why, t))
            if ok:
                break
    out["roundtrip_ok"] = any(x["ok"] for x in out["roundtrip"])
    out["ok"] = (out["auto_ok"] or out["thinking_ok"]) and out["roundtrip_ok"] and out["agent_params_ok"]
    print("tool smoke:", {k: out[k] for k in ("forced_ok", "auto_ok", "thinking_ok", "thinking_budget_retained",
                                              "roundtrip_ok", "roundtrip_mode", "agent_params_ok",
                                              "rescue_params_ok", "deadline_hit", "ok")}, flush=True)
    return out''')
patch11('''def start_server(cfg, extra):''', '''def start_server(cfg, extra, timeout=1500):''')
# v1.4: bounded install (pip had no timeout) and bounded GPU-drain polling after a kill (nvidia-smi had no timeout)
patch11('''                    "--disable-pip-version-check", "--no-warn-conflicts", "-q"], check=True)''',
        '''                    "--disable-pip-version-check", "--no-warn-conflicts", "-q"], check=True,
                   timeout=float(cfg.get("install_timeout_s", 1800)))   # v1.4''')
patch11('''                        "--disable-pip-version-check", "-q", *wheels], check=True)''',
        '''                        "--disable-pip-version-check", "-q", *wheels], check=True, timeout=600)   # v1.4''')
patch11('''    for _ in range(36):
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()
        if used and all(int(u) < 1500 for u in used if u.strip().isdigit()):
            return
        time.sleep(5)''', '''    t_end = time.time() + 240                       # v1.4: the whole drain wait is bounded
    while time.time() < t_end:
        try:
            used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=20).stdout.split()
        except Exception:
            used = []
        if used and all(int(u) < 1500 for u in used if u.strip().isdigit()):
            return
        time.sleep(5)''')
patch11('''               LIBRARY_PATH="/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", ""))''',
        '''               LIBRARY_PATH=f"{LINK_DIR}:/usr/local/nvidia/lib64:" + os.environ.get("LIBRARY_PATH", ""),
               VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="1")   # v1.1: libcuda shim dir first; graph memory in budget''')
patch11('''           "--max-num-seqs", str(cfg.get("max_num_seqs", 32)), *extra]''',
        '''           "--max-num-seqs", str(cfg.get("max_num_seqs", 32)),
           # v1.1: 4096 = the batch step that fixed Q27-XHIGH run 2's MTP-drafter OOM (default prefill step was larger)
           "--max-num-batched-tokens", str(cfg.get("max_num_batched_tokens", 4096)), *extra]''')
patch11('''                            start_new_session=True)
    while time.time() - t < 1500:''', '''                            start_new_session=True)
    CURRENT["proc"] = proc
    SERVER_PID.write_text(str(proc.pid))                 # v1.1: the notebook kills this group if we are killed
    while time.time() - t < timeout:''')
patch11('''        install(cfg)
        # A ladder of server option sets, most ambitious first; the last is plain serving.
        # Untested-on-L4 options (MTP, --reasoning-config) must never cost the whole run.
        ladder = server_ladder(cfg)
        for i, extra in enumerate(ladder):
            proc = start_server(cfg, extra)
            if proc is not None:
                print(f"server up with option set {i}: {extra}", flush=True)
                break
            print(f"option set {i} failed -> next", flush=True)
            time.sleep(20)
        if proc is None:
            return''', '''        import signal as _signal
        _signal.signal(_signal.SIGTERM, _on_sigterm)
        t_install = time.time()
        install(cfg)
        install_s = round(time.time() - t_install)
        link = {"probe": "SKIP (dryrun)"} if cfg.get("dryrun") else ensure_libcuda()   # v1.8
        if link.get("probe") == "FAIL":
            # v1.2: logged loudly, not fatal -- only FlashInfer's JIT needs -lcuda; other attention backends on the
            # ladder load without it, and every rung must still pass the tool smoke before the agent starts
            print("WARNING: -lcuda link probe FAILED; rungs that need FlashInfer JIT will fail", flush=True)
        # A ladder of server option sets, most ambitious first; the last is plain serving.
        # Untested-on-L4 options (MTP, --reasoning-config) must never cost the whole run.
        # v1.1: each rung must also pass a real tool-calling smoke test, and the ladder never eats the time the
        # agent needs (min_agent_s before the deadline).
        ladder = [[]] if cfg.get("dryrun") else server_ladder(cfg)      # v1.8 DRYRUN: one fake rung
        report = {"cuda_link": CUDA_LINK, "install_s": install_s, "rungs": []}
        reserve = float(cfg.get("min_agent_s", 1800))
        for i, extra in enumerate(ladder):
            left = cfg["deadline_epoch"] - time.time() - reserve
            if left < 300:
                print(f"no time for server option set {i} ({left:.0f}s before the agent reserve) -> stop", flush=True)
                break
            t_rung = time.time()
            proc = start_server(cfg, extra, timeout=min(1500.0, left))
            rung = {"index": i, "extra": extra, "server_up": proc is not None, "start_s": round(time.time() - t_rung)}
            if proc is not None:
                # v1.2: the smoke has its own hard deadline inside the time left before the agent reserve
                slack = cfg["deadline_epoch"] - time.time() - reserve
                if slack <= 60:
                    print(f"option set {i}: server up but no time left for the tool smoke -> stop", flush=True)
                    rung["note"] = "no time for the smoke"
                    report["rungs"].append(rung)
                    kill(proc)
                    proc = None
                    break
                t_smoke = time.time()
                sm = smoke(cfg, time.time() + min(240.0, slack - 30.0))   # v1.3: 240 s (adds the roundtrip)
                rung["smoke_s"] = round(time.time() - t_smoke)
                rung["smoke"] = sm
                if sm["ok"]:
                    if not sm["forced_ok"]:
                        cfg["force_submit_after"] = 0      # named tool_choice does not work on this rung
                    if not sm["thinking_ok"]:
                        cfg["thinking"] = False            # explicit fallback: no thinking, so no unbounded thinking
                        cfg["thinking_token_budget"] = None
                    report["rungs"].append(rung)
                    report["chosen"] = {"index": i, "extra": extra, "force_submit_after": cfg.get("force_submit_after"),
                                        "thinking": cfg.get("thinking", True),
                                        "thinking_budget_retained": sm.get("thinking_budget_retained", False)}
                    report["remaining_agent_s"] = round(cfg["deadline_epoch"] - time.time())
                    REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
                    print(f"server up with option set {i}: {extra}", flush=True)
                    break
                kill(proc)
                proc = None
                if sm.get("roundtrip_ok") and not sm.get("agent_params_ok"):
                    # v1.5: the server works, the AGENT's request is rejected (a config error, the same on every rung)
                    rung["failure"] = "AGENT_PARAMS"
                    report["rungs"].append(rung)
                    REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
                    print(f"option set {i}: agent request parameters rejected -> stop the ladder", flush=True)
                    break
                print(f"option set {i}: server up but tool smoke FAILED -> next", flush=True)
            else:
                # v1.3: classify the startup failure; no rung changes the attention backend, so a CUDA-linker failure
                # would repeat on every rung -> stop instead of reloading the 27B model again and again
                rung["failure"] = classify_startup_failure()
                if rung["failure"] == "CUDA_LINKER":
                    report["rungs"].append(rung)
                    REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
                    print(f"option set {i}: CUDA linker failure ({CUDA_LINK.get('probe')}) -> stop the ladder",
                          flush=True)
                    break
            report["rungs"].append(rung)
            REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
            print(f"option set {i} failed ({rung.get('failure', 'smoke')}) -> next", flush=True)
            time.sleep(20)
        REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
        if proc is None:
            return
        # v1.7 (GPT final audit): a rejected DIRECT-rescue request shape disables the rescue (the agent keeps the
        # whole time; the eval report marks the three-way comparison incomplete) instead of failing task by task
        chosen_sm = (report["rungs"][-1].get("smoke") or {}) if report["rungs"] else {}
        cfg["rescue_smoke_ok"] = chosen_sm.get("rescue_params_ok")
        # v1.8.2: only a LIVE rescue needs the request shape; the commit run's recorded DIRECT answers do not
        live_rescue = cfg.get("rescue") and (not cfg.get("eval") or cfg.get("eval_direct", "ref") == "live")
        if live_rescue and chosen_sm.get("rescue_params_ok") is False:
            cfg["rescue"] = False
            report["rescue_disabled"] = "rescue request shape rejected by the server"
            print("WARNING: DIRECT rescue requests rejected -> rescue disabled", flush=True)
        # v1.7 (GPT Patch B): the agent's REAL request end to end before any task (bounded, never fatal)
        e2e_budget = min(float(cfg.get("e2e_max_s", 600)),
                         cfg["deadline_epoch"] - time.time() - float(cfg.get("e2e_reserve_s", 1800)))
        if e2e_budget >= float(cfg.get("e2e_min_s", 120)):
            e2e = agent_e2e(cfg, time.time() + e2e_budget)
            report["agent_e2e"] = e2e
            cfg["agent_e2e_ok"] = e2e["ok"]
            if e2e["ok"] and e2e["effort"] and e2e["effort"] != (cfg.get("template_kwargs") or {}).get("reasoning_effort"):
                lad = list(cfg.get("effort_ladder") or [])
                cfg["template_kwargs"] = dict(cfg.get("template_kwargs") or {}, reasoning_effort=e2e["effort"])
                cfg["effort_ladder"] = lad[lad.index(e2e["effort"]) + 1:] if e2e["effort"] in lad else lad
                print(f"agent effort switched to {e2e['effort']} (the configured effort failed end to end)", flush=True)
            elif e2e["ok"] is False:
                # v1.8.2 (GPT audit): no effort completed python -> tool result -> verified submit on the real model.
                report["agent_gate"] = "AGENT_E2E_NO_GO"
                if cfg.get("eval"):            # the commit run would measure nothing: stop, keep the GPU
                    REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))
                    print("EVAL NO-GO: no working agent tool roundtrip on any effort -> evaluation skipped",
                          flush=True)
                    return
                if cfg.get("rescue") and cfg.get("rescue_smoke_ok") is True:
                    cfg["agent_disabled"] = True   # hidden run: the whole time goes to the DIRECT method
                    print("AGENT disabled (e2e NO-GO) -> DIRECT rescue on every task", flush=True)
                else:
                    print("WARNING: agent e2e failed and no DIRECT rescue is available -> agent runs as the last "
                          "resort", flush=True)
        else:
            report["agent_e2e"] = {"ok": None, "skipped": f"only {e2e_budget:.0f}s available"}
        report["agent_config"] = {k: cfg.get(k) for k in ("dryrun", "agent_disabled", "template_kwargs", "effort_ladder", "thinking",
                                                          "thinking_token_budget", "rescue", "rescue_smoke_ok")}
        REPORT_PATH.write_text(json.dumps(report, indent=1, default=str))''')

# v1.7 submit_main.py: agent_e2e (GPT Patch B) + an informational kernel-isolation probe
patch11('''def start_server(cfg, extra, timeout=1500):''', '''def agent_e2e(cfg, e2e_deadline):
    """v1.7 (GPT final audit, Patch B): the agent's REAL request -- its effort, sampling, thinking budget and full
    per-turn max_tokens, with the agent's tools -- on a fresh random task: a python call whose code really runs, the
    output sent back as a role "tool" message, then a submit that verifies. Tried with the configured effort first,
    then each effort-ladder step; returns the first effort that passes (ok False if none does). Each request is bounded
    (<= 300 s) and the whole probe by e2e_deadline."""
    sys.path.insert(0, HERE)
    from client import ChatClient
    from repl_agent import make_tools, verify
    from repl import run_snippet
    tools = make_tools(cfg.get("retain_functions", False))
    base_tk = dict(cfg.get("template_kwargs") or {})
    efforts = [base_tk.get("reasoning_effort")] + [e for e in (cfg.get("effort_ladder") or [])
                                                   if e != base_tk.get("reasoning_effort")]
    sysmsg = {"role": "system", "content": "You solve grid puzzles with the tools. Answer only with the requested "
                                           "tool call."}
    ask_py = {"role": "user", "content": "Call the `python` tool with code that prints train_inputs[0].shape."}
    out = {"ok": False, "effort": None, "tries": []}

    def first_call(r, name):
        calls = r.get("tool_calls") or []
        if "error" in r or not calls or calls[0].get("name") != name or not isinstance(calls[0].get("arguments"), dict):
            return None, None
        code = calls[0]["arguments"].get("code")
        return (calls[0], code) if isinstance(code, str) and code.strip() else (None, None)

    for eff in efforts:
        left = e2e_deadline - time.time()
        if left < 60:
            out["deadline_hit"] = True
            break
        task, shape = _smoke_task()
        cl = ChatClient("http://127.0.0.1:1234/v1", "qwen", temperature=cfg.get("temperature", 0.6),
                        top_p=cfg.get("top_p", 0.95), top_k=cfg.get("top_k", 20),
                        max_tokens=int(cfg.get("turn_max_tokens") or cfg.get("max_tokens") or 8192),
                        thinking=cfg.get("thinking", True), thinking_token_budget=cfg.get("thinking_token_budget"),
                        template_kwargs=dict(base_tk, **({"reasoning_effort": eff} if eff else {})))
        t = time.time()
        r1 = cl.chat([sysmsg, ask_py], tools=tools, timeout=min(300.0, left))
        rec = {"effort": eff, "python_s": round(time.time() - t, 1), "python_finish": r1.get("finish_reason"),
               "python_tokens": r1.get("completion_tokens"), "error": (r1.get("error") or "")[:300], "ok": False}
        c1, code1 = first_call(r1, "python")
        o = run_snippet(code1, task, timeout=8.0) if code1 else ""
        rec["python_ok"] = bool(code1) and shape in o and "Traceback" not in o and "[snippet crashed]" not in o
        if rec["python_ok"] and e2e_deadline - time.time() > 30:
            follow = [sysmsg, ask_py,
                      {"role": "assistant", "content": r1.get("content") or "",
                       "tool_calls": [{"id": c1["id"], "type": "function",
                                       "function": {"name": "python", "arguments": json.dumps(c1["arguments"])}}]},
                      {"role": "tool", "tool_call_id": c1["id"], "content": o},
                      {"role": "user", "content": "Good. Now call the `submit` tool with a Python program that "
                                                  "defines transform(grid) and returns grid unchanged."}]
            t = time.time()
            r2 = cl.chat(follow, tools=tools, timeout=min(300.0, e2e_deadline - time.time()))
            rec.update(submit_s=round(time.time() - t, 1), submit_finish=r2.get("finish_reason"),
                       submit_tokens=r2.get("completion_tokens"), error2=(r2.get("error") or "")[:300])
            _, code2 = first_call(r2, "submit")
            if code2:
                n_ok, test_out, _fb = verify(code2, task, timeout=8.0)
                rec["ok"] = n_ok == 1 and test_out == [task["test"][0]["input"]]
        rec["budget_dropped"] = bool(getattr(cl, "budget_dropped", False))
        rec["ok"] = rec["ok"] and not rec["budget_dropped"]
        out["tries"].append(rec)
        print("agent e2e:", rec, flush=True)
        if rec["ok"]:
            out.update(ok=True, effort=eff)
            break
    return out


def start_server(cfg, extra, timeout=1500):''')
patch11('''        install_s = round(time.time() - t_install)
''', '''        install_s = round(time.time() - t_install)
        try:                                 # v1.7: informational -- can a rootless namespace sandbox work here?
            _ns = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--fork", "true"],
                                 capture_output=True, text=True, timeout=10)
            KERNEL_ISO["unshare_rc"], KERNEL_ISO["stderr"] = _ns.returncode, _ns.stderr[-200:]
        except Exception as _e:
            KERNEL_ISO["unshare_rc"], KERNEL_ISO["stderr"] = None, repr(_e)[:200]
''')
patch11('''        report = {"cuda_link": CUDA_LINK, "install_s": install_s, "rungs": []}''',
        '''        report = {"cuda_link": CUDA_LINK, "install_s": install_s, "rungs": [], "kernel_isolation": KERNEL_ISO}''')
patch11('''CUDA_LINK = {}''', '''CUDA_LINK = {}
KERNEL_ISO = {}                                          # v1.7: unshare probe result (informational)''')

# v1.8 submit_main.py: DRYRUN (CPU, fake OpenAI server on the same port; no install, no CUDA, no GPU drain)
patch11('''def install(cfg):
    wh = one(''', '''def install(cfg):
    if cfg.get("dryrun"):                                 # v1.8 DRYRUN: no vLLM on a CPU run
        print("DRYRUN: install skipped", flush=True)
        return
    wh = one(''')
patch11('''def start_server(cfg, extra, timeout=1500):
    model = one(cfg["model_glob"]).parent''', '''def start_server(cfg, extra, timeout=1500):
    if cfg.get("dryrun"):                                 # v1.8 DRYRUN: fake_vllm.py, same port and API
        # v1.8.1: no `import urllib.request` here -- it made `urllib` LOCAL to the whole function, so the real
        # readiness poll below raised UnboundLocalError forever (GPU run 3: server up, never detected)
        proc = subprocess.Popen([sys.executable, os.path.join(HERE, "fake_vllm.py"), "1234"],
                                stdout=open(LOG, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        CURRENT["proc"] = proc
        SERVER_PID.write_text(str(proc.pid))
        t_f = time.time()
        while time.time() - t_f < 60:
            try:
                urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=2).read()
                print(f"DRYRUN: fake server ready after {time.time() - t_f:.1f}s", flush=True)
                return proc
            except Exception:
                time.sleep(0.5)
        kill(proc)
        return None
    model = one(cfg["model_glob"]).parent''')
patch11('''        try:
            urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5)
            print(f"server ready after {time.time() - t:.0f}s", flush=True)
            return proc
        except Exception:
            time.sleep(10)''', '''        try:
            urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=5)
            print(f"server ready after {time.time() - t:.0f}s", flush=True)
            return proc
        except (NameError, AttributeError, TypeError):
            raise                                     # v1.8.1: a code bug must never look like "not up yet"
        except Exception:
            time.sleep(10)''')
patch11('''    t_end = time.time() + 240                       # v1.4: the whole drain wait is bounded''',
        '''    if shutil.which("nvidia-smi") is None:           # v1.8: no GPU (DRYRUN) -> nothing to drain
        return
    t_end = time.time() + 240                       # v1.4: the whole drain wait is bounded''')


SERVER = {k: v for k, v in json.loads(cells[12].split('cfg = json.loads(r"""', 1)[1].split('""")', 1)[0]).items()
          if k in ("model_glob", "overlay", "parsers", "server_ladder", "max_model_len", "max_num_seqs")}
assert SERVER["max_model_len"] == 81920 and SERVER["max_num_seqs"] == 16 and len(SERVER["server_ladder"]) == 6

AGENT_CFG = {
    "mode": "repl", "attempts": 2, "concurrency": 16,
    "temperature": 0.6, "top_p": 0.95, "top_k": 20,
    # v1.5 (run 1 on esfersami50): Qwen3.8's chat template only knows xhigh (default) / medium / low -- "high" made
    # every agent request fail with HTTP 400. xhigh = the template default and the setting behind our LB 18.75 and
    # ARC Prize's 42.4 %; the per-turn thinking_token_budget still caps each turn; after a cut turn the ladder steps
    # down to medium, then low.
    "template_kwargs": {"reasoning_effort": "xhigh"},
    # v1.1 (audit H4): 8192 x 7 turns already filled the 60k cap before a forced submit could ever happen
    "thinking_token_budget": 6144,      # vLLM forces </think> after this many reasoning tokens in a turn
    # v1.8 (PIV v2.4G lesson: a 768-token action room after a forced close truncated tool calls): 4,096 tokens stay
    # for the tool call after the 6,144-token thinking budget forces </think>
    "max_tokens": 10240, "turn_max_tokens": 10240,
    "max_turns": 30, "context_budget": 28000, "keep_recent": 6, "trim_drain": 6000,
    "nudge_every": 4, "in_python": True, "force_submit_after": 5,
    "effort_ladder": ["medium", "low"], "retain_functions": True,
    "degenerate_guard": True, "timeout_as_cut": True, "snippet_timeout": 30.0,
    "tok_per_s": 300, "min_cap": 15000, "max_cap": 60000, "eval_cap": 60000, "cap_factor": 1.0,
    "max_num_batched_tokens": 4096,     # v1.1 (audit H5): Q27-XHIGH run 2 OOM fix
    # v1.6 (GPT deep audit): two-tier budget -- a 24k scout attempt for every task, a 36k deep attempt only where the
    # scout failed (attempt_caps overrides the supply-based cap per attempt index)
    "attempt_caps": [24000, 36000],
    # v1.7 (GPT final audit): 240 hidden tasks x 24k would not fit the agent's share of 12 h -> smaller scout in the
    # rerun; the eval report counts how many EVAL12 verifications happened within this scout cap (counterfactual)
    "rerun_attempt_caps": [16000, 32000],
    "e2e_max_s": 600,                   # v1.7: the agent's real request end to end before any task (GPT Patch B)
    # v1.6 HYBRID: after agent_frac of the time, a DIRECT answer (LB-18.75 method, run_direct.answer) for every task
    # without a verified program; merge order verified > direct > unverified > test input
    # v1.8: exactly the LB-18.75 budget per TEST INPUT (49,152 thinking + 8,192 answer; EVAL12 records: the correct
    # answers needed 26-37k thinking tokens, so v1.7's 32k per task would have lost 3 of 5)
    "rescue": True, "agent_frac": 0.5, "rescue_effort": "xhigh", "rescue_think": 49152, "rescue_answer": 8192,
    "rescue_commit_tokens": 8192, "rescue_temperature": 0.6, "rescue_min_start_s": 300, "rescue_concurrency": 16,
    # v1.8: the commit run merges with the recorded LB-18.75 answers on EVAL12 (no GPU for DIRECT); "live" re-runs it
    "eval_direct": "ref", "eval_direct_run": "run7", "e2e_reserve_s": 1800,
}

C12 = r'''# ==================================================================== launch: evaluation (commit run) or hidden test (rerun)
import json, os, subprocess, sys, time
from pathlib import Path
START = float(Path("/kaggle/working/start_epoch.txt").read_text())
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
EVAL12 = ''' + json.dumps(EVAL12) + r'''
SERVER = json.loads(r"""''' + json.dumps(SERVER) + r'''""")
AGENT = json.loads(r"""''' + json.dumps(AGENT_CFG) + r'''""")
cfg = dict(SERVER, **AGENT)
if RERUN:
    # hidden test: the agent stops at 11 h 35 min; the process is cut at 11 h 50 min (10 min before the 12 h limit)
    cfg["deadline_epoch"] = START + 12 * 3600 - 25 * 60
    cfg["min_agent_s"] = 3600            # the server ladder never starts a rung with < 1 h left for the agent
else:
    EVAL_HOURS = float(os.environ.get("Q27A_EVAL_HOURS", 1.5))   # v1.8: agent only (DIRECT from the run records)
    _ids = os.environ.get("Q27A_EVAL_TASKS", "eval12").strip()
    cfg.update(eval=True, deadline_epoch=START + EVAL_HOURS * 3600, min_agent_s=1800,
               eval_task_ids=EVAL12 if _ids == "eval12" else [t.strip() for t in _ids.split(",") if t.strip()])
    # v1.6: the commit run rescues EVERY task with a direct answer, so the report shows agent-only / direct-only /
    # hybrid scores and their overlap; the agent gets 60 % of the eval time
    cfg.update(rescue_all=True, agent_frac=float(os.environ.get("Q27A_AGENT_FRAC", 0.6)))


def _has_gpu():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        return out.returncode == 0 and "GPU" in out.stdout
    except Exception:
        return False


# v1.8 DRYRUN: a commit run WITHOUT a GPU (or Q27A_DRYRUN=1) drives the real path against fake_vllm.py on CPU: the
# exact cfg, smoke, e2e probe, agent loop, live DIRECT rescue, eval report, then the hidden-test path on 20 tasks into
# /kaggle/working/dryrun/. Never in a competition rerun.
DRYRUN = (not RERUN) and (os.environ.get("Q27A_DRYRUN", "") == "1" or not _has_gpu())
Path("/kaggle/working/dryrun_flag").write_text("1" if DRYRUN else "0")
if DRYRUN:
    cfg.update(dryrun=True, deadline_epoch=time.time() + 20 * 60, min_agent_s=60, e2e_reserve_s=60, e2e_max_s=120,
               e2e_min_s=10, eval_direct="live", rescue_all=True, rescue_min_start_s=5, install_timeout_s=60)
print("mode:", "COMPETITION RERUN (hidden test)" if RERUN else
      ("DRYRUN (CPU, fake server)" if DRYRUN else f"EVALUATION ({len(cfg['eval_task_ids'])} tasks)"),
      "| agent deadline in", round((cfg["deadline_epoch"] - time.time()) / 60), "min", flush=True)
print(json.dumps({k: v for k, v in cfg.items() if k not in ("server_ladder",)}, indent=1), flush=True)


def kill_leftover_server():
    """v1.1: subprocess.run's timeout SIGKILLs submit_main, whose finally-cleanup then never runs, and the vLLM
    server lives in its own session. Kill that recorded process group (only if it is a vLLM server or already gone)."""
    import signal
    pid_file = Path("/kaggle/working/vllm_server.pid")
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
        cmd = Path(f"/proc/{pid}/cmdline")
        if (not cmd.exists()) or ("vllm" in cmd.read_bytes().decode(errors="replace")):
            os.killpg(pid, signal.SIGKILL)
            print("killed leftover vLLM server process group", pid, flush=True)
    except ProcessLookupError:
        pass
    except Exception as e:
        print("server cleanup:", repr(e), flush=True)


try:
    rc = subprocess.run([sys.executable, "/kaggle/working/agent/submit_main.py", json.dumps(cfg)],
                        timeout=max(60, cfg["deadline_epoch"] + 15 * 60 - time.time())).returncode
    print("submit_main exit code", rc)
except Exception as e:
    print("submit_main did not finish cleanly:", repr(e))
finally:
    kill_leftover_server()
if DRYRUN:                                   # v1.8: the hidden-test path too, on 20 tasks, into its own folder
    Path("/kaggle/working/dryrun").mkdir(exist_ok=True)
    cfg2 = dict(cfg, eval=False, sub_path="/kaggle/working/dryrun/submission.json", interactive_tasks=20,
                deadline_epoch=time.time() + 15 * 60, rescue_log="/kaggle/working/dryrun/rescue_attempts.jsonl")
    try:
        rc2 = subprocess.run([sys.executable, "/kaggle/working/agent/submit_main.py", json.dumps(cfg2)],
                             timeout=20 * 60).returncode
        print("DRYRUN hidden-test path exit code", rc2, flush=True)
    except Exception as e:
        print("DRYRUN hidden-test path did not finish cleanly:", repr(e), flush=True)
    finally:
        kill_leftover_server()
for _p in ("/kaggle/working/agent_server_report.json", "/kaggle/working/agent_eval/eval_report.json",
           "/kaggle/working/agent_eval/progress.json"):
    try:
        _r = json.loads(Path(_p).read_text())
        if _p.endswith("eval_report.json"):
            _r = {k: _r.get(k) for k in ("official_score_pct", "query_hits", "queries", "task_macro_score_pct",
                                         "hybrid_all_rescued_pct", "agent_only_official_pct",
                                         "direct_only_official_pct", "complementarity", "verified_but_wrong_tasks",
                                         "tasks_with_verified_program", "status_counts", "circuit_breaker",
                                         "isolation_denied_tasks", "isolation_guard_failed_tasks",
                                         "agent_gen_tok_per_s", "rescue_gen_tok_per_s", "gen_tokens_total", "wall_s")}
        print(_p, json.dumps(_r, default=str)[:3000], flush=True)
    except Exception as e:
        print("no", _p, repr(e)[:120], flush=True)
'''

C13 = r'''# Last line of defence (v1.2, rewritten): whatever was left behind, submission.json ends up valid -- every hidden
# task, every test input, two grids of plain ints 0-9 with 1..30 rows and columns. A bad ROW is replaced by the
# fallback (test input); a non-dict file, null/str rows, ragged or out-of-range grids and unknown task keys are handled.
import json, glob, os
PATH = "/kaggle/working/submission.json"


def ok_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30
            and all(isinstance(r, list) and 1 <= len(r) <= 30 and len(r) == len(g[0]) for r in g)
            and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))


try:
    ch = json.load(open(glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True)[0]))
    try:
        raw = json.load(open(PATH))
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    fixed, n_bad, answered = {}, 0, 0
    for t, task in ch.items():
        rows = raw.get(t)
        rows = rows if isinstance(rows, list) and len(rows) == len(task["test"]) else [None] * len(task["test"])
        out_rows = []
        for r, x in zip(rows, task["test"]):
            if isinstance(r, dict) and ok_grid(r.get("attempt_1")) and ok_grid(r.get("attempt_2")):
                out_rows.append({"attempt_1": r["attempt_1"], "attempt_2": r["attempt_2"]})
                answered += int(r["attempt_1"] != x["input"])
            else:
                n_bad += 1
                out_rows.append({"attempt_1": x["input"], "attempt_2": x["input"]})
        fixed[t] = out_rows
    unknown = [t for t in raw if t not in ch]
    with open(PATH + ".tmp", "w") as f:
        json.dump(fixed, f)
    os.replace(PATH + ".tmp", PATH)
    print(f"submission.json valid: {len(fixed)} tasks, {answered} test outputs answered by the agent, "
          f"{n_bad} rows replaced by the fallback, {len(unknown)} unknown task keys dropped")
except Exception as e:
    print("validation error:", repr(e))
'''

FAKE = r'''%%writefile /kaggle/working/agent/fake_vllm.py
"""v1.8 DRYRUN: a stand-in for the vLLM OpenAI server on the same port and API, for a CPU-only Kaggle run (no GPU
quota) that drives the REAL notebook path end to end -- launcher cfg, smoke, agent e2e probe, agent loop, DIRECT
rescue, eval report, submission writing, guard -- on Kaggle's exact image. It checks plumbing, not model quality.

Behaviour: rejects unknown reasoning efforts like Qwen3.8's chat template; with tools it answers a named tool_choice
with that tool, otherwise python first, then submit (identity transform); without tools it answers the LB-18.75 prompt
with the FINAL INPUT grid in a ```grid block, except every third prompt (no grid -> exercises the commit call).
    python fake_vllm.py [port]
"""
import json
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATS = {"requests": 0, "tool": 0, "direct": 0, "commit": 0, "rejected": 0}
LOCK = threading.Lock()


def final_input_grid(text):
    if "FINAL INPUT:" not in text:
        return None
    rows = []
    for line in text.split("FINAL INPUT:", 1)[1].splitlines()[1:]:
        if not line.strip() or not line.strip()[0].isdigit():
            break
        rows.append(line.strip())
    return "\n".join(rows) or None


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": "qwen", "object": "model"}]})
        if self.path.startswith("/stats"):
            return self._send(200, STATS)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with LOCK:
            STATS["requests"] += 1
        tk = body.get("chat_template_kwargs") or {}
        eff = tk.get("reasoning_effort")
        if eff is not None and eff not in ("xhigh", "medium", "low"):
            with LOCK:
                STATS["rejected"] += 1
            return self._send(400, {"error": {"message": f"Unexpected reasoning effort {eff}. Supported types are "
                                                         "xhigh (default), medium, and low.", "code": 400}})
        msgs = body.get("messages") or []
        last_user = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "user"), "")
        max_tokens = int(body.get("max_tokens") or 1024)
        if body.get("tools"):
            with LOCK:
                STATS["tool"] += 1
            choice = body.get("tool_choice")
            prior_calls = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))
            if isinstance(choice, dict):
                name = choice["function"]["name"]
            elif "submit" in last_user or prior_calls:
                name = "submit"
            else:
                name = "python"
            code = ("def transform(grid):\n    return grid\n" if name == "submit"
                    else "print(train_inputs[0].shape)")
            msg = {"role": "assistant", "content": "", "reasoning_content": "fake reasoning",
                   "tool_calls": [{"id": f"call{STATS['requests']}", "type": "function",
                                   "function": {"name": name, "arguments": json.dumps({"code": code})}}]}
            return self._send(200, {"choices": [{"message": msg, "finish_reason": "tool_calls"}],
                                    "usage": {"prompt_tokens": 500, "completion_tokens": min(max_tokens, 700)}})
        if "Reply with the digit 7" in last_user:
            content = "7"
        elif "Stop analysing now" in last_user:
            with LOCK:
                STATS["commit"] += 1
            g = final_input_grid(last_user)
            content = f"```grid\n{g}\n```" if g else "no grid"
        else:
            with LOCK:
                STATS["direct"] += 1
            g = final_input_grid(last_user)
            no_grid = zlib.crc32(last_user.encode()) % 3 == 0
            content = "I could not finish." if (no_grid or not g) else f"```grid\n{g}\n```"
        msg = {"role": "assistant", "content": content, "reasoning_content": "fake reasoning about the rule"}
        return self._send(200, {"choices": [{"message": msg, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 800, "completion_tokens": min(max_tokens, 1200)}})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"fake vLLM server on {port}", flush=True)
    srv.serve_forever()
'''

DRYCHECK = r'''# v1.8 DRYRUN check: only in a dry run (CPU, fake server); a normal commit run or a competition rerun skips it.
import json, os, sys, glob
from pathlib import Path
W = Path("/kaggle/working")
if (W / "dryrun_flag").exists() and (W / "dryrun_flag").read_text().strip() == "1":
    checks = {}

    def ck(name, cond, info=""):
        checks[name] = {"ok": bool(cond), "info": str(info)[:300]}

    def load(p):
        try:
            return json.loads(Path(p).read_text())
        except Exception as e:
            return {"_error": repr(e)}

    def nlines(p):
        try:
            return sum(1 for _ in open(p))
        except Exception:
            return 0

    srv = load(W / "agent_server_report.json")
    sm = ((srv.get("rungs") or [{}])[-1].get("smoke") or {})
    ck("server_report", "rungs" in srv, list(srv)[:12])
    ck("smoke_ok", sm.get("ok") is True, {k: sm.get(k) for k in ("forced_ok", "auto_ok", "thinking_ok", "roundtrip_ok",
                                                                  "agent_params_ok", "rescue_params_ok")})
    ck("rescue_probe_ok", sm.get("rescue_params_ok") is True, sm.get("rescue_params"))
    ck("agent_e2e_ok", (srv.get("agent_e2e") or {}).get("ok") is True, srv.get("agent_e2e"))
    ck("kernel_isolation_probe_recorded", "unshare_rc" in (srv.get("kernel_isolation") or {}), srv.get("kernel_isolation"))
    rep = load(W / "agent_eval" / "eval_report.json")
    need = ["official_score_pct", "hybrid_all_rescued_pct", "agent_only_official_pct", "direct_only_official_pct",
            "complementarity", "verified_but_wrong_tasks", "isolation_denied_tasks", "agent_gen_tok_per_s",
            "rescue_gen_tok_per_s", "per_attempt"]
    ck("eval_report_fields", all(k in rep for k in need), [k for k in need if k not in rep] or rep.get("_error"))
    cm = rep.get("complementarity") or {}
    ck("eval_three_way_complete", cm.get("three_way_complete") is True, cm.get("rescue_statuses"))
    st = cm.get("rescue_statuses") or {}
    ck("rescue_answered_and_committed", st.get("answered", 0) > 0 and st.get("committed", 0) > 0, st)
    n_tasks = len((rep.get("config") or {}).get("eval_task_ids") or [])
    ck("attempts_logged", n_tasks > 0 and nlines(W / "agent_eval" / "attempts.jsonl") >= n_tasks,
       nlines(W / "agent_eval" / "attempts.jsonl"))
    ck("rescue_logged", nlines(W / "agent_eval" / "rescue_attempts.jsonl") > 0,
       nlines(W / "agent_eval" / "rescue_attempts.jsonl"))
    ck("agent_guard_never_failed", rep.get("isolation_guard_failed_tasks") == [], rep.get("isolation_guard_failed_tasks"))
    ck("no_breaker", not (rep.get("circuit_breaker") or {}).get("flag"), rep.get("circuit_breaker"))
    # the hidden-test path (run_submit) on 20 tasks
    ch = json.load(open(glob.glob("/kaggle/input/**/arc-agi_test_challenges.json", recursive=True)[0]))
    sub = load(W / "dryrun" / "submission.json")

    def okg(g):
        return (isinstance(g, list) and 1 <= len(g) <= 30
                and all(isinstance(r, list) and 1 <= len(r) <= 30 and len(r) == len(g[0]) for r in g)
                and all(type(v) is int and 0 <= v <= 9 for r in g for v in r))

    rows_ok = (isinstance(sub, dict) and "_error" not in sub and len(sub) > 0
               and all(t in ch and isinstance(rs, list) and len(rs) == len(ch[t]["test"])
                       and all(isinstance(r, dict) and okg(r.get("attempt_1")) and okg(r.get("attempt_2")) for r in rs)
                       for t, rs in sub.items()))
    ck("hidden_path_submission_valid", rows_ok, f"{len(sub)} tasks" if isinstance(sub, dict) else sub)
    rlog = W / "dryrun" / "rescue_attempts.jsonl"     # v1.8.2: was `>= 0` (always true) on a file never written
    ck("hidden_path_rescue_ran", rlog.is_file() and nlines(rlog) > 0, {"exists": rlog.is_file(), "records": nlines(rlog)})
    # the guard on Linux against the real competition files (nothing is printed from them)
    sys.path.insert(0, str(W / "agent"))
    import sandbox, repl
    solp = glob.glob("/kaggle/input/**/arc-agi_evaluation_solutions.json", recursive=True)
    target = solp[0] if solp else "/kaggle/input/missing/arc-agi_evaluation_solutions.json"
    INP = "/kaggle/input"
    RET = "\n    return grid\n"
    LIBC = "import ctypes, ctypes.util\n    lib = ctypes.CDLL(ctypes.util.find_library('c'))\n    "
    esc = {
        "open_solutions": f"def transform(grid):\n    open({target!r}).read(1){RET}",
        "listdir_input": f"import os\ndef transform(grid):\n    os.listdir('/kaggle/input'){RET}",
        "relative_after_chdir": (f"import os\ndef transform(grid):\n    os.chdir({os.path.dirname(INP)!r})\n"
                                 f"    os.listdir({os.path.basename(INP)!r}){RET}"),
        "ctypes_libc_open": f"def transform(grid):\n    {LIBC}f = lib.open{RET}",
        "ctypes_libc_fopen": f"def transform(grid):\n    {LIBC}f = lib.fopen{RET}",
        "fork_exec": f"import _posixsubprocess\ndef transform(grid):\n    _posixsubprocess.fork_exec(){RET}",
        "subprocess": f"import subprocess\ndef transform(grid):\n    subprocess.run(['cat', {target!r}]){RET}",
        "os_system": f"import os\ndef transform(grid):\n    os.system('ls /kaggle/input'){RET}",
        "symlink": f"import os\ndef transform(grid):\n    os.symlink({target!r}, '/tmp/q27_lnk'){RET}",
    }
    for k, code in esc.items():
        r = sandbox.run_one(code, [[[1]]], timeout=30)
        t = json.dumps(r)
        ck("guard_" + k, ("access denied" in t or "not allowed" in t) and r[0].get("output") is None, t[:200])
    r = sandbox.run_one("import numpy as np\nfrom scipy import ndimage\ndef transform(grid):\n"
                        "    lab, n = ndimage.label(np.array(grid) > 0)\n    return lab.tolist()\n",
                        [[[1, 0], [0, 1]]], timeout=60)
    ck("guard_numpy_scipy_still_work", r[0].get("output") == [[1, 0], [0, 2]], r)
    task = {"train": [{"input": [[1, 2]], "output": [[1, 2]]}], "test": [{"input": [[3, 4]]}]}
    o = repl.run_snippet(f"print(open({target!r}).read(5))", task, timeout=30)
    ck("guard_python_tool_open_solutions", "access denied" in o, o[:200])
    o = repl.run_snippet("print(train_inputs[0].shape)", task, timeout=30)
    ck("python_tool_works", "(1, 2)" in o, o[:200])
    os.environ["Q27_DENY_PATHS"] = ""
    r = sandbox.run_one("def transform(grid):\n    return grid\n", [[[1]]], timeout=30)
    os.environ.pop("Q27_DENY_PATHS", None)
    ck("guard_fail_closed", r[0].get("output") is None, r)
    passed = sum(c["ok"] for c in checks.values())
    (W / "dryrun_report.json").write_text(json.dumps({"passed": passed, "total": len(checks), "checks": checks},
                                                     indent=1, default=str))
    for k, c in checks.items():
        print(("PASS " if c["ok"] else "FAIL ") + k + ("" if c["ok"] else "  -> " + c["info"]))
    print(f"DRYRUN RESULT: {'PASS' if passed == len(checks) else 'FAIL'} ({passed}/{len(checks)})")
else:
    print("not a dry run -> DRYRUN check skipped")
'''

out_cells = [("markdown", HEAD), ("code", C1)] + [("code", c) for c in cells[2:11]] + \
            [("code", C11), ("code", RESCUE), ("code", DIRECT_REF), ("code", FAKE), ("code", RUN_EVAL), ("code", C12),
             ("code", C13), ("code", DRYCHECK)]
nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in out_cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
text = json.dumps(nb, indent=1, ensure_ascii=False)
DST.write_text(text, encoding="utf-8", newline="\n")      # v1.2: LF on every OS -> reproducible bytes
print("wrote", DST, hashlib.sha256(DST.read_bytes()).hexdigest())   # hash of the file on disk
