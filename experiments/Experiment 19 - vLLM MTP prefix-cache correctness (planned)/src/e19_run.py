# ==================================================================== CELL E19-2 - prompts, arms, probes, verdicts, report
# Prompts: the 12 tasks of the profile, iteration 0 / expert 0, rendered by S1's own e10_render_request (verbatim below:
# poetiq_build_message + the chat template with enable_thinking and reasoning_effort xhigh), tokenized as S1's e10_admit
# does. The token ids are identical on every arm and every account.
# Per arm (one vLLM 0.19 server), in this order:
#   serial  - 6 prompts (every other one by length), one at a time: reset -> cold (greedy 1,024) -> warm (the same prompt
#             again: prefix-cache hit) -> [prefix ON: reset -> cold2 (A/A)]
#   churn   - the batch-reordering probe (suspect: the async-scheduling race, vLLM #53919, num_accepted_tokens reordered
#             on every row move): the 6 serial prompts x 4 = 24 requests, up to 14 at once, max_tokens staggered
#             256..2,048, arrivals staggered and refilled as requests finish, so rows join / finish / move continuously;
#             each row is compared with the same prompt run alone (serial cold, or serial warm when it was a cache hit)
#   long8k  - the other 6 prompts / long30k - the 6 serial prompts, 8 at once: a phase-2-style context. natural = prompt +
#             a deterministic filler (the task's own grids written out inside <think>, identical ids on every arm) up to
#             256 tokens before a 1,600-token block boundary, greedy 1,024; cont = prompt + filler + the first 512 tokens
#             the server generated itself (the cached prefix holds generated tokens), greedy 512; cont must reproduce
#             natural's tokens 512-1,024.
#   /metrics every E19_SAMPLE_SEC: MTP acceptance per window (accepted / draft tokens), preemptions, running.
# Every row is saved as it ends (q27x/e19/e19_rows.jsonl); verdicts: e19_verdicts() (E19-J).
E19_PROMPTS = []
E19_VERDICTS = {}
E19_SAMPLE_SEC = float(os.environ.get("E19_SAMPLE_SEC", 10))

# @@S1_RENDER@@


def e19_filler_text(task, cycle):
    """Deterministic 'thinking' over the task's own training grids (identical on every arm)."""
    out = ["Let me carefully analyze the training examples.\n\n" if cycle == 0 else
           f"Let me re-check every example once more (pass {cycle + 1}).\n\n"]
    for k, ex in enumerate(task["train"]):
        gi, go = ex["input"], ex["output"]
        out.append(f"Example {k + 1} input ({len(gi)}x{len(gi[0])}):\n" + "\n".join(" ".join(map(str, r)) for r in gi) + "\n")
        out.append(f"Example {k + 1} output ({len(go)}x{len(go[0])}):\n" + "\n".join(" ".join(map(str, r)) for r in go) + "\n")
        if len(gi) == len(go) and len(gi[0]) == len(go[0]):
            ch = [(r, c, gi[r][c], go[r][c]) for r in range(len(gi)) for c in range(len(gi[0])) if gi[r][c] != go[r][c]]
            out.append(f"Comparing example {k + 1}: {len(ch)} cells change"
                       + ("".join(f", ({r},{c}) {a}->{b}" for r, c, a, b in ch[:24])) + ".\n\n")
        else:
            out.append(f"Comparing example {k + 1}: the size changes from {len(gi)}x{len(gi[0])} to {len(go)}x{len(go[0])}.\n\n")
    return "".join(out)


def e19_filler_ids(task, n):
    ids, cycle = [], 0
    while len(ids) < n:
        ids += tokenizer.encode(e19_filler_text(task, cycle), add_special_tokens=False)
        cycle += 1
    return ids[:n]


def e19_build_prompts():
    global E19_SERIAL_IDX, E19_LONG, E19_CHURN_JOBS
    for k, tid in enumerate(TASK_IDS):
        st = {"task_id": tid, "expert": 0, "task_index": k, "iteration": 0, "history": []}
        ids = tokenizer.encode(e10_render_request(st), add_special_tokens=False)
        E19_PROMPTS.append({"task": tid, "ids": ids, "len": len(ids)})
    by_len = sorted(range(len(E19_PROMPTS)), key=lambda i: (E19_PROMPTS[i]["len"], i))
    E19_SERIAL_IDX = sorted(by_len[::2][:E19_PLAN["n_serial"]])
    other = [i for i in range(len(E19_PROMPTS)) if i not in E19_SERIAL_IDX]
    lad, reps = E19_PLAN["churn_max_tokens"], E19_PLAN["churn_reps"]
    # job order interleaves prompts and lengths so that short and long requests alternate in the queue
    E19_CHURN_JOBS = [[i, r, lad[(k * reps + r * 3) % len(lad)]] for r in range(reps) for k, i in enumerate(E19_SERIAL_IDX)]
    bs, bo, gp = E19_PLAN["block"], E19_PLAN["backoff"], E19_PLAN["g_pre"]
    E19_LONG = []
    for name, n_ctx, which in E19_PLAN["long"]:
        idx = list(E19_SERIAL_IDX) if which == "serial" else other if which == "other" else list(range(len(E19_PROMPTS)))
        ctx = {}
        for i in idx:
            L = E19_PROMPTS[i]["len"]
            pf = -(-(L + n_ctx - gp) // bs) * bs - bo        # prompt + filler ends `backoff` tokens before a block boundary
            if pf - L < 1024 or pf + gp + E19_PLAN["cmp"] + 64 > MAX_MODEL_LEN:
                continue                                      # "where the context fits"
            ctx[i] = {"ids": E19_PROMPTS[i]["ids"] + e19_filler_ids(CHALLENGES[E19_PROMPTS[i]["task"]], pf - L), "pf_len": pf}
        E19_LONG.append({"name": name, "n_ctx": n_ctx, "idx": sorted(ctx), "ctx": ctx})
    E19_STATE["prompts"] = [{"task": p["task"], "len": p["len"],
                             "sha": hashlib.sha256(json.dumps(p["ids"]).encode()).hexdigest()[:16]} for p in E19_PROMPTS]
    E19_STATE["serial_idx"] = E19_SERIAL_IDX
    E19_STATE["churn_jobs"] = E19_CHURN_JOBS
    E19_STATE["long"] = [{"name": s["name"], "n_ctx": s["n_ctx"], "idx": s["idx"],
                          "pf_len": {str(i): s["ctx"][i]["pf_len"] for i in s["idx"]}} for s in E19_LONG]


def e19_meta(name):
    a = E19_STATE["arms"].get(name) or {}
    return {"arm": name, "prefix_caching": E19_ARMS[name]["prefix"], "spec": E19_ARMS[name]["spec"],
            "serial_idx": list(E19_SERIAL_IDX), "churn_jobs": [list(x) for x in E19_CHURN_JOBS],
            "long_stages": [{"name": s["name"], "idx": list(s["idx"])} for s in E19_LONG],
            "launch_failed": a.get("launch_failed"), "server_died": a.get("server_died"),
            "accept_windows": list(a.get("accept_windows") or [])}


def e19_parse_log(txt):
    g = lambda pat, f=str: (f(re.findall(pat, txt)[-1].replace(",", "")) if re.findall(pat, txt) else None)
    return {"version": g(r"vLLM API server version ([0-9][\w.+-]*)"),
            "kv_tokens": g(r"GPU KV cache size: ([\d,]+) tokens", int),
            "max_concurrency": g(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", float),
            "block_size": g(r"Setting attention block size to (\d+) tokens", int),
            "mamba_cache_mode": g(r"Mamba cache mode is set to '(\w+)'"),
            "async_lines": sorted(set(re.findall(r"[Aa]sync(?:hronous)? scheduling[^\n]{0,80}", txt)))[:6],
            "spec_lines": re.findall(r"SpecDecoding metrics:[^\n]{0,200}", txt)[-3:],
            "backend_lines": sorted(set(re.findall(r"(?:Using|Selected) [\w ]{0,40}?(?:backend|Backend)[^\n]{0,80}", txt)))[:12],
            "error_lines": [l[:300] for l in txt.splitlines() if re.search(r"\b(ERROR|Traceback|CUDA error|NaN)\b", l)][:30]}


def e19_conc_level(longest, want):
    """`want` streams, fewer when the KV pool cannot hold that many of the stage's longest requests (bf16 KV arm)."""
    a = E19_STATE["arms"][E19_CUR]
    cap = (a.get("log") or {}).get("max_concurrency")
    cap = cap * a.get("max_len", MAX_MODEL_LEN) if cap else (a.get("log") or {}).get("kv_tokens")
    if not cap:
        return want
    return int(max(1, min(want, (0.8 * cap) // max(1, longest))))


def e19_pool(jobs, level, deadline, stagger=0.0):
    """Run job(skip_status) callables on `level` worker threads (a worker takes the next job as soon as its request
    ends: refilled arrivals); worker k starts after k * stagger seconds. A job taken after the deadline or with a dead
    server is told to write skip rows instead of running."""
    q, lk = list(jobs), threading.Lock()

    def worker(k):
        time.sleep(k * stagger)
        while True:
            with lk:
                if not q:
                    return
                job = q.pop(0)
            if E19_SERVER is None or not E19_SERVER.alive():
                job("server_dead")
            elif time.perf_counter() > deadline - 5:
                job("not_tested_budget")
            else:
                job(None)
    th = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(max(1, level))]
    for t in th:
        t.start()
    for t in th:
        t.join()


class E19Sampler(threading.Thread):
    """/metrics every E19_SAMPLE_SEC while an arm runs: the per-window deltas show an MTP-acceptance collapse."""
    def __init__(self, arm):
        super().__init__(daemon=True)
        self.arm, self._halt, self.prev = arm, False, None

    def run(self):
        while not self._halt:
            m = e19_metrics()
            if "error" not in m:
                m["t"] = round(time.perf_counter() - NOTEBOOK_T0, 1)
                if self.prev is not None:
                    d = e19_delta(self.prev, m)
                    d.update(t=m["t"], running=m.get("running"))
                    with E19_LOCK:
                        E19_STATE["arms"][self.arm].setdefault("accept_windows", []).append(d)
                self.prev = m
            end = time.perf_counter() + E19_SAMPLE_SEC
            while not self._halt and time.perf_counter() < end:
                time.sleep(0.2)

    def stop(self):
        self._halt = True
        self.join(15)


def e19_alive(name):
    if E19_SERVER is not None and E19_SERVER.alive():
        return True
    a = E19_STATE["arms"][name]
    if not a.get("server_died"):
        a["server_died"] = "server process exited rc=%s at %.0fs; log tail: %s" % (
            E19_SERVER.proc.returncode if E19_SERVER is not None and E19_SERVER.proc else None,
            time.perf_counter() - NOTEBOOK_T0, (E19_SERVER.log_text()[-1500:] if E19_SERVER is not None else ""))
        e19_event(f"arm {name}: SERVER DIED")
    return False


def e19_stage_serial(name, deadline):
    pc = E19_ARMS[name]["prefix"]
    for i in E19_SERIAL_IDX:
        if not e19_alive(name):
            return
        if deadline - time.perf_counter() < E19_MIN_SERIAL_SEC:
            e19_event(f"arm {name}: serial stage stopped by the budget before prompt {i}")
            return
        ids = E19_PROMPTS[i]["ids"]
        for ps in ["cold", "warm"] + (["cold2"] if pc else []):
            if not e19_alive(name):
                return
            rs = e19_reset_cache() if ps != "warm" else None
            m0 = e19_metrics()
            g = e19_run_req(ids, E19_PLAN["max_new"], deadline)
            e19_write_row(e19_row(name, "serial", ps, i, ids, g, 1,
                                  {"reset": rs, "metric_delta": e19_delta(m0, e19_metrics())}))


def e19_stage_churn(name, deadline):
    if not e19_alive(name):
        return
    if deadline - time.perf_counter() < E19_MIN_CHURN_SEC:
        e19_event(f"arm {name}: churn stage not started (budget)")
        return
    lvl = e19_conc_level(max(p["len"] for p in E19_PROMPTS) + max(E19_PLAN["churn_max_tokens"]), E19_PLAN["churn_conc"])
    rs = e19_reset_cache()
    m0, t0 = e19_metrics(), time.perf_counter()

    def mk(i, rep, mx):
        def job(skip):
            ex = {"reset": rs, "rep": rep}
            if skip:
                e19_write_row(e19_skip_row(name, "churn", f"j{rep}", i, skip, skip, ex))
                return
            g = e19_run_req(E19_PROMPTS[i]["ids"], mx, deadline)
            e19_write_row(e19_row(name, "churn", f"j{rep}", i, E19_PROMPTS[i]["ids"], g, lvl, ex))
        return job
    e19_pool([mk(*j) for j in E19_CHURN_JOBS], lvl, deadline, stagger=E19_PLAN["churn_stagger_sec"])
    E19_STATE["arms"][name]["stages"]["churn"] = {"level": lvl, "sec": round(time.perf_counter() - t0, 1),
                                                 "metric_delta": e19_delta(m0, e19_metrics())}
    e19_save_state()


def e19_stage_long(name, st, deadline):
    if not e19_alive(name) or not st["idx"]:
        return
    if deadline - time.perf_counter() < E19_MIN_LONG_SEC:
        e19_event(f"arm {name}: {st['name']} not started (budget)")
        return
    gp, cm = E19_PLAN["g_pre"], E19_PLAN["cmp"]
    lvl = e19_conc_level(max(st["ctx"][i]["pf_len"] for i in st["idx"]) + gp + cm, E19_PLAN["long_conc"])
    rs = e19_reset_cache()
    m0, t0 = e19_metrics(), time.perf_counter()

    def mk(i):
        c = st["ctx"][i]
        ex = {"pf_len": c["pf_len"], "g_pre": gp, "n_ctx": st["n_ctx"], "reset": rs}

        def job(skip):
            if skip:
                e19_write_row(e19_skip_row(name, st["name"], "nat", i, skip, skip, ex))
                e19_write_row(e19_skip_row(name, st["name"], "cont", i, skip, skip, ex))
                return
            g = e19_run_req(c["ids"], gp + cm, deadline)
            r = e19_row(name, st["name"], "nat", i, c["ids"], g, lvl, ex)
            e19_write_row(r)
            if r["status"] != "ok":
                e19_write_row(e19_skip_row(name, st["name"], "cont", i, r["status"], "natural row " + r["status"], ex))
                return
            if len(g.tokens) < gp + 16:
                e19_write_row(e19_skip_row(name, st["name"], "cont", i, "na_short",
                                           f"natural stopped after {len(g.tokens)} tokens", ex))
                return
            ids2 = list(c["ids"]) + list(g.tokens[:gp])
            if not e19_alive(name):
                e19_write_row(e19_skip_row(name, st["name"], "cont", i, "server_dead", "server dead", ex))
                return
            g2 = e19_run_req(ids2, cm, deadline)
            e19_write_row(e19_row(name, st["name"], "cont", i, ids2, g2, lvl, ex))
        return job
    e19_pool([mk(i) for i in st["idx"]], lvl, deadline)
    E19_STATE["arms"][name]["stages"][st["name"]] = {"level": lvl, "sec": round(time.perf_counter() - t0, 1),
                                                    "metric_delta": e19_delta(m0, e19_metrics())}
    e19_save_state()


def e19_fill_missing(name):
    """Every planned row not written gets a row: server_dead after a death, else not_tested_budget."""
    done = set()
    if E19_ROWS_PATH.exists():
        for r in e19_load_rows(E19_ROWS_PATH):
            if r.get("arm") == name:
                done.add(e19_key(r))
    a = E19_STATE["arms"][name]
    st = "server_dead" if a.get("server_died") else "not_tested_budget"
    for k in e19_planned_keys(e19_meta(name)):
        if k not in done:
            e19_write_row(e19_skip_row(name, k[0], k[1], k[2], st, a.get("not_launched") or st))


def e19_launch(name, deadline):
    global E19_SERVER
    a = E19_STATE["arms"][name]
    backend = "FLASHINFER" if CUDA_LINK.get("probe") == "PASS" else "TRITON_ATTN"     # S1 CELL 5's rule
    for max_len in [None] + ([65536] if E19_ARMS[name]["kv"] == "auto" else []):
        flags = e19_flags(name, backend, max_len)
        a.update(flags=flags, backend=backend, max_len=max_len or MAX_MODEL_LEN)
        srv = E19Server(name if max_len is None else f"{name}_len{max_len}", flags)
        e19_event(f"arm {name}: launching ({shlex.join(flags)[:400]})")
        srv.start()
        try:
            ok, why = srv.wait_ready(max(60.0, min(E19_READY_MAX_SEC, deadline - time.perf_counter())))
        except BaseException:
            srv.kill()
            raise
        txt = srv.log_text()
        a["launch"] = {"ok": ok, "why": why, "sec": round(time.perf_counter() - srv.t0, 1)}
        a["log"] = e19_parse_log(txt)
        if ok:
            E19_SERVER = srv
            return True
        a["launch_tail"] = txt[-4000:]
        srv.kill()
        a["gpu_after_failed_launch"] = {"killed": e19_kill_gpu_pids(), "gpu": e19_wait_gpu_free()}
        # bf16 KV ("lower context if needed"): retry once with 65,536 only when the KV pool cannot hold 106,496
        if not (max_len is None and E19_ARMS[name]["kv"] == "auto"
                and re.search(r"(KV cache|max_model_len|max seq len|maximum number of tokens)", txt[-6000:], re.I)
                and re.search(r"(larger than|not enough|insufficient)", txt[-6000:], re.I)):
            break
    a["launch_failed"] = f"server launch failed: {why}"
    return False


def e19_arm(name, deadline):
    global E19_CUR
    E19_CUR = name
    a = E19_STATE["arms"][name] = {"arm": name, "cfg": E19_ARMS[name], "stages": {},
                                   "deadline_rel": round(deadline - NOTEBOOK_T0, 1),
                                   "start_rel": round(time.perf_counter() - NOTEBOOK_T0, 1)}
    E19_STATE["order"].append(name)
    e19_save_state()
    sampler = None
    try:
        if deadline - time.perf_counter() < E19_LAUNCH_MIN_SEC:
            a["not_launched"] = "NOT_TESTED_BUDGET: %.0f s left before the arm deadline" % (deadline - time.perf_counter())
            e19_event(f"arm {name}: {a['not_launched']}")
            return
        if not e19_launch(name, deadline):
            e19_event(f"arm {name}: LAUNCH FAILED: {a['launch_failed']}")
            return
        e19_event(f"arm {name}: {a['launch']['why']} | KV {a['log'].get('kv_tokens')} | block {a['log'].get('block_size')}"
                  f" | max concurrency {a['log'].get('max_concurrency')} | version {a['log'].get('version')}"
                  f" | async {a['log'].get('async_lines')}")
        # rev 2 (GPT audit 2 HIGH-4): the async A/B is only valid when the server really runs the intended mode
        _al = " ".join(a["log"].get("async_lines") or [])
        a["effective_async"] = ("enabled" if re.search(r"scheduling is\s+enabled", _al, re.I) else
                                "disabled" if re.search(r"scheduling is\s+disabled", _al, re.I) else "unknown")
        _want = "disabled" if E19_ARMS[name].get("async") is False else "enabled"
        if name in ("D", "E") and a["effective_async"] != _want and not (_want == "disabled" and a["effective_async"] == "unknown"):
            a["launch_failed"] = f"async mode mismatch: expected {_want}, observed {a['effective_async']} ({_al[:160]})"
            e19_event(f"arm {name}: LAUNCH FAILED: {a['launch_failed']}")
            return
        sm_ids = tokenizer.encode("Reply with the digits 1 2 3 4 5 6 7 8:", add_special_tokens=False)
        g = e19_run_req(sm_ids, 16, deadline)          # S1's engine smoke, through the server
        a["smoke"] = {"tokens": len(g.tokens), "error": g.err, "margins_present": any(m is not None for m in g.margins),
                      "usage_seen": g.usage is not None, "text": e19_decode(g.tokens)[:120]}
        if not g.tokens:
            a["launch_failed"] = f"engine smoke produced no tokens: {g.err}"
            e19_event(f"arm {name}: SMOKE FAILED {g.err}")
            return
        e19_save_state()
        sampler = E19Sampler(name)
        sampler.start()
        t0 = time.perf_counter()
        m0 = e19_metrics()
        e19_stage_serial(name, deadline)
        a["stages"]["serial"] = {"sec": round(time.perf_counter() - t0, 1), "metric_delta": e19_delta(m0, e19_metrics())}
        e19_stage_churn(name, deadline)
        for st in E19_LONG:
            e19_stage_long(name, st, deadline)
        e19_alive(name)
        a["log_end"] = e19_parse_log(E19_SERVER.log_text()) if E19_SERVER is not None else None
    except Exception as e:
        a["error"] = f"{type(e).__name__}: {e}"[:600]
        a["traceback"] = traceback.format_exc()[-3000:]
        e19_event(f"arm {name}: CRASH {a['error']}")
    finally:
        if sampler is not None:
            sampler.stop()
        try:
            e19_fill_missing(name)
        except Exception as e:
            a["fill_error"] = repr(e)[:300]
        a["stop"] = e19_stop_server()
        a["end_rel"] = round(time.perf_counter() - NOTEBOOK_T0, 1)
        a["meta"] = e19_meta(name)
        e19_save_state()
        e19_event(f"arm {name}: done ({a['end_rel'] - a['start_rel']:.0f} s); GPU released {a['stop'].get('gpu', {}).get('released')}")


def e19_report():
    rows = e19_load_rows(E19_ROWS_PATH) if E19_ROWS_PATH.exists() else []
    metas = {n: (E19_STATE["arms"][n].get("meta") or e19_meta(n)) for n in E19_STATE["order"]}
    E19_VERDICTS.update(e19_verdicts(rows, metas))
    E19_STATE["verdicts"] = E19_VERDICTS
    e19_save_state()
    L = e19_report_lines(E19_VERDICTS, E19_ARMS, E19_STATE["arms"],
                         f"# E19 vLLM MTP / prefix-cache correctness: profile {PROFILE}, run {RUN_ID}")
    L += ["", "Calibrators in this run: " + (", ".join(a for a in ("A", "B") if a in metas) or
                                             "none (run src/e19_compare.py over the three accounts' bundles)"), "",
          "## Stages", ""]
    for name in E19_STATE["order"]:
        a = E19_STATE["arms"][name]
        L.append(f"- {name}: smoke {a.get('smoke')}; stages " + json.dumps(a.get("stages"), default=str)[:1500])
    (E19_DIR / "e19_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)


if E19_ON:
    try:
        CUDA_LINK = ensure_libcuda_linkable()             # S1 CELL 5's -lcuda shim + link probe (FlashInfer JIT)
        CONFIG["cuda_link"] = CUDA_LINK
        E19_STATE["cuda_link"] = CUDA_LINK
        e19_event("CUDA/FlashInfer linker preflight: " + json.dumps(CUDA_LINK)[:300])
        e19_build_prompts()
        e19_event(f"prompts: {[p['len'] for p in E19_PROMPTS]} tokens; serial {E19_SERIAL_IDX}; churn "
                  f"{len(E19_CHURN_JOBS)} jobs; long " + str([(s['name'], len(s['idx'])) for s in E19_LONG]))
        _arms = [tuple(x) for x in P["e19_arms"]]
        E19_STATE["planned_arms"] = [x[0] for x in _arms]
        for _k, (_name, _reserve_min) in enumerate(_arms):
            _later = sum(r for _, r in _arms[_k + 1:]) * 60.0
            _deadline = HARD_STOP - E19_REPORT_RESERVE_SEC - _later
            e19_arm(_name, _deadline)
            if not (E19_STATE["arms"][_name].get("stop") or {}).get("gpu", {}).get("released", True):
                e19_event("GPU memory not released: no further arms")
                for _n2, _ in _arms[_k + 1:]:
                    E19_STATE["arms"][_n2] = {"arm": _n2, "cfg": E19_ARMS[_n2], "stages": {},
                                              "not_launched": "NOT_TESTED_BUDGET: GPU memory not released"}
                    E19_STATE["order"].append(_n2)
                    E19_STATE["arms"][_n2]["meta"] = e19_meta(_n2)
                break
    except Exception as _e:
        E19_STATE["fatal"] = f"{type(_e).__name__}: {_e}"[:800]
        E19_STATE["fatal_tb"] = traceback.format_exc()[-3000:]
        print("E19 FATAL:", E19_STATE["fatal_tb"], flush=True)
        try:
            e19_stop_server()
        except Exception:
            pass
    finally:
        try:
            e19_report()
        except Exception as _e:
            E19_STATE["report_error"] = traceback.format_exc()[-2000:]
            print("E19 report error:", E19_STATE["report_error"], flush=True)
        e19_save_state()
        try:
            with zipfile.ZipFile(WORKING / "Q27X_E19_BUNDLE.zip", "w", zipfile.ZIP_DEFLATED) as _z:
                for _p in sorted(E19_DIR.rglob("*")):
                    if _p.is_file():
                        _z.write(_p, _p.relative_to(OUT).as_posix())
            print("bundle:", WORKING / "Q27X_E19_BUNDLE.zip", flush=True)
        except Exception as _e:
            print("bundle error:", repr(_e)[:300], flush=True)
else:
    print("E19 not run:", "competition rerun (the CELL 0 identity submission stands)" if RERUN else f"profile {PROFILE} has no E19 arms",
          flush=True)
