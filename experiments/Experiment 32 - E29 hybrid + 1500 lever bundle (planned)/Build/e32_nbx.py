"""E32: the lever-bundle notebook functions of Q27-XHIGH-E32_LEVERS (executed into the notebook namespace by e32_setup(),
from the E32 payload dataset by name + sha; they read the notebook's globals). The notebook calls them only through
e32_call("<name>", default, *args) -> _e32x_<name>(*args) (the default is NOT passed on): a missing function / E32 not ON /
an exception = the default = the E29_DFLASH_D3S behaviour. The pure logic (switch parse, argv / env, the DFlash plan, the drop
rule, the read-back, the fingerprint) is e32_lib.py (E32L)."""
import copy as _e32copy

E32_MERLIN_FACTS = {}


def _e32x_rollback():
    """Back to the E29 DFlash plan (the base objects) and every lever off; used when the setup fails half-way."""
    b = E32_RT.get("base_df")
    if b and isinstance(E29_RT.get("df"), dict):
        E29_RT["df"].update({k_: _e32copy.deepcopy(v_) for k_, v_ in b.items()})
    E32_RT.update(active=False, cur=dict(E32L.OFF) if E32L is not None else {})
    _orig = getattr(E32L, "_e32_effective_orig", None) if E32L is not None else None
    if _orig is not None:
        E29L.effective = _orig


def _e32x_apply():
    """E29_RT["df"] (k, spec, captures, tail) <- the plan of the current lever values (OFF = the base objects exactly)."""
    pl = E32L.df_plan(E32_RT["base_df"], E32_RT["cur"], N.spec_json, N.compilation_json, E29L.df_spec_argv)
    pl["coverage"] = E32L.plan_coverage(pl)
    E29_RT["df"].update(k=pl["k"], spec=pl["spec"], captures=pl["captures"], tail=pl["tail"])
    E32_RT["plan"] = {k_: pl[k_] for k_ in ("k", "seqs", "spec_value", "captures", "coverage", "base")}
    return pl


def _e32x_setup():
    """After e29_setup: the CELL 1 switches -> the lever values; with a DFlash setup the E29 plan is rebuilt for them (the
    levers act on the D tries only), E29L.effective gains the E32 lever states + fingerprint (the LIVE start line)."""
    rt = E32_RT
    raw = {k_: globals().get(E32L.SWITCH_VAR[k_], E32L._MISSING) for k_ in E32L.LEVERS}
    p = E32L.parse(raw, os.environ.get("E32_LEVERS_JSON") if E11_HARNESS else None)
    sp = E32L.speed_cfg(globals().get("E32_SPEED"))
    rt.update(parsed=p, speed_cfg=sp, drops=[], readbacks=[], speed=None, speed_done=False, fingerprints=[], relaunch=[],
              lib=E32L.LIB_VERSION, cur=dict(E32L.OFF), active=False, last_argv=[])
    CONFIG["e32"] = {"parsed": p, "speed": sp, "lib": E32L.LIB_VERSION}
    df = E29_RT.get("df") or {}
    if "df" not in (E29_RT.get("on") or []) or not df.get("ok"):
        rt["state_why"] = "OFF_NO_DFLASH: the E29 DFlash setup did not succeed (MTP K3 = E29's fallback; no lever applies)"
        log("E32 levers: n/a (no DFlash try)", json.dumps({"parsed_on": p["on"], "invalid": p["invalid"]}, default=str))
        return
    rt["base_df"] = {k_: _e32copy.deepcopy(df[k_]) for k_ in ("spec", "k", "captures", "tail")}
    rt["cur"] = dict(p["cfg"])
    rt["active"] = True
    _e32x_apply()
    _e32x_patch_effective()
    log("E32 levers", json.dumps({"cfg": p["cfg"], "on": p["on"], "invalid": p["invalid"], "plan": {k_: rt["plan"].get(k_) for k_ in
                                  ("k", "seqs", "spec_value", "coverage")}, "argv": E32L.lever_argv(rt["cur"]),
                                  "env": E32L.lever_env(rt["cur"]), "speed": sp}, default=str)[:2000])


def _e32x_seqs():
    """nb5_boot: --max-num-seqs of a D try (E29: E29_CFG workers 24)."""
    return int(E32_RT["cur"]["SEQS"]) if E32_RT.get("active") else int(E29_CFG["workers"])


def _e32x_seqs_top():
    """nb5_capacity's top-rung rule (launched seqs -> workers) for the D try's E32 seqs (-1 = no extra value)."""
    return int(E32_RT["cur"]["SEQS"]) if E32_RT.get("active") and E29_RT.get("df_try") else -1


def _e32x_argv(df):
    if not E32_RT.get("active") or not df:
        return []
    a = E32L.lever_argv(E32_RT["cur"])
    E32_RT["last_argv"] = list(a)
    return a


def _e32x_env(df):
    return E32L.lever_env(E32_RT["cur"]) if E32_RT.get("active") and df else {}


def _e32x_tag(df):
    return E32L.tag(E32_RT["cur"]) if E32_RT.get("active") and df else ""


def _e32x_drop(lev, cls, status, try_name, stage):
    old = E32_RT["cur"].get(lev)
    E32_RT["cur"] = E32L.drop(E32_RT["cur"], lev)
    t_ = round(time.perf_counter() - NOTEBOOK_T0)
    E32_RT["drops"].append({"lever": lev, "value": old, "class": cls, "status": str(status)[:300], "try": try_name, "stage": stage,
                            "t_sec": t_, "why": f"{cls}: {status}"[:300]})
    E29_RT["dropped"].append({"lever": "E32:" + lev, "why": f"{cls}: {status}"[:600], "try": try_name, "t_sec": t_})
    _e32x_apply()
    log("!!!!!!!!!! E32 lever dropped:", lev, old, "->", E32L.OFF[lev], "|", cls, "|", str(status)[:200], "| try", try_name, "|",
        stage, "| still on", E32L.on_list(E32_RT["cur"]))


def _e32x_pick(trec, log_text, status):
    """A pre-phase failure of a D try with E32 levers on: one lever dropped (e32_lib.pick), the same D rung again (True).
    False = E29's rule (no E32 lever on, or a memory failure with no memory lever left -> the D ladder)."""
    if not E32_RT.get("active") or not trec.get("e29_df"):
        return False
    lt = log_text or ""
    lev, cls = E32L.pick(E32_RT["cur"], status, lt, E29L.is_oom(lt), E29L.is_device_fault(lt),
                         (trec.get("config_readback") or {}).get("checks"))
    trec["e32_pick"] = {"lever": lev, "class": cls, "levers": dict(E32_RT["cur"])}
    if not lev:
        return False
    _e32x_drop(lev, cls, status, trec.get("name"), "pre-phase")
    return True


def _e32x_df_dropped(why):
    """e29_drop('df'): DFlash is gone (MTP K3 = E29's fallback): every E32 lever still on is recorded as dropped with it."""
    if not E32_RT.get("active"):
        return
    for k_ in E32L.on_list(E32_RT["cur"]):
        E32_RT["drops"].append({"lever": k_, "value": E32_RT["cur"][k_], "class": "WITH_DFLASH", "status": str(why)[:300],
                                "try": None, "stage": "df_dropped", "t_sec": round(time.perf_counter() - NOTEBOOK_T0),
                                "why": "WITH_DFLASH: " + str(why)[:280]})
    E32_RT.update(active=False, cur=dict(E32L.OFF))


def _e32x_mtp():
    """e29_mtp_boot (in-run DFlash -> MTP K3): the E32 argv, the E32 env and the E32 seqs leave the booted config (E29's own
    code then strips the DFlash tail; the relaunch = the E28 top rung = E29's)."""
    if not E32_RT.get("active") or not NB5_ACTIVE_BOOT:
        return
    s_ = int(P.get("top_rung_seqs") or 24)
    x = E32L.strip_last(NB5_ACTIVE_BOOT.get("extra") or [], E32_RT.get("last_argv") or [])
    NB5_ACTIVE_BOOT.update(extra=E32L.set_last_flag(x, "--max-num-seqs", s_), max_num_seqs=s_,
                           levers={k_: v_ for k_, v_ in dict(NB5_ACTIVE_BOOT.get("levers") or {}).items() if k_ not in E32L.GDN_ENV.values()})
    E32_RT["last_argv"] = []


def _e32x_relaunch(old):
    """e21_relaunch (an in-run death of a DFlash server with E32 levers on): one lever dropped (e32_lib.pick on the dead
    server's log), the booted config rebuilt (argv / env / seqs / tail), the same D rung relaunched (True). False = E29's
    in-run plan (no lever on; a memory death with no memory lever left -> the next D rung / MTP K3)."""
    if not E32_RT.get("active") or FAKE or not E29_RT.get("df_live") or "df" not in (E29_RT.get("on") or []) or not NB5_ACTIVE_BOOT:
        return False
    if not E32L.on_list(E32_RT["cur"]):
        return False
    lt = old.log() or ""
    lev, cls = E32L.pick(E32_RT["cur"], "IN_RUN_DEATH", lt, E29L.is_oom(lt), E29L.is_device_fault(lt))
    E32_RT["relaunch"].append({"from": NB5_ACTIVE_BOOT.get("name"), "lever": lev, "class": cls,
                               "t_sec": round(time.perf_counter() - NOTEBOOK_T0)})
    if not lev:
        return False
    o_argv, o_tail = list(E32_RT.get("last_argv") or []), list(E29_RT["df"]["tail"])
    _e32x_drop(lev, cls, "IN_RUN_DEATH", NB5_ACTIVE_BOOT.get("name"), "in-run")
    new = E32_RT["cur"]
    E32_RT["last_argv"] = E32L.lever_argv(new)
    NB5_ACTIVE_BOOT.update(extra=E32L.rebuild_extra(NB5_ACTIVE_BOOT.get("extra") or [], o_argv, o_tail, new["SEQS"], E32_RT["last_argv"],
                                                    E29_RT["df"]["tail"]),
                           levers=dict({k_: v_ for k_, v_ in dict(NB5_ACTIVE_BOOT.get("levers") or {}).items()
                                        if k_ not in E32L.GDN_ENV.values()}, **E32L.lever_env(new)),
                           max_num_seqs=int(new["SEQS"]))
    log("!!!!!!!!!! E32 in-run death -> relaunch without", lev, json.dumps(E32_RT["relaunch"][-1], default=str))
    return True


def _e32x_merlin():
    if not E32_MERLIN_FACTS:
        try:
            tree = Path(((MERLIN_ENG or {}).get("kv") or {}).get("MERLIN_TREE") or "/nonexistent")
            lp = tree / "tools/merlin/launch.py"
            E32_MERLIN_FACTS.update(tree=str(tree), commit_pin=E32L.MERLIN_COMMIT_PIN, payload=str(globals().get("MERLIN_PAY")),
                                    launch_py_sha256=hashlib.sha256(lp.read_bytes()).hexdigest() if lp.is_file() else None)
        except Exception as e:
            return {"error": repr(e)[:200]}
    return dict(E32_MERLIN_FACTS)


def _e32x_states():
    return E32L.lever_states(E32_RT.get("parsed"), E32_RT.get("cur"), E32_RT.get("drops"),
                             bool(E32_RT.get("active")) or bool(E32_RT.get("drops")))


def _e32x_fp(srv=None):
    """The engine fingerprint of the booted server (argv, env, KV pin, Merlin, every lever's state)."""
    s = srv if srv is not None else globals().get("E11_SERVER")
    return E32L.fingerprint(getattr(s, "argv", None), getattr(s, "env", None), (NB5_ACTIVE_BOOT or {}).get("kv_bytes"),
                            _e32x_merlin(), E32L.short_states(_e32x_states()))


def _e32x_short():
    fp = _e32x_fp()
    return {"levers": E32L.short_states(_e32x_states()), "fp": fp.get("short"), "seqs": (E32_RT.get("cur") or {}).get("SEQS"),
            "k": (E29_RT.get("df") or {}).get("k"), "kv": fp.get("kv_bytes"), "drops": len(E32_RT.get("drops") or [])}


def _e32x_patch_effective():
    if getattr(E32L, "_e32_effective_orig", None) is None:
        E32L._e32_effective_orig = E29L.effective
    orig = E32L._e32_effective_orig

    def effective(rt, rerun=False):
        out = orig(rt, rerun)
        try:
            out["e32"] = _e32x_short()
        except Exception as e:
            out["e32"] = {"error": repr(e)[:120]}
        return out
    E29L.effective = effective


def _e32x_readback(s, trec):
    """Right after the DFlash read-back of a D try: every lever's own read-back (record) + the engine fingerprint; the DYN
    fallback lines = a real failure (status E32_DYN_FALLBACK -> the server stops, pick drops SPEC)."""
    if not E32_RT.get("active") or FAKE:
        return False
    t = s.log() or ""
    try:
        we = E29L.proc_env_readback(s.proc.pid, tuple(E32L.GDN_ENV.values()))
    except Exception:
        we = []
    try:
        ci = rpc(s.port, "nb1_cache_info")
    except Exception:
        ci = None
    rb = E32L.readback(E32_RT["cur"], t, (trec.get("config_readback") or {}).get("log"), we, ci)
    fp = _e32x_fp(s)
    rb.update(try_name=trec.get("name"), levers=dict(E32_RT["cur"]), fingerprint_short=fp.get("short"))
    trec["e32_readback"] = rb
    E32_RT["readbacks"].append({k_: rb.get(k_) for k_ in ("try_name", "levers", "applied", "real_failure", "fingerprint_short")})
    E32_RT["fingerprints"].append(dict(fp, try_name=trec.get("name")))
    log("E32 read-back", json.dumps({"try": trec.get("name"), "applied": rb["applied"], "real_failure": rb["real_failure"],
                                     "fingerprint": fp.get("short")}, default=str))
    print("E32 FINGERPRINT", fp.get("short"), json.dumps({k_: fp.get(k_) for k_ in ("kv_bytes", "merlin", "levers")}, default=str)[:1500],
          flush=True)
    if rb["real_failure"]:
        trec["status"] = rb["real_failure"]
        return True
    return False


def _e32x_speed(s, trec):
    """The fixed-window speed pass (record only), once per run, on the first D try that passed every gate incl. the F2 probe,
    before the phases; commit only (the hidden rerun skips it). Bands of the E32_SPEED switch (default 8k@16, 20k@24, 90k@8,
    116k@6); the cap skips only optional bands (90k / 116k always run, user rule); L.steady windows (WIN s) after the
    prefill settles; AL from the spec counters too. A server death here = this try's death (the boot gates handle it)."""
    sp = E32_RT.get("speed_cfg") or {}
    if E32_RT.get("speed_done") or not E32_RT.get("active"):
        return
    E32_RT["speed_done"] = True
    if RERUN or FAKE or not sp.get("on"):
        E32_RT["speed"] = {"status": "SKIPPED_" + ("RERUN" if RERUN else "FAKE" if FAKE else "SWITCH_OFF")}
        return
    pr = trec.get("e29_probe") or {}
    rate = float(pr["prompt"]) / float(pr["sec"]) if pr.get("prompt") and pr.get("sec") else float(PREFILL.get("tok_s") or 700.0)
    t0 = time.perf_counter()
    out = E32_RT["speed"] = {"status": "RUNNING", "try": trec.get("name"), "prefill_tok_s_used": round(rate, 1), "cap_min": sp["cap_min"],
                             "levers": dict(E32_RT["cur"]), "k": E29_RT["df"]["k"], "windows": {}}
    for band, n in sp["windows"]:
        key = f"{band}@{n}"
        need = E32L.speed_need_sec(band, n, rate, WIN)
        el = time.perf_counter() - t0
        if band not in E32L.SPEED_REQUIRED and el + need > sp["cap_min"] * 60:
            out["windows"][key] = {"status": "SKIPPED_CAP", "need_sec": round(need), "elapsed_sec": round(el)}
            continue
        if left() < need + 1800:
            out["windows"][key] = {"status": "SKIPPED_BUDGET", "need_sec": round(need)}
            continue
        if not s.alive():
            out["windows"][key] = {"status": "SKIPPED_SERVER_DEAD"}
            break
        tok = E32L.SPEED_BANDS[band]
        ps = [prompt(tok, v) for v in range(int(n))]
        c0, w0 = e29_counters(), time.perf_counter()
        try:
            r = L.steady(s.port, ps, int(n), secs=WIN, max_new=MAX_LEN - tok - 512, temperature=1.0, prefill_tok_s=rate)
        except Exception as e:
            r = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:300]}
        o = E32L.speed_record(band, n, r, E29L.canary(c0, e29_counters(), E29_RT["df"]["k"]))
        o["sec"] = round(time.perf_counter() - w0, 1)
        out["windows"][key] = o
        del ps
        print(E32L.speed_line((time.perf_counter() - NOTEBOOK_T0) / 60.0, key, o), flush=True)
    out.update(status="DONE", sec=round(time.perf_counter() - t0, 1), alive_after=s.alive())
    log("E32 speed pass", json.dumps({k_: (v_.get("tok_s"), v_.get("mean_acceptance_length"), v_.get("preemptions"), v_.get("status"))
                                      for k_, v_ in out["windows"].items()}, default=str), "| sec", out["sec"])


def _e32x_summary(sm=None):
    """The report record (q27x/e32_report.json + E28_OUT["e32"]) and the [E32 RUN SUMMARY] block; record only."""
    sm = sm or {}
    rt, cap = E32_RT, globals().get("NB5_CAPACITY") or {}
    st = _e32x_states()
    fp = _e32x_fp()
    out = {"lib": rt.get("lib"), "state": rt.get("state"), "state_why": rt.get("state_why"), "active_at_end": rt.get("active"),
           "parsed": rt.get("parsed"), "speed_cfg": rt.get("speed_cfg"), "levers": st, "levers_short": E32L.short_states(st),
           "drops": rt.get("drops"), "readbacks": rt.get("readbacks"), "speed": rt.get("speed"), "relaunch": rt.get("relaunch"),
           "plan": rt.get("plan"), "fingerprint": fp, "fingerprint_short": fp.get("short"), "fingerprints": rt.get("fingerprints"),
           "seqs": (rt.get("cur") or {}).get("SEQS"), "k": (E29_RT.get("df") or {}).get("k"), "kv_bytes": (NB5_ACTIVE_BOOT or {}).get("kv_bytes"),
           "pool_blocks": cap.get("pool_blocks"), "f_eff": cap.get("f_eff"), "workers": cap.get("workers"), "r2_workers": cap.get("r2_workers"),
           "tok_s_window": sm.get("tok_s_window"), "solver_tok_s": sm.get("solver_tok_s"), "tok_s_per_15min": sm.get("tok_s_per_15min"),
           "al": sm.get("al"), "preemptions": sm.get("preemptions"), "prefix_hit_pct": sm.get("prefix_hit_pct"),
           "booted_try": sm.get("booted_try"), "tries": sm.get("tries"), "errors": rt.get("errors")}
    try:
        write_json_atomic(BASE_OUT / "e32_report.json", out)
    except Exception as e:
        out["write_error"] = repr(e)[:200]
    print("[E32 RUN SUMMARY]\n" + E32L.summary_text(out), flush=True)
    return out
