# ==================================================================== CELL E17 — SGLang silent-accuracy tests (PLAN_E17 rev 2)
# One session, engine only, every server TP4 with the production numerics (RedHat INT4, FP8 KV, fp16 SSM,
# extra_buffer_lazy, overlap, decode CUDA graphs). The production launch made by the engine cell is the first launch (JIT
# warm, Phase 0 facts); every arm below is its own relaunch (71-97 s measured in EMS RISKY v1).
#   Phase 0      provenance + safety asserts (schedule_batch sha, #43567 state, DFlash taps, vocab, GDN backend)
#   L1-L6        unpatched: T1a small-slot upstream repro (interval 128, hook on / off, interval 256), T1b forced fail,
#                T1c DFlash W6 (interval 64), L6 = upstream Class 2 (no_buffer + MTP, #43387 known bug: harness check)
#   L12-L13      T3b: target-only vs DFlash, radix cache off, serial 4 + 16 concurrent, greedy 2,048 (T3a fp16 scores)
#   patch #43567 L7-L9: the T1 arms again (+ the patch-safety cases on L7)
#   decision A   BUG_EXPOSED_43567 -> T2 natural pressure (production flags, 16 streams), patched L11, revert, unpatched L10
#   L15          T3a fp32 SSM (only if time is left)
# Instrumentation: e17_hook (sitecustomize on PYTHONPATH of the server only; no SGLang source edit). Verdict ceiling:
# ENGINE_CORRECTNESS_VALIDATED. Every phase has its own deadline; whatever does not fit is skipped and recorded.
import random as _e17_rnd
import re
import shutil
import subprocess
import threading
import traceback
import shlex as _e17_shlex
import statistics as _e17_st
import numpy as np

E17 = {"plan": "PLAN_E17 rev 2 (2026-10-26)", "phase0": {}, "launches": {}, "arms": {}, "order": [], "skipped": {},
       "hook_sha256": E17_HOOK_SHA}
E14_RESULTS["e17"] = E17
E17_OUT = E14_OUT / "e17"
E17_OUT.mkdir(parents=True, exist_ok=True)
E17_HOOK_DIR = Path(os.environ.get("E17_HOOK_DIR", "/tmp/e17_hook"))
E17_HOOK_DIR.mkdir(parents=True, exist_ok=True)
(E17_HOOK_DIR / "sitecustomize.py").write_bytes(E17_HOOK_SRC.encode("utf-8"))      # bytes: no newline translation
assert hashlib.sha256((E17_HOOK_DIR / "sitecustomize.py").read_bytes()).hexdigest() == E17_HOOK_SHA
_SPEC = ("--speculative-algorithm", "--speculative-draft-model-path", "--speculative-draft-model-quantization",
         "--speculative-num-draft-tokens", "--speculative-num-steps", "--speculative-eagle-topk")
E17_TGT_DROP = _SPEC + ("--min-free-slots-delay",)
E17_TOL_FLOOR = 5e-3          # audit 4 (three GPT audits 2026-10-26): a FIXED reuse tolerance (never scaled by the A/A)
E17_AA_MAX = 1e-3             # a reference whose own A/A KL exceeds this is unstable: verdict INCONCLUSIVE, never clean
E17_P = {"W44": [44] * 128, "W42": [42] * 128}
E17_P["P255"] = [42] * 128 + [43] * 127          # upstream Class 1: prefill checkpoint 192, first decode crosses 256
E17_P["P511"] = [42] * 128 + [43] * 383          # interval-256 variant: checkpoint 448, first decode crosses 512
for _n in (246, 249, 252):                        # T1c (interval 64): checkpoint 192, boundary 256 is 4-10 tokens ahead
    E17_P[f"P{_n}"] = [42] * 128 + [43] * (_n - 128)
E17_CTX = None
E17_ARM = {"name": None}


def e17_save():
    e14_save()
    try:
        write_json_atomic(E17_OUT / "e17_report.json", E17)
    except Exception as _e:
        print("E17 save failed:", repr(_e)[:200], flush=True)


def e17_edit_flags(flags, drop=(), set_=None):
    """drop: flag names removed with their value(s); set_: {flag: value | [values] | None (bare flag)}."""
    out, i, set_ = [], 0, dict(set_ or {})
    names = set(drop) | set(set_)
    while i < len(flags):
        f = flags[i]
        if f in names:
            i += 1
            while i < len(flags) and not str(flags[i]).startswith("--"):
                i += 1
            continue
        out.append(f)
        i += 1
    for k, v in set_.items():
        out += [k] + ([str(x) for x in v] if isinstance(v, (list, tuple)) else ([str(v)] if v is not None else []))
    return out


def e17_left():
    return HARD_STOP - time.perf_counter()


def e17_ctx(cap):
    c = E14M.Ctx(SGL_URL, lambda: E11_SERVER.log_text() if E11_SERVER is not None else "", E17_OUT,
                 model_id="e17", deadline=time.perf_counter() + min(cap, max(30.0, e17_left() - 5 * 60)), log=print,
                 encode=None, normalize=lambda x: x)
    c.scale = E14_TS
    c.headers = {"Content-Type": "application/json"}      # no probe header: the real SGLang ignores it, the fake must not
    return c


def e17_launch(name, set_=None, drop=(), env=None, hook=True, verbose=True, timeout=None):
    """Kill the current server, launch the production command + this arm's changes (hook on: PYTHONPATH = the hook dir,
    E17_TRACE_DIR = a fresh per-launch dir), wait, record caps and the hook self-test."""
    global E11_SERVER, E17_CTX
    rec = {"arm": name, "set": {k: v for k, v in (set_ or {}).items()}, "drop": list(drop), "env": dict(env or {}),
           "hook": hook}
    if not E14M.launch_allowed(time.perf_counter(), HARD_STOP, relaunch_cost=240.0, floor=8 * 60, margin=4 * 60):
        rec["skipped"] = "NOT_TESTED_BUDGET"
        E17["launches"][name] = rec
        e17_save()
        print(f"E17 launch {name}: skipped (budget)", flush=True)
        return False, rec
    E14_STATE.enter("E17_LAUNCH:" + name, deadline=HARD_STOP)
    t0 = time.perf_counter()
    try:
        e11_kill_server(wait=True)
    except Exception:
        pass
    E11_SERVER = None
    rec["gpu_free"] = _e11_gpu_free()
    flags = e17_edit_flags(e11_flags(E14_VENV["cfg"]), drop=drop, set_=set_)
    envx = dict(E14_VENV["env"])
    envx.update(SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK="0", SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL="0")
    tdir = E17_OUT / "trace" / name
    if hook:
        shutil.rmtree(tdir, ignore_errors=True)
        tdir.mkdir(parents=True, exist_ok=True)
        envx.update(PYTHONPATH=str(E17_HOOK_DIR), E17_TRACE_DIR=str(tdir), E17_TRACE_VERBOSE="1" if verbose else "0")
    else:
        envx.pop("PYTHONPATH", None)
        envx.pop("E17_TRACE_DIR", None)
    envx.update(env or {})
    srv = E11Server(E14_VENV["launch"] + flags, envx, "e17_" + name)
    srv.start()
    try:
        ok, why = srv.wait_ready(min(timeout or SGL_RELAUNCH_TIMEOUT_SEC, max(60.0, e17_left() - 6 * 60)))
    except BaseException:
        try:
            srv.kill()
        except Exception:
            pass
        raise
    rec.update(ok=ok, why=why, sec=round(time.perf_counter() - t0, 1), flags=flags, trace_dir=str(tdir) if hook else None)
    if ok:
        E11_SERVER = srv
        rec["caps"] = e11_caps(srv.log_text())
        if hook:
            rec["hook_selftest"] = e17_hook_selftest(tdir)
    else:
        rec["log_tail"] = srv.log_text()[-3000:]
        srv.kill()
    E17["launches"][name] = rec
    E17_ARM["name"] = name if ok else None
    e17_save()
    print(f"E17 launch {name}: {why} ({rec['sec']} s) hook={rec.get('hook_selftest')}", flush=True)
    return ok, rec


def e17_hook_selftest(tdir, wait=60.0):
    """The hook is live when EVERY scheduler rank reports the three scheduler targets patched (audit 4: names merged
    over pids could pass a partial install). TP4 needs 4 fully patched processes (the CPU fake has one)."""
    need = {"ScheduleBatch", "SchedulerBatchResultProcessor", "MambaComponent"}
    want_ranks = 1 if E11_HARNESS else 4
    end = time.perf_counter() + wait * max(E14_TS, 0.05)
    seen, full = set(), 0
    while True:
        full, seen = 0, set()
        for p in Path(tdir).glob("counters_*.json"):
            try:
                names = {x.split("@")[0] for x in json.loads(p.read_text()).get("patched") or []}
            except Exception:
                continue
            seen |= names
            full += 1 if need <= names else 0
        if full >= want_ranks or time.perf_counter() > end:
            break
        time.sleep(1.0)
    fails = []
    for p in Path(tdir).glob("trace_*.jsonl"):
        try:
            fails += [json.loads(l) for l in p.read_text().splitlines() if '"patch_failed"' in l]
        except Exception:
            pass
    return {"ok": full >= want_ranks and not fails, "ranks_fully_patched": full, "ranks_wanted": want_ranks,
            "patched": sorted(seen), "patch_failed": fails[:4]}


def e17_trace(name, settle=3.0):
    """The hook counters of one launch. With TP4 every rank runs the scheduler: the primary process = the one with the
    most decisions (counters are per process, never summed)."""
    tdir = E17_OUT / "trace" / name
    time.sleep(settle * max(E14_TS, 0.05) + 0.5)
    procs = {}
    for p in tdir.glob("counters_*.json"):
        try:
            d = json.loads(p.read_text())
            procs[str(d.get("pid"))] = d
        except Exception:
            continue
    key = lambda d: sum((d.get("counters") or {}).get(k, 0) for k in ("optional_alloc_attempts", "donations_unfinished",
                                                                      "donations_finished", "spec_window_reachable"))
    prim = max(procs.values(), key=key) if procs else None
    out = {"processes": len(procs), "primary_pid": (prim or {}).get("pid"), "counters": dict((prim or {}).get("counters") or {}),
           "alloc": (prim or {}).get("alloc"), "per_pid": {k: (v.get("counters") or {}) for k, v in procs.items() if key(v)}}
    evs = []
    if prim is not None:
        tp = tdir / f"trace_{prim['pid']}.jsonl"
        if tp.exists():
            evs = [json.loads(l) for l in tp.read_text(encoding="utf-8").splitlines() if l.strip()]
    out["events_by_kind"] = {}
    for e in evs:
        out["events_by_kind"][e.get("ev")] = out["events_by_kind"].get(e.get("ev"), 0) + 1
    out["stale_events"] = [e for e in evs if e.get("ev") == "donation" and e.get("stale_candidate")][:20]
    out["fail_events"] = [e for e in evs if e.get("ev") == "alloc_failed"][:20]
    ks = []
    for p in tdir.glob("trace_*.jsonl"):
        ks += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if '"kv_scales"' in l]
    if ks:
        out["kv_scales"] = ks[0]
    return out


def e17_rid_evidence(tr, name):
    """Per-request evidence of one launch from the primary pid's full trace: rids whose optional slot failed, and rids
    with a CONFIRMED stale donation (decode path: at donation; spec path: a later spec_resolve with crossed=True), with
    the donated length(s)."""
    out = {"failed": set(), "confirmed": {}}
    pid = (tr or {}).get("primary_pid")
    tp = E17_OUT / "trace" / name / f"trace_{pid}.jsonl"
    if pid is None or not tp.exists():
        return out
    pend = {}
    for l in tp.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(l)
        except Exception:
            continue
        rid = e.get("rid")
        if e.get("ev") == "alloc_failed" and rid:
            out["failed"].add(rid)
        elif e.get("ev") == "donation" and e.get("stale_candidate") and rid:
            ln = e.get("returned") if isinstance(e.get("returned"), int) else e.get("cache_len")
            if ((e.get("marked") or {}).get("path")) == "decode":
                out["confirmed"].setdefault(rid, []).append(ln)
            else:
                pend.setdefault(rid, []).append(ln)
        elif e.get("ev") == "spec_resolve" and rid and e.get("confirmed"):
            out["confirmed"].setdefault(rid, []).extend(pend.pop(rid, []))
    return out


def e17_idle(ctx, wait=30.0):
    """Idle only when /metrics really reports running and queued = 0 (audit 4: missing fields are not idle)."""
    end = time.monotonic() + max(0.5, wait * max(E14_TS, 0.05))
    while True:
        try:
            p = E14M.prom(ctx, timeout=5)
            if "sglang:num_running_reqs" in p and "sglang:num_queue_reqs" in p and not p["sglang:num_running_reqs"] \
                    and not p["sglang:num_queue_reqs"]:
                return True
        except Exception:
            pass
        if time.monotonic() >= end:
            return False
        time.sleep(0.25)


# ---------------------------------------------------------------------------------------------- probes
def e17_post(ctx, body, timeout=600):
    r = E14M.post_json(ctx, "/generate", body, timeout=timeout)
    return r[0] if isinstance(r, list) else r


def e17_gen(ctx, ids, max_new, *, score=False, ignore_eos=True, temperature=0.0, rid=None):
    sp = {"temperature": temperature, "max_new_tokens": int(max_new), "ignore_eos": ignore_eos}
    if temperature > 0:
        sp.update(top_p=0.95, top_k=20)
    body = {"rid": rid or f"e17-{_e17_rnd.randrange(10 ** 12)}", "input_ids": list(ids), "sampling_params": sp}
    if score:
        body.update(return_logprob=True, logprob_start_len=-1, token_ids_logprob=E17_VOCAB_IDS)
    r = e17_post(ctx, body)
    meta = r.get("meta_info") or {}
    out = {"cached": meta.get("cached_tokens"), "n_out": len(r.get("output_ids") or []), "out": list(r.get("output_ids") or []),
           "finish": meta.get("finish_reason")}
    if score:
        out["lp"] = e17_full_lp(meta)
    return out


def e17_full_lp(meta):
    rows = meta.get("output_token_ids_logprobs") or []
    assert len(rows) >= 1, "no output_token_ids_logprobs"
    pairs = rows[0]
    assert len(pairs) == E17_V, (len(pairs), E17_V)
    ids = [int(x[1]) for x in pairs]
    assert ids == E17_VOCAB_IDS, "vocabulary order differs"
    lp = np.asarray([float(x[0]) for x in pairs], dtype=np.float64)
    assert np.isfinite(lp).all(), "non-finite logprobs"
    s = float(np.exp(lp).sum())
    lp = lp - np.logaddexp.reduce(lp)
    return {"lp": lp, "sum_raw": round(s, 6)}


def e17_kl(a, b):
    """KL(a || b) over the full vocabulary (a = the reference), plus max |dlogp| and top-1 change."""
    pa, la, lb = np.exp(a["lp"]), a["lp"], b["lp"]
    return {"kl": float(np.sum(pa * (la - lb))), "max_abs_dlogp": float(np.max(np.abs(la - lb))),
            "top1_changed": int(np.argmax(la)) != int(np.argmax(lb))}


def e17_flush(ctx):
    """flush_cache on a server that /metrics shows idle; valid = really flushed AND idle (audit 4: a cold control on a
    busy or unflushed server is no control)."""
    f = E14M.flush_cache(ctx, wait=30.0)
    if not f.get("ok"):
        time.sleep(2.0)
        f = E14M.flush_cache(ctx, wait=30.0)
    f["idle_metrics"] = e17_idle(ctx, wait=10.0)
    f["valid"] = bool(f.get("ok") and f.get("idle_metrics"))
    return f


def e17_warm(ctx, prefixes):
    return [e17_gen(ctx, p, 1)["cached"] for p in prefixes]


def e17_t1(ctx, pname, prefixes, seed_new, aa=3, cycles=3):
    """Upstream Class 1 (test_mamba_cache_kl.py): reference = flush, warm the prefixes, score P (aa times = the A/A);
    cycle = flush, warm, seed (P + seed_new greedy tokens), score P again (the reuse)."""
    P = E17_P[pname]
    tag = f"{pname}-{_e17_rnd.randrange(10 ** 9)}"
    out = {"prompt": pname, "len": len(P), "seed_new": seed_new, "ref_cached": [], "reuse_cached": [], "seed_cached": [],
           "flush_ok": [], "aa": [], "reuse": [], "cycles": []}
    out["safe_depth"] = max(len(p) for p in prefixes if P[:len(p)] == p) if prefixes else 0
    refs, ref_flush = [], []
    for _ in range(aa):
        f = e17_flush(ctx)
        ref_flush.append(f["valid"])
        out["flush_ok"].append(f.get("ok"))
        e17_warm(ctx, prefixes)
        r = e17_gen(ctx, P, 1, score=True)
        refs.append(r["lp"])
        out["ref_cached"].append(r["cached"])
        out["sum_raw"] = r["lp"]["sum_raw"]
    out["aa"] = [e17_kl(refs[0], x) for x in refs[1:]]
    out["aa_kl_max"] = max([x["kl"] for x in out["aa"]] or [0.0])
    out["tol"] = E17_TOL_FLOOR
    # audit 4: the reference must be clean (every flush valid, every reference hit exactly the safe prefix) and stable
    out["reference_valid"] = bool(all(ref_flush) and all(c == out["safe_depth"] for c in out["ref_cached"])
                                  and np.isfinite(out["aa_kl_max"]) and out["aa_kl_max"] <= E17_AA_MAX)
    for c in range(cycles):
        f = e17_flush(ctx)
        out["flush_ok"].append(f.get("ok"))
        e17_warm(ctx, prefixes)
        seed_rid, probe_rid = f"e17-seed-{tag}-{c}", f"e17-probe-{tag}-{c}"
        s = e17_gen(ctx, P, seed_new, rid=seed_rid)
        out["seed_cached"].append(s["cached"])
        r = e17_gen(ctx, P, 1, score=True, rid=probe_rid)
        kl = e17_kl(refs[0], r["lp"])
        out["reuse_cached"].append(r["cached"])
        out["reuse"].append(kl)
        out["cycles"].append({"cycle": c, "seed_rid": seed_rid, "probe_rid": probe_rid, "flush_valid": f["valid"],
                              "seed_cached": s["cached"], "probe_cached": r["cached"], "probe_kl": kl["kl"],
                              "top1_changed": kl["top1_changed"]})
    out["reuse_kl_max"] = max([x["kl"] for x in out["reuse"]] or [0.0])
    out["stale_depth_hits"] = sum(1 for c in out["reuse_cached"] if isinstance(c, int) and c > out["safe_depth"])
    out["_ref_lp"] = refs[0]
    return out


def e17_cycle_chain(t, cyc, ev):
    """audit 4: the causal chain of ONE seed/probe cycle: this seed's optional slot failed, this seed made a confirmed
    stale donation of length L, and this probe (after the cycle's own flush) hit exactly L > the safe prefix (the only
    node of that depth in a flushed cache) = the restore link."""
    lens = [x for x in (ev.get("confirmed") or {}).get(cyc["seed_rid"], []) if isinstance(x, int)]
    link = isinstance(cyc["probe_cached"], int) and cyc["probe_cached"] > t["safe_depth"] and cyc["probe_cached"] in lens
    return {"failed": cyc["seed_rid"] in ev.get("failed", set()), "stale_confirmed": bool(lens), "stale_lens": lens,
            "restore_link": link, "hot": cyc["probe_kl"] > t["tol"]}


def e17_verdict(t1s, ev, hook=True):
    """Per cycle, fail-closed (audit 4): BUG_EXPOSED only when ONE valid cycle holds the whole chain (failure -> confirmed
    stale donation -> restore link -> KL above the fixed tolerance). Evidence from different cycles never adds up."""
    if not t1s:
        return "NO_DATA"
    if not all(t["reference_valid"] for t in t1s):
        return "INCONCLUSIVE_UNSTABLE_OR_INVALID_REFERENCE"
    valid = [(t, c) for t in t1s for c in t["cycles"] if c["flush_valid"]]
    if not valid:
        return "INCONCLUSIVE_INVALID_COLD_CONTROL"
    if not hook:
        return "KL_ABOVE_TOL" if any(c["probe_kl"] > t["tol"] for t, c in valid) else "KL_WITHIN_TOL"
    ch = []
    for t, c in valid:
        c["chain"] = e17_cycle_chain(t, c, ev)
        ch.append(c["chain"])
    if any(x["failed"] and x["stale_confirmed"] and x["restore_link"] and x["hot"] for x in ch):
        return "BUG_EXPOSED"
    if any(x["failed"] and x["stale_confirmed"] and x["restore_link"] for x in ch):
        return "STALE_RESTORE_EXPOSED_NO_OUTPUT_FLIP"
    if any(x["hot"] for x in ch):
        return "KL_ABOVE_TOL_CHAIN_INCOMPLETE"
    if any(x["failed"] for x in ch) and not any(x["stale_confirmed"] for x in ch):
        return "TRIGGER_REACHED_NO_STALE"
    return "INCONCLUSIVE"


def e17_strip(t):
    return {k: v for k, v in t.items() if not k.startswith("_")}


def e17_safety(ctx, ref_t1):
    """Patch-safety cases on the 4-slot server (audits 1-3): normal EOS, abort mid-decode, max_new reached, 5 x reuse,
    cold after flush. After each: flush, the allocator's available slots vs the baseline (a leak shows as fewer), then
    warm + score P (a hit deeper than the safe prefix with KL above the tolerance = a stale restore)."""
    P, pref, ref = E17_P["P255"], [E17_P["W44"], E17_P["W42"]], ref_t1["_ref_lp"]
    tdir = Path(E17["launches"][E17_ARM["name"]].get("trace_dir") or E17_OUT)

    def avail():
        time.sleep(3.0 * max(E14_TS, 0.05) + 0.3)
        vals = []
        for p in tdir.glob("counters_*.json"):
            try:
                a = json.loads(p.read_text()).get("alloc") or {}
                if a.get("available") is not None:
                    vals.append(a["available"])
            except Exception:
                pass
        return max(vals) if vals else None
    e17_flush(ctx)
    base = avail()
    rows = []

    def after(case, extra=None):
        # the probe first (the case's own donation is still in the cache), then flush + the slot count
        e17_warm(ctx, pref)
        r = e17_gen(ctx, P, 1, score=True)
        kl = e17_kl(ref, r["lp"])
        f = e17_flush(ctx)
        a = avail()
        rows.append(dict({"case": case, "flush_ok": f.get("ok"), "flush_valid": f.get("valid"), "available": a, "available_base": base,
                          "leak": (a is not None and base is not None and a < base), "probe_cached": r["cached"],
                          "probe_kl": kl["kl"], "stale": bool(r["cached"] and r["cached"] > 128 and kl["kl"] > ref_t1["tol"])},
                         **(extra or {})))
    r = e17_gen(ctx, P, 64, ignore_eos=False)
    after("eos", {"finish": r["finish"]})
    rid = f"e17-abort-{_e17_rnd.randrange(10 ** 9)}"
    th = threading.Thread(target=lambda: e17_gen(ctx, P, 400, rid=rid), daemon=True)
    th.start()
    time.sleep(0.8)
    E14M.abort(ctx, [rid])
    th.join(60)
    after("abort")
    r = e17_gen(ctx, P, 48)
    after("max_new", {"n_out": r["n_out"]})
    for _ in range(5):
        e17_gen(ctx, P, 8)
    after("reuse5")
    e17_flush(ctx)
    r = e17_gen(ctx, P, 1, score=True)
    after("cold", {"cold_cached": r["cached"], "cold_kl": e17_kl(ref, r["lp"])["kl"]})
    return {"rows": rows, "leaks": sum(1 for x in rows if x["leak"]), "stale": sum(1 for x in rows if x["stale"]),
            "cold_cached_zero": rows[-1].get("cold_cached") == 0}


def e17_phase(name, fn, cap):
    """One measured block with its own deadline; recorded even on error; skipped when the budget is gone."""
    if e17_left() < 6 * 60:
        E17["skipped"][name] = "NOT_TESTED_BUDGET"
        e17_save()
        return None
    E14_STATE.enter("E17:" + name, deadline=HARD_STOP)
    t0 = time.perf_counter()
    try:
        res = fn(e17_ctx(cap))
    except BaseException as _e:
        res = {"error": f"{type(_e).__name__}: {_e}"[:500], "traceback": traceback.format_exc()[-1500:]}
    if isinstance(res, dict):
        res["phase_sec"] = round(time.perf_counter() - t0, 1)
        if E11_SERVER is not None:
            res["server_alive_after"] = E11_SERVER.alive()
    E17["arms"][name] = res
    E17["order"].append(name)
    e17_save()
    print(f"E17 {name}: {json.dumps(res, default=str)[:1200]}", flush=True)
    return res


def e17_small(interval, slots):
    return {"--max-mamba-cache-size": slots, "--max-running-requests": 1, "--mamba-track-interval": interval,
            "--chunked-prefill-size": 2048, "--cuda-graph-bs-decode": [1]}


def e17_small_launch(name, interval, spec, env=None, hook=True):
    """The small-slot server (upstream: 4 Mamba slots, 1 running). A failed launch steps up 4 -> 6 -> 8 slots."""
    for slots in (4, 6, 8):
        s = e17_small(interval, slots)
        ok, rec = e17_launch(f"{name}_m{slots}" if slots != 4 else name, set_=s,
                             drop=(("--cuda-graph-max-bs-decode",) + (() if spec else E17_TGT_DROP)), env=env, hook=hook)
        if ok or rec.get("skipped"):
            return ok, rec, slots
    return False, rec, slots


def e17_t1_arm(name, interval, prompts, seed_new, spec=False, env=None, hook=True, safety=False):
    ok, rec, slots = e17_small_launch(name, interval, spec, env=env, hook=hook)
    lname = rec.get("arm", name)
    if not ok:
        E17["arms"][name] = {"launch_ok": False, "why": rec.get("why") or rec.get("skipped"), "slots": slots}
        E17["order"].append(name)
        e17_save()
        return E17["arms"][name]
    if hook and not (rec.get("hook_selftest") or {}).get("ok"):
        E17["arms"][name] = {"launch_ok": True, "verdict": "HOOK_MISSING", "hook_selftest": rec.get("hook_selftest")}
        E17["order"].append(name)
        e17_save()
        return E17["arms"][name]

    def run(ctx):
        t1s = [e17_t1(ctx, p, [E17_P["W44"], E17_P["W42"]], seed_new) for p in prompts]
        res = {"launch": lname, "slots": slots, "interval": interval, "spec": spec, "env": dict(env or {}), "hook": hook,
               "t1": [e17_strip(t) for t in t1s]}
        if safety:
            res["safety"] = e17_safety(ctx, t1s[0])
        tr = e17_trace(lname) if hook else None
        res["trace"] = tr
        ev = e17_rid_evidence(tr, lname) if hook else {"failed": set(), "confirmed": {}}
        res["verdict"] = e17_verdict(t1s, ev, hook=hook)
        res["t1"] = [e17_strip(t) for t in t1s]          # again: e17_verdict added the per-cycle chain records
        res["rid_evidence"] = {"failed": sorted(ev["failed"])[:40], "confirmed": dict(list(ev["confirmed"].items())[:40])}
        if spec:
            res["accept"] = [l for l in (E11_SERVER.log_text() if E11_SERVER else "").splitlines() if "accept len" in l][-3:]
        return res
    return e17_phase(name, run, 14 * 60)


# ---------------------------------------------------------------------------------------------- Phase 0
def e17_pkg_dir():
    if E11_HARNESS:
        return Path(os.environ["E14_FAKE_SGLANG_PKG"])
    r = subprocess.run([E14_VENV["launch"][0], "-c", "import sglang, os; print(os.path.dirname(sglang.__file__))"],
                       capture_output=True, text=True, timeout=300, env=E14_VENV["env"])
    return Path(r.stdout.strip().splitlines()[-1])


def e17_phase0():
    p0 = E17["phase0"]
    try:
        pkg = e17_pkg_dir()
        p0["pkg_dir"] = str(pkg)
        sb = pkg / "srt" / "managers" / "schedule_batch.py"
        p0["schedule_batch_sha256"] = hashlib.sha256(sb.read_bytes()).hexdigest()
        p0["patch_43567_check"] = E14P.check(str(pkg), ("43567",))
        p0["patch_43567_state"] = p0["patch_43567_check"]["patches"]["43567"]["state"]
    except Exception as e:
        p0["pkg_error"] = repr(e)[:300]
    try:
        dc = _load_json(Path(SGL_DRAFT_PATH) / "config.json")
        tl = (dc.get("dflash_config") or {}).get("target_layer_ids")
        p0["dflash_target_layer_ids"], p0["dflash_num_target_layers"] = tl, dc.get("num_target_layers")
        p0["dflash_taps_ok"] = tl == [5, 19, 33, 47, 61] and dc.get("num_target_layers") == 64
        p0["dflash_block_size"] = (dc.get("dflash_config") or {}).get("block_size")
    except Exception as e:
        p0["dflash_taps_ok"], p0["dflash_error"] = False, repr(e)[:300]
    try:
        mc = _load_json(Path(SGL_MODEL_PATH) / "config.json")
        v = mc.get("vocab_size") or (mc.get("text_config") or {}).get("vocab_size")
        p0["vocab_size_config"] = v
        idx = Path(SGL_MODEL_PATH) / "model.safetensors.index.json"
        if idx.exists():
            wm = (_load_json(idx).get("weight_map") or {})
            p0["checkpoint_kv_scale_keys"] = sorted(k for k in wm if k.endswith(("k_scale", "v_scale")))[:8]
            p0["checkpoint_kv_scale_n"] = sum(1 for k in wm if k.endswith(("k_scale", "v_scale")))
    except Exception as e:
        p0["vocab_error"] = repr(e)[:300]
    lt = E11_SERVER.log_text() if E11_SERVER is not None else ""
    p0["gdn_backend_line"] = "Using hybrid linear attention backend for hybrid GDN models" in lt
    p0["kda_backend_line"] = "KDAAttnBackend" in lt
    p0["kv_scale_log_lines"] = [l[-200:] for l in lt.splitlines() if "scaling factor" in l.lower() or "k_scale" in l][:6]
    p0["production_engine_ready"] = bool(ENGINE_READY)
    p0["production_flags"] = [str(x) for x in (E11_SGL.get("launch_flags") or [])]
    e17_save()
    print("E17 phase 0:", json.dumps({k: v for k, v in p0.items() if k not in ("patch_43567_check", "production_flags")},
                                      default=str)[:1500], flush=True)
    return p0


# ---------------------------------------------------------------------------------------------- T3b / T3a / L6 / T2
def e17_arc_prompts(n=16):
    ask = "Find the transformation rule of this ARC-AGI-2 task and describe it, then give the test output grids."

    def rows(g):
        return "\n".join(" ".join(str(v) for v in r) for r in g)

    def text(t):
        s = "".join(f"\nExample {i + 1}\nInput:\n{rows(p['input'])}\nOutput:\n{rows(p['output'])}\n" for i, p in enumerate(t["train"]))
        return s + "".join(f"\nTest input {j + 1}:\n{rows(q['input'])}\n" for j, q in enumerate(t["test"]))
    try:
        allt = _load_json(ARC_ROOT / "arc-agi_evaluation_challenges.json")
    except Exception:
        allt = dict(CHALLENGES)
    keys = sorted(allt)
    msgs = [[{"role": "user", "content": ask + "\n" + text(allt[k])}] for k in keys[:n]]
    ids = [tokenizer.encode(tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                                          reasoning_effort=REASONING_EFFORT), add_special_tokens=False) for m in msgs]
    return ids, keys, text, allt


def e17_long_ids(target_tokens, skip=0):
    """One long user prompt from consecutive evaluation tasks (the T2 shared prompts, the T3a long prefix)."""
    _, keys, text, allt = e17_arc_prompts(1)          # keys = ALL evaluation task ids (sorted), not only the first n
    assert keys, "no ARC evaluation tasks loaded"
    parts, n = [], 0
    for off in range(len(keys)):                       # audit 4: wrap around the pool; never silently short
        parts.append(text(allt[keys[(skip + off) % len(keys)]]))
        n += len(tokenizer.encode(parts[-1], add_special_tokens=False))
        if n >= target_tokens:
            break
    m = [{"role": "user", "content": "Reference ARC tasks:\n" + "\n".join(parts) + "\nDescribe the rule of the last task."}]
    ids = tokenizer.encode(tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                                         reasoning_effort=REASONING_EFFORT), add_special_tokens=False)
    if len(ids) < target_tokens:
        raise RuntimeError(f"E17 long prompt too short: {len(ids)} < {target_tokens} (skip {skip}, {len(parts)} tasks)")
    return ids


def e17_text_bad(s):
    s = (s or "").strip()
    if not s:
        return True
    if re.search(r"(.)\1{31,}", s):
        return True
    return len(s) >= 200 and (len(set(s)) < 8 or sum(ch.isalnum() for ch in s) / len(s) < 0.10)


def e17_rows_summary(rows):
    ids = [r.get("ids") or [] for r in rows]
    return {"n": len(rows), "errors": sum(1 for r in rows if r.get("error")), "empty": sum(1 for x in ids if not x),
            "degenerate": sum(1 for x in ids if x and e17_text_bad(tokenizer.decode(x, skip_special_tokens=True))),
            "tokens": sum(len(x) for x in ids), "accept": [r.get("accept") for r in rows][:4]}


E17_T3 = {}


def e17_t3b_target(ctx):
    S4, S16 = E17_T3["ids"][:4], E17_T3["ids"]
    a = E14M.parity_pass(ctx, S4, max_new=E17_T3["n"], tag="t3b_tgt_serial_a")
    b = E14M.parity_pass(ctx, S4, max_new=E17_T3["n"], tag="t3b_tgt_serial_b")
    c = E14M.parity_pass(ctx, S16, max_new=E17_T3["n"], concurrent=True, tag="t3b_tgt_conc")
    E17_T3["tgt"] = (a, b, c)
    out = {"serial_aa": E14M.compare_passes(a, b, "target serial A/A"),
           "target_conc_vs_serial": E14M.compare_passes(a, c[:4], "target 16-conc vs serial"),
           "rows": {"serial_a": e17_rows_summary(a), "conc": e17_rows_summary(c)}}
    if e17_left() > 20 * 60:       # T3a fp16 half (radix is off on this server); the arrays stay out of the JSON
        E17_T3["fp16"] = e17_t3a_scores(ctx)
        out["t3a_fp16"] = {k: {"cached": v["cached"], "aa_kl": e17_kl(v["a"], v["b"])["kl"]} for k, v in E17_T3["fp16"].items()}
    return out


def e17_t3b_dflash(ctx):
    S4, S16 = E17_T3["ids"][:4], E17_T3["ids"]
    s = E14M.parity_pass(ctx, S4, max_new=E17_T3["n"], tag="t3b_dfl_serial")
    c = E14M.parity_pass(ctx, S16, max_new=E17_T3["n"], concurrent=True, tag="t3b_dfl_conc")
    out = {"rows": {"serial": e17_rows_summary(s), "conc": e17_rows_summary(c)}}
    if E17_T3.get("tgt"):
        ta, tb, tc = E17_T3["tgt"]
        out.update(dflash_serial_vs_target_serial=E14M.compare_passes(ta, s, "DFlash serial vs target serial"),
                   dflash_conc_vs_target_conc=E14M.compare_passes(tc, c, "DFlash 16-conc vs target 16-conc"),
                   dflash_conc_vs_dflash_serial=E14M.compare_passes(s, c[:4], "DFlash 16-conc vs DFlash serial"))
        out["target_serial_aa"] = E14M.compare_passes(ta, tb, "target serial A/A")
        out["verdict"] = e17_t3b_verdict(out, out["target_serial_aa"])
    out["accept_lines"] = [l[-160:] for l in (E11_SERVER.log_text() if E11_SERVER else "").splitlines() if "accept len" in l][-3:]
    return out


def e17_t3b_verdict(out, out_aa):
    """audit 4 (fail-closed): every comparison complete (expected > 0, every row compared, no error, no missing margin),
    the target A/A too; only then the large-margin flips decide."""
    comps = [out.get(k) or {} for k in ("dflash_serial_vs_target_serial", "dflash_conc_vs_target_conc", "dflash_conc_vs_dflash_serial")]
    comps.append(out_aa or {})
    ok = all((c.get("expected") or 0) > 0 and c.get("compared") == c.get("expected") and c.get("errors", 1) == 0
             and c.get("missing_margins", 1) == 0 for c in comps)
    if not ok:
        return "INCOMPLETE"
    aa = (out_aa or {}).get("large_margin_flips") or 0
    return "DFLASH_HIGH_MARGIN_DIVERGENCE" if max(c.get("large_margin_flips") or 0 for c in comps[:2]) > aa + 1 \
        else "NO_HIGH_MARGIN_DIVERGENCE"


def e17_t3a_scores(ctx):
    ids, out = E17_T3["long"], {}
    for cut in E17_T3["cuts"]:
        if cut > len(ids):
            continue
        a = e17_gen(ctx, ids[:cut], 1, score=True)
        b = e17_gen(ctx, ids[:cut], 1, score=True)
        out[str(cut)] = {"a": a["lp"], "b": b["lp"], "cached": [a["cached"], b["cached"]]}
    return out


def e17_t3a_compare(fp16, fp32):
    rows = {}
    for cut in fp16:
        if cut in fp32:
            rows[cut] = {"aa_fp16": e17_kl(fp16[cut]["a"], fp16[cut]["b"])["kl"], "aa_fp32": e17_kl(fp32[cut]["a"], fp32[cut]["b"])["kl"],
                         "fp16_vs_fp32": e17_kl(fp32[cut]["a"], fp16[cut]["a"])}
    return rows


def e17_l6(ctx):
    """Upstream Class 2 (TestMambaSpecTruncationKL): no_buffer + MTP, a short seed, probe = prompt + output + newline,
    warm vs cold. 0.5.21 has no #43387 guard: a large KL here means the harness can see a stale restore."""
    res = {}
    for case, q, n in (("count", "Count from 1 to 20, separated by commas.", 9), ("story", "Tell a short story about a cat and a fish.", 5)):
        ids = tokenizer.encode(tokenizer.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                                             add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)
        e17_flush(ctx)
        s = e17_gen(ctx, ids, n)
        probe = ids + s["out"] + tokenizer.encode("\n", add_special_tokens=False)
        w = e17_gen(ctx, probe, 1, score=True)
        e17_flush(ctx)
        c = e17_gen(ctx, probe, 1, score=True)
        e17_flush(ctx)
        c2 = e17_gen(ctx, probe, 1, score=True)
        res[case] = {"seed_out": s["n_out"], "seed_finish": s["finish"], "warm_cached": w["cached"], "cold_cached": [c["cached"], c2["cached"]],
                     "kl_cold_warm": e17_kl(c["lp"], w["lp"]), "kl_cold_cold": e17_kl(c["lp"], c2["lp"])["kl"]}
    hot = [k for k, v in res.items() if v["warm_cached"] and v["kl_cold_warm"]["kl"] > max(E17_TOL_FLOOR, 10 * v["kl_cold_cold"])]
    res["verdict"] = "STALE_RESTORE_DETECTED" if hot else "NOT_DETECTED"
    return res


def e17_t2(ctx, tag, minutes):
    """T2: 16 streams, 4 shared task prompts x 4 streams, production sampling; each turn appends its output and a
    feedback block (every turn is a prefix hit; > 45k restarts the chain from the shared prompt = the Poetiq pattern).
    At the end: per shared prompt a warm probe, then flush and two cold probes (full vocabulary)."""
    bases, fb = E17_T2["bases"], E17_T2["fb"]
    stop_at = time.perf_counter() + minutes * 60 * max(E14_TS, 0.02)
    stats = {"requests": 0, "prefix_hits": 0, "errors": 0, "tokens_out": 0, "restarts": 0, "cached_sum": 0, "prompt_sum": 0,
             "error_samples": []}
    lock = threading.Lock()
    stop_ev, active = threading.Event(), set()

    def stream(si):
        ids = list(bases[si % 4])
        n = 0
        while time.perf_counter() < stop_at - 120 * max(E14_TS, 0.02) and not stop_ev.is_set():
            rid = f"e17-t2-{tag}-{si}-{n}"
            n += 1
            with lock:
                active.add(rid)
            try:
                r = e17_gen(ctx, ids, E17_T2["max_new"], temperature=1.0, ignore_eos=False, rid=rid)
            except Exception as e:
                with lock:
                    stats["errors"] += 1
                    if len(stats["error_samples"]) < 6:
                        stats["error_samples"].append(repr(e)[:240])
                time.sleep(1.0)
                continue
            finally:
                with lock:
                    active.discard(rid)
            with lock:
                stats["requests"] += 1
                stats["prefix_hits"] += 1 if (r["cached"] or 0) > 0 else 0
                stats["tokens_out"] += r["n_out"]
                stats["cached_sum"] += r["cached"] or 0
                stats["prompt_sum"] += len(ids)
            ids = ids + r["out"] + fb
            if len(ids) > E17_T2["max_ctx"]:
                ids = list(bases[si % 4])
                with lock:
                    stats["restarts"] += 1
    ths = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(16)]
    for t in ths:
        t.start()
    E14M.join_all(ths, max(1.0, stop_at - time.perf_counter() + 60))
    # audit 4: the cleanup gate before any probe: stop new turns, abort what still runs, every thread joined, idle
    stop_ev.set()
    with lock:
        live = list(active)
    if live:
        E14M.abort(ctx, live, budget=8)
    alive = E14M.join_all(ths, 40)
    idle = e17_idle(ctx, wait=60)
    base_out = {"stats": stats, "alive_threads": alive, "server_idle": idle, "aborted": len(live),
                "base_lens": [len(b) for b in bases]}
    if alive or not idle or stats["errors"]:
        base_out["verdict"] = "INCOMPLETE"
        base_out["why"] = "cleanup gate: alive threads / server not idle / stream errors"
        base_out["trace"] = e17_trace(tag)
        return base_out
    probes = []
    for bi, b in enumerate(bases):
        w = e17_gen(ctx, b, 1, score=True)
        f1 = e17_flush(ctx)
        c1 = e17_gen(ctx, b, 1, score=True)
        f2 = e17_flush(ctx)
        c2 = e17_gen(ctx, b, 1, score=True)
        probes.append({"base": bi, "len": len(b), "warm_cached": w["cached"], "cold_cached": [c1["cached"], c2["cached"]],
                       "cold_valid": bool(f1["valid"] and f2["valid"] and c1["cached"] == 0 and c2["cached"] == 0),
                       "kl_cold_warm": e17_kl(c1["lp"], w["lp"]), "kl_cold_cold": e17_kl(c1["lp"], c2["lp"])["kl"]})
    tr = e17_trace(tag)
    k = tr.get("counters") or {}
    att, fail = k.get("optional_alloc_attempts", 0), k.get("optional_alloc_failed", 0)
    stale = k.get("stale_donation_confirmed", 0)
    if not all(p["cold_valid"] and p["kl_cold_cold"] <= E17_AA_MAX for p in probes):
        verdict = "INCOMPLETE_COLD_CONTROL"
    else:
        # T2 has no per-request provenance: a warm divergence is reported next to the counters, never as a confirmed chain
        div = any(p["warm_cached"] and p["kl_cold_warm"]["kl"] > E17_TOL_FLOOR for p in probes)
        verdict = ("NO_FAILURE_OBSERVED" if not fail else "FAILURE_OBSERVED_NO_STALE_RESTORE" if not stale else
                   "STALE_DONATIONS_AND_WARM_DIVERGENCE" if div else "STALE_RESTORE_EXPOSED_NO_OUTPUT_FLIP")
    return dict(base_out, probes=probes, trace=tr, verdict=verdict,
                fail_per_1000_boundaries=round(1000 * fail / att, 2) if att else None,
                stale_per_1000_prefix_hits=round(1000 * stale / stats["prefix_hits"], 2) if stats["prefix_hits"] else None)


def e17_revert_verified(pkg):
    """audit 4: kill the server, revert, then require revert ok + state not_applied + the Phase-0 sha of
    schedule_batch.py; otherwise raise (no "unpatched" arm may run on a patched build)."""
    try:
        e11_kill_server(wait=True)
    except Exception:
        pass
    res = E14P.revert(str(pkg))
    state = E14P.check(str(pkg), ("43567",))["patches"]["43567"]["state"]
    sha_now = hashlib.sha256((Path(pkg) / "srt" / "managers" / "schedule_batch.py").read_bytes()).hexdigest()
    rec = {"revert": res, "state": state, "sha_ok": sha_now == E17["phase0"].get("schedule_batch_sha256")}
    E17.setdefault("reverts", []).append(rec)
    if not res.get("ok") or state != "not_applied" or not rec["sha_ok"]:
        E17["patch_revert_failed"] = rec
        e17_save()
        raise RuntimeError(f"PATCH_REVERT_FAILED: ok={res.get('ok')} state={state} sha_ok={rec['sha_ok']}")
    return rec


# ---------------------------------------------------------------------------------------------- run
try:
    E17_ON = bool(P.get("e17")) and not RERUN and bool(E14_VENV.get("launch"))
    if E17_ON:
        e17_phase0()
        E17_V = int(os.environ.get("E17_VOCAB") or E17["phase0"].get("vocab_size_config") or 248320)
        E17_VOCAB_IDS = list(range(E17_V))
        E17["vocab_used"] = E17_V
        E17["phase0"]["gate"] = {"dflash_taps_ok": E17["phase0"].get("dflash_taps_ok"),
                                 "unpatched_at_start": E17["phase0"].get("patch_43567_state") == "not_applied"}
        if not E17["phase0"].get("dflash_taps_ok"):
            E17["notes_dflash"] = "DFlash taps not verified: the DFlash arms still run, their results are flagged"
        PKG17 = Path(E17["phase0"]["pkg_dir"])
        if E17["phase0"].get("patch_43567_state") != "not_applied":
            raise RuntimeError(f"E17: SGLang is not the unpatched original at start: {E17['phase0'].get('patch_43567_state')}")
        # user 2026-10-26 (after ~9 GPU-h of engine sweeps with no speed gain): the first E17 session is SHORT (<= 60
        # min): only the decisive #43567 pairs L1 / L7 (ordinary decode, + the safety cases) and L5 / L9 (DFlash W6).
        # The profile E17 keeps the full plan (L2-L4, L6, T3b, T2, T3a) for a follow-up only if the bug shows.
        E17_SHORT = bool(P.get("e17_short"))
        E17["mode"] = "SHORT" if E17_SHORT else "FULL"
        if not E17_SHORT:
            E17_T3["ids"], _, _, _ = e17_arc_prompts(16)
            E17_T3["n"] = int(os.environ.get("E17_T3B_TOKENS", 2048))
            E17_T3["long"] = e17_long_ids(int(os.environ.get("E17_T3A_TOKENS", 50000)))
            E17_T3["cuts"] = [c for c in (8192, 16384, 32768, 49152) if c < len(E17_T3["long"])] or [len(E17_T3["long"]) - 1]
            E17["t3a_long_len"], E17["t3a_cuts"] = len(E17_T3["long"]), list(E17_T3["cuts"])

        # ---- unpatched block
        e17_t1_arm("L1_t1a_128", 128, ["P255"], 32, safety=True)
        if not E17_SHORT:
            e17_t1_arm("L2_t1a_128_hook_off", 128, ["P255"], 32, hook=False)
            e17_t1_arm("L3_t1a_256", 256, ["P255", "P511"], 32)
            e17_t1_arm("L4_t1b_forced", 128, ["P255"], 32, env={"SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL": "1"})
        e17_t1_arm("L5_t1c_dflash", 64, ["P246", "P249", "P252"], 160, spec=True)
        if E17_SHORT:
            for _k in ("L2_t1a_128_hook_off", "L3_t1a_256", "L4_t1b_forced", "L6_class2_mtp", "L12_t3b_target", "L13_t3b_dflash",
                       "L8_t1a_256_patched", "T2", "L15_t3a_fp32"):
                E17["skipped"][_k] = "SHORT session (user 2026-10-26): follow-up only if the bug shows"
        elif e17_left() > 70 * 60:
            ok6, _ = e17_launch("L6_class2_mtp", set_={
                "--speculative-algorithm": "EAGLE", "--speculative-draft-model-path": str(SGL_MODEL_PATH),
                "--speculative-num-steps": 3, "--speculative-eagle-topk": 1, "--speculative-num-draft-tokens": 4,
                "--mamba-radix-cache-strategy": "no_buffer", "--page-size": 1, "--disable-overlap-schedule": None,
                "--disable-cuda-graph": None, "--max-running-requests": 8, "--max-mamba-cache-size": 128},
                drop=("--cuda-graph-bs-decode", "--cuda-graph-max-bs-decode", "--disable-prefill-cuda-graph",
                      "--min-free-slots-delay"), hook=False)
            if ok6:
                e17_phase("L6_class2_mtp", e17_l6, 8 * 60)
        else:
            E17["skipped"]["L6_class2_mtp"] = "NOT_TESTED_BUDGET"
        if not E17_SHORT:
            okt, _ = e17_launch("L12_t3b_target", set_={"--disable-radix-cache": None}, drop=E17_TGT_DROP, hook=False)
            if okt:
                e17_phase("L12_t3b_target", e17_t3b_target, 22 * 60)
            okd, _ = e17_launch("L13_t3b_dflash", set_={"--disable-radix-cache": None}, hook=False)
            if okd:
                e17_phase("L13_t3b_dflash", e17_t3b_dflash, 14 * 60)

        # ---- #43567 patched block
        E17["patch_apply"] = E14P.apply(str(PKG17), ("43567",))
        E17["patch_check_after_apply"] = E14P.check(str(PKG17), ("43567",))["patches"]["43567"]["state"]
        e17_save()
        print("E17 patch #43567:", E17["patch_apply"].get("ok"), E17["patch_check_after_apply"], flush=True)
        if E17["patch_check_after_apply"] == "applied":
            e17_t1_arm("L7_t1a_128_patched", 128, ["P255"], 32, safety=True)
            if not E17_SHORT:
                e17_t1_arm("L8_t1a_256_patched", 256, ["P255", "P511"], 32)
            e17_t1_arm("L9_t1c_dflash_patched", 64, ["P246", "P249", "P252"], 160, spec=True)
            for _a in ("L7_t1a_128_patched", "L8_t1a_256_patched", "L9_t1c_dflash_patched"):
                _ln = (E17["arms"].get(_a) or {}).get("launch") or _a
                try:
                    _lt = (OUT / f"sglang_server_e17_{_ln}.log").read_text(encoding="utf-8", errors="replace")
                except Exception:
                    _lt = ""
                if "Can not alloc mamba cache" in _lt and isinstance(E17["arms"].get(_a), dict):
                    E17["arms"][_a]["crash_path_43559"] = True       # #43559's assertion: a crash path, not a verdict

        # ---- decision point A
        _v = lambda a: (E17["arms"].get(a) or {}).get("verdict")
        _un = [a for a in ("L1_t1a_128", "L3_t1a_256", "L5_t1c_dflash") if _v(a) == "BUG_EXPOSED"]
        _pa = {"L1_t1a_128": "L7_t1a_128_patched", "L3_t1a_256": "L8_t1a_256_patched", "L5_t1c_dflash": "L9_t1c_dflash_patched"}
        _clean = [a for a in _un if _v(_pa[a]) == "TRIGGER_REACHED_NO_STALE" and
                  ((E17["arms"].get(_pa[a]) or {}).get("trace") or {}).get("counters", {}).get("stale_donation_confirmed", 0) == 0]
        E17["decision_A"] = {"unpatched_bug_arms": _un, "patched_clean": _clean,
                             "verdict": "BUG_EXPOSED_43567" if _clean else ("BUG_SEEN_PATCH_NOT_CLEAN" if _un else "NOT_EXPOSED_OR_INCONCLUSIVE")}
        e17_save()
        print("E17 decision A:", E17["decision_A"], flush=True)

        if E17["decision_A"]["verdict"] == "BUG_EXPOSED_43567" and not E17_SHORT and e17_left() > 40 * 60:
            E17_T2 = {"bases": [e17_long_ids(int(os.environ.get("E17_T2_TOKENS", 30000)), skip=12 * i) for i in range(4)],
                      "fb": tokenizer.encode("\nFeedback: the program failed on example 2; the output grid differs in 3 "
                                             "cells. Revise the rule and try again.\n" * 30, add_special_tokens=False),
                      "max_new": int(os.environ.get("E17_T2_MAX_NEW", 3000)), "max_ctx": 45000}
            E17["t2_base_lens"] = [len(b) for b in E17_T2["bases"]]
            assert len({tuple(b) for b in E17_T2["bases"]}) == 4, "T2 bases are not distinct"
            if E14P.check(str(PKG17), ("43567",))["patches"]["43567"]["state"] == "applied":
                ok11, _ = e17_launch("L11_t2_patched", hook=True, verbose=False)
                if ok11:
                    e17_phase("L11_t2_patched", lambda c: e17_t2(c, "L11_t2_patched", 12), 16 * 60)
            E17["patch_revert"] = e17_revert_verified(PKG17)      # raises: no unpatched arm on a patched build
            ok10, _ = e17_launch("L10_t2_unpatched", hook=True, verbose=False)
            if ok10:
                e17_phase("L10_t2_unpatched", lambda c: e17_t2(c, "L10_t2_unpatched", 12), 16 * 60)
        elif not E17_SHORT:
            E17["skipped"]["T2"] = "decision A: " + E17["decision_A"]["verdict"] if E17["decision_A"]["verdict"] != "BUG_EXPOSED_43567" else "NOT_TESTED_BUDGET"
        if "patch_revert" not in E17:
            E17["patch_revert"] = e17_revert_verified(PKG17)
        E17["patch_state_end"] = E14P.check(str(PKG17), ("43567",))["patches"]["43567"]["state"]

        # ---- T3a fp32 (only with time left)
        if E17_SHORT:
            pass
        elif E17_T3.get("fp16") and e17_left() > 22 * 60:
            ok15, _ = e17_launch("L15_t3a_fp32", set_={"--disable-radix-cache": None, "--mamba-ssm-dtype": "float32"},
                                 drop=E17_TGT_DROP, hook=False)
            if ok15:
                def _t3a(ctx):
                    f32 = e17_t3a_scores(ctx)
                    return {"rows": e17_t3a_compare(E17_T3["fp16"], f32)}
                e17_phase("L15_t3a_fp32", _t3a, 12 * 60)
        else:
            E17["skipped"]["L15_t3a_fp32"] = "NOT_TESTED_BUDGET or no fp16 scores"
        E14_STATE.enter("E17_DONE")
        e17_save()
    else:
        E17["skipped"]["all"] = "rerun" if RERUN else ("no SGLang venv" if not E14_VENV.get("launch") else "E17 profile off")
        print("E17 skipped:", E17["skipped"]["all"], flush=True)
except BaseException:
    E17["error"] = traceback.format_exc()[-3000:]
    print("E17 cell error:", E17["error"][-1500:], flush=True)
    try:
        if "PKG17" in globals() and E14P.check(str(PKG17), ("43567",))["patches"]["43567"]["state"] != "not_applied":
            E17["patch_revert_after_error"] = e17_revert_verified(PKG17)
    except Exception:
        pass
    try:
        e17_save()
    except Exception:
        pass
