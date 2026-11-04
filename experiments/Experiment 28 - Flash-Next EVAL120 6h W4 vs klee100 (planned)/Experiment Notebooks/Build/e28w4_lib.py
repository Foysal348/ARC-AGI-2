"""E28 W4 library (2026-11-01): exec'd into the notebook namespace at the end of the engine cell (the nb6q_lib.py pattern:
packed verbatim, sha256 checked; a load failure is logged and the run goes on without these RECORD-ONLY functions).

  e28_i8_parse     the NB4T I8 read-back on one server log (nb4t_lib.log_config's regexes, verbatim): the non-default args'
                   speculative_config (draft_lm_head_quant), every 'Draft lm_head: INT8 weight-only copy ...' build line
                   (mode, shard, MiB, worker), language_model_only, max_num_seqs, max_model_len, kv_cache_memory_bytes
  e28_i8_gate      the boot gate: I8 must read back ENGAGED (else I8_READBACK_UNVERIFIED -> the same pin with the BF16 head)
  e28_readback     after the warm-up and at the end of every booted try: the I8 label (ENGAGED / SPEC_ONLY / NOT_ENGAGED /
                   CONTRADICTED), the launch knobs the server reports (C12 seqs 12, max-model-len 131072, text-only), and
                   this boot's pool / F / workers (NB5_CAPACITY); recorded in the try record, NB5_REPORT["e28"] and the log.
                   Never raises, never stops a run (commit or rerun).
  e28_summary      the NB5_REPORT["e28"] block printed by CELL 9.
  e28_sched_report CELL 9: the scheduler's own record (SCHED_LAST["e28"]: phases, rows, B2 rows, slots) -> e28_sched_report.json
Stdlib only.
"""
import ast as _e28_ast
import json as _e28_json
import re as _e28_re
import time as _e28_time
from pathlib import Path as _E28Path

E28_LIB_VERSION = "e28w4_lib 1.0 (2026-11-01: I8 read-back, C12 / 131072 / text-only launch read-back, boot pool / F)"
# = nb4t_lib.py _WTAG / _HEAD_RX / _SPEC_RX (the NB4T I8 read-back that ran on GPU, 675b1b9d), verbatim
_E28_WTAG = _e28_re.compile(r"\(Worker_PP(\d+)_TP(\d+)")
_E28_HEAD_RX = _e28_re.compile(r"Draft lm_head: (INT8|INT4) weight-only copy of the (\d+)x(\d+) shard \(([\d.]+) MiB")
_E28_SPEC_RX = _e28_re.compile(r"'speculative_config': (\{[^{}]*\})")


def _e28_literal(s):
    try:
        v = _e28_ast.literal_eval(s)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def e28_i8_parse(log_text, want="int8"):
    """NB4T's config read-back for the draft head (nb4t_lib.log_config + config_readback's two draft-head checks) on one
    server log: ENGAGED = the non-default args' speculative_config carries draft_lm_head_quant == want AND every
    'Draft lm_head:' build line says want.upper() (NB4T I8: two lines, the PP1 workers, 124160x2560, 307.9 MiB);
    SPEC_ONLY = the argument is there, no build line (yet); CONTRADICTED = a line or the spec shows another mode;
    NOT_ENGAGED = neither."""
    t = log_text or ""
    lines = t.splitlines()
    na = next((l for l in lines if "non-default args:" in l), "")
    out = {"non_default_found": bool(na), "want": want}
    m = _E28_SPEC_RX.search(na)
    sp = _e28_literal(m.group(1)) if m else None
    out["speculative_config"] = sp
    out["spec_draft_head"] = sp.get("draft_lm_head_quant") if isinstance(sp, dict) else None
    out["spec_k"] = sp.get("num_speculative_tokens") if isinstance(sp, dict) else None
    out["spec_sample"] = sp.get("draft_sample_method") if isinstance(sp, dict) else None
    for key, rx in (("max_num_seqs", r"'max_num_seqs': (\d+)"), ("max_model_len", r"'max_model_len': (\d+)"),
                    ("kv_cache_memory_bytes", r"'kv_cache_memory_bytes': (\d+)"), ("max_num_batched_tokens", r"'max_num_batched_tokens': (\d+)")):
        mm = _e28_re.search(rx, na)
        out[key] = int(mm.group(1)) if mm else None
    mm = _e28_re.search(r"'language_model_only': (True|False)", na)
    out["language_model_only"] = (mm.group(1) == "True") if mm else None
    heads = []
    for l in lines:
        h = _E28_HEAD_RX.search(l)
        if h:
            w = _E28_WTAG.search(l)
            heads.append({"mode": h.group(1), "n": int(h.group(2)), "k": int(h.group(3)), "mib": float(h.group(4)),
                          "worker": f"PP{w.group(1)}_TP{w.group(2)}" if w else None})
    out["draft_head_lines"] = heads[:8]
    out["draft_head_line_count"] = len(heads)
    out["draft_head_any"] = any("Draft lm_head:" in l for l in lines)
    modes = sorted({h["mode"] for h in heads})
    out["modes"] = modes
    spec_ok = out["spec_draft_head"] == want
    built_ok = modes == [str(want).upper()]
    if (out["spec_draft_head"] not in (None, want)) or (modes and not built_ok):
        out["label"] = "CONTRADICTED"
    elif spec_ok and built_ok:
        out["label"] = "ENGAGED"
    elif spec_ok:
        out["label"] = "SPEC_ONLY"
    else:
        out["label"] = "NOT_ENGAGED"
    return out


def e28_i8_gate(log_text, want="int8"):
    """The I8 boot gate (lead 2026-11-01, after the K100 audits): with DRAFT_HEAD=int8 the server log must show NB4T's read-back
    ENGAGED (spec draft_lm_head_quant == int8 AND every 'Draft lm_head:' build line INT8). Anything else (SPEC_ONLY,
    CONTRADICTED, NOT_ENGAGED = BF16 evidence) -> (False, I8_READBACK_UNVERIFIED): the caller stops this server and retries the
    SAME pin with the BF16 head. Returns (ok, record)."""
    rb = e28_i8_parse(log_text, want)
    rec = {"label": rb["label"], "spec_draft_head": rb.get("spec_draft_head"), "draft_head_line_count": rb.get("draft_head_line_count"),
           "modes": rb.get("modes"), "verdict": "ENGAGED" if rb["label"] == "ENGAGED" else "I8_READBACK_UNVERIFIED"}
    return rb["label"] == "ENGAGED", rec


def _e28_block():
    rep = globals().get("NB5_REPORT")
    if not isinstance(rep, dict):
        return None
    return rep.setdefault("e28", {"lib": E28_LIB_VERSION, "readbacks": [], "capacity": []})


def e28_readback(trec, s, stage):
    """Record only: the I8 / C12 / 131072 / text-only read-back of this launch + the boot's pool / F / workers."""
    try:
        P_ = globals().get("P") or {}
        want_seqs = ((trec or {}).get("max_num_seqs_launch") if isinstance(trec, dict) and trec.get("max_num_seqs_launch")
                     is not None else P_.get("seqs_fixed"))     # this try's launch seqs (12 on the C12 rung, NB6Q's table below)
        want_len = P_.get("server_max_model_len") or 131072   # NB6Q's server max-model-len (user 2026-11-01)
        want_head = (globals().get("E28_BUILD_ENV") or {}).get("DRAFT_HEAD")   # the CURRENT head (BF16 after an I8 fallback)
        try:
            txt = s.log() if s is not None else ""
        except Exception as _e:
            txt = ""
        rb = e28_i8_parse(txt, want_head or "int8")
        cap = globals().get("NB5_CAPACITY") or {}
        caps = cap.get("caps") or {}
        rec = {"stage": stage, "t_sec": round(_e28_time.perf_counter() - globals().get("NOTEBOOK_T0", 0.0)),
               "launch": getattr(s, "name", None), "i8_label": rb["label"] if want_head else "NOT_REQUESTED",
               "i8": {k: rb.get(k) for k in ("spec_draft_head", "spec_k", "spec_sample", "draft_head_line_count", "modes",
                                             "draft_head_lines")},
               "launch_knobs": {"max_num_seqs": rb.get("max_num_seqs"), "want_max_num_seqs": want_seqs,
                                "max_model_len": rb.get("max_model_len"), "want_max_model_len": want_len,
                                "language_model_only": rb.get("language_model_only"),
                                "kv_cache_memory_bytes": rb.get("kv_cache_memory_bytes"),
                                "max_num_batched_tokens": rb.get("max_num_batched_tokens")},
               "rung": dict((trec or {}).get("e28_rung") or {}) if isinstance(trec, dict) else {},
               "capacity": {"pool_blocks": cap.get("pool_blocks"), "pool_source": cap.get("pool_source"),
                            "e28_workers": cap.get("e28_workers"),
                            "f_eff": cap.get("f_eff"), "f_log": cap.get("f_log"), "f_source": cap.get("f_source"),
                            "capacity_blocks": cap.get("capacity_blocks"), "workers": cap.get("workers"),
                            "r2_workers": cap.get("r2_workers"), "kv_tokens": caps.get("kv_tokens"),
                            "conc_tokens": caps.get("conc_tokens"), "conc_x": caps.get("conc_x"),
                            "block_size": cap.get("block_size"), "per_attempt": cap.get("per_attempt")}}
        rec["knobs_ok"] = {"c12_seqs": (want_seqs is None) or rb.get("max_num_seqs") == int(want_seqs),
                           "max_model_len": (want_len is None) or rb.get("max_model_len") == int(want_len),
                           "language_model_only": "--language-model-only" not in (P_.get("e28_extra_args") or [])
                           or rb.get("language_model_only") is True}
        if trec is not None:
            trec.setdefault("e28_readback", {})[stage] = rec
        blk = _e28_block()
        if blk is not None:
            blk["readbacks"].append({k: rec[k] for k in ("stage", "t_sec", "launch", "i8_label", "launch_knobs", "knobs_ok")})
            if stage == "after_warmup":
                blk["capacity"].append(dict(rec["capacity"], launch=rec["launch"], kv_bytes=(trec or {}).get("kv_bytes"),
                                            pin_gib=(trec or {}).get("pin_gib")))
        loud = (want_head and rec["i8_label"] != "ENGAGED") or not all(rec["knobs_ok"].values())
        print(("!!!!!!!!!! " if loud else "") + "E28 W4 read-back [" + str(stage) + "] " + _e28_json.dumps(
            {"launch": rec["launch"], "i8": rec["i8_label"], "knobs": rec["launch_knobs"], "knobs_ok": rec["knobs_ok"],
             "pool": rec["capacity"]["pool_blocks"], "F": rec["capacity"]["f_eff"], "f_log": rec["capacity"]["f_log"],
             "workers": rec["capacity"]["workers"], "kv_tokens": rec["capacity"]["kv_tokens"]}, default=str), flush=True)
        return rec
    except Exception as _e:
        try:
            if trec is not None:
                trec.setdefault("e28_readback", {})[stage] = {"error": repr(_e)[:400]}
        except Exception:
            pass
        print("E28 W4 read-back error (record only):", repr(_e)[:300], flush=True)
        return None


def e28_summary():
    """CELL 9: the E28 block of nb5_report.json, one line."""
    blk = _e28_block() or {}
    last = (blk.get("readbacks") or [{}])[-1]
    capl = (blk.get("capacity") or [{}])[-1]
    return {"lib": blk.get("lib"), "i8_label_last": last.get("i8_label"), "knobs_last": last.get("launch_knobs"),
            "knobs_ok_last": last.get("knobs_ok"), "pool_blocks": capl.get("pool_blocks"), "f_eff": capl.get("f_eff"),
            "workers": capl.get("workers"), "kv_bytes": capl.get("kv_bytes"), "pin_gib": capl.get("pin_gib"),
            "readbacks": len(blk.get("readbacks") or []),
            "sched": {k: (globals().get("E28_SCHED_REC") or {}).get(k) for k in ("state", "requested", "applied", "verify", "env",
                                                                                "error", "restored")}}


def _d(x):                                       # = e28k100_lib.py
    return x if isinstance(x, dict) else {}


def _l(x):                                       # = e28k100_lib.py
    return x if isinstance(x, list) else []


def preempt_events(sampler_rows):
    """[(t_prev, t, delta)] of the sampler's cumulative vllm:num_preemptions_total (a counter reset = a relaunch: ignored).
    (= e28k100_lib.py preempt_events, verbatim)"""
    ev, last = [], None
    for s in sorted([r for r in _l(sampler_rows) if isinstance(r, dict) and isinstance(r.get("preempt"), (int, float))
                     and isinstance(r.get("t"), (int, float))], key=lambda r: r["t"]):
        if last is not None and s["preempt"] > last["preempt"]:
            ev.append((float(last["t"]), float(s["t"]), int(s["preempt"] - last["preempt"])))
        last = s
    return ev


def flag_preemptions(rows, records, events):
    """Per attempt row: preempt_while_running = the preemptions counted in sampler intervals that overlap [t0, t1] of the
    attempt (an upper bound of its own: vLLM does not name the preempted request); None without sampler data.
    (= e28k100_lib.py flag_preemptions, verbatim)"""
    tt = {}
    for t_, rs in _d(records).items():
        for r in _l(rs):
            if isinstance(r, dict):
                tt[(t_, r.get("branch"))] = (r.get("t0"), r.get("t1"))
    for row in rows:
        a, b = tt.get((row["task"], row["branch"]), (None, None))
        if events is None or not a or not b:
            row["preempt_while_running"] = None
            continue
        row["preempt_while_running"] = sum(d for t0_, t1_, d in events if t1_ >= float(a) and t0_ <= float(b))
    return rows


def e28_sched_report(path=None, sampler_path=None):
    """CELL 9 (lead 2026-11-01): the A / B / C scheduler's own record (E12_RUN["rev6_scheduler"]["e28"] = arc2_duck.SCHED_LAST
    ["e28"]) instead of CELL 7's E15B metrics, which count b2 as an 'attempt 3+' with the b1 budget. Per phase: started / done
    / queued / left; the A classification and the B classes; per-attempt rows grouped by phase (count, tokens, end reasons,
    the B2 rows in full); the answer-slot sources; config, ignored args, warnings. The full block (every row, the slot table)
    goes to `path` (JSON); the return value is the compact summary. Never raises."""
    try:
        run = globals().get("E12_RUN") or {}
        e = (run.get("rev6_scheduler") or {}).get("e28")
        if not isinstance(e, dict):
            return {"state": "NO_E28_RECORD", "sched": (globals().get("E28_SCHED_REC") or {}).get("state")}
        rows = [dict(r) for r in (e.get("attempts") or []) if isinstance(r, dict)]
        ev = None                                 # preemptions (slots: the engine schedules KV) from the phase's 15-s sampler
        if sampler_path is not None and _E28Path(str(sampler_path)).is_file():
            ev = preempt_events([_e28_json.loads(l) for l in _E28Path(str(sampler_path)).read_text(encoding="utf-8").splitlines()
                                 if l.strip()])
        flag_preemptions(rows, globals().get("E12_RECORDS") or {}, ev)
        by = {}
        for r in rows:
            b = by.setdefault(str(r.get("phase")), {"attempts": 0, "tokens": 0, "kinds": {}, "end_reasons": {}})
            b["attempts"] += 1
            b["tokens"] += int(r.get("tokens") or 0)
            b["kinds"][str(r.get("kind"))] = b["kinds"].get(str(r.get("kind")), 0) + 1
            b["end_reasons"][str(r.get("end_reason"))] = b["end_reasons"].get(str(r.get("end_reason")), 0) + 1
        src = {"attempt_1": {}, "attempt_2": {}}
        for t, sl in (e.get("slots") or {}).items() if isinstance(e.get("slots"), dict) else []:
            for x in (sl if isinstance(sl, list) else []):
                if isinstance(x, dict):
                    for k in ("attempt_1", "attempt_2"):
                        v = str(x.get(k))
                        src[k][v] = src[k].get(v, 0) + 1
        a_cls = {}
        for v in (e.get("a_class") or {}).values() if isinstance(e.get("a_class"), dict) else []:
            k_ = str(v.get("kind") if isinstance(v, dict) else v)
            a_cls[k_] = a_cls.get(k_, 0) + 1
        out = {"state": "OK", "config": e.get("config"), "started": e.get("started"), "done": e.get("done"),
               "queued": e.get("queued"), "left": e.get("left"), "a_class_counts": a_cls, "b_class": e.get("b_class"),
               "phases": by, "b2_rows": [r for r in rows if r.get("phase") == "B2"][:60], "slot_sources": src,
               "warnings": e.get("warnings"), "ignored_args": e.get("ignored_args"), "setup_failed": e.get("setup_failed"),
               "order_holds": e.get("order_holds"), "rows": len(rows),
               "idle_worker_seconds": e.get("idle_worker_seconds"), "idle_sec_by_reason": e.get("idle_sec_by_reason"),
               "backfill_starts": e.get("backfill_starts"),
               "preemptions": {"intervals": len(ev) if ev is not None else None,
                               "sampler_total": sum(d for _, _, d in ev) if ev is not None else None,
                               "server_counter_phase": (((globals().get("NB5_REPORT") or {}).get("phases") or {}).get("EVAL", {})
                                                        .get("counters") or {}).get("vllm:num_preemptions_total"),
                               "attempts_with_preemption": sum(1 for r in rows if r.get("preempt_while_running")),
                               "note": "a preemption = a recompute (speed only), not a failure; per attempt = an upper bound"}}
        if path is not None:
            try:
                p = _E28Path(str(path))
                tmp = p.with_name(p.name + ".tmp")
                tmp.write_text(_e28_json.dumps({"summary": out, "e28": e, "attempts_with_preemption_flags": rows}, default=str),
                               encoding="utf-8")
                tmp.replace(p)
                out["file"] = str(p)
            except Exception as _e:
                out["file_error"] = repr(_e)[:200]
        return out
    except Exception as _e:
        return {"state": "ERROR", "error": repr(_e)[:400]}
