"""Extract numeric-only records from the K100Q run (no response/prompt text is ever printed or saved).

Outputs (CSV/JSON) into OUT dir:
  requests.csv   one row per client request (nb5_requests.jsonl + branch_result.requests meta, matched by order)
  branches.csv   one row per attempt (branch)
  sampler.csv    nb5_sampler.jsonl (15 s client+server gauges)
  srvstats.csv   server log 10 s metric lines (loggers.py + metrics.py SpecDecoding)
  sched_gauges.csv  scheduler gauges (30 s)
Usage: python -I extract.py <run_q27x_dir> <out_dir>
"""
import csv, json, os, re, sys, datetime as dt
from pathlib import Path

RUN = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
PH = RUN / "phase_Q_K100_FP8_K3"
DUCK = PH / "duck"

TEXT_KEYS = {"reasoning", "content", "tool_calls", "text", "summary", "test_outputs", "arguments"}


def w(name, rows, fields):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


req_rows, br_rows = [], []
for d in sorted(DUCK.iterdir()):
    if not d.is_dir():
        continue
    task, b = d.name.rsplit("_b", 1)
    b = int(b)
    nb5 = []
    p = d / "nb5_requests.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                x = json.loads(line)
            except Exception:
                continue
            g = ((x.get("info") or {}).get("guard") or {})
            nb5.append({"t0": x.get("t0"), "t1": x.get("t1"), "kind": x.get("kind"), "prompt": x.get("prompt"),
                        "cached": x.get("cached"), "completion": x.get("completion"), "finish": x.get("finish"),
                        "error": (type(x.get("error")).__name__ if x.get("error") is not None else ""),
                        "error_len": len(str(x.get("error"))) if x.get("error") is not None else 0,
                        "g_prompt": g.get("prompt"), "g_requested": g.get("requested"), "g_cap": g.get("cap"),
                        "g_clamped": g.get("clamped"), "g_how": g.get("how"),
                        "n_tool_calls": len(x.get("tool_calls") or []),
                        "reasoning_chars": len(x.get("reasoning") or ""),
                        "content_chars": len(x.get("content") or ""),
                        "toolarg_chars": sum(len(str((tc or {}).get("arguments") or "")) for tc in (x.get("tool_calls") or [])),
                        "gen_after": x.get("generated_tokens_after")})
    br = json.loads((d / "branch_result.json").read_text(encoding="utf-8", errors="replace"))
    meta = br.get("requests") or []
    subs = br.get("submissions") or []
    chk = br.get("checked") or []
    verify = [s.get("verify_sec") for s in subs + chk if isinstance(s.get("verify_sec"), (int, float))]
    for i, r in enumerate(nb5):
        m = meta[i] if i < len(meta) else {}
        r.update(task=task, branch=b, i=i, n_nb5=len(nb5), n_meta=len(meta),
                 m_t=m.get("t"), m_sec=m.get("sec"), m_idle=m.get("idle_sec"), m_chars=m.get("chars"),
                 m_msgs=m.get("msgs"), m_prefix_ok=m.get("prefix_ok"), m_prefix_msgs=m.get("prefix_msgs"),
                 m_prev_msgs=m.get("prev_msgs"), m_ser_prefix=m.get("serialized_char_prefix"),
                 m_prev_chars=m.get("prev_chars"), m_trimmed=m.get("trimmed"), m_running=m.get("running"),
                 m_queue=m.get("queue"), m_kv=m.get("kv"), m_prompt=m.get("prompt"), m_cached=m.get("cached"),
                 m_completion=m.get("completion"), m_finish=m.get("finish"))
        req_rows.append(r)
    runs = br.get("runs") or []
    ev6 = br.get("rev6_events") or {}
    br_rows.append({"task": task, "branch": b, "t0": br.get("t0"), "t1": br.get("t1"), "status": br.get("status"),
                    "turns": br.get("turns"), "n_runs": len(runs),
                    "run_sec": sum((r.get("sec") or 0) for r in runs), "run_tokens": sum((r.get("tokens") or 0) for r in runs),
                    "generated_tokens": br.get("generated_tokens"), "total_tokens": br.get("total_tokens"),
                    "accepted": br.get("accepted"), "n_requests_meta": len(meta), "n_requests_nb5": len(nb5),
                    "n_subs": len(subs), "n_checked": len(chk), "verify_sec_sum": round(sum(verify), 3),
                    "verify_n": len(verify),
                    "first_sub_t": min([s.get("t") for s in subs if s.get("t")] or [None]) if subs else None,
                    "tool_calls_ev": ev6.get("tool_calls"), "requests_started": ev6.get("requests_started"),
                    "nb5_verdict": br.get("nb5_verdict"), "failures": br.get("failures"),
                    "note_len": len(br.get("note") or "")})

reqf = ["task", "branch", "i", "n_nb5", "n_meta", "t0", "t1", "kind", "prompt", "cached", "completion", "finish", "error",
        "error_len", "g_prompt", "g_requested", "g_cap", "g_clamped", "g_how", "n_tool_calls", "reasoning_chars",
        "content_chars", "toolarg_chars", "gen_after", "m_t", "m_sec", "m_idle", "m_chars", "m_msgs", "m_prefix_ok",
        "m_prefix_msgs", "m_prev_msgs", "m_ser_prefix", "m_prev_chars", "m_trimmed", "m_running", "m_queue", "m_kv",
        "m_prompt", "m_cached", "m_completion", "m_finish"]
w("requests.csv", req_rows, reqf)
w("branches.csv", br_rows, list(br_rows[0].keys()))

# sampler
srows = [json.loads(l) for l in (PH / "nb5_sampler.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
sf = ["t", "running", "waiting", "kv_server", "gen", "preempt", "inflight", "workers", "duck_waiting", "duck_held",
      "free_blocks", "work_left", "free_not_admitted", "reserved", "error"]
w("sampler.csv", srows, sf)

# scheduler gauges + events
rep = json.loads((PH / "e12_run_report.json").read_text(encoding="utf-8"))
er = rep["e12_run"]
sch = er["rev6_scheduler"]
gf = sch["gauge_fields"]
w("sched_gauges.csv", [dict(zip(gf, g)) for g in sch["gauges"]], gf)
json.dump({"events": [{k: v for k, v in e.items() if k in ("t", "kind", "kv", "running", "task", "branch")} for e in sch["events"]],
           "durations": sch["durations"], "kv_holds": sch["kv_holds"], "kv_hold_sec": sch["kv_hold_sec"],
           "started_sec": er.get("started_sec"), "finished_sec": er.get("finished_sec"),
           "window_shift": er.get("window_shift"), "solve_end_epoch": er.get("solve_end_epoch"),
           "planned_solve_end_sec": er.get("planned_solve_end_sec")},
          open(OUT / "sched_meta.json", "w"), indent=1)

# server log metric lines
LOG = RUN / "engine" / "logs" / "server_merlin_fp8_mtp_pin508.log"
pat = re.compile(r"INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[loggers\.py:\d+\] Engine 000: Avg prompt throughput: ([\d.]+) tokens/s, "
                 r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: "
                 r"([\d.]+)%, Prefix cache hit rate: ([\d.]+)%")
spat = re.compile(r"INFO (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d) \[metrics\.py:\d+\] SpecDecoding metrics: Mean acceptance length: ([\d.]+),"
                  r".*?Accepted: (\d+) tokens, Drafted: (\d+) tokens")
st, sp = [], {}
for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
    m = pat.search(line)
    if m:
        mo, da, hh, mi, ss = map(int, m.groups()[:5])
        t = dt.datetime(2026, mo, da, hh, mi, ss, tzinfo=dt.timezone.utc).timestamp()
        st.append({"t": t, "prompt_tps": float(m.group(6)), "gen_tps": float(m.group(7)), "running": int(m.group(8)),
                   "waiting": int(m.group(9)), "kv_pct": float(m.group(10)), "prefix_hit_pct": float(m.group(11))})
        continue
    m = spat.search(line)
    if m:
        mo, da, hh, mi, ss = map(int, m.groups()[:5])
        t = dt.datetime(2026, mo, da, hh, mi, ss, tzinfo=dt.timezone.utc).timestamp()
        sp[t] = {"al": float(m.group(6)), "accepted": int(m.group(7)), "drafted": int(m.group(8))}
for r in st:
    s = sp.get(r["t"]) or {}
    r.update(al=s.get("al"), accepted=s.get("accepted"), drafted=s.get("drafted"))
w("srvstats.csv", st, ["t", "prompt_tps", "gen_tps", "running", "waiting", "kv_pct", "prefix_hit_pct", "al", "accepted", "drafted"])

# telemetry gpu (t is notebook-relative seconds)
tel = []
for l in (RUN / "engine" / "telemetry.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        x = json.loads(l)
    except Exception:
        continue
    row = {"t_rel": x.get("t")}
    for gi, g in enumerate(x.get("gpu") or []):
        for k, v in enumerate(g):
            row[f"g{gi}_{k}"] = v
    tel.append(row)
w("telemetry_gpu.csv", tel, list(tel[0].keys()) if tel else ["t_rel"])

print("requests", len(req_rows), "branches", len(br_rows), "sampler", len(srows), "srvstats", len(st), "tel", len(tel))
mism = [(r["task"], r["branch"], r["n_nb5"], r["n_meta"]) for r in req_rows if r["i"] == 0 and r["n_nb5"] != r["n_meta"]]
print("branches with nb5 != meta request counts:", len(set(mism)), sorted(set(mism))[:20])
