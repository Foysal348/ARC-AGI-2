"""E25 NB4S library (2026-11-01): which change removes the ~21 ms in-graph stall on stage 0 and speeds up decode?

Arms (each = B + exactly one change, run in this priority order): B (NB1 R2 B3 + profiler), S (CUDA_DEVICE_MAX_CONNECTIONS
=32), QP (Q-pack: NB4_PPACK=1 on the nb4_ppack-patched tree), QD (Q-defer: NB4_PPDEFER=1 on the same patched tree).

Pure logic, stdlib only (embedded verbatim in the notebook and unit-tested locally / on the Py3.12 CPU kit):
  - the arm plan and the never-do gate (env + argv), the launch check on the final env / argv;
  - the worker env read-back (POST /collective_rpc nb4s_env) parser;
  - the nb4_ppack patch sequencing (apply before the first Q arm, keep between QP and QD, revert at the end; a failed
    apply makes every Q arm NOT_INTEGRATED without a boot);
  - /metrics parsing with the iteration-token histogram buckets -> the micro-batch split of a steady window;
  - a per-graph stall parser of the torch-profiler traces (largest in-graph gap of every FULL graph per stage: node index,
    stream, the NCCL kernel that released it, the offset to the nearest pp_broadcast kernel end; stage-1 SendRecv time and
    pp_broadcast send-side kernel durations; per-step request counts per parity from the scope names);
  - a compact summary of nb4_timeline.analyze() (agent C's module, embedded verbatim, sha-checked);
  - the speed comparison, the budget rule, the per-arm verdicts and the best arm.
Q arms read back through nb4_ppack.readback_label; the verdict uses the LATEST read-back of the arm (q_readback_final:
the whole log after the server stopped, incl. the final stats line > q_readback_end > q_readback). Only ENGAGED counts;
ENGAGED_FALLBACK (Q-defer collapsed to post-at-consume: flush_exec < 0.9 x defer_park or flush_recv + flush_consume >
0.05 x defer_park on a non-last rank) is its own verdict and never a clean engage or a best-arm candidate.
Definitions used by the verdicts (all from the profiled 8k@8 window of each arm):
  stall of a graph = the largest gap inside one FULL graph replay (>= FULL_MIN_NODES kernels) during which none of the
  graph's own kernels runs; steady graphs = every replay except the first and the last of each rank (profiler edges).
"""
import gzip
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

LIB_VERSION = "nb4s_lib 2.2 (2026-11-01, arms B / S / QP / QD; nb4_ppack rev 2 read-back: ENGAGED_FALLBACK; best arm = throughput winner)"
FLAG = "CUDA_DEVICE_MAX_CONNECTIONS"
ENV_PACK, ENV_DEFER = "NB4_PPACK", "NB4_PPDEFER"           # = nb4_ppack.ENV_PACK / ENV_DEFER
WATCHED = (FLAG, ENV_PACK, ENV_DEFER)                      # read back on every arm (expected value or absent)
ORDER = ("B", "S", "QP", "QD")                             # run order = priority when time is short
Q_ARMS = ("QP", "QD")
Q_LABEL = {"QP": "Q-pack", "QD": "Q-defer"}                # nb4_ppack.server_env / readback_label arm names
STALL_LT_MS = 1.0
FIXED_SHARE = 0.95
BASE_STALLED_SHARE_MAX = 0.5
BASE_STALL_MIN_MS = 5.0
REDUCED_FRACTION = 0.5
SPEED_MIN = 0.02
FULL_MIN_NODES = 500
RELEASE_WINDOW_US = 200.0
START_RESERVE_SEC = 300                                    # step(): an arm starts only with est_min x 60 + 300 s left
BODY_RESERVE_SEC = 300                                     # body cap clipped to left - 300 s (teardown + analysis + verdict)
HARD_STOP_MIN = 95
CLEAN = ("EXACT", "NEAR_TIE")
WINDOWS = ("8k@8#1", "8k@8#2", "profile_8k@8", "40k@8")    # the same windows in the same order in every arm

# ------------------------------------------------------------------ never-do list (NB4_RESEARCH_folders §2 item 9 + lead)
NEVER_DO_ENV = {
    "NCCL_GRAPH_MIXING_SUPPORT": lambda v: str(v).strip() == "0",          # unsafe: tp comm mixes eager prefill + graphs
    "NCCL_GRAPH_REGISTER": lambda v: str(v).strip() == "1",                # deadlock risk (SGLang keeps 0)
    "NCCL_PROTO": lambda v: "ll128" in str(v).lower(),                     # forced LL128: corruption risk
    "NCCL_P2P_DISABLE": lambda v: str(v).strip() == "0",                   # P2P on
    "NCCL_P2P_LEVEL": lambda v: True,                                      # P2P on
    "P2P": lambda v: str(v).strip().lower() not in ("off", "0", "false", ""),
    "NCCL_SHM_USE_CUDA_MEMCPY": lambda v: str(v).strip() == "1",           # slower; memcpy hangs in capture
    "NCCL_LAUNCH_ORDER_IMPLICIT": lambda v: True,                          # not a fix (graphs research); never added
    "NCCL_MIN_CTAS": lambda v: True, "NCCL_MAX_CTAS": lambda v: True,      # NCCL CTA tuning: measured negative
    "NCCL_MIN_NCHANNELS": lambda v: True, "NCCL_MAX_NCHANNELS": lambda v: True, "NCCL_NTHREADS": lambda v: True,
    "COUNTERS": lambda v: str(v).strip() == "1",                           # disables the fused QSA writer
}
NEVER_DO_ARGS = ("--no-async-scheduling", "--no-enable-prefix-caching", "--enable-hc-sp")
NEVER_DO_ARG_VALUES = {"--mamba-ssm-cache-dtype": ("bfloat16",)}         # C3 SSM bf16: parked (0 capacity + B2 FAIL)


def never_do_violations(env, argv=()):
    """Every never-do item set in `env` (a dict) or `argv` (a list; the last occurrence of a flag wins)."""
    bad = []
    for k, pred in NEVER_DO_ENV.items():
        if k in (env or {}) and env[k] is not None:
            try:
                hit = pred(env[k])
            except Exception:
                hit = True
            if hit:
                bad.append(f"env {k}={env[k]}")
    for k in (env or {}):
        if k.endswith("_COUNTERS") and str(env[k]).strip() == "1":
            bad.append(f"env {k}={env[k]}")
    a = [str(x) for x in (argv or ())]
    for f in NEVER_DO_ARGS:
        if f in a:
            bad.append(f"argv {f}")
    for f, vals in NEVER_DO_ARG_VALUES.items():
        idx = [i for i, x in enumerate(a) if x == f]
        if idx and idx[-1] + 1 < len(a) and a[idx[-1] + 1] in vals:
            bad.append(f"argv {f} {a[idx[-1] + 1]}")
    return bad


# ------------------------------------------------------------------ arm plan
def arm_plan(base_extra, base_env, kv_bytes, q_env=None, ports=(8011, 8021, 8031, 8041)):
    """B = NB1 B3 (base_extra / base_env / the B3 pin); S = B + CUDA_DEVICE_MAX_CONNECTIONS=32; QP / QD = B + the
    nb4_ppack patch + NB4_PPACK=1 / NB4_PPDEFER=1 (q_env = {"QP": nb4_ppack.server_env("Q-pack"), "QD": ...}).
    Every arm: the same flags (profiler included), the same pin, the same windows in the same order."""
    q_env = q_env or {"QP": {ENV_PACK: "1"}, "QD": {ENV_DEFER: "1"}}
    change = {"B": {}, "S": {FLAG: "32"}, "QP": dict(q_env["QP"]), "QD": dict(q_env["QD"])}
    title = {"B": "NB1 B3 config (FP8 KV, pin 3,553,580,524 B, text-only, #60490, MTP K3, seqs 8) + profiler",
             "S": "B + CUDA_DEVICE_MAX_CONNECTIONS=32 (32 hardware work queues per device)",
             "QP": "B + Q-pack (nb4_ppack patch, NB4_PPACK=1: one packed broadcast per step, sent after the drafts)",
             "QD": "B + Q-defer (nb4_ppack patch, NB4_PPDEFER=1: the receive posted after the next forward launch)"}
    est = {"B": 22, "S": 18, "QP": 18, "QD": 18}            # the full minimum: launch + gates + body core + teardown
    arms = {}
    for i, k in enumerate(ORDER):
        env = dict(base_env, **change[k])
        arms[k] = {"key": k, "name": f"nb4s_{k.lower()}", "port": ports[i], "extra": list(base_extra), "env": env,
                   "change": change[k], "expect": {w: env.get(w) for w in WATCHED}, "kv_bytes": int(kv_bytes),
                   "needs_patch": k in Q_ARMS, "q_label": Q_LABEL.get(k), "est_min": est[k], "cap_min": 14 if k != "B" else 16,
                   "title": title[k], "flag_value": env.get(FLAG)}
    return arms


def check_arm_plan(arms):
    """Fail closed (raises ValueError): every arm free of never-do items; each non-B arm differs from B by exactly ONE
    env key (its change) and nothing else (flags, pin); B sets none of the watched keys; QP / QD never carry the S env;
    ports distinct; the order is B > S > QP > QD."""
    errs = []
    b = arms["B"]
    if tuple(arms) != ORDER:
        errs.append(f"order {tuple(arms)} != {ORDER}")
    for k, a in arms.items():
        v = never_do_violations(a["env"], a["extra"])
        if v:
            errs.append(f"{k}: never-do {v}")
        if a["extra"] != b["extra"]:
            errs.append(f"{k}: flags differ from B")
        if a["kv_bytes"] != b["kv_bytes"]:
            errs.append(f"{k}: pin differs from B")
        diff = {x for x in set(a["env"]) | set(b["env"]) if a["env"].get(x) != b["env"].get(x)}
        if k == "B":
            if any(w in a["env"] for w in WATCHED):
                errs.append("B sets a watched key")
        elif len(diff) != 1 or diff != set(a["change"]) or any(a["env"].get(x) != v for x, v in a["change"].items()):
            errs.append(f"{k}: env diff vs B is {sorted(diff)} (must be exactly {a['change']})")
        if k in Q_ARMS and FLAG in a["env"]:
            errs.append(f"{k} carries the S env")
        if a["needs_patch"] != (k in Q_ARMS):
            errs.append(f"{k}: needs_patch {a['needs_patch']}")
    if len({a["port"] for a in arms.values()}) != len(arms):
        errs.append("ports not distinct")
    if errs:
        raise ValueError("ARM_PLAN_INVALID: " + "; ".join(errs))
    return True


def launch_check(arm, env, argv):
    """At launch, on the FULL final env / argv of the server process: never-do items + every watched key equal to the
    arm's expected value (absent when expected None). Fail closed."""
    bad = never_do_violations(env, argv)
    for w, want in arm["expect"].items():
        have = env.get(w)
        if have != want:
            bad.append(f"{arm['key']} launch env {w}={have!r}, expected {want!r}")
    return bad


# ------------------------------------------------------------------ worker env read-back
def parse_env_readback(rpc, expect, n_workers=4):
    """rpc = {"results": [per-rank dict from nb4s_env]} or {"error": ...}; expect = {key: value or None (absent)}.
    OK only when every one of n_workers distinct ranks answered and each reports exactly the expected values."""
    out = {"expected": dict(expect), "ok": False, "definitive": False}
    if not isinstance(rpc, dict) or rpc.get("error"):
        out["why"] = "rpc error: " + str((rpc or {}).get("error") if isinstance(rpc, dict) else rpc)[:200]
        return out
    res = rpc.get("results")
    if not isinstance(res, list):
        out["why"] = "no results list"
        return out
    rows = []
    for w in res:
        w = w if isinstance(w, dict) else {}
        envd = w.get("env") if isinstance(w.get("env"), dict) else {}
        rows.append({"rank": w.get("rank"), "pid": w.get("pid"), "values": {k: envd.get(k) for k in expect},
                     "error": w.get("error")})
    out["workers"] = rows
    ranks = [r["rank"] for r in rows]
    if len(rows) != n_workers:
        out["why"] = f"{len(rows)} workers answered, expected {n_workers}"
    elif any(r["error"] for r in rows):
        out["why"] = "worker error: " + str([r["error"] for r in rows if r["error"]])[:200]
    elif len(set(ranks)) != n_workers or None in ranks:
        out["why"] = f"ranks {ranks} not {n_workers} distinct"
    elif any(r["values"] != dict(expect) for r in rows):
        out["why"] = f"values {[r['values'] for r in rows]} != {dict(expect)}"
        out["definitive"] = True                      # the workers answered: the env is really not as planned
    else:
        out["ok"] = True
        out["why"] = f"all {n_workers} workers: {dict(expect)}"
    return out


def parse_proc_environ(blobs, expect):
    """Side record: {pid: raw /proc/<pid>/environ bytes or None} of the engine processes -> each process's values."""
    vals = {}
    for pid, raw in (blobs or {}).items():
        if raw is None:
            vals[str(pid)] = "UNREADABLE"
            continue
        d = {}
        for item in (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)).split("\0"):
            if "=" in item:
                k, v = item.split("=", 1)
                d[k] = v
        vals[str(pid)] = {k: d.get(k) for k in expect}
    readable = [v for v in vals.values() if v != "UNREADABLE"]
    return {"values": vals, "all_expected": bool(readable) and all(v == dict(expect) for v in readable), "readable": len(readable)}


# ------------------------------------------------------------------ nb4_ppack patch sequencing
class PatchSeq:
    """The nb4_ppack patch on the installed tree: applied right before the first Q arm, kept between QP and QD (only the
    env changes), reverted (pre-shas verified by revert_fn) after the last Q arm or in finish(); a non-Q arm never runs on
    the patched tree. A failed apply -> every Q arm NOT_INTEGRATED (no boot). apply_fn / revert_fn raise on any miss."""

    def __init__(self, apply_fn, revert_fn):
        self.apply_fn, self.revert_fn = apply_fn, revert_fn
        self.state, self.log, self.apply_rec, self.revert_rec, self.error = "pristine", [], None, None, None

    def before_arm(self, arm):
        """(ok, why). Q arm: the tree must be patched (applies once). Non-Q arm: the tree must be pristine."""
        if arm.get("needs_patch"):
            if self.state == "applied":
                return True, "already applied"
            if self.state in ("failed", "revert_failed"):
                return False, f"NOT_INTEGRATED: patch {self.state}: {self.error}"
            try:
                self.apply_rec = self.apply_fn()
                self.state = "applied"
                self.log.append(("apply", arm["key"]))
                return True, "applied"
            except Exception as e:
                self.state, self.error = "failed", f"{type(e).__name__}: {e}"[:500]
                self.log.append(("apply_failed", arm["key"]))
                return False, "NOT_INTEGRATED: " + self.error
        if self.state == "applied":
            ok = self._revert(arm["key"])
            if not ok:
                return False, "tree not restored before a non-Q arm: " + str(self.error)
        return True, "pristine"

    def _revert(self, tag):
        try:
            self.revert_rec = self.revert_fn()
            self.state = "reverted"
            self.log.append(("revert", tag))
            return True
        except Exception as e:
            self.state, self.error = "revert_failed", f"{type(e).__name__}: {e}"[:500]
            self.log.append(("revert_failed", tag))
            return False

    def finish(self):
        if self.state == "applied":
            self._revert("finish")
        return {"state": self.state, "log": self.log, "error": self.error}


# ------------------------------------------------------------------ /metrics: the lab's aggregation + histogram buckets
_PROM_RX = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
ITER_BUCKET = "vllm:iteration_tokens_total_bucket"


def prom_parse(txt):
    """(agg, buckets): agg = exactly e23_lab.prom's dict (sum over labels + '@max'); buckets = {le (float): cumulative
    count} of the iteration-token histogram (summed over label sets)."""
    out, buckets = {}, {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_RX.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        if v != v:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
        out[m.group(1) + "@max"] = max(out.get(m.group(1) + "@max", v), v)
        if m.group(1) == ITER_BUCKET and m.group(2):
            le = re.search(r'le="([^"]+)"', m.group(2))
            if le:
                try:
                    k = float(le.group(1))
                except ValueError:
                    continue
                buckets[k] = buckets.get(k, 0.0) + v
    return out, buckets


def hist_delta(h0, h1):
    """Per-bucket (non-cumulative) step counts between two cumulative snapshots: [[lo, hi, count], ...]."""
    if not h0 or not h1:
        return None
    les = sorted(set(h0) | set(h1))
    cum = [(le, h1.get(le, 0.0) - h0.get(le, 0.0)) for le in les]
    out, prev_le, prev_c = [], 0.0, 0.0
    for le, c in cum:
        out.append([prev_le, le, round(c - prev_c, 1)])
        prev_le, prev_c = le, c
    return out


def parity_from_hist(delta, n, pp=2, k=3):
    """Micro-batch split of a decode window from its iteration-token histogram: balanced = every step schedules
    ceil(n / pp) x (1 + k) tokens (4 x 4 = 16 at 8 streams). BALANCED when >= 90 % of the steps fall in the bucket that
    holds that count; IMBALANCED otherwise; UNRESOLVED when that bucket also holds the 2x count or there is no data."""
    if not delta:
        return {"label": "UNRESOLVED", "why": "no histogram"}
    t_bal = math.ceil(n / pp) * (1 + k)
    tot = sum(c for _, _, c in delta if c > 0)
    if tot <= 0:
        return {"label": "UNRESOLVED", "why": "no steps", "balanced_tokens": t_bal}
    hb = next(((lo, hi, c) for lo, hi, c in delta if lo < t_bal <= hi), None)
    if hb is None:
        return {"label": "UNRESOLVED", "why": f"no bucket holds {t_bal}", "balanced_tokens": t_bal}
    if hb[0] < 2 * t_bal <= hb[1]:
        return {"label": "UNRESOLVED", "why": f"bucket ({hb[0]}, {hb[1]}] cannot separate {t_bal} from {2 * t_bal}",
                "balanced_tokens": t_bal}
    share = hb[2] / tot
    return {"label": "BALANCED" if share >= 0.9 else "IMBALANCED", "share_at_balanced_bucket": round(share, 3),
            "balanced_tokens": t_bal, "bucket": [hb[0], hb[1]], "steps": int(tot),
            "dist": [[lo, hi, c] for lo, hi, c in delta if c > 0][:12]}


# ------------------------------------------------------------------ per-graph stall parser (torch-profiler traces)
def _open(p):
    with (gzip.open if str(p).endswith(".gz") else open)(p, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _rank_of(name, tr, k):
    m_pp, m_tp, m_r = re.search(r"pp(\d+)", name), re.search(r"_tp(\d+)", name), re.search(r"rank[-_]?(\d+)", name)
    di = tr.get("distributedInfo") if isinstance(tr.get("distributedInfo"), dict) else {}
    rank = di.get("rank") if isinstance(di.get("rank"), int) else (int(m_r.group(1)) if m_r else k)
    return rank, (int(m_pp.group(1)) if m_pp else rank // 2), (int(m_tp.group(1)) if m_tp else rank % 2)


def _a(e):
    a = e.get("args")
    return a if isinstance(a, dict) else {}


SCOPE_RX = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def rank_graphs(tr):
    """One rank: graph replays (kernels grouped by the cudaGraphLaunch correlation), NCCL kernels, decode scopes."""
    ev = [e for e in (tr.get("traceEvents") or []) if e.get("ph") == "X" and "ts" in e]
    launches = {}
    for e in ev:
        if str(e.get("cat", "")).lower() in ("cuda_runtime", "cuda_driver") and "GraphLaunch" in str(e.get("name", "")):
            c = _a(e).get("correlation")
            if c is not None:
                launches[c] = e
    groups, nccl = defaultdict(list), []
    for e in ev:
        if str(e.get("cat", "")).lower() not in ("kernel", "gpu_kernel"):
            continue
        a = _a(e)
        c = a.get("correlation")
        if c in launches:
            groups[c].append(e)
        if "nccl" in str(e.get("name", "")).lower():
            nccl.append({"a": float(e["ts"]), "b": float(e["ts"]) + float(e.get("dur", 0.0)), "name": str(e.get("name", "")).split("(")[0],
                         "coll": a.get("Collective name"), "dtype": a.get("dtype"), "nelems": a.get("In msg nelems"),
                         "pg": a.get("Process Group Description"), "group": a.get("Process Group Ranks"),
                         "stream": a.get("stream", e.get("tid")), "in_graph": bool(a.get("graph id"))})
    nccl.sort(key=lambda d: d["b"])
    graphs = []
    for c, ks in groups.items():
        ks.sort(key=lambda e: float(e["ts"]))
        graphs.append({"corr": c, "ks": ks, "n": len(ks), "start": float(ks[0]["ts"]),
                       "end": max(float(k["ts"]) + float(k.get("dur", 0.0)) for k in ks)})
    graphs.sort(key=lambda g: g["start"])
    scopes = sorted((e for e in ev if e.get("cat") == "user_annotation" and SCOPE_RX.match(str(e.get("name", "")))),
                    key=lambda e: float(e["ts"]))
    return graphs, nccl, scopes


def largest_gap(g, nccl):
    """The largest gap inside one graph replay (none of its own kernels running): node index (0-based, by start), the
    stream / kernel after the gap, the releasing NCCL kernel (latest end within RELEASE_WINDOW_US before the gap end) and
    the offset of the gap end to the nearest preceding pp_broadcast kernel end."""
    ks = g["ks"]
    cur = float(ks[0]["ts"]) + float(ks[0].get("dur", 0.0))
    best = (0.0, None, None, None)
    for i in range(1, len(ks)):
        t = float(ks[i]["ts"])
        if t - cur > best[0]:
            best = (t - cur, i, cur, t)
        cur = max(cur, t + float(ks[i].get("dur", 0.0)))
    gap, idx, a, b = best
    out = {"nodes": len(ks), "gap_ms": round(gap / 1000.0, 3), "node_index": idx}
    if idx is None:
        return out
    k = ks[idx]
    out.update(stream=_a(k).get("stream", k.get("tid")), kernel=str(k.get("name", "")).split("(")[0].replace("void ", "")[:60],
               gap_start_ms_in_graph=round((a - g["start"]) / 1000.0, 3))
    if gap * 1e-3 >= STALL_LT_MS:
        cand = [x for x in nccl if b - RELEASE_WINDOW_US <= x["b"] <= b + 5.0]
        if cand:
            r = max(cand, key=lambda x: x["b"])
            out["released_by"] = f"{r['name']} {r['coll']} {r['dtype']}x{r['nelems']} {r['pg']}"
            out["release_lag_us"] = round(b - r["b"], 1)
    ppb = [x["b"] for x in nccl if x["pg"] == "pp_broadcast" and x["b"] <= b + 5.0]
    if ppb:
        out["offset_to_pp_broadcast_end_us"] = round(b - max(ppb), 1)
    return out


def _share_below(vals, lim):
    return round(sum(1 for v in vals if v < lim) / len(vals), 3) if vals else None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def _stage_stall(rows_by_rank):
    """Aggregate per-graph rows (steady graphs only) of one stage's ranks."""
    gaps, nodes, rel, offs = [], Counter(), Counter(), []
    for rows in rows_by_rank:
        steady = rows[1:-1] if len(rows) > 2 else rows
        for x in steady:
            gaps.append(x["gap_ms"])
            if x["gap_ms"] >= STALL_LT_MS:
                nodes[x["node_index"]] += 1
                rel[x.get("released_by")] += 1
                if x.get("offset_to_pp_broadcast_end_us") is not None:
                    offs.append(x["offset_to_pp_broadcast_end_us"])
    return {"graphs": len(gaps), "stall_ms_median": _med(gaps),
            "stall_ms_mean": round(statistics.mean(gaps), 3) if gaps else None,
            "share_below_1ms": _share_below(gaps, STALL_LT_MS),
            "node_index_mode": nodes.most_common(1)[0][0] if nodes else None, "node_index_hist": dict(nodes.most_common(5)),
            "released_by_mode": rel.most_common(1)[0][0] if rel else None,
            "offset_to_pp_broadcast_end_us_median": _med(offs)}


def graph_stalls(src, full_min=FULL_MIN_NODES):
    """Per-boot stall facts from the traces of the profiled window (a directory or a list of files): the stage-0 stall
    (the question), the stage-1 stall (where Q-defer could move it), stage-1 SendRecv and pp_broadcast send kernels."""
    p = Path(src) if not isinstance(src, (list, tuple)) else None
    files = sorted(p.rglob("*.pt.trace.json*")) if p is not None and p.is_dir() else ([p] if p is not None else [Path(x) for x in src])
    if not files:
        return {"error": "NO_TRACE_FILE"}
    ranks = {}
    for k, f in enumerate(files):
        tr = _open(f)
        rank, pp, tp = _rank_of(f.name, tr, k)
        graphs, nccl, scopes = rank_graphs(tr)
        ranks[rank] = {"pp": pp, "tp": tp, "graphs": graphs, "nccl": nccl, "scopes": scopes, "file": f.name}
    stage0 = min(r["pp"] for r in ranks.values())
    out = {"files": [r["file"] for r in ranks.values()], "by_rank": {}}
    rows_stage = defaultdict(list)
    for rk in sorted(ranks):
        r = ranks[rk]
        tg = [g for g in r["graphs"] if g["n"] >= full_min]
        rows = [largest_gap(g, r["nccl"]) for g in tg]
        rows_stage[r["pp"]].append(rows)
        steady = rows[1:-1] if len(rows) > 2 else rows
        gv = [x["gap_ms"] for x in steady]
        rec = {"stage": r["pp"], "target_graphs": len(tg), "graph_nodes": dict(Counter(g["n"] for g in tg).most_common(3)),
               "small_graphs": len(r["graphs"]) - len(tg), "stall_ms_median": _med(gv), "share_below_1ms": _share_below(gv, STALL_LT_MS),
               "steady_graphs": len(steady), "first_graph_gap_ms": rows[0]["gap_ms"] if rows else None,
               "node_index_mode": Counter(x["node_index"] for x in steady if x["gap_ms"] >= STALL_LT_MS).most_common(3),
               "stream_mode": Counter(str(x.get("stream")) for x in steady if x["gap_ms"] >= STALL_LT_MS).most_common(3),
               "kernel_after_gap": Counter(x.get("kernel") for x in steady if x["gap_ms"] >= STALL_LT_MS).most_common(2),
               "released_by": Counter(x.get("released_by") for x in steady if x["gap_ms"] >= STALL_LT_MS).most_common(3),
               "rows": rows[:60]}
        if len(tg) >= 2:
            rec["graph_period_ms_median"] = round(statistics.median([tg[i + 1]["start"] - tg[i]["start"] for i in range(len(tg) - 1)]) / 1000.0, 3)
            lo, hi, nt = tg[0]["start"], tg[-1]["start"], len(tg) - 1
            inwin = [x for x in r["nccl"] if x["b"] > lo and x["a"] < hi]
            sr = sum(min(x["b"], hi) - max(x["a"], lo) for x in inwin if "SendRecv" in x["name"])
            rec["sendrecv_ms_per_tick"] = round(sr / 1000.0 / nt, 3)
            pb = [x["b"] - x["a"] for x in inwin if x["pg"] == "pp_broadcast"]
            rec["pp_broadcast_kernels"] = {"per_tick": round(len(pb) / nt, 2), "ms_per_tick": round(sum(pb) / 1000.0 / nt, 3),
                                           "median_us": round(statistics.median(pb), 1) if pb else None,
                                           "max_ms": round(max(pb) / 1000.0, 3) if pb else None}
        gen = [int(SCOPE_RX.match(str(e.get("name", ""))).group(3)) for e in r["scopes"]]
        rec["requests_by_parity"] = {str(q): dict(Counter(gen[q::2])) for q in (0, 1)}
        out["by_rank"][rk] = rec
    out["stage0"] = _stage_stall(rows_stage[stage0])
    s1 = [v for v in out["by_rank"].values() if v["stage"] != stage0]
    out["stage1"] = dict(_stage_stall([rr for st, lst in rows_stage.items() if st != stage0 for rr in lst]),
                         sendrecv_ms_per_tick_mean=_med([v.get("sendrecv_ms_per_tick") for v in s1]),
                         pp_broadcast_send_ms_per_tick=_med([(v.get("pp_broadcast_kernels") or {}).get("ms_per_tick") for v in s1]),
                         pp_broadcast_send_max_ms=max([(v.get("pp_broadcast_kernels") or {}).get("max_ms") or 0 for v in s1] or [0]))
    ref = out["by_rank"].get(min(rk for rk in ranks if ranks[rk]["pp"] == stage0), {})
    out["parity"] = parity_from_scopes(ref.get("requests_by_parity"))
    return out


def parity_from_scopes(par):
    """{"0": {n_requests: steps}, "1": {...}} -> BALANCED when both parities have requests and their modal counts differ
    by <= 1 (8 streams: 4 + 4); IMBALANCED otherwise (e.g. 8 + 0, NB2 SEED2)."""
    if not par:
        return {"label": "UNRESOLVED", "why": "no scopes"}
    mode = {}
    for q in ("0", "1"):
        c = {int(k): v for k, v in (par.get(q) or {}).items()}
        mode[q] = max(c, key=lambda x: c[x]) if c else 0
    lab = "BALANCED" if mode["0"] > 0 and mode["1"] > 0 and abs(mode["0"] - mode["1"]) <= 1 else "IMBALANCED"
    return {"label": lab, "modal_requests": mode, "by_parity": par}


# ------------------------------------------------------------------ nb4_timeline.analyze() -> the per-boot report fields
def _g(d, *ks):
    for k in ks:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def summarize_timeline(rep):
    """Compact per-boot fields from agent C's analyze() output; every field is read defensively (None when missing)."""
    if not isinstance(rep, dict) or rep.get("error"):
        return {"error": (rep or {}).get("error") if isinstance(rep, dict) else "no report"}
    s = rep.get("summary") or {}
    out = {"tick_ms_mean": s.get("tick_ms_mean"), "tick_ms_median": s.get("tick_ms_median"),
           "steady_ticks": s.get("steady_ticks"), "graph_modes_steady": s.get("graph_modes"),
           "graph_modes_by_rank": rep.get("graph_modes"), "clock": _g(rep, "clock", "method"), "stages": {}}
    for st in ("stage0", "stage1"):
        if isinstance(s.get(st), dict):
            out["stages"][st] = {c: _g(s, st, c, "mean_ms") for c in ("compute", "copy", "in_graph_stall", "nccl_peer_wait",
                                                                       "nccl_tp_skew", "nccl_transfer", "host_launch_gap",
                                                                       "host_python_gap", "idle_no_host")}
            hs = [out["stages"][st].get(c) for c in ("host_launch_gap", "host_python_gap", "idle_no_host")]
            out["stages"][st]["host_total"] = round(sum(x for x in hs if x is not None), 3) if any(x is not None for x in hs) else None
    ticks = [t for t in (rep.get("ticks") or []) if isinstance(t, dict)]
    st_t = ticks[1:-1] if len(ticks) > 2 else ticks
    sv = [_g(t, "stall", "in_graph_stall_ms_ref_rank") for t in st_t]
    sv = [x for x in sv if isinstance(x, (int, float))]
    out["tick_stall_ms_median"] = round(statistics.median(sv), 3) if sv else None
    out["tick_share_stall_below_1ms"] = _share_below(sv, STALL_LT_MS)
    by_rank = defaultdict(lambda: defaultdict(list))
    for t in st_t:
        for r, row in (t.get("by_rank") or {}).items():
            for c in ("compute", "copy", "in_graph_stall", "nccl_peer_wait", "nccl_transfer"):
                if isinstance(row.get(c), (int, float)):
                    by_rank[str(r)][c].append(row[c])
            by_rank[str(r)]["tick"].append(t.get("tick_ms"))
    out["per_rank"] = {}
    for r, d in sorted(by_rank.items()):
        m = {c: round(statistics.mean([x for x in v if x is not None]), 3) for c, v in d.items() if [x for x in v if x is not None]}
        if "tick" in m and "compute" in m:
            m["busy"] = round(m["compute"] + m.get("copy", 0.0), 3)
            m["idle"] = round(m["tick"] - m["busy"], 3)
        out["per_rank"][r] = m
    out["stalls"] = rep.get("stalls")
    par = rep.get("parity") or {}
    out["parity_verdict"] = par.get("verdict") if isinstance(par, dict) else None
    out["parity_requests"] = {k: _g(v, "n_hist_by_parity") for k, v in ((par.get("stages") or {}) if isinstance(par, dict) else {}).items()}
    dep = rep.get("dependency") or {}
    out["dependency_verdict"] = dep.get("verdict") if isinstance(dep, dict) else None
    out["ideal_tick"] = s.get("ideal_tick")
    out["pipeline"] = s.get("pipeline")
    out["stage1_idle_ms_mean"] = s.get("stage1_idle_ms_mean")
    return out


# ------------------------------------------------------------------ speed
def window_ok(w):
    """A window counts for speed only when STEADY, running == level, prefill done, 0 preemptions, a tok/s value, and its
    micro-batch split BALANCED (an IMBALANCED / UNRESOLVED split is labelled and never compared)."""
    if not isinstance(w, dict):
        return False, "missing"
    if w.get("status") != "STEADY":
        return False, f"status {w.get('status')}"
    if w.get("running_eq_level") is not True:
        return False, "running != level"
    if (w.get("preemptions") or 0) != 0:
        return False, "preemptions"
    if not isinstance(w.get("tok_s"), (int, float)) or w["tok_s"] <= 0:
        return False, "no tok/s"
    lab = (w.get("parity") or {}).get("label")
    if lab != "BALANCED":
        return False, f"parity {lab}"
    return True, "ok"


def speed_compare(b, s, keys=("8k@8#1", "8k@8#2")):
    """b / s: {window key: window record}. Gain = mean(X) / mean(B) - 1 over the valid windows; B's spread = |w1 - w2| /
    mean (needs both B windows valid). PASS when gain > max(spread, SPEED_MIN)."""
    bv = {k: b.get(k) for k in keys if window_ok(b.get(k))[0]}
    sv = {k: s.get(k) for k in keys if window_ok(s.get(k))[0]}
    out = {"b_valid": sorted(bv), "x_valid": sorted(sv),
           "why_invalid": {f"{arm}:{k}": window_ok(d.get(k))[1] for arm, d in (("B", b), ("X", s)) for k in keys if not window_ok(d.get(k))[0]}}
    if len(bv) < 2:
        out.update(status="INCONCLUSIVE_B_SPREAD_UNKNOWN", why="B needs both 8k@8 windows valid for its own spread")
        return out
    if len(sv) != len(keys):                          # GPT audit 2026-11-01: every X window must be valid (one is not enough)
        out.update(status="INCONCLUSIVE_X_WINDOWS" if sv else "INCONCLUSIVE_NO_X_WINDOW",
                   why=f"all {len(keys)} X windows must be valid; valid {sorted(sv)}")
        return out
    bt = [w["tok_s"] for w in bv.values()]
    st = [w["tok_s"] for w in sv.values()]
    bm, sm = statistics.mean(bt), statistics.mean(st)
    spread = abs(bt[0] - bt[1]) / bm
    gain = sm / bm - 1.0
    thr = max(spread, SPEED_MIN)
    out.update(b_tok_s=[round(x, 1) for x in bt], x_tok_s=[round(x, 1) for x in st], b_mean=round(bm, 1), x_mean=round(sm, 1),
               b_spread=round(spread, 4), gain=round(gain, 4), threshold=round(thr, 4),
               status="PASS" if gain > thr else "FAIL",
               b_tick_ms=[w.get("tick_ms") for w in bv.values()], x_tick_ms=[w.get("tick_ms") for w in sv.values()])
    return out


def side_window(b, s, key="40k@8"):
    """The 40k@8 window: reported (gain + validity), not gating."""
    ob, os_ = window_ok(b.get(key)), window_ok(s.get(key))
    out = {"b_ok": ob[0], "x_ok": os_[0], "why": [ob[1], os_[1]]}
    if ob[0] and os_[0]:
        out["gain"] = round(s[key]["tok_s"] / b[key]["tok_s"] - 1.0, 4)
        out["tok_s"] = [b[key]["tok_s"], s[key]["tok_s"]]
        out["tick_ms"] = [b[key].get("tick_ms"), s[key].get("tick_ms")]
    return out


# ------------------------------------------------------------------ budget
def arm_fits(left_sec, arm):
    """step()'s own rule: an arm starts only with its full minimum left: est_min x 60 + START_RESERVE_SEC."""
    return left_sec >= arm["est_min"] * 60 + START_RESERVE_SEC


def body_cap_sec(cap_min, left_sec):
    """The body cap at body start: the arm's cap, clipped so teardown + analysis + verdict keep BODY_RESERVE_SEC."""
    return max(60.0, min(cap_min * 60.0, left_sec - BODY_RESERVE_SEC))


# ------------------------------------------------------------------ verdicts
def stall_class(b_st, x_st, b_s1=None, x_s1=None):
    """STALL_FIXED / STALL_MOVED / STALL_REDUCED / NO_EFFECT from the per-graph stall facts (stage0 dicts; stage1 dicts
    for the move-to-stage-1 check). A stage-0 fix with a NEW stage-1 stall (median >= BASE_STALL_MIN_MS where B had
    none) is STALL_MOVED (the main Q-defer risk)."""
    sh = x_st.get("share_below_1ms")
    if sh is None:
        return "INCONCLUSIVE_NO_TRACE", "no stage-0 graphs"
    b_s1, x_s1 = b_s1 or {}, x_s1 or {}
    s1_new = ((x_s1.get("stall_ms_median") or 0.0) >= BASE_STALL_MIN_MS and (b_s1.get("stall_ms_median") or 0.0) < STALL_LT_MS)
    if sh >= FIXED_SHARE:
        if s1_new:
            return "STALL_MOVED", f"stage 0 fixed ({sh:.0%} < {STALL_LT_MS} ms) but stage 1 now stalls {x_s1.get('stall_ms_median')} ms (B {b_s1.get('stall_ms_median')})"
        return "STALL_FIXED", f"{sh:.0%} of stage-0 graphs stall < {STALL_LT_MS} ms"
    moved_node = x_st.get("node_index_mode") is not None and x_st.get("node_index_mode") != b_st.get("node_index_mode")
    moved_rel = (x_st.get("released_by_mode") or "").split(" ")[:4] != (b_st.get("released_by_mode") or "").split(" ")[:4]
    if moved_node or moved_rel or s1_new:
        return "STALL_MOVED", (f"stall at node {x_st.get('node_index_mode')} released by {x_st.get('released_by_mode')} vs B node "
                               f"{b_st.get('node_index_mode')} released by {b_st.get('released_by_mode')}; stage-1 stall "
                               f"{x_s1.get('stall_ms_median')} ms (B {b_s1.get('stall_ms_median')})")
    bm, sm = b_st.get("stall_ms_median") or 0.0, x_st.get("stall_ms_median") or 0.0
    if bm and sm <= REDUCED_FRACTION * bm:
        return "STALL_REDUCED", f"same node; median stall {sm} ms vs B {bm} ms"
    return "NO_EFFECT", f"same node {x_st.get('node_index_mode')}; median stall {sm} ms vs B {bm} ms"


def _win(rec):
    """The arm's windows: rec["steady"] in the notebook (so RC2's garbage-under-load gate sees them), rec["windows"] accepted."""
    w = (rec or {}).get("steady")
    return w if isinstance(w, dict) else ((rec or {}).get("windows") or {})


def base_check(B):
    """(ok, label, why): B must be clean (gate PASS, env as planned) and reproduce the stall with a balanced profile."""
    if not B or str(B.get("status", "")).startswith("SKIPPED") or B.get("status") is None:
        return False, "INCONCLUSIVE_NOT_RUN", f"B {B.get('status') if B else None}"
    if B.get("quality_gate") != "PASS":
        return False, "INCONCLUSIVE_BASE_GATE", f"B gate {B.get('quality_gate')} / status {B.get('status')}"
    if B.get("status") != "DONE":                     # GPT audit 2026-11-01: a failed B run (ERROR / LAUNCH_FAILED / ...) never counts
        return False, "INCONCLUSIVE_BASE_FAILED", f"B status {B.get('status')}"
    if not (B.get("env_readback") or {}).get("ok"):
        return False, "INCONCLUSIVE_BASE_ENV", str((B.get("env_readback") or {}).get("why"))
    bst = (B.get("stall") or {}).get("stage0") or {}
    if not bst.get("graphs"):
        return False, "INCONCLUSIVE_NO_TRACE", "B has no stage-0 graphs"
    bp = ((B.get("stall") or {}).get("parity") or {}).get("label")
    if bp != "BALANCED":                              # GPT audit 2026-11-01: the base profile must be balanced, as every arm's is
        return False, "INCONCLUSIVE_BASE_IMBALANCED", f"B profiled window parity {bp}"
    if not ((bst.get("share_below_1ms") or 0) <= BASE_STALLED_SHARE_MAX and (bst.get("stall_ms_median") or 0) >= BASE_STALL_MIN_MS):
        return False, "INCONCLUSIVE_BASE_NO_STALL", ("B did not reproduce the stall (profile parity "
                                                     f"{((B.get('stall') or {}).get('parity') or {}).get('label')})")
    return True, "OK", "B clean and stalled"


def arm_verdict(B, X, key, base=None):
    """One change arm vs B. Order: not run / not integrated / not engaged -> correctness -> evidence -> stall -> speed."""
    base = base or base_check(B)
    out = {"arm": key}
    st = str((X or {}).get("status") or "")
    if not X or st.startswith("SKIPPED") or not st:
        out.update(verdict="SKIPPED_BUDGET" if "BUDGET" in st else ("NOT_INTEGRATED" if (X or {}).get("not_integrated") else
                                                                    "SKIPPED_DEPENDENCY" if X else "SKIPPED_BUDGET"),
                   why=(X or {}).get("why") or "not run")
        return out
    if X.get("not_integrated") or st == "NOT_INTEGRATED":
        out.update(verdict="NOT_INTEGRATED", why=X.get("why") or (X.get("q_readback") or {}).get("label"))
        return out
    so = X.get("oracle_vs")
    if so == "FAIL_HIGH_MARGIN" or str(X.get("correctness", "")).startswith("NO_GO") or str(X.get("quality_gate", "")).startswith("FAIL"):
        out.update(verdict="CORRECTNESS_FAIL", why=f"oracle vs B {so}, correctness {X.get('correctness')}, gate {X.get('quality_gate')}")
        return out
    qr = next((X[k] for k in ("q_readback_final", "q_readback_end", "q_readback") if isinstance(X.get(k), dict)), None)
    if key in Q_ARMS and (not isinstance(qr, dict) or qr.get("label") != "ENGAGED"):
        lab = (qr or {}).get("label")
        out.update(verdict=lab if lab in ("NOT_INTEGRATED", "ENGAGED_FALLBACK") else "NOT_ENGAGED",
                   why=f"ppack read-back {lab}: {(qr or {}).get('detail')}")
        return out
    if not (X.get("env_readback") or {}).get("ok"):
        out.update(verdict="NOT_ENGAGED", why=str((X.get("env_readback") or {}).get("why")))
        return out
    if st != "DONE":                                  # GPT audit 2026-11-01: an abnormal final status never gets a positive verdict
        out.update(verdict="INCONCLUSIVE_ARM_FAILED", why=f"arm status {st}")
        return out
    if so not in CLEAN:
        out.update(verdict="INCONCLUSIVE_ORACLE", why=f"oracle vs B {so} (status {X.get('status')})")
        return out
    if not base[0]:
        out.update(verdict=base[1], why=base[2])
        return out
    if B.get("pin") != X.get("pin"):
        out.update(verdict="INCONCLUSIVE_PIN_MISMATCH", why=f"pin B {B.get('pin')} vs {key} {X.get('pin')}")
        return out
    bst, xst = (B.get("stall") or {}).get("stage0") or {}, (X.get("stall") or {}).get("stage0") or {}
    if not xst.get("graphs"):
        out.update(verdict="INCONCLUSIVE_NO_TRACE", why=f"{key} has no stage-0 graphs")
        return out
    xp = ((X.get("stall") or {}).get("parity") or {}).get("label")
    if xp != "BALANCED":
        out.update(verdict="INCONCLUSIVE_IMBALANCED", why=f"{key} profiled window split {xp}: not comparable with B")
        return out
    out["b_stall"] = {k: bst.get(k) for k in ("stall_ms_median", "share_below_1ms", "node_index_mode", "released_by_mode")}
    out["x_stall"] = {k: xst.get(k) for k in ("stall_ms_median", "share_below_1ms", "node_index_mode", "released_by_mode")}
    b1, x1 = (B.get("stall") or {}).get("stage1") or {}, (X.get("stall") or {}).get("stage1") or {}
    out["stage1"] = {"b": {k: b1.get(k) for k in ("stall_ms_median", "sendrecv_ms_per_tick_mean", "pp_broadcast_send_ms_per_tick")},
                     "x": {k: x1.get(k) for k in ("stall_ms_median", "sendrecv_ms_per_tick_mean", "pp_broadcast_send_ms_per_tick")}}
    cls, why = stall_class(bst, xst, b1, x1)
    spd = speed_compare(_win(B), _win(X))
    out["speed"] = spd
    out["side_40k"] = side_window(_win(B), _win(X))
    if cls == "STALL_FIXED":
        out.update(verdict="STALL_FIXED_WIN" if spd.get("status") == "PASS" else "STALL_FIXED_NO_SPEED",
                   why=why + f"; speed {spd.get('status')} (gain {spd.get('gain')}, threshold {spd.get('threshold')})")
    else:
        out.update(verdict=cls, why=why + f"; speed {spd.get('status')} (gain {spd.get('gain')})")
    return out


RANK = {"STALL_FIXED_WIN": 0, "STALL_FIXED_NO_SPEED": 1, "STALL_REDUCED": 2, "STALL_MOVED": 3}
# GPT audit 2026-11-01 (verdict selection): the best arm is the THROUGHPUT winner, not the best stall class. Eligible = a
# measured, clean verdict (RANK or NO_EFFECT: correctness, engagement, base and trace all passed), 8k@8 speed PASS, both
# 40k@8 windows valid and the 40k gain not below -SPEED_MIN. The old stall-class ranking stays as stall_best_arm.
MEASURED = set(RANK) | {"NO_EFFECT"}


def throughput_eligible(v):
    """(ok, why) for one arm verdict record."""
    if v.get("verdict") not in MEASURED:
        return False, f"verdict {v.get('verdict')}"
    spd, side = v.get("speed") or {}, v.get("side_40k") or {}
    if spd.get("status") != "PASS":
        return False, f"8k@8 speed {spd.get('status')} (gain {spd.get('gain')})"
    if not (side.get("b_ok") and side.get("x_ok")):
        return False, f"40k@8 window invalid {side.get('why')}"
    if (side.get("gain") or 0.0) < -SPEED_MIN:
        return False, f"40k@8 regression {side.get('gain')}"
    return True, "ok"


def verdict(arms):
    """arms: {"B": rec, "S": rec, "QP": rec, "QD": rec} (missing = not run). Per-arm verdicts + the best arm = the highest
    8k@8 gain among throughput-eligible arms (throughput_eligible), ties by stall class. stall_best_arm = the diagnostic
    ranking: the best stall class (STALL_FIXED_WIN > STALL_FIXED_NO_SPEED > STALL_REDUCED > STALL_MOVED), then the gain."""
    B = arms.get("B") or {}
    base = base_check(B)
    out = {"lib": LIB_VERSION, "base": {"ok": base[0], "label": base[1], "why": base[2]},
           "criteria": {"fixed": f">= {FIXED_SHARE:.0%} of the arm's steady stage-0 graphs stall < {STALL_LT_MS} ms (no new stage-1 stall)",
                        "speed": f"8k@8 gain > max(B window-to-window spread, {SPEED_MIN:.0%})", "oracle": "vs B EXACT or NEAR_TIE"},
           "arms": {}}
    for k in ORDER[1:]:
        out["arms"][k] = arm_verdict(B, arms.get(k), k, base)
    cands = [(RANK[v["verdict"]], -((v.get("speed") or {}).get("gain") or -9.0), k) for k, v in out["arms"].items() if v["verdict"] in RANK]
    if cands:
        sb = min(cands)
        out["stall_best_arm"], out["stall_best_verdict"] = sb[2], out["arms"][sb[2]]["verdict"]
    else:
        out["stall_best_arm"], out["stall_best_verdict"] = None, ("NO_ARM_FIXED" if base[0] else base[1])
    out["throughput_eligible"] = {k: dict(zip(("ok", "why"), throughput_eligible(v))) for k, v in out["arms"].items()}
    elig = [(-(v["speed"]["gain"]), RANK.get(v["verdict"], len(RANK)), k) for k, v in out["arms"].items()
            if out["throughput_eligible"][k]["ok"]]
    if elig:
        best = min(elig)
        out["best_arm"] = best[2]
        out["best_verdict"] = out["arms"][best[2]]["verdict"]
    else:
        out["best_arm"], out["best_verdict"] = None, ("NO_THROUGHPUT_WINNER" if base[0] else base[1])
    out["verdict"] = f"{out['best_arm']}:{out['best_verdict']}" if out["best_arm"] else out["best_verdict"]
    out["why"] = "; ".join(f"{k} {v['verdict']}" for k, v in out["arms"].items())
    return out
