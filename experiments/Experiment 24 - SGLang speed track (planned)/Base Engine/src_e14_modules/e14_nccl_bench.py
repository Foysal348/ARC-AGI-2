"""E14 NCCL bench for 4x L4 on PCIe (Experiment 14, decision D2; 2026-10-23).

What it answers (Notes/15):
- the real cost of ONE 983,040 B bf16 all-reduce (96 rows x 5120 hidden) when 140 of them run back to back inside a
  CUDA graph, as SGLang's decode graph does per step (graph_983040.per_call_us = the number that matters);
- the logits all-gather (11.9 MB per rank, 47.7 MB total), eager and inside a graph;
- eager all-reduce latency/bandwidth at 8 KiB .. 64 MiB (algbw and busbw = algbw * 2(n-1)/n);
- the NCCL transport per channel (P2P/CUMEM, P2P/IPC, SHM, NET), parsed from NCCL_DEBUG=INFO lines;
- whether CUDA P2P works (torch.cuda.can_device_access_peer matrix) and the PCIe topology (nvidia-smi);
- whether two env arms help: NCCL_P2P_LEVEL=SYS (only when P2P works) and NCCL_SHM_USE_CUDA_MEMCPY=1.

Process layout (no CUDA in the driver):
  driver (this file, --out)                      stdlib only; never raises; always writes the JSON
    -> probe child  (--child-probe)              imports torch: versions, device list, P2P matrix
    -> per arm: launcher child (--child-run)     own process group / session, hard timeout -> whole tree killed
         -> torch.multiprocessing.spawn 4 ranks  NCCL via tcp://127.0.0.1:<free port>; each rank writes
                                                 rank_<r>.json after EVERY phase, so a hang keeps partial results
The driver aggregates the rank files (max over ranks = the slowest rank defines the collective) and parses the arm's
stdout+stderr for NCCL lines (NCCL writes its debug log to stdout by default).

Every arm inherits the env SGLang 0.5.21 sets for its own NCCL communicators unless --no-sglang-env:
  NCCL_CUMEM_ENABLE=0, NCCL_NVLS_ENABLE=0 (entrypoints/engine.py:1801-1810, only when not already set),
  NCCL_GRAPH_REGISTER=0 (arg_groups/parallel_hook.py:326, setdefault), CUDA_DEVICE_MAX_CONNECTIONS=8 (engine.py:1816).

Usage (venv python, torch there):
  python e14_nccl_bench.py --out result.json [--arms baseline,p2p_sys,shm_memcpy] [--timeout-per-arm 240] [--fake]
--fake: no torch / CUDA / nvidia-smi at all; the same driver path with synthetic children; JSON marked "fake": true.
"""
import argparse
import json
import os
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback

BENCH_VERSION = "e14-nccl-bench-1"

EAGER_SIZES = [("8KiB", 8192), ("80KiB", 81920), ("256KiB", 262144), ("983040B", 983040),
               ("2MiB", 2097152), ("64MiB", 67108864)]
GRAPH_BYTES = 983040            # 96 rows x 5120 hidden x 2 B (bf16), one SGLang decode all-reduce
GRAPH_CALLS = 140               # all-reduces per decode step (measured in SGLang)
AG_BYTES_PER_RANK = 11_919_360  # 96 rows x 62,080 vocab rows per rank x 2 B = 11.9 MB per rank, 47.7 MB total

COMMON_ENV = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,ENV"}
SGLANG_ENV_DEFAULTS = {"NCCL_CUMEM_ENABLE": "0", "NCCL_NVLS_ENABLE": "0", "NCCL_GRAPH_REGISTER": "0"}
SGLANG_ENV_FORCED = {"CUDA_DEVICE_MAX_CONNECTIONS": "8"}
ARMS = {
    "baseline": {},
    "p2p_sys": {"NCCL_P2P_LEVEL": "SYS"},
    "shm_memcpy": {"NCCL_SHM_USE_CUDA_MEMCPY": "1"},
}
P2P_ARMS = {"p2p_sys"}          # run only when every GPU pair reports can_device_access_peer
DEFAULT_ARMS = "baseline,p2p_sys,shm_memcpy"
ADOPT_RULE_PCT = 20.0           # Notes/13 D2: an arm >= 20 % faster at 983,040 B (graph per call) with the right sum
TAIL_CHARS = 6000


# ============================================================================================ small helpers (stdlib)
def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def write_json_atomic(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=str)
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def tail_text(path, n=TAIL_CHARS):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "replace")
    except Exception:
        return ""


def read_text(path, limit=64 << 20):
    try:
        with open(path, "rb") as f:
            return f.read(limit).decode("utf-8", "replace")
    except Exception:
        return ""


def percentile(vals, q):
    """Linear-interpolated percentile (q in 0..100), like numpy's default."""
    v = sorted(float(x) for x in vals)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def stats_us(samples):
    s = [float(x) for x in samples if x is not None]
    if not s:
        return None
    return {"n": len(s), "median_us": statistics.median(s), "p90_us": percentile(s, 90),
            "min_us": min(s), "max_us": max(s)}


def algbw_GBps(nbytes, us):
    """Algorithm bandwidth in GB/s (decimal, as nccl-tests): bytes / time."""
    if not us or us <= 0:
        return None
    return nbytes / (us * 1e-6) / 1e9


def busbw_allreduce(algbw, n):
    """nccl-tests bus bandwidth for all-reduce: algbw * 2(n-1)/n."""
    if algbw is None or n < 1:
        return None
    return algbw * 2.0 * (n - 1) / n


def busbw_allgather(algbw, n):
    """nccl-tests bus bandwidth for all-gather (algbw over the TOTAL output bytes): algbw * (n-1)/n."""
    if algbw is None or n < 1:
        return None
    return algbw * (n - 1) / n


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


# ============================================================================================ NCCL log parsing
RE_PREFIX = re.compile(r"^(?P<host>[^\s:]+):(?P<pid>\d+):(?P<tid>\d+)\s+\[(?P<dev>\d+)\]\s+NCCL\s+"
                       r"(?P<lvl>INFO|WARN|TRACE|ABORT)\s?(?P<msg>.*)$")
RE_CONN = re.compile(r"Channel\s+(?P<ch>\d+)/(?P<sub>\d+)\s*:\s*(?P<src>\d+)\[(?P<sdev>\d+)\]\s*->\s*"
                     r"(?P<dst>\d+)\[(?P<ddev>\d+)\]\s*(?:\[(?P<dir>send|receive)\]\s*)?via\s+(?P<tr>\S+)")
RE_RING = re.compile(r"Channel\s+(?P<ch>\d+)/(?P<n>\d+)\s*:\s*(?P<order>\d+(?:\s+\d+)+)\s*$")
RE_VERSION = re.compile(r"NCCL version\s+(\S+)")
RE_ENV = re.compile(r"(NCCL_[A-Z0-9_]+)\s+set by environment to\s+(\S+)")
RE_PATTERN = re.compile(r"Pattern\s+(?P<pattern>\d+),\s*crossNic\s+(?P<crossnic>\d+),\s*nChannels\s+(?P<nch>\d+),"
                        r"\s*bw\s+(?P<bw>[\d.]+)/(?P<bwinter>[\d.]+),\s*type\s+(?P<type>\S+?)/(?P<typeinter>\S+?),"
                        r"\s*sameChannels\s+(?P<same>\d+)")
RE_COLLCH = re.compile(r"(?P<coll>\d+)\s+coll channels,\s*(?:(?P<collnet>\d+)\s+collnet channels,\s*)?"
                       r"(?:(?P<nvls>\d+)\s+nvls channels,\s*)?(?P<p2p>\d+)\s+p2p channels")
RE_P2PTYPE = re.compile(r"Check P2P Type\s+(.*)$")
RE_CUDADRV = re.compile(r"cudaDriverVersion\s+(\d+)")
RE_INIT_TIMINGS = re.compile(r"Init timings.*?total\s+([\d.]+)")
RE_CONNECTED = re.compile(r"Connected all (rings|trees)(.*)$")
RE_P2P_NOTE = re.compile(r"(P2P\b.*(disabled|not supported|unavailable|fail)|peer access.*(not|fail))", re.I)


def parse_nccl_log(text):
    """Parse NCCL_DEBUG=INFO output (any mix of ranks, stdout/stderr interleaved).

    Returns the per-channel connections (deduplicated), a transport summary, version, env lines, graph search
    results, channel counts, WARN lines and P2P notes. Never raises."""
    out = {"nccl_version": None, "cuda_driver": None, "env": {}, "connections": [], "transport": [],
           "families": [], "primary": None, "rings": [], "patterns": [], "coll_channels": [], "p2p_type": [],
           "connected": [], "init_complete": 0, "init_total_s": [], "warnings": [], "p2p_notes": [],
           "nccl_lines": 0}
    if not text:
        return out
    seen = set()
    conns = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if "NCCL" not in line:
            continue
        m = RE_PREFIX.match(line.strip())
        if m:
            lvl, msg, dev = m.group("lvl"), m.group("msg"), m.group("dev")
        else:
            i = line.find("NCCL INFO")
            j = line.find("NCCL WARN")
            if i < 0 and j < 0:
                continue
            lvl = "INFO" if i >= 0 else "WARN"
            msg = line[(i if i >= 0 else j) + 9:].strip()
            dev = None
        out["nccl_lines"] += 1
        if lvl in ("WARN", "ABORT"):
            if len(out["warnings"]) < 50:
                out["warnings"].append(msg.strip()[:300])
            continue
        mc = RE_CONN.search(msg)
        if mc:
            tr = mc.group("tr")
            key = (int(mc.group("ch")), int(mc.group("sub")), int(mc.group("src")), int(mc.group("dst")),
                   mc.group("dir") or "", tr)
            if key not in seen:
                seen.add(key)
                conns.append({"channel": key[0], "sub": key[1], "src": key[2], "dst": key[3],
                              "src_dev": int(mc.group("sdev")), "dst_dev": int(mc.group("ddev")),
                              "dir": key[4] or None, "transport": tr, "family": tr.split("/")[0],
                              "log_dev": None if dev is None else int(dev)})
            continue
        mr = RE_RING.search(msg)
        if mr and "->" not in msg:
            item = {"channel": int(mr.group("ch")), "order": [int(x) for x in mr.group("order").split()]}
            if item not in out["rings"]:
                out["rings"].append(item)
            continue
        mv = RE_VERSION.search(msg)
        if mv:
            out["nccl_version"] = mv.group(1)
            continue
        md = RE_CUDADRV.search(msg)
        if md:
            out["cuda_driver"] = int(md.group(1))
            continue
        me = RE_ENV.search(msg)
        if me:
            out["env"][me.group(1)] = me.group(2).rstrip(".")     # NCCL ends the line with '.'
            continue
        mp_ = RE_PATTERN.search(msg)
        if mp_:
            item = {"pattern": int(mp_.group("pattern")), "crossNic": int(mp_.group("crossnic")),
                    "nChannels": int(mp_.group("nch")), "bw_intra": float(mp_.group("bw")),
                    "bw_inter": float(mp_.group("bwinter")), "type_intra": mp_.group("type"),
                    "type_inter": mp_.group("typeinter"), "sameChannels": int(mp_.group("same"))}
            if item not in out["patterns"]:
                out["patterns"].append(item)
            continue
        mch = RE_COLLCH.search(msg)
        if mch:
            item = {k: (int(v) if v is not None else None) for k, v in mch.groupdict().items()}
            if item not in out["coll_channels"]:
                out["coll_channels"].append(item)
            continue
        mt = RE_P2PTYPE.search(msg)
        if mt:
            s = mt.group(1).strip()
            if s not in out["p2p_type"]:
                out["p2p_type"].append(s)
            continue
        mco = RE_CONNECTED.search(msg)
        if mco:
            s = ("all " + mco.group(1) + mco.group(2)).strip()
            if s not in out["connected"]:
                out["connected"].append(s)
            continue
        if "Init COMPLETE" in msg:
            out["init_complete"] += 1
        mi = RE_INIT_TIMINGS.search(msg)
        if mi:
            out["init_total_s"].append(float(mi.group(1)))
            continue
        if RE_P2P_NOTE.search(msg) and len(out["p2p_notes"]) < 30:
            s = msg.strip()[:300]
            if s not in out["p2p_notes"]:
                out["p2p_notes"].append(s)
    out["connections"] = conns
    out["transport"] = summarize_transport(conns)
    out["families"] = sorted({c["family"] for c in conns})
    if out["transport"]:
        out["primary"] = out["transport"][0]["transport"]
    return out


def summarize_transport(conns):
    """One row per transport string, most frequent first."""
    rows = {}
    for c in conns:
        r = rows.setdefault(c["transport"], {"transport": c["transport"], "family": c["family"], "count": 0,
                                             "channels": set(), "pairs": set()})
        r["count"] += 1
        r["channels"].add(c["channel"])
        r["pairs"].add("%d->%d" % (c["src"], c["dst"]))
    res = []
    for r in sorted(rows.values(), key=lambda x: (-x["count"], x["transport"])):
        res.append({"transport": r["transport"], "family": r["family"], "count": r["count"],
                    "channels": sorted(r["channels"]),
                    "pairs": sorted(r["pairs"], key=lambda p: tuple(int(x) for x in p.split("->")))})
    return res


# ============================================================================================ aggregation (stdlib)
def _combine(per_rank_samples, nbytes=None, world=None, kind="allreduce"):
    """per_rank_samples: {rank: [us,...]} -> slowest-rank headline + per-rank stats + bandwidth."""
    per_rank = {}
    for r, s in sorted(per_rank_samples.items(), key=lambda kv: int(kv[0])):
        st = stats_us(s or [])
        if st:
            per_rank[str(r)] = st
    if not per_rank:
        return None
    med = max(v["median_us"] for v in per_rank.values())
    p90 = max(v["p90_us"] for v in per_rank.values())
    out = {"median_us": med, "p90_us": p90,
           "min_rank_median_us": min(v["median_us"] for v in per_rank.values()),
           "slowest_rank": max(per_rank, key=lambda r: per_rank[r]["median_us"]),
           "ranks_reported": len(per_rank)}
    lens = {len(s) for s in per_rank_samples.values() if s}
    if len(lens) == 1 and len(per_rank_samples) > 1:
        n = lens.pop()
        rows = [list(s) for s in per_rank_samples.values() if s]
        out["per_iter_max_median_us"] = statistics.median(max(row[i] for row in rows) for i in range(n))
    if nbytes is not None and world:
        alg = algbw_GBps(nbytes, med)
        out["algbw_GBps"] = alg
        out["busbw_GBps"] = busbw_allreduce(alg, world) if kind == "allreduce" else busbw_allgather(alg, world)
    out["per_rank"] = per_rank
    return out


def aggregate_ranks(ranks, world=None):
    """ranks: list of rank_<r>.json dicts (possibly partial). Max over ranks for every timing."""
    ranks = [r for r in ranks if isinstance(r, dict) and "rank" in r]
    ranks.sort(key=lambda r: int(r["rank"]))
    world = world or (max((int(r.get("world") or 0) for r in ranks), default=0)) or len(ranks)
    agg = {"ranks_reported": [int(r["rank"]) for r in ranks], "world": world,
           "phases_done": {str(r["rank"]): r.get("phases_done", []) for r in ranks},
           "phase_errors": {}, "devices": {str(r["rank"]): r.get("device_name") for r in ranks}}
    for r in ranks:
        for ph, err in (r.get("errors") or {}).items():
            agg["phase_errors"].setdefault(ph, {})[str(r["rank"])] = err

    # a) correctness
    cor = [r.get("correct") for r in ranks if isinstance(r.get("correct"), dict)]
    if cor and len(cor) == world:
        agg["correct"] = all(bool(c.get("pass")) for c in cor)
    else:
        agg["correct"] = False if cor else None
    agg["correct_detail"] = {str(r["rank"]): r.get("correct") for r in ranks if r.get("correct") is not None}

    # b) eager all-reduce
    eager = {}
    for label, nbytes in EAGER_SIZES:
        samples = {r["rank"]: ((r.get("eager") or {}).get(label) or {}).get("samples_us") for r in ranks}
        samples = {k: v for k, v in samples.items() if v}
        c = _combine(samples, nbytes, world, "allreduce")
        if c:
            loops = [((r.get("eager") or {}).get(label) or {}).get("loop_us_per_call") for r in ranks]
            loops = [x for x in loops if x is not None]
            c["bytes"] = nbytes
            c["loop_us_per_call"] = max(loops) if loops else None
            eager[label] = c
    agg["eager"] = eager

    # c) CUDA-graph replay, 140 x 983,040 B
    g = {"bytes": GRAPH_BYTES}
    gr = [r for r in ranks if isinstance(r.get("graph_983040"), dict)]
    if gr:
        calls = gr[0]["graph_983040"].get("calls", GRAPH_CALLS)
        per_call = {r["rank"]: [x / calls for x in r["graph_983040"].get("replay_us", [])] for r in gr}
        per_call = {k: v for k, v in per_call.items() if v}
        c = _combine(per_call, GRAPH_BYTES, world, "allreduce")
        g["calls_per_graph"] = calls
        g["replays"] = max(len(r["graph_983040"].get("replay_us", [])) for r in gr)
        if c:
            g["per_call_us"] = c["median_us"]
            g["per_call_p90_us"] = c["p90_us"]
            g["min_rank_per_call_us"] = c["min_rank_median_us"]
            g["per_iter_max_per_call_us"] = c.get("per_iter_max_median_us")
            g["replay_ms_median"] = c["median_us"] * calls / 1000.0
            g["algbw_GBps"] = c.get("algbw_GBps")
            g["busbw_GBps"] = c.get("busbw_GBps")
            g["per_rank"] = c["per_rank"]
        loops = [r["graph_983040"].get("eager_loop_us_per_call") for r in gr]
        loops = [x for x in loops if x is not None]
        g["eager_loop_per_call_us"] = max(loops) if loops else None
        gc = [r["graph_983040"].get("graph_correct") for r in gr]
        g["graph_correct"] = (all(bool(x) for x in gc) and len(gc) == world) if all(x is not None for x in gc) else None
        g["capture_mode"] = gr[0]["graph_983040"].get("capture_mode")
    errs = agg["phase_errors"].get("graph_983040")
    if errs:
        g["error"] = errs
    agg["graph_983040"] = g

    # d) all-gather 11.9 MB per rank
    ag = {"bytes_per_rank": None, "total_bytes": None}
    are = [r for r in ranks if isinstance(r.get("allgather_eager"), dict)]
    if are:
        bpr = are[0]["allgather_eager"].get("bytes_per_rank", AG_BYTES_PER_RANK)
        ag["bytes_per_rank"] = bpr
        ag["total_bytes"] = bpr * world
        c = _combine({r["rank"]: r["allgather_eager"].get("samples_us") for r in are}, bpr * world, world,
                     "allgather")
        if c:
            ag["eager"] = _to_ms(c)
    arg = [r for r in ranks if isinstance(r.get("allgather_graph"), dict)]
    if arg:
        bpr = ag["bytes_per_rank"] or arg[0]["allgather_graph"].get("bytes_per_rank", AG_BYTES_PER_RANK)
        c = _combine({r["rank"]: r["allgather_graph"].get("samples_us") for r in arg}, bpr * world, world,
                     "allgather")
        if c:
            ag["graph"] = _to_ms(c)
            ag["graph"]["status"] = "ok"
    for ph in ("allgather_eager", "allgather_graph"):
        e = agg["phase_errors"].get(ph)
        if e:
            key = "eager" if ph == "allgather_eager" else "graph"
            ag.setdefault(key, {})
            ag[key]["status"] = "capture_failed" if key == "graph" else "failed"
            ag[key]["error"] = e
    agg["allgather_11_9MB"] = ag
    return agg


def _to_ms(c):
    out = {"median_ms": c["median_us"] / 1000.0, "p90_ms": c["p90_us"] / 1000.0,
           "min_rank_median_ms": c["min_rank_median_us"] / 1000.0, "algbw_GBps": c.get("algbw_GBps"),
           "busbw_GBps": c.get("busbw_GBps"), "slowest_rank": c.get("slowest_rank"),
           "ranks_reported": c.get("ranks_reported")}
    if c.get("per_iter_max_median_us") is not None:
        out["per_iter_max_median_ms"] = c["per_iter_max_median_us"] / 1000.0
    out["per_rank"] = {r: {"median_ms": v["median_us"] / 1000.0, "p90_ms": v["p90_us"] / 1000.0, "n": v["n"]}
                       for r, v in c["per_rank"].items()}
    return out


def build_summary(result):
    arms = result.get("arms") or {}
    per_arm = {}
    for name, a in arms.items():
        g = a.get("graph_983040") or {}
        ag = a.get("allgather_11_9MB") or {}
        e = (a.get("eager") or {}).get("983040B") or {}
        ag_ms = (ag.get("graph") or {}).get("median_ms")
        if ag_ms is None:
            ag_ms = (ag.get("eager") or {}).get("median_ms")
        pc = g.get("per_call_us")
        per_arm[name] = {
            "status": a.get("status"), "correct": a.get("correct"), "graph_correct": g.get("graph_correct"),
            "graph_per_call_us": pc, "graph_per_call_p90_us": g.get("per_call_p90_us"),
            "eager_983040_median_us": e.get("median_us"),
            "eager_64MiB_busbw_GBps": ((a.get("eager") or {}).get("64MiB") or {}).get("busbw_GBps"),
            "allgather_eager_ms": (ag.get("eager") or {}).get("median_ms"),
            "allgather_graph_ms": (ag.get("graph") or {}).get("median_ms"),
            "primary_transport": a.get("primary_transport"),
            "per_step_comm_ms_est": (GRAPH_CALLS * pc / 1000.0 + (ag_ms or 0.0)) if pc is not None else None,
        }
    # the workload runs inside CUDA graphs: an arm counts only with eager AND graph results correct on every rank
    # (GPT audit 2 P0: a graph-wrong arm could win before; status / missing flags are rejected too)
    cands = {n: v for n, v in per_arm.items() if v["status"] == "ok" and v["correct"] is True
             and v["graph_correct"] is True and v["graph_per_call_us"] is not None}
    best = min(cands, key=lambda n: cands[n]["graph_per_call_us"]) if cands else None
    base = cands.get("baseline", {}).get("graph_per_call_us")   # an invalid baseline is no reference (review 10)
    s = {"metric": "graph_983040.per_call_us = slowest rank's median over replays of (graph replay time / %d)"
                   % GRAPH_CALLS,
         "best_arm": best, "best_per_call_us": cands[best]["graph_per_call_us"] if best else None,
         "baseline_per_call_us": base, "per_arm": per_arm, "adopt_rule_pct": ADOPT_RULE_PCT}
    if best and base:
        b = cands[best]["graph_per_call_us"]
        s["improvement_vs_baseline_pct"] = (1.0 - b / base) * 100.0
        s["adopt_candidate"] = best if (best != "baseline" and
                                        s["improvement_vs_baseline_pct"] >= ADOPT_RULE_PCT - 1e-9) else None
    else:
        s["improvement_vs_baseline_pct"] = None
        s["adopt_candidate"] = None
    if result.get("fake"):
        s["fake"] = True
    return s


# ============================================================================================ process control
def kill_tree(proc):
    """Kill the child and everything it spawned (POSIX: its own session/process group; Windows: taskkill /T)."""
    try:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                proc.kill()
        else:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=30)
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
    except Exception:
        pass


def run_with_timeout(cmd, env, timeout, stdout_path, stderr_path, cwd=None):
    """Run cmd in its own process group; on timeout kill the whole group. Never raises."""
    res = {"cmd": cmd, "status": None, "returncode": None, "elapsed_s": None, "stdout_path": stdout_path,
           "stderr_path": stderr_path}
    kw = {}
    if os.name == "posix":
        kw["start_new_session"] = True
    else:
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    t0 = time.monotonic()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(stdout_path)), exist_ok=True)
        with open(stdout_path, "wb") as fo, open(stderr_path, "wb") as fe:
            try:
                p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL, **kw)
            except Exception as e:
                res.update(status="error", error="spawn failed: %r" % (e,), elapsed_s=time.monotonic() - t0)
                return res
            try:
                rc = p.wait(timeout=timeout)
                res["returncode"] = rc
                res["status"] = "ok" if rc == 0 else "error"
            except subprocess.TimeoutExpired:
                kill_tree(p)
                res["status"] = "timeout"
                try:
                    p.wait(timeout=30)
                except Exception:
                    pass
                res["returncode"] = p.returncode
    except Exception as e:
        res["status"] = "error"
        res["error"] = "runner: %r" % (e,)
    res["elapsed_s"] = time.monotonic() - t0
    return res


def run_cmd(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return {"rc": p.returncode, "out": p.stdout.decode("utf-8", "replace"),
                "err": p.stderr.decode("utf-8", "replace")[-2000:]}
    except Exception as e:
        return {"rc": None, "out": "", "err": repr(e)}


def parse_csv(text):
    lines = [l for l in (text or "").splitlines() if l.strip()]
    if len(lines) < 2:
        return []
    hdr = [h.strip() for h in lines[0].split(",")]
    return [dict(zip(hdr, [x.strip() for x in l.split(",")])) for l in lines[1:]]


# ============================================================================================ driver
def build_arm_env(arm_env, args, base_env=None):
    """Full child env + the record of what we set. SGLang defaults only when the key is not inherited."""
    env = dict(os.environ if base_env is None else base_env)
    applied = {}
    if not args.no_sglang_env:
        for k, v in SGLANG_ENV_DEFAULTS.items():
            if k not in env:
                env[k] = v
                applied[k] = v
            else:
                applied[k] = env[k] + " (inherited)"
        for k, v in SGLANG_ENV_FORCED.items():
            env[k] = v
            applied[k] = v
    dbg = dict(COMMON_ENV)
    if args.nccl_debug_subsys:
        dbg["NCCL_DEBUG_SUBSYS"] = args.nccl_debug_subsys
    for k, v in list(dbg.items()) + list(arm_env.items()):
        env[k] = v
        applied[k] = v
    for k, v in env.items():
        if k.startswith("NCCL_") and k not in applied:
            applied[k] = v + " (inherited)"
    env["PYTHONUNBUFFERED"] = "1"
    return env, applied


def probe_topology(args, workroot):
    topo = {"fake": bool(args.fake)}
    if args.fake:
        n = args.world or 4
        ok = args.fake_p2p == "yes"
        topo["nvidia_smi_topo_m"] = {"rc": 0, "out": _FAKE_TOPO, "err": ""}
        topo["nvidia_smi_topo_p2p_r"] = {"rc": 0, "out": "(fake)", "err": ""}
        topo["gpus"] = [{"index": str(i), "name": "FAKE L4", "pci.bus_id": "00000000:%02X:00.0" % (i + 1),
                         "pcie.link.gen.current": "4", "pcie.link.width.current": "16", "clocks.sm [MHz]": "2040",
                         "power.limit [W]": "72.00 W"} for i in range(n)]
        probe = {"fake": True, "torch": "fake", "cuda": "fake", "nccl": "2.29.7", "driver_version": None,
                 "device_count": n, "devices": [{"name": "FAKE L4"} for _ in range(n)],
                 "can_access_peer": [[(i != j) and ok for j in range(n)] for i in range(n)]}
        topo["probe"] = {"status": "ok", **probe}
    else:
        topo["nvidia_smi_topo_m"] = run_cmd(["nvidia-smi", "topo", "-m"])
        topo["nvidia_smi_topo_p2p_r"] = run_cmd(["nvidia-smi", "topo", "-p2p", "r"])
        q_full = ("index,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,"
                  "pcie.link.width.max,clocks.sm,clocks.max.sm,power.limit,driver_version")
        q_min = "pci.bus_id,pcie.link.gen.current,pcie.link.width.current,clocks.sm,power.limit"
        q = run_cmd(["nvidia-smi", "--query-gpu=" + q_full, "--format=csv"])
        if q["rc"] != 0:
            q = run_cmd(["nvidia-smi", "--query-gpu=" + q_min, "--format=csv"])
        topo["nvidia_smi_query"] = q
        topo["gpus"] = parse_csv(q["out"]) if q["rc"] == 0 else []
        pdir = os.path.join(workroot, "_probe")
        os.makedirs(pdir, exist_ok=True)
        pout = os.path.join(pdir, "probe.json")
        if os.path.exists(pout):
            os.remove(pout)
        cmd = [sys.executable, os.path.abspath(__file__), "--child-probe", "--child-out", pout]
        r = run_with_timeout(cmd, dict(os.environ), args.probe_timeout, os.path.join(pdir, "stdout.log"),
                             os.path.join(pdir, "stderr.log"))
        probe = read_json(pout) or {}
        probe["status"] = "ok" if (r["status"] == "ok" and probe.get("can_access_peer") is not None) else \
            (r["status"] if r["status"] != "ok" else "error")
        probe["elapsed_s"] = r["elapsed_s"]
        if probe["status"] != "ok":
            probe["stderr_tail"] = tail_text(os.path.join(pdir, "stderr.log"), 3000)
        topo["probe"] = probe
    m = topo["probe"].get("can_access_peer")
    topo["p2p_matrix"] = m
    topo["p2p_all"] = bool(m) and len(m) > 1 and all(bool(m[i][j]) for i in range(len(m))
                                                       for j in range(len(m)) if i != j)
    topo["p2p_any"] = bool(m) and any(bool(m[i][j]) for i in range(len(m)) for j in range(len(m)) if i != j)
    return topo


def run_one_arm(name, args, topo, workroot, arm_table, cmd_override=None):
    """Run one arm in a killable child; aggregate whatever the ranks wrote. Never raises."""
    t0 = time.monotonic()
    rec = {"status": None, "env": {}, "transport": [], "correct": None, "eager": {}, "graph_983040": {},
           "allgather_11_9MB": {}, "stderr_tail": "", "stdout_tail": ""}
    try:
        if name not in arm_table:
            rec.update(status="error", error="unknown arm %r (known: %s)" % (name, ", ".join(sorted(arm_table))))
            return rec
        env, applied = build_arm_env(arm_table[name], args)
        rec["env"] = applied
        if name in P2P_ARMS and not topo.get("p2p_all") and not args.force_p2p:
            why = "skipped: no P2P"
            if (topo.get("probe") or {}).get("status") != "ok":
                why += " (probe %s)" % (topo.get("probe") or {}).get("status")
            rec.update(status="skipped", reason=why)
            return rec
        wd = os.path.join(workroot, name)
        if os.path.isdir(wd):
            shutil.rmtree(wd, ignore_errors=True)
        os.makedirs(wd, exist_ok=True)
        cfg = {"world": args.world, "warmup": args.warmup, "iters": args.iters, "graph_calls": args.graph_calls,
               "replays": args.replays, "ag_bytes": args.ag_bytes, "capture_mode": args.capture_mode,
               "arm": name}
        if cmd_override is not None:
            cmd = list(cmd_override)
        else:
            cmd = [sys.executable, os.path.abspath(__file__), "--child-run", "--arm", name, "--workdir", wd,
                   "--cfg-json", json.dumps(cfg)]
            if args.fake:
                cmd.append("--child-fake")
        rec["cmd"] = cmd
        out_p, err_p = os.path.join(wd, "stdout.log"), os.path.join(wd, "stderr.log")
        r = run_with_timeout(cmd, env, args.timeout_per_arm, out_p, err_p, cwd=wd)
        rec["returncode"] = r["returncode"]
        rec["elapsed_s"] = r["elapsed_s"]
        rec["stdout_log"], rec["stderr_log"] = out_p, err_p
        rec["stderr_tail"] = tail_text(err_p)
        rec["stdout_tail"] = tail_text(out_p)
        log = read_text(out_p) + "\n" + read_text(err_p)
        nccl = parse_nccl_log(log)
        rec["transport"] = nccl.pop("transport")
        rec["primary_transport"] = nccl.get("primary")
        conns = nccl.pop("connections")
        nccl["connections_n"] = len(conns)
        nccl["connections_sample"] = conns[:12]
        rec["nccl"] = nccl
        launcher = read_json(os.path.join(wd, "launcher.json")) or {}
        rec["launcher"] = launcher
        ranks = []
        for fn in sorted(os.listdir(wd)):
            if re.match(r"rank_\d+\.json$", fn):
                d = read_json(os.path.join(wd, fn))
                if d:
                    ranks.append(d)
        agg = aggregate_ranks(ranks, launcher.get("world") or args.world)   # else the ranks' own "world" field
        world = agg["world"]
        rec.update({k: agg[k] for k in ("correct", "eager", "graph_983040", "allgather_11_9MB")})
        rec["correct_detail"] = agg["correct_detail"]
        rec["ranks_reported"] = agg["ranks_reported"]
        rec["phases_done"] = agg["phases_done"]
        rec["devices"] = agg["devices"]
        if agg["phase_errors"]:
            rec["phase_errors"] = agg["phase_errors"]
        if r["status"] == "timeout":
            rec["status"] = "timeout"
            rec["error"] = "timeout after %.0f s; whole process group killed; partial rank results kept" % \
                args.timeout_per_arm
        elif r["status"] != "ok" or launcher.get("error"):
            rec["status"] = "error"
            rec["error"] = launcher.get("error") or r.get("error") or "child exit code %s" % r["returncode"]
        elif world and len(ranks) < world:
            rec["status"] = "error"
            rec["error"] = "only %d of %d ranks reported" % (len(ranks), world)
        else:
            rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "error"
        rec["error"] = "driver: %r" % (e,)
        rec["traceback"] = traceback.format_exc()[-3000:]
    finally:
        rec["elapsed_s"] = rec.get("elapsed_s") or (time.monotonic() - t0)
    return rec


def parse_arm_env_specs(specs):
    """--arm-env 'name:K=V,K2=V2' -> {name: {K: V}}."""
    table = {}
    for s in specs or []:
        if ":" not in s:
            raise ValueError("bad --arm-env %r (want name:K=V,...)" % s)
        name, kvs = s.split(":", 1)
        env = {}
        for kv in [x for x in kvs.split(",") if x.strip()]:
            k, _, v = kv.partition("=")
            env[k.strip()] = v.strip()
        table[name.strip()] = env
    return table


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", help="result JSON path (driver mode)")
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument("--arm-env", action="append", default=[], help="extra arm, 'name:K=V,K2=V2' (repeatable)")
    ap.add_argument("--timeout-per-arm", type=float, default=240.0)
    ap.add_argument("--probe-timeout", type=float, default=120.0)
    ap.add_argument("--fake", action="store_true", help="CPU-only synthetic run (no torch, no CUDA)")
    ap.add_argument("--fake-p2p", choices=["yes", "no"], default="yes", help="fake probe: P2P matrix all true/false")
    ap.add_argument("--world", type=int, default=None, help="ranks (default: torch.cuda.device_count(), must be 4)")
    ap.add_argument("--workdir", default=None, help="driver: work root (default <out dir>/e14_nccl_work)")
    ap.add_argument("--no-sglang-env", action="store_true", help="do not apply SGLang's NCCL env defaults")
    ap.add_argument("--nccl-debug-subsys", default=None, help="override NCCL_DEBUG_SUBSYS (default INIT,GRAPH,ENV)")
    ap.add_argument("--force-p2p", action="store_true", help="run P2P arms even if the probe says no P2P")
    ap.add_argument("--capture-mode", default="thread_local", choices=["thread_local", "global", "relaxed"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--graph-calls", type=int, default=GRAPH_CALLS)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--ag-bytes", type=int, default=AG_BYTES_PER_RANK, help="all-gather bytes per rank")
    # internal child modes
    ap.add_argument("--child-run", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--child-probe", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--child-fake", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--arm", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--cfg-json", default="{}", help=argparse.SUPPRESS)
    return ap


def driver_main(args, argv=None, cmd_overrides=None):
    """Run all arms; ALWAYS writes args.out (also after a failure). Returns the result dict."""
    out = os.path.abspath(args.out)
    workroot = os.path.abspath(args.workdir or os.path.join(os.path.dirname(out), "e14_nccl_work"))
    result = {"bench_version": BENCH_VERSION, "fake": bool(args.fake), "started": now_iso(),
              "argv": list(argv) if argv is not None else sys.argv[1:], "python": sys.executable,
              "config": {"arms": args.arms, "timeout_per_arm_s": args.timeout_per_arm, "world": args.world,
                         "eager_sizes": EAGER_SIZES, "graph_bytes": GRAPH_BYTES, "graph_calls": args.graph_calls,
                         "replays": args.replays, "warmup": args.warmup, "iters": args.iters,
                         "ag_bytes_per_rank": args.ag_bytes, "capture_mode": args.capture_mode,
                         "sglang_env": not args.no_sglang_env, "workroot": workroot},
              "topology": {}, "arms": {}, "summary": {}}
    try:
        os.makedirs(workroot, exist_ok=True)
        _safe_write(out, result)
        arm_table = dict(ARMS)
        try:
            arm_table.update(parse_arm_env_specs(args.arm_env))
        except Exception as e:
            result["arm_env_error"] = repr(e)
        try:
            result["topology"] = probe_topology(args, workroot)
        except Exception as e:
            result["topology"] = {"error": repr(e), "p2p_all": False}
        _safe_write(out, result)
        names = [a.strip() for a in (args.arms or "").split(",") if a.strip()]
        for name in names:
            co = (cmd_overrides or {}).get(name)
            result["arms"][name] = run_one_arm(name, args, result["topology"], workroot, arm_table, co)
            result["summary"] = build_summary(result)
            _safe_write(out, result)
            if not args.fake and result["arms"][name].get("status") == "timeout":
                time.sleep(3.0)          # let the killed ranks release GPU memory before the next arm
    except BaseException as e:           # includes KeyboardInterrupt: still write what we have
        result["driver_error"] = repr(e)
        result["driver_traceback"] = traceback.format_exc()[-3000:]
    finally:
        try:
            result["summary"] = build_summary(result)
        except Exception as e:
            result["summary"] = {"error": repr(e)}
        result["finished"] = now_iso()
        result["torch_imported_in_driver"] = "torch" in sys.modules
        _safe_write(out, result)
    return result


def _safe_write(path, obj):
    try:
        write_json_atomic(path, obj)
    except Exception as e:
        sys.stderr.write("e14_nccl_bench: could not write %s: %r\n" % (path, e))


# ============================================================================================ fake children (CPU)
_FAKE_TOPO = """\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tPHB\tPHB\tPHB\t0-47\t0\t\tN/A
GPU1\tPHB\t X \tPHB\tPHB\t0-47\t0\t\tN/A
GPU2\tPHB\tPHB\t X \tPHB\t0-47\t0\t\tN/A
GPU3\tPHB\tPHB\tPHB\t X \t0-47\t0\t\tN/A
(fake)"""

_FAKE_BASE_US = {"baseline": 290.0, "p2p_sys": 180.0, "shm_memcpy": 300.0}


def fake_nccl_lines(arm, world):
    tr = "P2P/IPC/read" if arm == "p2p_sys" else "SHM/direct/direct"
    lines = []
    for r in range(world):
        p = "fakehost:%d:%d [%d] NCCL INFO " % (1000 + r, 1000 + r, r)
        if r == 0:
            lines.append(p + "NCCL version 2.29.7+cuda13.0")
        lines.append(p + "cudaDriverVersion 13000")
        for k, v in (("NCCL_CUMEM_ENABLE", "0"), ("NCCL_GRAPH_REGISTER", "0")):
            lines.append(p + "%s set by environment to %s." % (k, v))
        lines.append(p + "Pattern 4, crossNic 0, nChannels 2, bw 6.000000/6.000000, type %s/PIX, sameChannels 1"
                     % ("PHB" if arm == "p2p_sys" else "SYS"))
        for ch in range(2):
            lines.append(p + "Channel %02d/02 : %s" % (ch, " ".join(str(i) for i in range(world))))
        for ch in range(2):
            lines.append(p + "Channel %02d/0 : %d[%d] -> %d[%d] via %s" % (ch, r, r, (r + 1) % world,
                                                                          (r + 1) % world, tr))
        lines.append(p + "Connected all rings, use ring PXN 0 GDR 0")
        lines.append(p + "2 coll channels, 2 collnet channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer")
        lines.append(p + "ncclCommInitRankConfig comm 0x5f%04x rank %d nranks %d cudaDev %d nvmlDev %d busId %d000 "
                     "commId 0xfake - Init COMPLETE" % (r, r, world, r, r, r + 1))
    return lines


def fake_child_run(args):
    cfg = json.loads(args.cfg_json or "{}")
    arm = args.arm or "baseline"
    world = cfg.get("world") or 4
    wd = args.workdir
    rng = random.Random(sum(ord(c) for c in arm) * 7919)
    base = _FAKE_BASE_US.get(arm, 250.0)
    iters, replays, calls = cfg.get("iters", 20), cfg.get("replays", 10), cfg.get("graph_calls", GRAPH_CALLS)
    agb = cfg.get("ag_bytes", AG_BYTES_PER_RANK)
    print("\n".join(fake_nccl_lines(arm, world)), flush=True)
    sys.stderr.write("fake child %s: synthetic numbers, no torch\n" % arm)
    for r in range(world):
        jit = 1.0 + 0.03 * r
        res = {"rank": r, "world": world, "fake": True, "device_name": "FAKE L4", "errors": {},
               "correct": {"pass": True, "expected": world * (world + 1) / 2, "min": world * (world + 1) / 2,
                           "max": world * (world + 1) / 2, "dtype": "bf16", "numel": GRAPH_BYTES // 2},
               "eager": {}}
        for label, nb in EAGER_SIZES:
            lat = (40.0 + nb / 3.2e3) * jit
            res["eager"][label] = {"bytes": nb, "samples_us": [lat * (1 + 0.05 * rng.random()) for _ in range(iters)],
                                   "loop_us_per_call": lat * 0.9}
        res["graph_983040"] = {"calls": calls, "capture_mode": cfg.get("capture_mode", "thread_local"),
                               "replay_us": [calls * base * jit * (1 + 0.02 * rng.random()) for _ in range(replays)],
                               "eager_loop_us_per_call": base * 1.15 * jit, "graph_correct": True}
        ag_us = (40.0 + agb * world / 3.2e3) * jit
        res["allgather_eager"] = {"bytes_per_rank": agb,
                                  "samples_us": [ag_us * (1 + 0.04 * rng.random()) for _ in range(iters)]}
        res["allgather_graph"] = {"bytes_per_rank": agb,
                                  "samples_us": [ag_us * 0.97 * (1 + 0.04 * rng.random()) for _ in range(iters)]}
        res["phases_done"] = ["init", "correct", "eager", "graph_983040", "allgather_eager", "allgather_graph"]
        write_json_atomic(os.path.join(wd, "rank_%d.json" % r), res)
    write_json_atomic(os.path.join(wd, "launcher.json"), {"arm": arm, "world": world, "error": None, "fake": True})
    return 0


# ============================================================================================ real children (torch)
def child_probe(args):
    out = {"fake": False}
    try:
        import torch  # noqa: lazy, child only
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        try:
            v = torch.cuda.nccl.version()
            out["nccl"] = ".".join(str(x) for x in v) if isinstance(v, (tuple, list)) else str(v)
        except Exception as e:
            out["nccl"] = "error: %r" % (e,)
        try:
            dv = torch._C._cuda_getDriverVersion()
            out["driver_version"] = dv
        except Exception:
            out["driver_version"] = None
        n = torch.cuda.device_count()
        out["device_count"] = n
        out["devices"] = []
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            out["devices"].append({"name": p.name, "capability": "%d.%d" % (p.major, p.minor),
                                   "total_mem_GiB": round(p.total_memory / 2 ** 30, 2),
                                   "sm_count": getattr(p, "multi_processor_count", None)})
        out["can_access_peer"] = [[(i != j) and bool(torch.cuda.can_device_access_peer(i, j)) for j in range(n)]
                                  for i in range(n)]
    except Exception as e:
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()[-2000:]
    write_json_atomic(args.child_out, out)
    return 0 if "error" not in out else 1


def child_run(args):
    """Launcher: spawns the ranks. Writes launcher.json (world, error)."""
    if args.child_fake:
        return fake_child_run(args)
    wd = args.workdir
    cfg = json.loads(args.cfg_json or "{}")
    info = {"arm": args.arm, "world": None, "error": None, "started": now_iso()}
    rc = 0
    try:
        import torch  # noqa: lazy, child only
        import torch.multiprocessing as mp
        n = torch.cuda.device_count()
        info["device_count"] = n
        world = cfg.get("world") or n
        info["world"] = world
        write_json_atomic(os.path.join(wd, "launcher.json"), info)
        if not cfg.get("world") and n != 4:
            raise RuntimeError("expected 4 GPUs, torch.cuda.device_count() = %d (pass --world to override)" % n)
        if world > n or world < 2:
            raise RuntimeError("world %d not possible with %d GPUs" % (world, n))
        port = free_port()
        info["port"] = port
        write_json_atomic(os.path.join(wd, "launcher.json"), info)
        mp.spawn(_rank_worker, args=(world, port, wd, cfg), nprocs=world, join=True)
    except Exception as e:
        info["error"] = repr(e)[:2000]
        info["traceback"] = traceback.format_exc()[-3000:]
        rc = 1
    info["finished"] = now_iso()
    write_json_atomic(os.path.join(wd, "launcher.json"), info)
    return rc


def _rank_worker(rank, world, port, wd, cfg):
    res = {"rank": rank, "world": world, "errors": {}, "phases_done": [], "phase_s": {}}
    path = os.path.join(wd, "rank_%d.json" % rank)

    def save():
        try:
            write_json_atomic(path, res)
        except Exception:
            pass

    dist = None
    try:
        import torch
        import torch.distributed as dist
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
        res["device_name"] = torch.cuda.get_device_name(rank)
        kw = dict(backend="nccl", init_method="tcp://127.0.0.1:%d" % port, rank=rank, world_size=world)
        t0 = time.monotonic()
        try:
            dist.init_process_group(device_id=dev, **kw)     # eager NCCL comm init (torch >= 2.3)
        except TypeError:
            dist.init_process_group(**kw)
        res["phase_s"]["init"] = time.monotonic() - t0
        res["phases_done"].append("init")
        save()
        ctx = _Ctx(torch, dist, rank, world, dev, cfg)
        for name, fn in (("correct", _ph_correct), ("eager", _ph_eager), ("graph_983040", _ph_graph),
                         ("allgather_eager", _ph_ag_eager), ("allgather_graph", _ph_ag_graph)):
            t0 = time.monotonic()
            try:
                res[name] = fn(ctx)
            except Exception as e:
                res["errors"][name] = (repr(e) + " | " + traceback.format_exc()[-1500:])[:2500]
            res["phase_s"][name] = time.monotonic() - t0
            res["phases_done"].append(name)
            save()
    except Exception as e:
        res["errors"]["init"] = (repr(e) + " | " + traceback.format_exc()[-1500:])[:2500]
    finally:
        save()
        try:
            if dist is not None and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


class _Ctx:
    def __init__(self, torch, dist, rank, world, dev, cfg):
        self.torch, self.dist, self.rank, self.world, self.dev, self.cfg = torch, dist, rank, world, dev, cfg
        self.small = torch.ones(1, dtype=torch.float32, device=dev)

    def align(self):
        """Line the ranks up before a timed region (tiny all-reduce + device sync)."""
        self.dist.all_reduce(self.small)
        self.torch.cuda.synchronize()

    def ev(self):
        return self.torch.cuda.Event(enable_timing=True)

    def time_eager(self, op, warmup, iters):
        torch = self.torch
        for _ in range(warmup):
            op()
        torch.cuda.synchronize()
        st, en = self.ev(), self.ev()
        samples = []
        for _ in range(iters):
            self.align()
            st.record()
            op()
            en.record()
            en.synchronize()
            samples.append(st.elapsed_time(en) * 1000.0)
        self.align()
        st.record()
        for _ in range(iters):
            op()
        en.record()
        en.synchronize()
        return {"samples_us": samples, "loop_us_per_call": st.elapsed_time(en) * 1000.0 / iters}

    def capture(self, graph, body):
        """Capture body() into graph on torch's side capture stream; NCCL comm already warmed eagerly."""
        torch = self.torch
        mode = self.cfg.get("capture_mode", "thread_local")
        try:
            ctxm = torch.cuda.graph(graph, capture_error_mode=mode)
        except TypeError:                     # very old torch: no capture_error_mode
            ctxm = torch.cuda.graph(graph)
            mode = "default"
        with ctxm:
            body()
        torch.cuda.synchronize()
        return mode


def _ph_correct(c):
    torch = c.torch
    n = GRAPH_BYTES // 2
    x = torch.full((n,), float(c.rank + 1), dtype=torch.bfloat16, device=c.dev)
    c.dist.all_reduce(x)
    torch.cuda.synchronize()
    exp = c.world * (c.world + 1) / 2.0
    mn, mx = float(x.min().item()), float(x.max().item())
    return {"pass": mn == exp and mx == exp, "expected": exp, "min": mn, "max": mx, "dtype": "bf16", "numel": n}


def _ph_eager(c):
    torch = c.torch
    out = {}
    for label, nb in EAGER_SIZES:
        x = torch.zeros(nb // 2, dtype=torch.bfloat16, device=c.dev)
        r = c.time_eager(lambda: c.dist.all_reduce(x), c.cfg.get("warmup", 5), c.cfg.get("iters", 20))
        r["bytes"] = nb
        out[label] = r
        del x
    torch.cuda.synchronize()
    return out


def _ph_graph(c):
    torch, dist = c.torch, c.dist
    calls = int(c.cfg.get("graph_calls", GRAPH_CALLS))
    replays = int(c.cfg.get("replays", 10))
    n = GRAPH_BYTES // 2
    buf = torch.zeros(n, dtype=torch.bfloat16, device=c.dev)      # zeros: 140 in-place sums stay 0
    # warm up on a side stream (torch docs: warm up before capture; NCCL comm + buffers exist before capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            dist.all_reduce(buf)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    c.align()
    g = torch.cuda.CUDAGraph()

    def body():
        for _ in range(calls):
            dist.all_reduce(buf)

    mode = c.capture(g, body)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    st, en = c.ev(), c.ev()
    rep = []
    for _ in range(replays):
        c.align()
        st.record()
        g.replay()
        en.record()
        en.synchronize()
        rep.append(st.elapsed_time(en) * 1000.0)
    # the same 140 calls launched eagerly back to back (CPU launch + PG overhead visible)
    c.align()
    st.record()
    for _ in range(calls):
        dist.all_reduce(buf)
    en.record()
    en.synchronize()
    eager_loop = st.elapsed_time(en) * 1000.0 / calls
    # correctness inside a graph: one captured all-reduce, input refilled before replay
    cb = torch.zeros(n, dtype=torch.bfloat16, device=c.dev)
    g1 = torch.cuda.CUDAGraph()
    c.capture(g1, lambda: dist.all_reduce(cb))
    cb.fill_(float(c.rank + 1))
    torch.cuda.synchronize()
    c.align()
    g1.replay()
    torch.cuda.synchronize()
    exp = c.world * (c.world + 1) / 2.0
    ok = float(cb.min().item()) == exp and float(cb.max().item()) == exp
    del g, g1
    return {"calls": calls, "capture_mode": mode, "replay_us": rep, "eager_loop_us_per_call": eager_loop,
            "graph_correct": ok}


def _ph_ag_eager(c):
    torch = c.torch
    nb = int(c.cfg.get("ag_bytes", AG_BYTES_PER_RANK))
    inp = torch.zeros(nb // 2, dtype=torch.bfloat16, device=c.dev)
    out = torch.empty(c.world * (nb // 2), dtype=torch.bfloat16, device=c.dev)
    r = c.time_eager(lambda: c.dist.all_gather_into_tensor(out, inp), c.cfg.get("warmup", 5), c.cfg.get("iters", 20))
    r["bytes_per_rank"] = nb
    return r


def _ph_ag_graph(c):
    torch, dist = c.torch, c.dist
    nb = int(c.cfg.get("ag_bytes", AG_BYTES_PER_RANK))
    inp = torch.zeros(nb // 2, dtype=torch.bfloat16, device=c.dev)
    out = torch.empty(c.world * (nb // 2), dtype=torch.bfloat16, device=c.dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            dist.all_gather_into_tensor(out, inp)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    c.align()
    g = torch.cuda.CUDAGraph()
    mode = c.capture(g, lambda: dist.all_gather_into_tensor(out, inp))
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    st, en = c.ev(), c.ev()
    samples = []
    for _ in range(int(c.cfg.get("iters", 20))):
        c.align()
        st.record()
        g.replay()
        en.record()
        en.synchronize()
        samples.append(st.elapsed_time(en) * 1000.0)
    del g
    return {"bytes_per_rank": nb, "samples_us": samples, "capture_mode": mode}


# ============================================================================================ entry
def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.child_probe:
        if args.child_fake:
            write_json_atomic(args.child_out, {"fake": True, "can_access_peer": [[False, True], [True, False]]})
            return 0
        return child_probe(args)
    if args.child_run:
        return child_run(args)
    if not args.out:
        ap.error("--out is required")
    res = driver_main(args, argv)
    s = res.get("summary") or {}
    print("e14_nccl_bench: wrote %s | best arm %s at %s us/call (graph, 983,040 B) | arms: %s" % (
        args.out, s.get("best_arm"), _fmt(s.get("best_per_call_us")),
        ", ".join("%s=%s" % (k, v.get("status")) for k, v in (res.get("arms") or {}).items())))
    return 0


def _fmt(x):
    return "n/a" if x is None else "%.1f" % x


if __name__ == "__main__":
    sys.exit(main())
