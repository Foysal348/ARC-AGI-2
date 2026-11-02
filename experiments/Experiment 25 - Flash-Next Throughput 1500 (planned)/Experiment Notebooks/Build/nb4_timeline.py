"""NB4 timeline analysis (E25, 2026-11-01): where does the stage-0 time of a PP2 tick go?

Input: the torch-profiler Chrome traces of the vLLM workers (one *.pt.trace.json[.gz] per rank; TP2 x PP2 = 4 ranks).
Stdlib only (json / gzip / re / bisect / statistics), so it runs inside the notebook kernel without extra packages.

Definitions (all times in microseconds inside, milliseconds in the report):
- Clock. Every file is put on one clock: ts + (baseTimeNanoseconds_file - baseTimeNanoseconds_ref) / 1000. The residual
  offset between ranks is then MEASURED from collectives that complete together on every member (TP AllReduce kernel
  ends for TP partners, PP Broadcast / SendRecv kernel ends across stages): the mode of the end-time differences, refined
  by their median. |offset| <= CLOCK_TOL_US -> SHARED_CLOCK (no correction); otherwise the offset is applied
  (CORRECTED_BY_COLLECTIVES) and the residual spread is reported.
- Kernel classes on a rank's GPU: compute = every non-NCCL kernel; copy = gpu_memcpy / gpu_memset; nccl_pp = NCCL kernels
  whose peer is the other pipeline stage (Broadcast / SendRecv / Send / Recv, or an eager kernel whose process group spans
  stages); nccl_tp = the other NCCL kernels (AllReduce / AllGather / ReduceScatter inside the stage).
- Tick. Stage-0 tick k = [GPU start of forward k, GPU start of forward k+1) on the reference rank (stage 0, TP 0). A
  forward = one execute_context_* annotation of the model runner (one micro-batch); its GPU start = the first kernel /
  copy launched (by CUPTI correlation) from a runtime call inside that annotation on its thread.
- Exclusive attribution of every instant of a tick on every rank (priority order, the categories sum to the tick):
    compute          a compute kernel runs;
    copy             only a memcpy / memset runs;
    in_graph_stall   inside a CUDA-graph replay's GPU span, nothing of the graph runs (no compute, copy or TP
                     collective): the graph's work was already submitted but is blocked (whatever else spins meanwhile,
                     e.g. a PP broadcast; the side measures split it by what the peer stage does);
    nccl_peer_wait   only NCCL runs, it is a PP kernel, and the other stage is computing (waiting for the peer);
    nccl_tp_skew     only NCCL runs, it is a TP kernel, and a TP partner is computing (rank skew);
    nccl_transfer    only NCCL runs and nobody it waits for is computing (the transfer itself / spin on a host-late peer);
    host_launch_gap  the GPU is idle and this rank's CPU threads are inside an op / CUDA launch / graph replay call;
    host_python_gap  the GPU is idle, no op is recorded, but the worker is inside the step annotation (Python between ops;
                     only resolvable with with_stack=True);
    idle_no_host     the GPU is idle and the worker is outside any step and op: waiting for the next step from the
                     engine core (scheduler / IPC) or blocked in a CPU-side comm (gloo) wait.
  Each in-graph stall > STALL_MIN_US is matched to the NCCL kernel whose end released it (latest end within
  RELEASE_WINDOW_US before the stall end), with that collective's Seq and the step that posted it.
- Graph mode of a forward (rev 2, agent D's source research): EMPTY = a 0-token step (generation_0(0), no context
  tokens); EAGER = no cudaGraphLaunch in the step annotation; FULL = exactly one graph launch with >= FULL_MIN_NODES
  kernel nodes; PIECEWISE = several graph launches; SMALL_GRAPH = one launch below FULL_MIN_NODES. Eager kernels around
  the graph (RecoverSSM commit, input prep, the PP send) do not change the mode; they are counted (compute_eager).
  The lab rule (e23_lab rev 3h _scope_kind: any eager model-compute launch -> PIECEWISE) is kept only inside
  nb1_reference() so NB1's verdict is reproduced.
- Micro-batch parity (pp_size = number of stages): step k on a stage has parity k % pp_size. vLLM V2's PPHandler
  consumes the sampled-token receive of step k at step k + pp_size (pp_utils.py ring), so step k's real dependency is
  the receive posted by step k - pp_size; a stall released by any other step's collective is a false dependency.
- NB1 numbers (nb1_reference) are recomputed with NB1's own code path (build_e25_nb1.microbatch_costs + the lab's
  trace_summary categories with NB1's A7 classifier order), so they are directly comparable.
"""
import bisect
import gzip
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLOCK_TOL_US = 30.0
CLOCK_SEARCH_US = 5000.0
STALL_MIN_US = 1000.0
RELEASE_WINDOW_US = 200.0
FULL_MIN_NODES = 500
GPU_CATS = ("kernel", "gpu_kernel", "gpu_memcpy", "gpu_memset")
KERNEL_CATS = ("kernel", "gpu_kernel")
RUNTIME_CATS = ("cuda_runtime", "cuda_driver")
HOST_CATS = ("cpu_op", "python_function", "cuda_runtime", "cuda_driver")
# runtime calls that poll / wait / query: the host is not preparing or launching work while inside them
HOST_NOT_LAUNCH = re.compile(r"Synchronize|Query|StreamIsCapturing|GetCaptureInfo|ExchangeStreamCaptureMode|"
                             r"PointerGetAttributes|GetDevice|GetLastError|PeekAtLastError")
LAUNCH_RE = re.compile(r"LaunchKernel|cuLaunchKernel")
SCOPE_RE = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
PP_KERNEL_RE = re.compile(r"Broadcast|SendRecv|ncclDevKernel_Send|ncclDevKernel_Recv", re.I)
SIDE = ("in_graph_stall_peer_computing", "in_graph_stall_pp_nccl_running", "kernel_union", "nccl_transfer_pp", "nccl_transfer_tp", "nccl_transfer_peer_in_nccl",
        "nccl_transfer_peer_host", "nccl_transfer_peer_idle")
CATEGORIES = ("compute", "copy", "in_graph_stall", "nccl_peer_wait", "nccl_tp_skew", "nccl_transfer", "host_launch_gap",
              "host_python_gap", "idle_no_host")

# ---- NB1's kernel classifier (lab rev 3h CATS + NB1's A7 prefix), copied so the numbers are comparable
LAB_CATS = [("nccl", ("nccl", "allreduce", "all_reduce", "allgather", "all_gather")),
            ("attention", ("flashinfer", "flash_fwd", "unified_attention", "paged", "attn", "attention", "batchdecode",
                           "batchprefill", "reshape_and_cache")),
            ("gdn_linear_attn", ("gdn", "delta_rule", "chunk_", "recurrent", "causal_conv", "l2norm", "solve_tril",
                                 "merge_16x16", "recompute_w_u", "kkt", "cumsum", "post_conv", "layer_norm_gated",
                                 "gated_delta")),
            ("gemm_marlin", ("marlin",)),
            ("gemm_other", ("gemm", "cutlass", "xmma", "cublas", "sm89_", "ampere_", "sm80_")),
            ("sampling", ("sampl", "topk", "top_k", "argmax", "softmax", "rejection"))]
NB1_CATS = ([("nccl_sendrecv", ("sendrecv", "send_recv", "nccldevkernel_send", "nccldevkernel_recv")), LAB_CATS[0],
             ("recoverssm_commit", ("recoverssm", "commit_", "boundary_fold", "final_fold", "conv_compaction")),
             ("qsa", ("qsa",)), ("moe_marlin", ("marlin_moe", "fused_marlin_moe", "moe_wna16")),
             ("topk", ("persistent_topk", "cooperative_topk"))] + LAB_CATS[1:])
NB1_COMPUTE_CATS = ("nccl", "attention", "gdn_linear_attn", "gemm_marlin", "gemm_other", "nccl_sendrecv",
                    "recoverssm_commit", "qsa", "moe_marlin", "topk")


def kernel_base(name):
    s = re.sub(r"\(anonymous namespace\)::", "", str(name))
    s = re.sub(r"^\s*void\s+", "", s)
    return s.split("<")[0].split("(")[0]


def nb1_cat(name):
    base = kernel_base(name).lower()
    return next((k for k, pats in NB1_CATS if any(p in base for p in pats)), "other")


# ------------------------------------------------------------------ interval algebra (sorted, disjoint [a, b] lists)
def merge(iv):
    out = []
    for a, b in sorted(iv):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1][1] = b
        else:
            out.append([a, b])
    return out


def total(iv):
    return sum(b - a for a, b in iv)


def clip(m, lo, hi, starts=None):
    """m merged; returns the part of m inside [lo, hi]. `starts` = [a for a, _ in m] for a bisect start."""
    if not m or hi <= lo:
        return []
    i = bisect.bisect_right(starts, lo) - 1 if starts is not None else 0
    i = max(i, 0)
    out = []
    while i < len(m) and m[i][0] < hi:
        a, b = m[i]
        if b > lo:
            out.append([max(a, lo), min(b, hi)])
        i += 1
    return out


def inter(x, y):
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        lo, hi = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if hi > lo:
            out.append([lo, hi])
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def sub(x, y):
    """x minus y (both merged)."""
    out = []
    j = 0
    for a, b in x:
        cur = a
        while j < len(y) and y[j][1] <= cur:
            j += 1
        k = j
        while k < len(y) and y[k][0] < b:
            if y[k][0] > cur:
                out.append([cur, y[k][0]])
            cur = max(cur, y[k][1])
            if cur >= b:
                break
            k += 1
        if cur < b:
            out.append([cur, b])
    return out


def union(*lists):
    return merge([list(v) for m in lists for v in m])


class Ivs:
    """A merged interval list with bisect clipping."""

    def __init__(self, iv):
        self.m = merge(iv)
        self.starts = [a for a, _ in self.m]

    def clip(self, lo, hi):
        return clip(self.m, lo, hi, self.starts)


# ------------------------------------------------------------------ loading
def _open(path):
    path = Path(path)
    with (gzip.open if path.name.endswith(".gz") else open)(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _base_ns(tr):
    try:
        return int(tr.get("baseTimeNanoseconds") or 0)
    except (TypeError, ValueError):
        return 0


def rank_identity(name, tr, k=0, tp=2):
    """rank, pp stage, tp index: vLLM's worker file name (dp0_pp1_tp0_..._rank2) first, then distributedInfo."""
    m_pp, m_tp = re.search(r"pp(\d+)", name), re.search(r"_tp(\d+)", name)
    m_r = re.search(r"rank[-_]?(\d+)", name)
    di = tr.get("distributedInfo") or {}
    if isinstance(di, str):
        di = {}
    rank = di.get("rank") if isinstance(di.get("rank"), int) else (int(m_r.group(1)) if m_r else k)
    pp = int(m_pp.group(1)) if m_pp else rank // tp
    tpi = int(m_tp.group(1)) if m_tp else rank % tp
    return rank, pp, tpi


def load_traces(src, tp=2):
    """src: a directory (rglob *.pt.trace.json*), one file, or a list of files / already-loaded trace dicts.
    Returns {rank: {"name", "rank", "pp", "tp", "base_ns", "trace"}}."""
    items = []
    if isinstance(src, (list, tuple)):
        for k, x in enumerate(src):
            if isinstance(x, dict):
                items.append((x.get("_name", f"trace{k}"), x))
            else:
                items.append((Path(x).name, _open(x)))
    else:
        p = Path(src)
        files = sorted(p.rglob("*.pt.trace.json*")) if p.is_dir() else [p]
        items = [(f.name, _open(f)) for f in files]
    out = {}
    for k, (name, tr) in enumerate(items):
        rank, pp, tpi = rank_identity(name, tr, k, tp)
        out[rank] = {"name": name, "rank": rank, "pp": pp, "tp": tpi, "base_ns": _base_ns(tr), "trace": tr}
    return out


# ------------------------------------------------------------------ per-rank timeline
def _args(e):
    a = e.get("args")
    return a if isinstance(a, dict) else {}


def _group_ranks(a):
    g = a.get("Process Group Ranks")
    if isinstance(g, str):
        try:
            g = json.loads(g)
        except ValueError:
            return None
    return g if isinstance(g, list) else None


def build_rank(rec, stage_of, shift_us=0.0):
    """One rank's timeline on the common clock (ts + shift_us)."""
    ev = rec["trace"].get("traceEvents") or []
    X = [e for e in ev if e.get("ph") == "X" and "ts" in e]
    sh = float(shift_us)
    rt_by_corr, rt_by_tid = {}, defaultdict(list)
    for e in X:
        if str(e.get("cat", "")).lower() in RUNTIME_CATS:
            c = _args(e).get("correlation")
            if c is not None:
                rt_by_corr[c] = e
                rt_by_tid[e.get("tid")].append(e)
    for v in rt_by_tid.values():
        v.sort(key=lambda e: e["ts"])
    rt_ts = {t: [e["ts"] for e in v] for t, v in rt_by_tid.items()}
    gpu = sorted((e for e in X if str(e.get("cat", "")).lower() in GPU_CATS), key=lambda e: e["ts"])
    comp, copy, ncpp, nctp, nccl_k = [], [], [], [], []
    by_corr = defaultdict(list)
    for e in gpu:
        a, b = e["ts"] + sh, e["ts"] + sh + float(e.get("dur", 0.0))
        cat = str(e.get("cat", "")).lower()
        name = str(e.get("name", ""))
        c = _args(e).get("correlation")
        if c is not None:
            by_corr[c].append(e)
        if cat in ("gpu_memcpy", "gpu_memset"):
            copy.append([a, b])
        elif "nccl" in name.lower():
            ga = _args(e)
            grp = _group_ranks(ga)
            if grp is not None and len({stage_of.get(r) for r in grp}) > 1:
                pp_k = True
            elif grp is not None:
                pp_k = False
            else:
                pp_k = PP_KERNEL_RE.search(name) is not None
            (ncpp if pp_k else nctp).append([a, b])
            nccl_k.append({"a": a, "b": b, "pp": pp_k, "name": kernel_base(name), "coll": ga.get("Collective name"),
                           "group": ga.get("Process Group Ranks"), "dtype": ga.get("dtype"),
                           "nelems": ga.get("In msg nelems"), "seq": ga.get("Seq"),
                           "pg": ga.get("Process Group Description")})
        else:
            comp.append([a, b])
    # graph replay GPU spans
    gspans = []
    for c, ks in by_corr.items():
        l = rt_by_corr.get(c)
        if l is not None and "GraphLaunch" in str(l.get("name", "")):
            gspans.append([min(k["ts"] for k in ks) + sh, max(k["ts"] + float(k.get("dur", 0.0)) for k in ks) + sh])
    # host activity (CPU threads of the worker): ops and launch calls, not polling / sync
    host, scopes, posts = [], [], []
    for e in X:
        cat = str(e.get("cat", "")).lower()
        if cat == "cpu_op" and e.get("name") == "record_param_comms":
            a = _args(e)
            if a.get("Collective name") not in (None, "wait"):
                posts.append({"ts": e["ts"] + sh, "coll": a.get("Collective name"), "dtype": a.get("dtype"),
                              "nelems": a.get("In msg nelems"), "seq": a.get("Seq"),
                              "pg": a.get("Process Group Description"), "group": a.get("Process Group Ranks")})
        if cat in HOST_CATS:
            if cat in RUNTIME_CATS and HOST_NOT_LAUNCH.search(str(e.get("name", ""))):
                continue
            host.append([e["ts"] + sh, e["ts"] + sh + float(e.get("dur", 0.0))])
        elif cat == "user_annotation" and SCOPE_RE.match(str(e.get("name", ""))):
            scopes.append(e)
    scopes.sort(key=lambda e: e["ts"])
    fwds = []
    for s in scopes:
        lo, hi, tid = s["ts"], s["ts"] + float(s.get("dur", 0.0)), s.get("tid")
        ts_l = rt_ts.get(tid, [])
        i0, i1 = bisect.bisect_left(ts_l, lo), bisect.bisect_right(ts_l, hi)
        g = kl = ce = un = 0
        gnodes, gspan = [], []
        aux = Counter()
        ks_all, cats = [], defaultdict(float)
        for r in rt_by_tid.get(tid, [])[i0:i1]:
            nm = str(r.get("name", ""))
            c = _args(r).get("correlation")
            ks = by_corr.get(c, [])
            if "GraphLaunch" in nm:
                g += 1
                gk = [k for k in ks if str(k.get("cat", "")).lower() in KERNEL_CATS]
                gnodes.append(len(gk))
                gspan.extend([k["ts"] + sh, k["ts"] + sh + float(k.get("dur", 0.0))] for k in gk)
            elif LAUNCH_RE.search(nm):
                kl += 1
                kn = next((str(k.get("name", "")) for k in ks if str(k.get("cat", "")).lower() in KERNEL_CATS), None)
                if kn is None:
                    un += 1
                elif nb1_cat(kn) in NB1_COMPUTE_CATS:
                    ce += 1
                else:
                    aux[kernel_base(kn)[:60]] += 1
            ks_all.extend(ks)
            for k in ks:
                if str(k.get("cat", "")).lower() in KERNEL_CATS:
                    cats[nb1_cat(k.get("name", ""))] += float(k.get("dur", 0.0))
        kiv = [[k["ts"] + sh, k["ts"] + sh + float(k.get("dur", 0.0))] for k in ks_all]
        kk = [[k["ts"] + sh, k["ts"] + sh + float(k.get("dur", 0.0))] for k in ks_all
              if str(k.get("cat", "")).lower() in KERNEL_CATS]
        mk = merge(kk)
        mm = re.match(SCOPE_RE, str(s.get("name", "")))
        fwds.append({"name": s.get("name"), "cpu_ts": s["ts"] + sh, "cpu_dur": float(s.get("dur", 0.0)),
                     "ctx_requests": int(mm.group(1)), "ctx_tokens": int(mm.group(2)), "gen_requests": int(mm.group(3)),
                     "gen_tokens": int(mm.group(4)), "graph_nodes": gnodes,
                     "graph_span": [min(a for a, _ in gspan), max(b for _, b in gspan)] if gspan else None,
                     "gpu_start": min((a for a, _ in kiv), default=None), "gpu_end": max((b for _, b in kiv), default=None),
                     "kernel_wall_us": (mk[-1][1] - mk[0][0]) if mk else 0.0, "kernel_union_us": total(mk),
                     "kernel_sum_us": sum(cats.values()), "kernel_us_by_cat": dict(cats),
                     "graph_launches": g, "kernel_launches": kl, "compute_eager": ce, "unresolved_eager": un,
                     "mode": graph_mode(gnodes, int(mm.group(2)) + int(mm.group(4))),
                     "mode_lab_rule": scope_kind(g, ce, un), "aux_eager": dict(aux)})
    # collectives posted by each step: host calls from this step's annotation start to the next one's
    sts = [f["cpu_ts"] for f in fwds]
    for f in fwds:
        f["posts"] = []
    for p_ in sorted(posts, key=lambda d: d["ts"]):
        i = bisect.bisect_right(sts, p_["ts"]) - 1
        if i >= 0:
            fwds[i]["posts"].append(p_)
    return {"rank": rec["rank"], "pp": rec["pp"], "tp": rec["tp"], "name": rec["name"],
            "compute": Ivs(comp), "copy": Ivs(copy), "nccl_pp": Ivs(ncpp), "nccl_tp": Ivs(nctp),
            "nccl_kernels": sorted(nccl_k, key=lambda d: d["b"]), "graph_spans": Ivs(gspans),
            "host": Ivs(host), "scope": Ivs([[s["ts"] + sh, s["ts"] + sh + float(s.get("dur", 0.0))] for s in scopes]),
            "forwards": fwds, "posts": posts,
            "span": (gpu[0]["ts"] + sh, max(e["ts"] + float(e.get("dur", 0.0)) for e in gpu) + sh) if gpu else None}


def graph_mode(graph_nodes, tokens, full_min=None):
    """rev 2 rule: EMPTY (0-token step) / EAGER (no graph) / PIECEWISE (several graph launches) / FULL (one launch of
    >= full_min kernel nodes, default FULL_MIN_NODES) / SMALL_GRAPH (one launch below full_min)."""
    full_min = FULL_MIN_NODES if full_min is None else full_min
    if tokens == 0:
        return "EMPTY"
    if not graph_nodes:
        return "EAGER"
    if len(graph_nodes) > 1:
        return "PIECEWISE"
    return "FULL" if graph_nodes[0] >= full_min else "SMALL_GRAPH"


def scope_kind(g, compute_eager, unresolved):
    """The lab rule (e23_lab rev 3h): FULL = graph-launched, no eager model compute inside the execution scope."""
    if g == 0:
        return "EAGER"
    if compute_eager:
        return "PIECEWISE"
    if unresolved:
        return "PIECEWISE" if g > 1 else "MIXED"
    return "FULL"


# ------------------------------------------------------------------ clock
def _collective_ends(rec, kind):
    ev = rec["trace"].get("traceEvents") or []
    pat = {"tp": re.compile(r"AllReduce|AllGather|ReduceScatter"), "pp": PP_KERNEL_RE}[kind]
    return sorted(e["ts"] + float(e.get("dur", 0.0)) for e in ev
                  if e.get("ph") == "X" and str(e.get("cat", "")).lower() in KERNEL_CATS
                  and "nccl" in str(e.get("name", "")).lower() and pat.search(str(e.get("name", ""))))


def pair_offset(ea, eb, search_us=CLOCK_SEARCH_US, cap=600):
    """Clock offset of b vs a from collective end times that complete together: mode of all differences within
    +-search_us (bin 5 us), refined by the median of the differences within +-25 us of the mode."""
    if not ea or not eb:
        return None
    ea = ea[:cap] if len(ea) > cap else ea
    diffs = []
    for x in ea:
        i0, i1 = bisect.bisect_left(eb, x - search_us), bisect.bisect_right(eb, x + search_us)
        diffs.extend(y - x for y in eb[i0:i1])
    if not diffs:
        return None
    hist = Counter(round(d / 5.0) for d in diffs)
    mode = hist.most_common(1)[0][0] * 5.0
    near = sorted(d for d in diffs if abs(d - mode) <= 25.0)
    if not near:
        return None
    # per-a nearest difference to the refined offset: the residual spread
    off = statistics.median(near)
    res = []
    for x in ea:
        i = bisect.bisect_left(eb, x + off)
        c = [eb[k] - x - off for k in (i - 1, i) if 0 <= k < len(eb)]
        if c:
            res.append(min(c, key=abs))
    res.sort()
    return {"offset_us": round(off, 2), "matched": len(near), "candidates": len(ea),
            "residual_p10_us": round(res[len(res) // 10], 2) if res else None,
            "residual_p50_us": round(res[len(res) // 2], 2) if res else None,
            "residual_p90_us": round(res[9 * len(res) // 10], 2) if res else None}


def align_clocks(ranks, tol_us=CLOCK_TOL_US):
    """Shifts (us) that put every rank on the reference rank's clock. Step 1: baseTimeNanoseconds differences.
    Step 2: measured residual offsets from collectives (TP partners: AllReduce ends; across stages: PP kernel ends),
    propagated from the lowest rank by BFS. Offsets within tol_us are reported but not applied."""
    order = sorted(ranks)
    ref = order[0]
    base_shift = {r: (ranks[r]["base_ns"] - ranks[ref]["base_ns"]) / 1000.0 for r in order}
    ends = {}
    for r in order:
        ends[r] = {k: [t + base_shift[r] for t in _collective_ends(ranks[r], k)] for k in ("tp", "pp")}
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            kind = "tp" if ranks[a]["pp"] == ranks[b]["pp"] else "pp"
            if kind == "pp" and ranks[a]["tp"] != ranks[b]["tp"]:
                continue                       # PP peers are the same TP index (PP group = {tp i of each stage})
            po = pair_offset(ends[a][kind], ends[b][kind])
            if po:
                pairs.append(dict(po, a=a, b=b, kind=kind))
    shift = {ref: 0.0}
    frontier = [ref]
    while frontier:
        nxt = []
        for a in frontier:
            for p in pairs:
                for x, y, sgn in ((p["a"], p["b"], 1), (p["b"], p["a"], -1)):
                    if x == a and y not in shift:
                        shift[y] = shift[a] + sgn * p["offset_us"]
                        nxt.append(y)
        frontier = nxt
    resid = {r: shift.get(r, 0.0) for r in order}
    applied = any(abs(v) > tol_us for v in resid.values())
    total_shift = {r: base_shift[r] - (resid[r] if applied else 0.0) for r in order}
    method = ("CORRECTED_BY_COLLECTIVES" if applied else "SHARED_CLOCK") + \
             ("" if len(shift) == len(order) else " (some ranks without a matched collective: base shift only)")
    return {"method": method, "reference_rank": ref, "tolerance_us": tol_us,
            "base_ns": {r: ranks[r]["base_ns"] for r in order}, "base_shift_us": base_shift,
            "measured_offset_us": resid, "applied_shift_us": total_shift, "pairs": pairs,
            "note": "offset = median end-time difference of collectives that complete together on all members; "
                    "applied only when above the tolerance"}


# ------------------------------------------------------------------ attribution
def attribute(tl, lo, hi, peer_compute, partner_compute, peer_nccl=(), peer_host=()):
    """Exclusive split of [lo, hi] on one rank. peer_compute / partner_compute: merged compute of the other stage /
    of the TP partners, already clipped to [lo, hi]. peer_nccl / peer_host: the PP peer ranks' NCCL kernels and host
    activity (clipped), used only for the side split of nccl_transfer (what the peer does meanwhile)."""
    win = [[lo, hi]]
    c = tl["compute"].clip(lo, hi)
    cp = sub(tl["copy"].clip(lo, hi), c)
    busy = union(c, cp)
    tpc = tl["nccl_tp"].clip(lo, hi)
    ppc = tl["nccl_pp"].clip(lo, hi)
    stall = inter(sub(win, union(busy, tpc)), tl["graph_spans"].clip(lo, hi))
    npp = sub(sub(ppc, busy), stall)
    ntp = sub(sub(sub(tpc, busy), npp), stall)
    nccl = union(npp, ntp)
    peer_wait = inter(npp, peer_compute)
    skew = inter(ntp, partner_compute)
    transfer = sub(nccl, union(peer_wait, skew))
    idle = sub(win, union(busy, nccl, stall))
    h = tl["host"].clip(lo, hi)
    launch = inter(idle, h)
    rest = sub(idle, launch)
    py = inter(rest, tl["scope"].clip(lo, hi))
    none = sub(rest, py)
    out = {"compute": total(c), "copy": total(cp), "in_graph_stall": total(stall), "nccl_peer_wait": total(peer_wait), "nccl_tp_skew": total(skew),
           "nccl_transfer": total(transfer), "host_launch_gap": total(launch), "host_python_gap": total(py),
           "idle_no_host": total(none)}
    out["_stall"] = stall
    out["in_graph_stall_peer_computing"] = total(inter(stall, peer_compute))
    out["in_graph_stall_pp_nccl_running"] = total(inter(stall, ppc))
    out["kernel_union"] = total(union(c, ppc, tpc))
    # side split of nccl_transfer: PP vs TP kernel, and the PP peer's state meanwhile (in NCCL too = a communication
    # bubble on both stages; on the host = the peer's host is late; neither = the peer is idle)
    out["nccl_transfer_pp"] = total(inter(transfer, npp))
    out["nccl_transfer_tp"] = total(transfer) - out["nccl_transfer_pp"]
    t_peer_nccl = inter(transfer, merge([list(v) for v in peer_nccl]))
    rest_t = sub(transfer, t_peer_nccl)
    t_peer_host = inter(rest_t, merge([list(v) for v in peer_host]))
    out["nccl_transfer_peer_in_nccl"] = total(t_peer_nccl)
    out["nccl_transfer_peer_host"] = total(t_peer_host)
    out["nccl_transfer_peer_idle"] = total(rest_t) - total(t_peer_host)
    return out


def stall_release(tl, stalls, min_us=STALL_MIN_US, window_us=RELEASE_WINDOW_US):
    """For each stall interval >= min_us: the NCCL kernel on this rank whose end is the latest within window_us
    before the stall end (the one that released the blocked graph work)."""
    ends = [k["b"] for k in tl["nccl_kernels"]]
    out = []
    for a, b in stalls:
        if b - a < min_us:
            continue
        i = bisect.bisect_right(ends, b)
        cand = [tl["nccl_kernels"][j] for j in range(max(0, i - 8), i) if b - window_us <= ends[j] <= b]
        k = max(cand, key=lambda d: d["b"]) if cand else None
        lbl = None
        if k:
            lbl = f"{k['name']}" + (f" {k['coll']} {k['dtype']}x{k['nelems']} group {k['group']}" if k["coll"] else "")
        out.append({"stall": [a, b], "stall_us": b - a, "released_by": lbl, "lag_us": (b - k["b"]) if k else None,
                    "release_kernel_us": (k["b"] - k["a"]) if k else None, "seq": k.get("seq") if k else None,
                    "pg": k.get("pg") if k else None})
    return out


def _ms(x):
    return round(x / 1000.0, 3)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def analyze(src, tp=2, ref_rank=None, max_ticks=10000, tol_us=CLOCK_TOL_US):
    """The full NB4 tick analysis. src: see load_traces (or the dict load_traces returned). Returns a JSON-able dict:
    clock, cpu_activity, ticks (per tick: categories per stage, graph mode, critical path, stage-1 idle, stall),
    summary (steady-tick means / medians), stalls, nb1_reference."""
    ranks = src if (isinstance(src, dict) and src and all(isinstance(v, dict) and "trace" in v for v in src.values())) \
        else load_traces(src, tp)
    if not ranks:
        return {"error": "NO_TRACE_FILE"}
    clock = align_clocks(ranks, tol_us)
    stage_of = {r: ranks[r]["pp"] for r in ranks}
    tls = {r: build_rank(ranks[r], stage_of, clock["applied_shift_us"][r]) for r in ranks}
    stages = sorted({tls[r]["pp"] for r in tls})
    s0 = stages[0]
    if ref_rank is None:
        ref_rank = min(r for r in tls if tls[r]["pp"] == s0)
    ref = tls[ref_rank]
    fw = [f for f in ref["forwards"] if f["gpu_start"] is not None]
    fw.sort(key=lambda f: f["gpu_start"])
    starts = [f["gpu_start"] for f in fw]
    ticks = []
    for k in range(min(len(fw) - 1, max_ticks)):
        lo, hi = starts[k], starts[k + 1]
        comp_by_rank = {r: tls[r]["compute"].clip(lo, hi) for r in tls}
        stage_comp = {s: union(*[comp_by_rank[r] for r in tls if tls[r]["pp"] == s]) for s in stages}
        per_rank = {}
        for r, tl in tls.items():
            peer = union(*[stage_comp[s] for s in stages if s != tl["pp"]]) if len(stages) > 1 else []
            partner = union(*[comp_by_rank[q] for q in tls if q != r and tls[q]["pp"] == tl["pp"]])
            pr = [q for q in tls if tls[q]["pp"] != tl["pp"] and tls[q]["tp"] == tl["tp"]]
            pn = union(*[union(tls[q]["nccl_pp"].clip(lo, hi), tls[q]["nccl_tp"].clip(lo, hi)) for q in pr])
            ph = union(*[tls[q]["host"].clip(lo, hi) for q in pr])
            per_rank[r] = attribute(tl, lo, hi, peer, partner, pn, ph)
        row = {"tick": k, "t0_ms": _ms(lo - starts[0]), "tick_ms": _ms(hi - lo), "forward": fw[k]["name"],
               "graph_mode": fw[k]["mode"],
               "graph_mode_by_rank": {}, "stages": {},
               "by_rank": {r: {c: _ms(per_rank[r][c]) for c in CATEGORIES + SIDE} for r in sorted(per_rank)}}
        for s in stages:
            rs = [r for r in tls if tls[r]["pp"] == s]
            row["stages"][f"stage{s}"] = {c: _ms(statistics.mean(per_rank[r][c] for r in rs))
                                          for c in CATEGORIES + SIDE}
            row["stages"][f"stage{s}"]["gpu_busy_stage_union"] = _ms(total(stage_comp[s]))
        for r in tls:
            f2 = [f for f in tls[r]["forwards"] if f["gpu_start"] is not None and lo <= f["gpu_start"] < hi]
            row["graph_mode_by_rank"][r] = [f["mode"] for f in f2]
        if len(stages) > 1:
            a, b = stage_comp[stages[0]], union(*[stage_comp[s] for s in stages[1:]])
            both = total(inter(a, b))
            row["pipeline"] = {"stage0_only_ms": _ms(total(a) - both), "other_stage_only_ms": _ms(total(b) - both),
                               "both_busy_ms": _ms(both), "neither_ms": _ms((hi - lo) - total(union(a, b)))}
            s1 = f"stage{stages[1]}"
            row["stage1_idle_ms"] = round(row["tick_ms"] - row["stages"][s1]["compute"] - row["stages"][s1]["copy"], 3)
            ideal = max(row["stages"][f"stage{s}"]["compute"] + row["stages"][f"stage{s}"]["copy"] for s in stages)
        else:
            ideal = row["stages"][f"stage{s0}"]["compute"]
        row["critical_path"] = {"tick_ms": row["tick_ms"], "forward_kernel_wall_ms": _ms(fw[k]["kernel_wall_us"]),
                                "forward_kernel_union_ms": _ms(fw[k]["kernel_union_us"]),
                                "forward_kernel_sum_ms": _ms(fw[k]["kernel_sum_us"]),
                                "ideal_overlapped_tick_ms": round(ideal, 3),
                                "serialization_ms": round(row["tick_ms"] - ideal, 3)}
        rel = stall_release(ref, per_rank[ref_rank]["_stall"])
        row["stall"] = {"in_graph_stall_ms_ref_rank": _ms(per_rank[ref_rank]["in_graph_stall"]),
                        "stalls_over_1ms": len(rel),
                        "released_by": [x["released_by"] for x in rel], "lag_us": [round(x["lag_us"], 1) if x["lag_us"]
                                                                                   is not None else None for x in rel]}
        ticks.append(row)
    summ = summarize(ticks, stages)
    if len(stages) > 1 and summ.get("steady_ticks"):
        summ["ideal_tick"] = ideal_tick(summ, stages)
    return {"clock": clock, "cpu_activity": {r: cpu_activity(ranks[r]) for r in sorted(ranks)},
            "reference_rank": ref_rank, "stages": stages, "forwards_ref_rank": len(fw),
            "graph_modes": {r: dict(Counter(f["mode"] for f in tls[r]["forwards"])) for r in sorted(tls)},
            "graph_nodes": {r: {str(k): v for k, v in Counter(tuple(f["graph_nodes"]) for f in
                                                              tls[r]["forwards"]).most_common(4)} for r in sorted(tls)},
            "ticks": ticks, "summary": summ, "stalls": stall_summary(ticks),
            "parity": microbatch_parity(tls, stages), "dependency": dependency_check(tls, ref_rank, len(stages)),
            "nb1_reference": nb1_reference(ranks, tp, tls),
            "definitions": __doc__.split("Definitions")[1].strip()[:6000]}


def steady_ticks(ticks):
    """Drop the first and last tick and ticks outside 0.5-1.5x the median tick (profiler start / stop edges)."""
    if len(ticks) <= 2:
        return list(ticks)
    mid = statistics.median(t["tick_ms"] for t in ticks)
    return [t for t in ticks[1:-1] if 0.5 * mid <= t["tick_ms"] <= 1.5 * mid]


def summarize(ticks, stages):
    st = steady_ticks(ticks)
    if not st:
        return {"steady_ticks": 0}
    out = {"steady_ticks": len(st), "all_ticks": len(ticks), "tick_ms_mean": _mean([t["tick_ms"] for t in st]),
           "tick_ms_median": _median([t["tick_ms"] for t in st]),
           "graph_modes": dict(Counter(t["graph_mode"] for t in st))}
    for s in stages:
        key = f"stage{s}"
        out[key] = {c: {"mean_ms": _mean([t["stages"][key][c] for t in st]),
                        "median_ms": _median([t["stages"][key][c] for t in st])}
                    for c in CATEGORIES + SIDE + ("gpu_busy_stage_union",)}
        tm = out["tick_ms_mean"]
        out[key]["share_of_tick_pct"] = {c: round(100 * out[key][c]["mean_ms"] / tm, 1) if tm else None
                                         for c in CATEGORIES}
        out[key]["gap_ms_mean"] = round(tm - out[key]["compute"]["mean_ms"] - out[key]["copy"]["mean_ms"], 3)
    if len(stages) > 1:
        out["stage1_idle_ms_mean"] = _mean([t.get("stage1_idle_ms") for t in st])
        out["pipeline"] = {k: _mean([t["pipeline"][k] for t in st]) for k in st[0]["pipeline"]}
    out["critical_path"] = {k: _mean([t["critical_path"][k] for t in st]) for k in st[0]["critical_path"]}
    return out


def stall_summary(ticks):
    st = steady_ticks(ticks)
    rel = Counter(x for t in st for x in t["stall"]["released_by"])
    lags = [x for t in st for x in t["stall"]["lag_us"] if x is not None]
    return {"ticks_with_stall_over_1ms": sum(1 for t in st if t["stall"]["stalls_over_1ms"]),
            "steady_ticks": len(st), "in_graph_stall_ms_mean": _mean([t["stall"]["in_graph_stall_ms_ref_rank"] for t in st]),
            "released_by": dict(rel.most_common(6)), "release_lag_us_median": _median(lags),
            "release_lag_us_p90": round(sorted(lags)[9 * len(lags) // 10], 1) if lags else None}


def ideal_tick(summ, stages):
    """Tick bounds from the measured per-step stage work (compute + copy per stage-0 tick, steady means).
    balanced_parity_ideal = max(slowest stage, per-micro-batch chain / pp): with pp micro-batches of equal size and no
    false dependency, a stage can overlap its next micro-batch with the other stage's work, but one micro-batch's own
    chain (stage 0 -> stage 1 -> broadcast -> stage 0) still spans all stages once per pp steps."""
    pp = len(stages)
    busy = {f"stage{s}": round(summ[f"stage{s}"]["compute"]["mean_ms"] + summ[f"stage{s}"]["copy"]["mean_ms"], 3)
            for s in stages}
    tot = sum(busy.values())
    slow = max(busy, key=busy.get)
    neither = (summ.get("pipeline") or {}).get("neither_ms") or 0.0
    chain = tot / pp
    ideal = max(busy[slow], chain)
    return {"per_stage_busy_ms": busy, "observed_tick_ms": summ["tick_ms_mean"],
            "balanced_parity_ideal_ms": round(ideal, 3),
            "bound_by": f"{slow} work" if busy[slow] >= chain else "micro-batch chain",
            "chain_bound_ms": round(chain, 3),
            "chain_bound_with_observed_bubble_ms": round(chain + neither, 3),
            "rebalanced_stages_ms": round(chain, 3),
            "one_parity_serial_tick_ms": round(tot + neither, 3),
            "gain_vs_observed_pct": round(100 * (summ["tick_ms_mean"] / ideal - 1), 1) if ideal else None,
            "note": "rebalanced_stages = equal per-stage work (layer split / where the MTP draft and lm-head run); "
                    "one_parity_serial = every step's work on both stages back to back (no overlap), with this run's "
                    "per-step work; with all requests in one parity a step carries more requests, so its work is "
                    "larger than measured here"}


def _tp0(tls, stage):
    rs = [r for r in tls if tls[r]["pp"] == stage]
    return min(rs, key=lambda r: (tls[r]["tp"], r)) if rs else None


def microbatch_parity(tls, stages):
    """Per step on each stage (TP0 rank): requests n (context + generation requests of the execute_context
    annotation), tokens, parity = step % pp, and the PP broadcasts the step posted (dtype x elements). Histograms of
    n per parity, a verdict, and the stage-to-stage step match (the broadcast Seqs a stage-1 step sends = the ones
    the same-index stage-0 step receives -> same micro-batch)."""
    pp = len(stages)
    out = {"pp_size": pp, "stages": {}}
    seqsets = {}
    for s in stages:
        r = _tp0(tls, s)
        fw = sorted(tls[r]["forwards"], key=lambda f: f["cpu_ts"])
        hist, bsz, steps = defaultdict(Counter), defaultdict(Counter), []
        seqsets[s] = []
        for i, f in enumerate(fw):
            n = f["ctx_requests"] + f["gen_requests"]
            par = i % pp
            bc = [p_ for p_ in f["posts"] if p_["coll"] == "broadcast"]
            sig = " ".join(f"{p_['dtype']}x{p_['nelems']}" for p_ in bc)
            hist[par][n] += 1
            bsz[par][sig or "none"] += 1
            seqsets[s].append(frozenset(p_["seq"] for p_ in bc if p_["seq"] is not None))
            steps.append({"step": i, "parity": par, "n_requests": n, "tokens": f["ctx_tokens"] + f["gen_tokens"],
                          "mode": f["mode"], "broadcasts": sig})
        out["stages"][f"stage{s}"] = {"rank": r, "steps": len(fw),
                                      "n_hist_by_parity": {par: dict(c) for par, c in sorted(hist.items())},
                                      "broadcast_sizes_by_parity": {par: dict(c) for par, c in sorted(bsz.items())},
                                      "first_steps": steps[:6]}
    if pp > 1:
        a, b = seqsets[stages[0]], seqsets[stages[-1]]
        m = [x == y for x, y in zip(a, b) if x and y]
        out["stage_step_match"] = {"compared": len(m), "matched": sum(m),
                                   "meaning": "the stage-1 step with the same index sends exactly the broadcast Seqs "
                                              "this stage-0 step receives (same micro-batch)"}
    verdicts = []
    for s in stages:
        h = out["stages"][f"stage{s}"]["n_hist_by_parity"]
        if pp < 2:
            verdicts.append("SINGLE_STAGE")
            continue
        ns = {par: set(c) for par, c in h.items()}
        if len(ns) < pp:
            verdicts.append("TOO_FEW_STEPS")
        elif any(v == {0} for v in ns.values()) and any(v - {0} for v in ns.values()):
            verdicts.append("ONE_PARITY")
        elif all(len(v) == 1 for v in ns.values()) and len({next(iter(v)) for v in ns.values()}) == 1 \
                and 0 not in ns[0]:
            verdicts.append("BALANCED_" + "+".join(str(next(iter(ns[par]))) for par in sorted(ns)))
        else:
            verdicts.append("UNBALANCED")
    out["verdict"] = verdicts[0] if len(set(verdicts)) == 1 else "STAGES_DISAGREE:" + ",".join(verdicts)
    out["note"] = ("request ids are not in the trace: consecutive steps carrying disjoint request sets follows from "
                   "async PP scheduling (a request in flight is not scheduled again) and is cross-checked by the "
                   "running count of the profiled window (running == sum of n over one parity cycle)")
    return out


def dependency_check(tls, ref_rank, pp):
    """REAL vs FALSE dependency of the stage-0 in-graph stalls. For stage-0 step k (TP0 rank): the largest stalls in
    its graph span, the NCCL kernel that released each (its Seq -> the step that posted that collective), and the
    readiness of the step's real input: the broadcasts posted by step k - pp (vLLM V2 PPHandler consumes step T's
    receive at T + pp). REAL = released by the consumed slot's collective; FALSE = released by another step's
    collective while the consumed slot had already completed before the stall began; MIXED = another step's
    collective, but the consumed slot completed only during the stall."""
    ref = tls[ref_rank]
    fw = sorted(ref["forwards"], key=lambda f: f["cpu_ts"])
    owner, by_seq = {}, {}
    for i, f in enumerate(fw):
        for p_ in f["posts"]:
            if p_["seq"] is not None and p_["coll"] == "broadcast":
                owner[(p_["pg"], p_["seq"])] = i
    for k in ref["nccl_kernels"]:
        if k.get("seq") is not None:
            by_seq[(k.get("pg"), k["seq"])] = k
    rows, unreleased = [], 0
    for i, f in enumerate(fw):
        if not f.get("graph_span"):
            continue
        a, b = f["graph_span"]
        st = sub([[a, b]], union(ref["compute"].clip(a, b), ref["copy"].clip(a, b), ref["nccl_tp"].clip(a, b)))
        rel = stall_release(ref, st)
        real = None
        if i - pp >= 0:
            ks = [by_seq.get((p_["pg"], p_["seq"])) for p_ in fw[i - pp]["posts"]
                  if p_["coll"] == "broadcast" and p_["seq"] is not None]
            ks = [x for x in ks if x]
            real = max(x["b"] for x in ks) if ks else None
        for x in rel:
            if x["released_by"] is None:
                unreleased += 1
                continue
            o = owner.get((x["pg"], x["seq"]))
            delta = (i - o) if o is not None else None
            s0, s1 = x["stall"]
            if delta is None:
                v = "UNRESOLVED"
            elif delta == pp:
                v = "REAL"
            elif real is None:
                v = "FALSE_NO_REAL_INPUT" if i - pp < 0 else "UNRESOLVED"
            elif real <= s0:
                v = "FALSE"
            else:
                v = "MIXED"
            rows.append({"step": i, "stall_ms": _ms(s1 - s0), "released_by": x["released_by"],
                         "release_seq": x["seq"], "posted_by_step": o, "step_delta": delta,
                         "lag_us": round(x["lag_us"], 1) if x["lag_us"] is not None else None,
                         "real_input_ready_before_graph_start_ms": _ms(a - real) if real is not None else None,
                         "real_input_ready_before_stall_start_ms": _ms(s0 - real) if real is not None else None,
                         "verdict": v})
    vc = Counter(r_["verdict"] for r_ in rows)
    return {"pp_size": pp, "consumed_slot_delta": pp, "stalls_checked": len(rows),
            "stalls_without_release_kernel": unreleased, "verdicts": dict(vc),
            "step_delta_hist": dict(Counter(r_["step_delta"] for r_ in rows)),
            "released_by": dict(Counter(r_["released_by"] for r_ in rows).most_common(4)),
            "stall_ms_median": _median([r_["stall_ms"] for r_ in rows]),
            "real_input_ready_before_graph_start_ms_median": _median(
                [r_["real_input_ready_before_graph_start_ms"] for r_ in rows]),
            "real_input_ready_before_graph_start_ms_min": min(
                (r_["real_input_ready_before_graph_start_ms"] for r_ in rows
                 if r_["real_input_ready_before_graph_start_ms"] is not None), default=None),
            "lag_us_median": _median([r_["lag_us"] for r_ in rows]),
            "verdict": (vc.most_common(1)[0][0] if rows else "NO_STALL"),
            "rows": rows[:8]}


def cpu_activity(rec):
    """What the trace holds on the host side: is a host-gap split possible at all, and at which resolution."""
    ev = rec["trace"].get("traceEvents") or []
    cats = Counter(str(e.get("cat", "")) for e in ev if e.get("ph") == "X")
    shapes = any("Input Dims" in _args(e) for e in ev if e.get("cat") == "cpu_op")
    stack = cats.get("python_function", 0) > 0 or any("Call stack" in _args(e) for e in ev if e.get("cat") == "cpu_op")
    scopes = Counter(str(e.get("name", ""))[:40] for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation")
    host_threads = Counter((e.get("pid"), e.get("tid")) for e in ev if e.get("ph") == "X"
                           and str(e.get("cat", "")).lower() in HOST_CATS)
    return {"cpu_op": cats.get("cpu_op", 0), "python_function": cats.get("python_function", 0),
            "cuda_runtime": cats.get("cuda_runtime", 0), "cuda_driver": cats.get("cuda_driver", 0),
            "user_annotation": cats.get("user_annotation", 0), "kernel": cats.get("kernel", 0),
            "has_cpu_activity": cats.get("cpu_op", 0) > 0 or cats.get("cuda_runtime", 0) > 0,
            "with_stack": stack, "record_shapes": shapes,
            "custom_scopes": any(k.startswith("gpu_model_runner:") for k in scopes),
            "user_annotation_names": dict(scopes.most_common(8)), "host_threads": len(host_threads)}


# ------------------------------------------------------------------ NB1's own numbers (comparability)
def nb1_microbatch_costs(ev):
    """build_e25_nb1.microbatch_costs, same joins and means (decode-only scopes grouped by verify tokens)."""
    rt, kern, names = defaultdict(list), defaultdict(list), {}
    for e in ev:
        if e.get("ph") == "X" and str(e.get("cat", "")).lower() in RUNTIME_CATS:
            c = _args(e).get("correlation")
            if c is not None:
                rt[e.get("tid")].append((e.get("ts", 0), c))
                names[c] = str(e.get("name", ""))
        elif e.get("ph") == "X" and str(e.get("cat", "")).lower() in KERNEL_CATS:
            c = _args(e).get("correlation")
            if c is not None:
                kern[c].append((nb1_cat(e.get("name", "")), float(e.get("dur", 0.0)), float(e.get("ts", 0.0))))
    scopes = []
    for e in ev:
        if e.get("ph") == "X" and str(e.get("cat", "")) == "user_annotation":
            mm = re.match(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$", str(e.get("name", "")))
            if mm:
                scopes.append((e, int(mm.group(2)), int(mm.group(3)), int(mm.group(4))))
    per = []
    for e, nctx, nreq, ntok in scopes[:600]:
        lo, hi, tid = e.get("ts", 0), e.get("ts", 0) + e.get("dur", 0), e.get("tid")
        cats, ivl = defaultdict(float), []
        for ts, c in rt.get(tid, []):
            if lo <= ts <= hi:
                for k, d, kts in kern.get(c, []):
                    cats[k] += d
                    ivl.append([kts, kts + d])
        mi = merge(ivl)
        per.append({"ctx_tokens": nctx, "gen_tokens": ntok, "wall": (mi[-1][1] - mi[0][0]) if mi else 0.0,
                    "union": total(mi), "sum": sum(cats.values())})
    by = defaultdict(list)
    for x in per:
        if x["ctx_tokens"] == 0:
            by[x["gen_tokens"]].append(x)
    agg = {}
    for ntok, xs in sorted(by.items()):
        agg[ntok] = {"microbatches": len(xs), "critical_path_ms": _ms(statistics.mean(x["wall"] for x in xs)),
                     "gpu_busy_union_ms": _ms(statistics.mean(x["union"] for x in xs)),
                     "summed_kernel_ms": _ms(statistics.mean(x["sum"] for x in xs))}
    return agg


def nb1_category_pct(ev):
    per, tot = Counter(), 0.0
    for e in ev:
        if e.get("ph") == "X" and str(e.get("cat", "")).lower() in KERNEL_CATS:
            d = float(e.get("dur", 0.0))
            tot += d
            per[nb1_cat(e.get("name", ""))] += d
    return {k: round(100 * v / tot, 1) for k, v in per.most_common()} if tot else {}


def nb1_reference(ranks, tp=2, tls=None):
    """NB1's per-stage table (TP0 rank of each stage, verify-token group with the most micro-batches), the
    kernel-category shares of every rank (NB1 reported rank 0's) and NB1's graph verdict with the lab rule
    (e23_lab rev 3h _scope_kind; superseded by graph_mode() everywhere else)."""
    out = {"by_rank": {}}
    if tls is None:
        stage_of = {r: ranks[r]["pp"] for r in ranks}
        tls = {r: build_rank(ranks[r], stage_of) for r in ranks}
    out["graph_verdict_lab_rule"] = {}
    for r in sorted(tls):
        kinds = [f["mode_lab_rule"] for f in tls[r]["forwards"]]
        ks = set(kinds)
        out["graph_verdict_lab_rule"][r] = {"verdict": (ks.pop() if len(ks) == 1 else "MIXED") if kinds else "UNKNOWN",
                                            "forward_kinds": dict(Counter(kinds))}
    for r in sorted(ranks):
        ev = ranks[r]["trace"].get("traceEvents") or []
        agg = nb1_microbatch_costs(ev)
        out["by_rank"][r] = {"stage": ranks[r]["pp"], "decode_by_verify_tokens": agg, "category_pct": nb1_category_pct(ev)}
    for r in sorted(ranks):
        if ranks[r]["tp"] == 0:
            agg = out["by_rank"][r]["decode_by_verify_tokens"]
            if agg:
                k = max(agg, key=lambda x: agg[x]["microbatches"])
                out[f"stage{ranks[r]['pp']}"] = dict(agg[k], verify_tokens=k, rank=r)
    return out


# ------------------------------------------------------------------ report
def format_report(rep):
    """Markdown tables of the summary (what the notebook prints and the results note copies)."""
    L = []
    s = rep.get("summary") or {}
    ck = rep.get("clock") or {}
    L.append(f"clock: {ck.get('method')} (measured offsets us: "
             f"{ {r: round(v, 1) for r, v in (ck.get('measured_offset_us') or {}).items()} })")
    nb = rep.get("nb1_reference") or {}
    L.append("\nNB1 reproduction (critical path / GPU-busy union / summed kernels, ms per micro-batch):")
    for k in sorted(x for x in nb if x.startswith("stage")):
        v = nb[k]
        L.append(f"  {k} (rank {v['rank']}, {v['verify_tokens']} verify tokens, n={v['microbatches']}): "
                 f"{v['critical_path_ms']} / {v['gpu_busy_union_ms']} / {v['summed_kernel_ms']}")
    r0 = (nb.get("by_rank") or {}).get(rep.get("reference_rank"), {})
    if r0:
        L.append(f"  rank {rep.get('reference_rank')} kernel time by category %: {r0.get('category_pct')}")
    L.append(f"\nsteady ticks {s.get('steady_ticks')} of {s.get('all_ticks')}; tick ms mean {s.get('tick_ms_mean')} "
             f"median {s.get('tick_ms_median')}; graph modes {s.get('graph_modes')}")
    stg = [k for k in s if k.startswith("stage") and isinstance(s[k], dict)]
    if stg:
        L.append("\n| category | " + " | ".join(f"{k} ms (mean)" for k in stg) + " | " +
                 " | ".join(f"{k} % of tick" for k in stg) + " |")
        L.append("|---|" + "---|" * (2 * len(stg)))
        for c in CATEGORIES:
            L.append(f"| {c} | " + " | ".join(str(s[k][c]["mean_ms"]) for k in stg) + " | " +
                     " | ".join(str(s[k]["share_of_tick_pct"][c]) for k in stg) + " |")
        for c in SIDE:
            if c == "kernel_union":
                continue
            L.append(f"| ({c}, side measure) | " + " | ".join(str(s[k][c]["mean_ms"]) for k in stg) + " |" +
                     " |" * len(stg))
        L.append("| (gap = tick - compute - copy) | " + " | ".join(str(s[k]["gap_ms_mean"]) for k in stg) + " |" +
                 " |" * len(stg))
    if "pipeline" in s:
        L.append(f"\npipeline per tick (ms): {s['pipeline']}; stage-1 idle {s.get('stage1_idle_ms_mean')} ms")
    L.append(f"critical path per tick (ms): {s.get('critical_path')}")
    L.append(f"stalls: {rep.get('stalls')}")
    if s.get("ideal_tick"):
        L.append(f"ideal tick: {s['ideal_tick']}")
    par = rep.get("parity") or {}
    if par:
        L.append(f"micro-batch parity: {par.get('verdict')}; n per parity: "
                 f"{ {k: v['n_hist_by_parity'] for k, v in (par.get('stages') or {}).items()} }; "
                 f"stage step match {par.get('stage_step_match')}")
    dep = rep.get("dependency") or {}
    if dep:
        L.append(f"dependency: {dep.get('verdict')} {dep.get('verdicts')}; step delta {dep.get('step_delta_hist')}; "
                 f"real input ready before graph start (ms, median / min) "
                 f"{dep.get('real_input_ready_before_graph_start_ms_median')} / "
                 f"{dep.get('real_input_ready_before_graph_start_ms_min')}")
    lab = (rep.get("nb1_reference") or {}).get("graph_verdict_lab_rule")
    if lab:
        L.append(f"graph mode (rev 2 rule): {rep.get('graph_modes')}; NB1 lab rule: "
                 f"{ {r: v['verdict'] for r, v in lab.items()} }")
    return "\n".join(L)


def strip_private(rep):
    """Drop interval lists (not JSON-sized) before json.dump."""
    return json.loads(json.dumps(rep, default=lambda o: None))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python nb4_timeline.py <trace dir or files...> [--json out.json] [--tp 2]")
        sys.exit(2)
    args = sys.argv[1:]
    out_json, tp_ = None, 2
    if "--json" in args:
        i = args.index("--json")
        out_json = args[i + 1]
        del args[i:i + 2]
    if "--tp" in args:
        i = args.index("--tp")
        tp_ = int(args[i + 1])
        del args[i:i + 2]
    src_ = args[0] if len(args) == 1 else args
    rep_ = analyze(src_, tp=tp_)
    print(format_report(rep_))
    if out_json:
        Path(out_json).write_text(json.dumps(strip_private(rep_), indent=1), encoding="utf-8", newline="\n")
