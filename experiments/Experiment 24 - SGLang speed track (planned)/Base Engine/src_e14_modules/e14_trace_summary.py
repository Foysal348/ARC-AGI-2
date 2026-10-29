"""E14 trace summarizer for SGLang 0.5.21 /start_profile chrome traces (Experiment 14, decision D1; 2026-10-23).

stdlib only. Streams the "traceEvents" array (no json.load of the whole file), so traces of a few hundred MB do
not blow up the notebook's RAM; a configurable on-disk size guard skips anything larger with a message.

What SGLang 0.5.21 writes (verified in the local source, Notes/15 has the line refs):
- legacy path (SGLANG_PROFILE_V2 unset, the default): managers/scheduler_components/profiler_manager.py:350-359
    <output_dir>/[<profile_prefix>-]<profile_id>-TP-<tp_rank>[-DP-<r>][-PP-<r>][-EP-<r>][-EXTEND|-DECODE].trace.json.gz
  profile_id defaults to str(time.time()) (managers/tokenizer_control_mixin.py:412); the -EXTEND/-DECODE suffix only
  with profile_by_stage; output_dir defaults to $SGLANG_TORCH_PROFILER_DIR or /tmp (profiler_manager.py:126-127).
- V2 path (SGLANG_PROFILE_V2=1): utils/profile_utils.py:357-372, same layout with suffix -prefill / -decode.
- merge_profiles=true adds merged-<profile_id>.trace.json.gz (utils/profile_merger.py:20-21): skipped by default
  (it repeats every rank's events).
- One file per TP rank; each is torch.profiler's export_chrome_trace (kineto JSON, gzip by the .gz name).

API:  summarize(trace_dir, num_steps, top=30, ...) -> dict          CLI:  python e14_trace_summary.py <dir> --steps 20
"""
import argparse
import array
import gzip
import json
import os
import re
import sys
from collections import Counter

CATEGORY_RULES = [
    # Order matters: the first match wins (case-insensitive search). Deliberate order (Notes/15):
    # - GDN before sampling: FLA's chunk_local_cumsum (prefill) would otherwise hit "cumsum";
    # - sampling and norm before attention: FlashInfer names its sampling/norm/activation kernels
    #   flashinfer::sampling::*, flashinfer::norm::*, flashinfer::activation::*, so bare "flashinfer" is only a
    #   late fallback for attention (after elementwise);
    # - attention before other_gemm: FA-style kernels are often cutlass::device_kernel<flash::...>.
    ("nccl_allreduce", r"ncclDevKernel_AllReduce|ncclKernel_AllReduce|AllReduce|all_reduce|cross_device_reduce"),
    ("nccl_allgather", r"AllGather|all_gather"),
    ("nccl_other", r"nccl"),
    ("marlin_gemm", r"marlin"),
    ("gdn_triton", r"fused_sigmoid_gating|fused_recurrent|chunk_|gated_delta|causal_conv|mamba|ssm|gdn"),
    ("sampling", r"topk|top_k|softmax|sampling|argmax|multinomial|cumsum|sort"),
    ("norm", r"rmsnorm|layernorm|norm"),
    ("attention", r"BatchDecode|BatchPrefill|SingleDecode|SinglePrefill|attention|fmha|flash_fwd|flash::|"
                  r"MergeState|merge_state|PagedKV|paged_kv"),
    ("other_gemm", r"gemm|cutlass|sm89_xmma|sm80_xmma|ampere_|cublas|splitKreduce"),
    ("elementwise_other", r"elementwise|vectorized|unrolled|act_and_mul|silu|gelu|rotary|rope|CatArray|"
                          r"copy_kernel|fill_kernel|index|scatter|gather|reduce_kernel|triton_"),
    ("attention", r"flashinfer"),
]
CATEGORIES = ["nccl_allreduce", "nccl_allgather", "nccl_other", "marlin_gemm", "other_gemm", "attention",
              "gdn_triton", "sampling", "norm", "memcpy", "memset", "elementwise_other", "other"]
COLLECTIVES = ("nccl_allreduce", "nccl_allgather", "nccl_other")
_RULES = [(c, re.compile(p, re.I)) for c, p in CATEGORY_RULES]
KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
TRACE_SUFFIXES = (".trace.json.gz", ".trace.json", ".json.gz", ".json")
DEFAULT_MAX_FILE_MB = 2048.0
DEFAULT_MAX_INTERVALS = 5_000_000
NAME_MAX = 240
STEP_RE = re.compile(r"^step\[(\S+)")

_cls_cache = {}


def classify(name, cat="kernel"):
    """Kernel category by event cat (memcpy/memset) then by name regex (first match)."""
    cat = (cat or "").lower()
    key = (name, cat)
    hit = _cls_cache.get(key)
    if hit is not None:
        return hit
    if cat == "gpu_memcpy" or re.match(r"^Memcpy\b", name or ""):
        res = "memcpy"
    elif cat == "gpu_memset" or re.match(r"^Memset\b", name or ""):
        res = "memset"
    else:
        res = "other"
        for c, rx in _RULES:
            if rx.search(name or ""):
                res = c
                break
    if len(_cls_cache) < 200_000:
        _cls_cache[key] = res
    return res


# ============================================================================================ streaming reader
def _open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def iter_trace_events(path, chunk_chars=1 << 22, meta=None):
    """Yield each object of the top-level "traceEvents" array (or of a top-level array) without loading the file.

    meta (dict, optional) receives "preamble" (text before the array, capped) for rank lookup."""
    dec = json.JSONDecoder()
    with _open_text(path) as f:
        buf = f.read(chunk_chars)
        eof = len(buf) < chunk_chars
        # locate the array start
        stripped = buf.lstrip()
        if stripped.startswith("["):
            pos = len(buf) - len(stripped) + 1
        else:
            pre = ""
            while True:
                i = buf.find('"traceEvents"')
                if i >= 0:
                    break
                if eof:
                    if meta is not None:
                        meta["preamble"] = (pre + buf)[-20000:]
                        meta["no_trace_events"] = True
                    return
                pre = (pre + buf[:-32])[-20000:]
                more = f.read(chunk_chars)
                eof = len(more) < chunk_chars
                buf = buf[-32:] + more
            if meta is not None:
                meta["preamble"] = (pre + buf[:i])[-20000:]
            j = i + len('"traceEvents"')
            while True:
                k = buf.find("[", j)
                if k >= 0:
                    pos = k + 1
                    break
                if eof:
                    return
                more = f.read(chunk_chars)
                eof = len(more) < chunk_chars
                buf = buf[j:] + more
                j = 0
        while True:
            n = len(buf)
            while pos < n and buf[pos] in " \t\r\n,":
                pos += 1
            if pos >= n:
                if eof:
                    return
                buf = f.read(chunk_chars)
                eof = len(buf) < chunk_chars
                pos = 0
                continue
            if buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("truncated or invalid trace near char %d of the last chunk" % pos)
                more = f.read(chunk_chars)
                eof = len(more) < chunk_chars
                buf = buf[pos:] + more
                pos = 0
                continue
            yield obj
            pos = end
            if pos > chunk_chars:
                buf = buf[pos:]
                pos = 0


# ============================================================================================ per-file stats
class FileStats:
    def __init__(self, path, rank, max_intervals):
        self.path = path
        self.rank = rank
        self.cat_us = Counter()
        self.cat_n = Counter()
        self.kern = {}                      # name -> [count, total_us, category]
        self.first_ts = None
        self.last_end = None
        self.events = 0
        self.kernels = 0
        self.max_intervals = max_intervals
        self.starts = array.array("d")
        self.ends = array.array("d")
        self.union_dropped = False
        self.step_spans = {}                # label -> [count, cpu_us, gpu_us]

    def add(self, ev):
        self.events += 1
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            return
        cat = str(ev.get("cat") or "").lower()
        name = ev.get("name") or ""
        if cat in KERNEL_CATS:
            try:
                dur = float(ev.get("dur") or 0.0)
                ts = float(ev.get("ts"))
            except (TypeError, ValueError):
                return
            c = classify(name, cat)
            self.kernels += 1
            self.cat_us[c] += dur
            self.cat_n[c] += 1
            k = self.kern.get(name)
            if k is None:
                self.kern[name] = [1, dur, c]
            else:
                k[0] += 1
                k[1] += dur
            end = ts + dur
            if self.first_ts is None or ts < self.first_ts:
                self.first_ts = ts
            if self.last_end is None or end > self.last_end:
                self.last_end = end
            if not self.union_dropped:
                if len(self.starts) >= self.max_intervals:
                    self.union_dropped = True
                    self.starts = array.array("d")
                    self.ends = array.array("d")
                else:
                    self.starts.append(ts)
                    self.ends.append(end)
        elif cat in ("user_annotation", "gpu_user_annotation"):
            m = STEP_RE.match(name)
            if m:
                rec = self.step_spans.setdefault(m.group(1), [0, 0.0, 0.0])
                try:
                    dur = float(ev.get("dur") or 0.0)
                except (TypeError, ValueError):
                    dur = 0.0
                if cat == "user_annotation":
                    rec[0] += 1
                    rec[1] += dur
                else:
                    rec[2] += dur

    def busy_us(self):
        """GPU busy time = union of kernel/memcpy/memset intervals (overlapping streams counted once)."""
        if self.union_dropped:
            return None
        n = len(self.starts)
        if n == 0:
            return 0.0
        order = sorted(range(n), key=self.starts.__getitem__)
        total = 0.0
        cs, ce = self.starts[order[0]], self.ends[order[0]]
        for i in order[1:]:
            s, e = self.starts[i], self.ends[i]
            if s > ce:
                total += ce - cs
                cs, ce = s, e
            elif e > ce:
                ce = e
        total += ce - cs
        return total

    def detected_steps(self):
        """Scheduler forwards seen as CPU 'step[...]' spans (model_runner.py:1729): VERIFY if present, else all
        non-DRAFT labels."""
        if not self.step_spans:
            return None
        if "VERIFY" in self.step_spans:
            return self.step_spans["VERIFY"][0]
        return sum(v[0] for k, v in self.step_spans.items() if k != "DRAFT") or None


def _rank_from_name(path):
    m = re.search(r"TP-(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _group_key(path):
    b = os.path.basename(path)
    for suf in TRACE_SUFFIXES:
        if b.endswith(suf):
            b = b[: -len(suf)]
            break
    if re.search(r"TP-\d+", b):
        return re.sub(r"TP-\d+", "TP-*", b)
    return "ungrouped"


def discover(trace_dir, recursive=False, include_merged=False):
    """Trace files under trace_dir (or trace_dir itself if it is a file); returns (files, skipped)."""
    files, skipped = [], []
    if os.path.isfile(trace_dir):
        return [trace_dir], skipped
    if not os.path.isdir(trace_dir):
        return [], [{"file": trace_dir, "reason": "not a file or directory"}]
    if recursive:
        cand = [os.path.join(dp, fn) for dp, _, fns in os.walk(trace_dir) for fn in fns]
    else:
        cand = [os.path.join(trace_dir, fn) for fn in os.listdir(trace_dir)]
    for p in sorted(cand):
        if not os.path.isfile(p):
            continue
        b = os.path.basename(p)
        if not b.endswith(TRACE_SUFFIXES) or b.endswith((".tmp",)):
            continue
        if b.startswith("merged-") and not include_merged:
            skipped.append({"file": p, "reason": "merged multi-rank trace skipped (include_merged=False)"})
            continue
        files.append(p)
    return files, skipped


def read_file(path, max_file_mb=DEFAULT_MAX_FILE_MB, max_intervals=DEFAULT_MAX_INTERVALS):
    """Stream one trace file into FileStats. Returns (stats or None, skip_reason or None)."""
    try:
        size_mb = os.path.getsize(path) / 1e6
    except OSError as e:
        return None, "stat failed: %r" % (e,)
    if max_file_mb is not None and size_mb > max_file_mb:
        return None, "skipped: %.1f MB on disk > max_file_mb %.1f (raise max_file_mb to read it)" % (
            size_mb, max_file_mb)
    meta = {}
    st = FileStats(path, _rank_from_name(path), max_intervals)
    try:
        for ev in iter_trace_events(path, meta=meta):
            st.add(ev)
    except Exception as e:              # truncated gzip (EOFError), zlib.error, bad JSON (ValueError), I/O
        return None, "read failed: %r" % (e,)
    if st.rank is None:
        m = re.search(r'"rank"\s*:\s*(\d+)', meta.get("preamble", ""))
        if m:
            st.rank = int(m.group(1))
    st.size_mb = size_mb
    st.no_trace_events = bool(meta.get("no_trace_events"))
    return st, None


# ============================================================================================ summary
def _r(x, nd=4):
    return None if x is None else round(x, nd)


def _summarize_group(stats, num_steps, top):
    stats = sorted(stats, key=lambda s: (s.rank is None, s.rank if s.rank is not None else 0, s.path))
    for i, s in enumerate(stats):
        if s.rank is None:
            s.rank = i
    det = {str(s.rank): s.detected_steps() for s in stats}
    if num_steps and num_steps > 0:
        steps, src = float(num_steps), "argument"
    else:
        vals = [v for v in det.values() if v]
        if vals:
            steps, src = float(Counter(vals).most_common(1)[0][0]), "detected step[] spans"
        else:
            steps, src = 1.0, "none (values are totals over the trace)"
    ranks = {}
    for s in stats:
        ksum = sum(s.cat_us.values())
        busy = s.busy_us()
        wall = (s.last_end - s.first_ts) if s.first_ts is not None else 0.0
        cats = {}
        for c in CATEGORIES:
            us = s.cat_us.get(c, 0.0)
            cats[c] = {"ms_per_step": us / 1000.0 / steps, "calls_per_step": s.cat_n.get(c, 0) / steps,
                       "share_of_kernel_sum": (us / ksum) if ksum else 0.0,
                       "share_of_busy": (us / busy) if busy else None}
        ranks[str(s.rank)] = {
            "file": s.path, "size_mb": _r(getattr(s, "size_mb", None), 2), "events": s.events, "kernels": s.kernels,
            "kernel_sum_ms_per_step": ksum / 1000.0 / steps,
            "gpu_busy_ms_per_step": None if busy is None else busy / 1000.0 / steps,
            "gpu_busy_note": "interval union skipped (too many kernels)" if busy is None else None,
            "wall_ms_per_step": wall / 1000.0 / steps,
            "detected_steps": det[str(s.rank)],
            "step_spans": {k: {"count": v[0], "cpu_ms_total": v[1] / 1000.0, "gpu_ms_total": v[2] / 1000.0}
                           for k, v in sorted(s.step_spans.items())},
            "categories": cats,
            "no_trace_events": getattr(s, "no_trace_events", False),
        }
    n = len(ranks)
    agg = {}
    for c in CATEGORIES:
        v = [ranks[r]["categories"][c]["ms_per_step"] for r in ranks]
        sh = [ranks[r]["categories"][c]["share_of_busy"] for r in ranks]
        sh = [x for x in sh if x is not None]
        agg[c] = {"ms_per_step_min": min(v) if v else 0.0, "ms_per_step_mean": sum(v) / n if n else 0.0,
                  "ms_per_step_max": max(v) if v else 0.0,
                  "share_of_busy_mean": (sum(sh) / len(sh)) if sh else None}
    coll = {}
    for c in COLLECTIVES:
        v = {r: ranks[r]["categories"][c]["ms_per_step"] for r in ranks}
        if not v:
            continue
        mn, mx = min(v.values()), max(v.values())
        coll[c] = {"min_over_ranks_ms_per_step": mn, "max_over_ranks_ms_per_step": mx,
                   "skew_ms_per_step": mx - mn, "min_rank": min(v, key=v.get), "max_rank": max(v, key=v.get),
                   "calls_per_step": ranks[min(v, key=v.get)]["categories"][c]["calls_per_step"]}
    busy = [ranks[r]["gpu_busy_ms_per_step"] for r in ranks if ranks[r]["gpu_busy_ms_per_step"] is not None]
    kern = {}
    for s in stats:
        for name, (cnt, us, c) in s.kern.items():
            k = kern.setdefault(name, [0, 0.0, c])
            k[0] += cnt
            k[1] += us
    top_rows = []
    for name, (cnt, us, c) in sorted(kern.items(), key=lambda kv: -kv[1][1])[: max(0, int(top))]:
        top_rows.append({"name": name if len(name) <= NAME_MAX else name[:NAME_MAX] + "...", "category": c,
                         "count_total": cnt, "calls_per_step_per_rank": cnt / steps / n,
                         "total_ms_all_ranks": us / 1000.0, "ms_per_step_per_rank": us / 1000.0 / steps / n,
                         "mean_us": us / cnt if cnt else 0.0})
    return {"num_steps": steps, "num_steps_source": src, "detected_steps": det, "n_ranks": n,
            "ranks": ranks, "categories": agg, "collectives": coll,
            "gpu_busy_ms_per_step_max": max(busy) if busy else None,
            "gpu_busy_ms_per_step_min": min(busy) if busy else None,
            "top_kernels": top_rows}


def summarize(trace_dir, num_steps, top=30, max_file_mb=DEFAULT_MAX_FILE_MB, recursive=False,
              include_merged=False, max_intervals=DEFAULT_MAX_INTERVALS):
    """Summarize all SGLang chrome traces in trace_dir; one group per <prefix><profile_id>[-stage] with one entry per
    TP rank. Values are ms per step (= total / num_steps). Never raises for a bad file (it is listed in "skipped")."""
    files, skipped = discover(trace_dir, recursive=recursive, include_merged=include_merged)
    groups = {}
    for p in files:
        st, why = read_file(p, max_file_mb=max_file_mb, max_intervals=max_intervals)
        if st is None:
            skipped.append({"file": p, "reason": why})
            continue
        groups.setdefault(_group_key(p), []).append(st)
    out = {"trace_dir": str(trace_dir), "files_read": sum(len(v) for v in groups.values()), "skipped": skipped,
           "categories_order": CATEGORIES, "category_rules": CATEGORY_RULES,
           "notes": ["ms_per_step = summed kernel duration / num_steps per rank",
                     "collectives: min over ranks is closer to the real transfer cost; max - min is time spent "
                     "waiting inside NCCL kernels for the slowest rank (skew)",
                     "gpu_busy = union of kernel/memcpy/memset intervals on that rank"],
           "groups": {}}
    for key, stats in sorted(groups.items()):
        out["groups"][key] = _summarize_group(stats, num_steps, top)
    if len(out["groups"]) == 1:
        out["main_group"] = next(iter(out["groups"]))
    elif out["groups"]:
        # the decode group if there is one, else the biggest
        dec = [k for k in out["groups"] if k.endswith(("-DECODE", "-decode"))]
        out["main_group"] = dec[0] if dec else max(out["groups"], key=lambda k: out["groups"][k]["n_ranks"])
    else:
        out["main_group"] = None
    return out


def format_table(summary):
    lines = []
    for key, g in summary.get("groups", {}).items():
        lines.append("group %s: %d ranks, %s steps (%s)" % (key, g["n_ranks"], g["num_steps"], g["num_steps_source"]))
        lines.append("  %-18s %10s %10s %10s %8s" % ("category", "min ms", "mean ms", "max ms", "share"))
        for c in summary.get("categories_order", CATEGORIES):
            a = g["categories"][c]
            if a["ms_per_step_max"] <= 0:
                continue
            sh = a["share_of_busy_mean"]
            lines.append("  %-18s %10.3f %10.3f %10.3f %7s" % (c, a["ms_per_step_min"], a["ms_per_step_mean"],
                                                              a["ms_per_step_max"],
                                                              "-" if sh is None else "%.1f%%" % (100 * sh)))
        lines.append("  busy ms/step: min %s max %s" % (_r(g["gpu_busy_ms_per_step_min"], 3),
                                                        _r(g["gpu_busy_ms_per_step_max"], 3)))
        for t in g["top_kernels"][:10]:
            lines.append("  %8.3f ms/step  %6.1f calls  %8.1f us  [%s] %s" % (
                t["ms_per_step_per_rank"], t["calls_per_step_per_rank"], t["mean_us"], t["category"], t["name"][:90]))
    for s in summary.get("skipped", []):
        lines.append("skipped %s: %s" % (s["file"], s["reason"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Summarize SGLang /start_profile chrome traces (one per TP rank).")
    ap.add_argument("trace_dir")
    ap.add_argument("--steps", type=int, default=0, help="profiled steps (0 = detect from step[] spans)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-file-mb", type=float, default=DEFAULT_MAX_FILE_MB)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--include-merged", action="store_true")
    a = ap.parse_args(argv)
    s = summarize(a.trace_dir, a.steps, top=a.top, max_file_mb=a.max_file_mb, recursive=a.recursive,
                  include_merged=a.include_merged)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1)
    print(format_table(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
