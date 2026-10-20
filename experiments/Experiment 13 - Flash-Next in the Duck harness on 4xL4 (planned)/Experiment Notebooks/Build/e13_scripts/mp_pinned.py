"""E13 micro-preflight P: can 4 ranks each pin their BF16 PLE shard in host RAM at the same time? (no model)

SGLang 0.5.21 allocates the offloaded PLE table with torch.empty(..., pin_memory=True) per TP rank
(qwen4_exp_ple_table.py:225-245). This test allocates the real per-rank size (sum of the model's PLE shard files / 4)
in 4 processes concurrently (one per GPU), records MemAvailable / cgroup memory before, at peak and after release, and
the time to pin. It never touches the model weights. Usage:
  python -I mp_pinned.py --out RESULT.json --model-dir DIR [--ranks 4] [--per-rank-bytes N]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

CHILD = r'''
import os, sys, time, torch
rank, nbytes, d = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
torch.cuda.set_device(rank)
t = time.perf_counter()
try:
    x = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    x[:: 1 << 20] = 1                                  # touch one byte per MiB
    msg = "ok %.1f %s" % (time.perf_counter() - t, x.is_pinned())
except Exception as e:
    msg = "fail " + type(e).__name__ + ": " + str(e)[:300]
open(os.path.join(d, "ready_%d" % rank), "w").write(msg)
while not os.path.exists(os.path.join(d, "release")):
    time.sleep(0.5)
'''


def meminfo():
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            if k in ("MemTotal", "MemAvailable", "MemFree", "Cached", "Mlocked", "Unevictable"):
                out[k + "_GB"] = round(int(v.split()[0]) * 1024 / 1e9, 2)
    except Exception as e:
        out["error"] = str(e)
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current",
              "/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            v = open(p).read().strip()
            out[os.path.basename(p)] = v if not v.isdigit() else round(int(v) / 1e9, 2)
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--model-dir", default="")
    ap.add_argument("--ranks", type=int, default=4); ap.add_argument("--per-rank-bytes", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    ple_files = sorted(glob.glob(os.path.join(a.model_dir, "model-ple-*.safetensors"))) if a.model_dir else []
    ple_total = sum(os.path.getsize(f) for f in ple_files)
    per_rank = a.per_rank_bytes or (ple_total // a.ranks)
    rep = {"ple_files": len(ple_files), "ple_total_GB": round(ple_total / 1e9, 2), "per_rank_GB": round(per_rank / 1e9, 2),
           "ranks": a.ranks, "before": meminfo(), "peak": None, "after": None, "children": {}, "pass": False}
    d = tempfile.mkdtemp(prefix="e13pin_")
    script = os.path.join(d, "child.py"); open(script, "w").write(CHILD)
    t0 = time.perf_counter()
    # Ranks are pinned one after another and held together. Before each rank: MemAvailable must exceed the rank's size
    # plus a 15 GB margin, else stop (an OOM kill here would take down the notebook kernel). The drop per rank shows
    # whether the host allocator rounds the size up.
    procs, low, rep["drop_per_rank_GB"], rep["stopped_reason"] = [], None, [], None
    for r in range(a.ranks):
        avail = meminfo().get("MemAvailable_GB")
        if avail is None or avail < per_rank / 1e9 + 15:
            rep["stopped_reason"] = f"rank {r}: MemAvailable {avail} GB < {per_rank / 1e9:.1f} + 15 GB margin"
            break
        procs.append(subprocess.Popen([sys.executable, "-I", script, str(r), str(per_rank), d]))
        while time.perf_counter() - t0 < a.timeout and procs[-1].poll() is None \
                and not os.path.exists(os.path.join(d, f"ready_{r}")):
            m = meminfo().get("MemAvailable_GB")
            if m is not None:
                low = m if low is None else min(low, m)
            time.sleep(1)
        after = meminfo().get("MemAvailable_GB")
        rep["drop_per_rank_GB"].append(round(avail - after, 2) if after is not None else None)
        if not os.path.exists(os.path.join(d, f"ready_{r}")) or not open(os.path.join(d, f"ready_{r}")).read().startswith("ok"):
            rep["stopped_reason"] = rep["stopped_reason"] or f"rank {r} did not pin"
            break
    rep["peak"] = meminfo(); rep["min_MemAvailable_GB"] = low; rep["sec"] = round(time.perf_counter() - t0, 1)
    for r in range(a.ranks):
        f = os.path.join(d, f"ready_{r}")
        if os.path.exists(f):
            rep["children"][r] = open(f).read()
        else:
            rep["children"][r] = f"no report (rc={procs[r].poll()})" if r < len(procs) else "not started"
    open(os.path.join(d, "release"), "w").write("1")
    for p in procs:
        try:
            p.wait(timeout=120)
        except Exception:
            p.kill()
    time.sleep(3)
    rep["after"] = meminfo()
    rep["pass"] = per_rank > 0 and all(str(v).startswith("ok") and v.endswith("True") for v in rep["children"].values())
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("ple_total_GB", "per_rank_GB", "min_MemAvailable_GB", "sec", "pass")}), rep["children"])


if __name__ == "__main__":
    sys.exit(main())
