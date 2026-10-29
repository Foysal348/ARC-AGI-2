#!/usr/bin/env python3
"""P5 benchmark: 4-rank host-staged all-reduce (p5_host_ar.so) against NCCL on one node (Kaggle 4x L4, no P2P).

    python bench_host_ar.py --so /path/p5_host_ar.so --out /kaggle/working/p5_bench.json

Runs the user's test sequence, in order:
  (a) numerical oracle: FP32 / BF16 vs NCCL and vs an exact FP64 sum, random + adversarial inputs, sizes
      64 KB .. 3.9 MB plus odd sizes (partial units / groups / empty chunks), out-of-place and in-place, block variants;
      strict check: bit-identical to the ordered FP32-accumulate reference and identical on all ranks;
  (b) latency per call vs NCCL (eager and CUDA graph) at the six sizes, a block sweep at 983,040 B, a PCIe copy probe,
      then NCCL again in separate launches with NCCL_MIN_CTAS = NCCL_MAX_CTAS in {4, 8, 16, 32};
  (c) CUDA graph capture of the host-staged op: 1 call and 50 back-to-back calls, each graph replayed 1000x with new
      input data before every replay, every replay verified bit-exactly; epoch continuity checked;
  (d) INT8 payload arm (mode 1): error vs the FP64 oracle with a per-group analytic bound, cross-rank identity,
      latency vs the native host path and NCCL.
Writes ONE JSON (--out). Never hangs: the kernel's waits have timeouts (abort flag + status words), every worker group
runs under a wall-clock budget, and a global watchdog kills the process groups of stuck workers.
Exit codes: 0 all steps ran and every correctness check passed; 1 a correctness check failed; 2 watchdog kill /
worker crash / step error; 3 setup error (no GPUs, .so missing or not loadable).
The parent process never initializes CUDA; workers are separate processes (one per GPU).
"""
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

SIZES = [65536, 245760, 491520, 983040, 1966080, 3932160]
DECODE = 983040
# odd sizes in bytes: partial 8-element units, partial 32-element groups, empty rank chunks, decode size + 1 element
EDGE_BYTES = [2, 4, 16, 30, 62, 64, 126, 1000, 4098, 65538, 983042, 983136]
MAX_BYTES = 4 * 1024 * 1024
NCCL_CTAS = [4, 8, 16, 32]
PATTERNS = ["randn", "wide", "cancel", "large", "spikes", "tiny", "ramp", "nan_inf"]
INT8_PATTERNS = ["randn", "wide", "cancel", "spikes", "ramp", "outlier", "tiny", "tiny37", "nan_inf"]
INT8_FLOOR = 1e-36  # host_ar.cu kQuantFloor: 0 < amax < floor is quantized as amax = floor
EXIT_OK, EXIT_FAIL, EXIT_TIMEOUT, EXIT_SETUP = 0, 1, 2, 3
TARGET_US = 250.0
MEASURED_NCCL_US = 343.88


# ===================================================================================================== parent side
def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def sh(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"rc": r.returncode, "out": (r.stdout + r.stderr)[-6000:]}
    except Exception as e:  # noqa: BLE001
        return {"rc": -1, "out": f"{type(e).__name__}: {e}"}


def env_probe():
    env = {"python": sys.version.split()[0], "executable": sys.executable}
    env["nvidia_smi_L"] = sh(["nvidia-smi", "-L"])
    env["nvidia_smi_topo"] = sh(["nvidia-smi", "topo", "-m"])
    env["nvidia_smi_q"] = sh(["nvidia-smi", "--query-gpu=index,name,driver_version,pcie.link.gen.current,"
                              "pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max", "--format=csv"])
    env["numa"] = sh(["numactl", "-H"]) if shutil.which("numactl") else {"rc": -1, "out": "numactl not installed"}
    env["lscpu"] = sh(["lscpu"])
    try:
        st = os.statvfs("/dev/shm")
        env["dev_shm"] = {"total_mb": round(st.f_blocks * st.f_frsize / 2**20, 1),
                          "free_mb": round(st.f_bavail * st.f_frsize / 2**20, 1)}
    except Exception as e:  # noqa: BLE001
        env["dev_shm"] = {"error": str(e)}
    return env


def kill_group(procs):
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass
    for p in procs:
        try:
            p.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass


def nccl_log_summary(partdir):
    lines = []
    via = {}
    for f in sorted(Path(partdir).glob("nccl.*.txt")):
        try:
            for l in f.read_text(errors="replace").splitlines():
                if " via " in l:
                    tok = l.split(" via ", 1)[1].split()[0] if l.split(" via ", 1)[1].split() else "?"
                    via[tok] = via.get(tok, 0) + 1
                if ("channels" in l and ("coll" in l or "p2p" in l)) or "Init COMPLETE" in l or "NCCL_" in l \
                        and "set by environment" in l:
                    if len(lines) < 40:
                        lines.append(l.strip()[-300:])
        except Exception:  # noqa: BLE001
            pass
    return {"via_counts": via, "lines": lines}


def run_group(a, gname, steps, extra_env, budget_s, outdir):
    partdir = outdir / "parts" / gname
    if partdir.exists():
        shutil.rmtree(partdir)
    partdir.mkdir(parents=True)
    env = dict(os.environ)
    env.update({k: str(v) for k, v in extra_env.items()})
    env.setdefault("NCCL_DEBUG", "INFO")
    env.setdefault("NCCL_DEBUG_SUBSYS", "INIT,ENV,GRAPH")
    env["NCCL_DEBUG_FILE"] = str(partdir / "nccl.%h.%p.txt")
    env["PYTHONUNBUFFERED"] = "1"
    port = free_port()
    procs, logs = [], []
    t0 = time.time()
    for r in range(a.world):
        lf = open(partdir / f"rank{r}.log", "wb")
        logs.append(lf)
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--rank", str(r), "--world", str(a.world),
               "--port", str(port), "--so", a.so, "--partdir", str(partdir), "--steps", steps, "--group", gname,
               "--kernel-timeout-ms", str(a.kernel_timeout_ms)]
        if a.quick:
            cmd.append("--quick")
        procs.append(subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True))
    status = "ok"
    first_exit = None
    last_scan = time.time()
    while True:
        rcs = [p.poll() for p in procs]
        if all(rc is not None for rc in rcs):
            break
        now = time.time()
        if any(rc not in (None, 0) for rc in rcs) and first_exit is None:
            first_exit = now
        if first_exit is None and now - last_scan > 5:
            last_scan = now
            for pf in partdir.glob("part_r*.json"):
                try:
                    if json.loads(pf.read_text()).get("fatal"):
                        first_exit = now
                        break
                except Exception:  # noqa: BLE001 - file being replaced
                    pass
        if first_exit is not None and now - first_exit > 60:  # a crashed rank leaves the others blocked in NCCL
            status = "worker_crash_kill"
            kill_group(procs)
            break
        if now - t0 > budget_s:
            status = "watchdog_kill"
            kill_group(procs)
            break
        time.sleep(0.5)
    for lf in logs:
        lf.close()
    rcs = [p.returncode for p in procs]
    if status == "ok" and any(rc != 0 for rc in rcs):
        status = "worker_error"
    parts = {}
    for r in range(a.world):
        pf = partdir / f"part_r{r}.json"
        try:
            parts[r] = json.loads(pf.read_text())
        except Exception as e:  # noqa: BLE001
            parts[r] = {"missing": str(e)}
    tails = {}
    if status != "ok":
        for r in range(a.world):
            try:
                tails[r] = (partdir / f"rank{r}.log").read_text(errors="replace")[-4000:]
            except Exception:  # noqa: BLE001
                pass
    return {"status": status, "rcs": rcs, "sec": round(time.time() - t0, 1), "env": extra_env,
            "steps": steps, "nccl_log": nccl_log_summary(partdir), "rank0": parts.get(0, {}),
            "other_ranks": {r: {k: v for k, v in parts[r].items() if k in ("errors", "checks", "status")}
                            for r in parts if r != 0}, "log_tails": tails}


def parent(a):
    t0 = time.time()
    outp = Path(a.out)
    outdir = outp.parent / (outp.stem + "_work")
    outdir.mkdir(parents=True, exist_ok=True)
    res = {"meta": {"tool": "bench_host_ar.py", "version": 1, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                         time.gmtime()),
                    "so": a.so, "world": a.world, "quick": a.quick, "timeout_s": a.timeout_s,
                    "sizes": SIZES, "edge_bytes": EDGE_BYTES, "decode_bytes": DECODE,
                    "target_us": TARGET_US, "measured_nccl_us_reference": MEASURED_NCCL_US},
           "env": env_probe(), "groups": {}}
    code = EXIT_OK

    def finish(c):
        res["exit_code"] = c
        res["sec"] = round(time.time() - t0, 1)
        try:
            for f in Path("/dev/shm").glob("p5har_*"):  # names are unlinked after attach; this only catches crashes
                f.unlink()
        except Exception:  # noqa: BLE001
            pass
        tmp = outp.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=str))
        tmp.replace(outp)
        print("P5_BENCH", "exit", c, "| json", outp, "| sec", res["sec"], flush=True)
        print("P5_SUMMARY " + json.dumps(res.get("summary", {}), default=str)[:6000], flush=True)
        sys.exit(c)

    if not Path(a.so).exists():
        res["setup_error"] = f"missing {a.so}"
        finish(EXIT_SETUP)
    ngpu = res["env"]["nvidia_smi_L"]["out"].count("GPU ") if res["env"]["nvidia_smi_L"]["rc"] == 0 else 0
    if ngpu < a.world:
        res["setup_error"] = f"need {a.world} GPUs, nvidia-smi lists {ngpu}"
        finish(EXIT_SETUP)
    want = set(a.steps.split(","))
    plan = []
    if want & {"a", "b"}:
        plan.append(("main_ab", ",".join(s for s in ("a", "b") if s in want), {}, 900))
    if "b" in want:
        for x in NCCL_CTAS:
            plan.append((f"nccl_ctas{x}", "b_nccl", {"NCCL_MIN_CTAS": x, "NCCL_MAX_CTAS": x}, 240))
    if want & {"c", "d"}:
        plan.append(("main_cd", ",".join(s for s in ("c", "d") if s in want), {}, 900))
    for gname, steps, extra, gbudget in plan:
        remaining = t0 + a.timeout_s - time.time()
        if remaining < 45:
            res["groups"][gname] = {"status": "skipped_no_time"}
            code = max(code, EXIT_TIMEOUT)
            continue
        print(f"[p5] group {gname} steps={steps} env={extra} budget={min(gbudget, remaining):.0f}s", flush=True)
        g = run_group(a, gname, steps, extra, min(gbudget, remaining - 15), outdir)
        res["groups"][gname] = g
        print(f"[p5] group {gname} -> {g['status']} rcs={g['rcs']} {g['sec']}s", flush=True)
        if g["status"] != "ok":
            code = max(code, EXIT_TIMEOUT)
        if g["rank0"].get("setup_error"):
            code = max(code, EXIT_SETUP)
    res["summary"] = summarize(res)
    if code == EXIT_OK and not res["summary"].get("all_checks_pass"):
        code = EXIT_FAIL
    if code == EXIT_TIMEOUT and res["summary"].get("any_check_failed"):
        res["summary"]["note"] = "a correctness check failed AND a group did not finish (exit 2 wins)"
    finish(code)


def summarize(res):
    s = {}
    g = res["groups"]
    r0 = lambda name: (g.get(name) or {}).get("rank0", {})  # noqa: E731
    checks = {}
    for name in g:
        for k, v in (r0(name).get("checks") or {}).items():
            checks[f"{name}.{k}"] = v
        for rk, other in ((g[name] or {}).get("other_ranks") or {}).items():
            for k, v in (other.get("checks") or {}).items():
                checks[f"{name}.r{rk}.{k}"] = v
    s["checks"] = checks
    s["any_check_failed"] = any(v is False for v in checks.values())
    s["all_checks_pass"] = bool(checks) and all(v is True for v in checks.values())
    s["step_errors"] = {name: r0(name).get("errors") for name in g if r0(name).get("errors")}
    if s["step_errors"]:
        s["all_checks_pass"] = False
    b = r0("main_ab").get("b") or {}
    lat = b.get("latency") or {}
    dec = lat.get(str(DECODE)) or {}
    s["decode_983040_us"] = {k: dec.get(k) for k in ("nccl_eager", "nccl_graph", "torch_nccl_eager", "host_eager",
                                                     "host_graph")}
    s["host_graph_best_us_983040"] = (b.get("block_sweep") or {}).get("best_us")
    s["host_graph_best_blocks"] = (b.get("block_sweep") or {}).get("best_blocks")
    tuned = {}
    for x in NCCL_CTAS:
        t = (r0(f"nccl_ctas{x}").get("b_nccl") or {}).get(str(DECODE)) or {}
        tuned[x] = {"nccl_eager": t.get("nccl_eager"), "nccl_graph": t.get("nccl_graph")}
    s["nccl_tuned_983040_us"] = tuned
    d = r0("main_cd").get("d") or {}
    s["int8_graph_us_983040"] = ((d.get("latency") or {}).get(str(DECODE)) or {}).get("host_int8_graph")
    best = [v for v in (s["host_graph_best_us_983040"], (dec or {}).get("host_graph")) if isinstance(v, (int, float))]
    s["host_best_us_983040"] = min(best) if best else None
    s["target_250us_met_native"] = (s["host_best_us_983040"] is not None and s["host_best_us_983040"] <= TARGET_US)
    s["target_250us_met_int8"] = (isinstance(s["int8_graph_us_983040"], (int, float))
                                  and s["int8_graph_us_983040"] <= TARGET_US)
    ng = [v for v in [dec.get("nccl_graph")] + [t["nccl_graph"] for t in tuned.values()] if isinstance(v, (int, float))]
    s["nccl_best_graph_us_983040"] = min(ng) if ng else None
    if s["host_best_us_983040"] and s["nccl_best_graph_us_983040"]:
        s["speedup_vs_best_nccl"] = round(s["nccl_best_graph_us_983040"] / s["host_best_us_983040"], 3)
    return s


# ===================================================================================================== worker side
class Part:
    def __init__(self, partdir, rank):
        self.path = Path(partdir) / f"part_r{rank}.json"
        self.d = {"rank": rank, "errors": [], "checks": {}, "progress": []}
        self.save()

    def __getitem__(self, k):
        return self.d[k]

    def __setitem__(self, k, v):
        self.d[k] = v
        self.save()

    def check(self, name, ok):
        self.d["checks"][name] = bool(ok)
        self.save()

    def progress(self, msg):
        self.d["progress"].append([round(time.time(), 1), msg])
        self.d["progress"] = self.d["progress"][-60:]
        self.save()
        print(f"[r{self.d['rank']}] {msg}", flush=True)

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1, default=str))
        tmp.replace(self.path)


class NcclC:
    """Minimal ctypes NCCL communicator (same libnccl as torch) for eager + graph-captured baselines on the current
    torch stream (vLLM's pynccl does the same)."""
    TYPES = {"torch.float32": 7, "torch.float16": 6, "torch.bfloat16": 9}

    def __init__(self, torch, dist, rank, world):
        import ctypes
        self.ct = ctypes
        self.torch = torch
        path = None
        try:
            import nvidia
            for base in nvidia.__path__:
                for c in sorted(Path(base).glob("nccl/lib/libnccl.so*")):
                    path = str(c)
                    break
        except Exception:  # noqa: BLE001
            pass
        self.lib = ctypes.CDLL(path or "libnccl.so.2")
        self.path = path or "libnccl.so.2"

        class UID(ctypes.Structure):
            _fields_ = [("internal", ctypes.c_byte * 128)]
        self.UID = UID
        lib = self.lib
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(UID)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UID, ctypes.c_int]
        lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_void_p]
        lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
        lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        uid = UID()
        if rank == 0:
            assert lib.ncclGetUniqueId(ctypes.byref(uid)) == 0
        obj = [bytes(uid.internal) if rank == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        ctypes.memmove(uid.internal, obj[0], 128)
        self.comm = ctypes.c_void_p()
        rc = lib.ncclCommInitRank(ctypes.byref(self.comm), world, uid, rank)
        assert rc == 0, f"ncclCommInitRank rc={rc}"
        v = ctypes.c_int()
        lib.ncclGetVersion(ctypes.byref(v))
        self.version = v.value

    def all_reduce(self, x, y):
        s = self.torch.cuda.current_stream().cuda_stream
        rc = self.lib.ncclAllReduce(self.ct.c_void_p(x.data_ptr()), self.ct.c_void_p(y.data_ptr()), x.numel(),
                                    self.TYPES[str(x.dtype)], 0, self.comm, self.ct.c_void_p(s))
        if rc != 0:
            raise RuntimeError(f"ncclAllReduce rc={rc}")

    def close(self):
        try:
            self.lib.ncclCommDestroy(self.comm)
        except Exception:  # noqa: BLE001
            pass


class Ctx:
    def __init__(self, a):
        import torch
        import torch.distributed as dist
        from datetime import timedelta
        self.a, self.torch, self.dist = a, torch, dist
        self.rank, self.world = a.rank, a.world
        self.part = Part(a.partdir, a.rank)
        torch.cuda.set_device(self.rank)
        self.dev = torch.device("cuda", self.rank)
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{a.port}", rank=self.rank,
                                world_size=self.world, timeout=timedelta(seconds=300), device_id=self.dev)
        torch.ops.load_library(a.so)
        self.ops = torch.ops.p5_host_ar
        self.part["info"] = self.ops.info()
        self.part["torch"] = torch.__version__
        self.part["torch_nccl_version"] = ".".join(map(str, torch.cuda.nccl.version()))
        self.part["gpu"] = torch.cuda.get_device_name(self.rank)
        self.part["env_nccl"] = {k: v for k, v in os.environ.items() if k.startswith("NCCL_") and "FILE" not in k}
        self.nc = NcclC(torch, dist, self.rank, self.world)
        self.part["nccl_ctypes"] = {"path": self.nc.path, "version": self.nc.version}
        self.h = None
        self.n_handles = 0
        self.new_handle()
        x = torch.ones(1024, device=self.dev)
        dist.all_reduce(x)  # warm the torch NCCL communicator
        torch.cuda.synchronize()

    # -- handle management (collective)
    def new_handle(self):
        if self.h is not None:
            try:
                self.torch.cuda.synchronize()
                self.ops.destroy(self.h)
            except Exception as e:  # noqa: BLE001
                self.part["errors"].append(f"destroy: {e}")
        name = [f"/p5har_{os.getpid()}_{time.time_ns()}_{self.n_handles}" if self.rank == 0 else None]
        self.dist.broadcast_object_list(name, src=0)
        self.h = self.ops.init(self.rank, self.world, name[0], MAX_BYTES, self.rank, self.a.kernel_timeout_ms)
        self.n_handles += 1
        st = self.ops.status(self.h)
        self.max_blocks = int(st[-1])
        self.part["max_coresident_blocks"] = self.max_blocks
        self.part["handles_created"] = self.n_handles

    def status_ok(self, where):
        """Collective: every rank learns whether any rank's handle reported an error; on error re-init."""
        st = list(self.ops.status(self.h))
        bad = 1 if (st[0] != 0 or any(v != 0 for v in st[1:-1])) else 0
        t = self.torch.tensor([bad], device=self.dev, dtype=self.torch.int32)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.MAX)
        if int(t.item()):
            self.part["errors"].append({"where": where, "status": st,
                                        "decoded": [decode_status(v) for v in st[1:-1]]})
            self.new_handle()
            return False
        return True

    def barrier(self):
        self.torch.cuda.synchronize()
        self.dist.barrier(device_ids=[self.rank])
        self.torch.cuda.synchronize()

    def gather_obj(self, obj):
        out = [None] * self.world
        self.dist.all_gather_object(out, obj)
        return out


def decode_status(v):
    v = int(v)
    if v == 0:
        return "ok"
    names = {1: "timeout", 2: "desync", 3: "peer_abort", 4: "signature_mismatch"}
    return f"{names.get(v >> 24, v >> 24)} phase={(v >> 16) & 0xFF} block={v & 0xFFFF}"


# ----------------------------------------------------------------------------------------------- inputs / refs
def gen(torch, dev, seed, rank, n, dtype, pattern):
    g = torch.Generator(device=dev)
    g.manual_seed((seed * 1000003 + rank * 7919 + 17) % (2**63 - 1))
    x = torch.randn(n, generator=g, device=dev, dtype=torch.float32)
    finfo = torch.finfo(dtype)
    if pattern == "wide":
        x = x * torch.pow(10.0, torch.rand(n, generator=g, device=dev) * 10.0 - 6.0)
    elif pattern == "cancel":
        g2 = torch.Generator(device=dev)
        g2.manual_seed(seed * 31 + 5)
        base = torch.randn(n, generator=g2, device=dev) * 1e3
        x = base * (1.0 if rank % 2 == 0 else -1.0) + 1e-3 * x
    elif pattern == "large":
        x = (x * (finfo.max / 16)).clamp(-finfo.max / 4, finfo.max / 4)
    elif pattern == "spikes":
        m = torch.rand(n, generator=g, device=dev) < 0.002
        x = torch.where(m, torch.sign(x) * 1e4, torch.zeros_like(x))
    elif pattern == "tiny":
        x = x * 1e-39
    elif pattern == "tiny37":  # amax ~1e-37: below the INT8 floor, where 127 / amax would overflow fp32
        x = x * 3e-38
    elif pattern == "ramp":
        x = (torch.arange(n, device=dev, dtype=torch.float32) % 251) * float(rank + 1) - 100.0
    elif pattern == "nan_inf":
        k = torch.rand(n, generator=g, device=dev)
        x = torch.where(k < 0.001, torch.full_like(x, float("nan")), x)
        x = torch.where((k >= 0.001) & (k < 0.002), torch.full_like(x, float("inf")), x)
        x = torch.where((k >= 0.002) & (k < 0.003), torch.full_like(x, float("-inf")), x)
    elif pattern == "outlier":  # one large value per 32-group: the rest quantize coarsely (worst case for INT8)
        idx = torch.arange(n, device=dev)
        x = torch.where(idx % 32 == (rank * 7) % 32, x * 1e3, x)
    return x.to(dtype)


def ordered_ref(torch, xs, dtype):
    acc = xs[0].float()
    for x in xs[1:]:
        acc = acc + x.float()
    return acc.to(dtype)


def bits(torch, t):
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def bit_equal(torch, a, b):
    """bit-identical, except NaN == NaN regardless of payload; returns (all_equal, n_diff)."""
    same = bits(torch, a) == bits(torch, b)
    both_nan = torch.isnan(a) & torch.isnan(b)
    diff = int((~(same | both_nan)).sum().item())
    return diff == 0, diff


def err_stats(torch, out, exact, addend_abs_sum=None):
    o = out.double()
    fin = torch.isfinite(exact) & torch.isfinite(o)
    nonfinite_mismatch = int(((torch.isnan(o) != torch.isnan(exact)) |
                              (torch.isinf(o) & torch.isinf(exact) & (o != exact)) |
                              (torch.isinf(o) != torch.isinf(exact))).sum().item())
    if int(fin.sum().item()) == 0:
        return {"max_abs": 0.0, "max_rel": 0.0, "rel_l2": 0.0, "nonfinite_mismatch": nonfinite_mismatch}
    e = (o - exact)[fin]
    ex = exact[fin]
    r = {"max_abs": float(e.abs().max().item()),
         "max_rel": float((e.abs() / ex.abs().clamp_min(1e-30)).max().item()),
         "rel_l2": float((e.norm() / ex.norm().clamp_min(1e-300)).item()),
         "nonfinite_mismatch": nonfinite_mismatch}
    if addend_abs_sum is not None:
        r["max_err_over_abs_addends"] = float((e.abs() / addend_abs_sum[fin].clamp_min(1e-300)).max().item())
    return r


# ----------------------------------------------------------------------------------------------- step (a)
def step_a(c):
    torch, P = c.torch, c.part
    res = {"cases": [], "fail": []}
    dts = [("float32", torch.float32), ("bfloat16", torch.bfloat16)]
    pats = PATTERNS if not c.a.quick else ["randn", "cancel", "nan_inf"]
    cases = []
    for dname, dt in dts:
        for nbytes in SIZES:
            for pat in pats:
                cases.append((dname, dt, nbytes, pat, False, 0))
        for nbytes in EDGE_BYTES:
            if nbytes % torch.tensor([], dtype=dt).element_size() == 0:
                cases.append((dname, dt, nbytes, "randn", False, 0))
        cases.append((dname, dt, DECODE, "randn", True, 0))  # in place
        for b in (1, 7, c.max_blocks):  # explicit grids, up to the co-resident limit (auto = 30 at this size)
            cases.append((dname, dt, DECODE, "randn", False, b))
    all_bitexact = all_ident = all_status = True
    for ci, (dname, dt, nbytes, pat, inplace, blocks) in enumerate(cases):
        n = nbytes // torch.tensor([], dtype=dt).element_size()
        seed = 1000 + ci
        xs = [gen(torch, c.dev, seed, j, n, dt, pat) for j in range(c.world)]
        mine = xs[c.rank].clone()
        y_nccl = mine.clone()
        c.nc.all_reduce(y_nccl, y_nccl)
        if inplace:
            out = mine.clone()
            c.ops.all_reduce(c.h, out, out, 0, blocks)
        else:
            out = torch.empty_like(mine)
            c.ops.all_reduce(c.h, mine, out, 0, blocks)
        torch.cuda.synchronize()
        ok_status = c.status_ok(f"a:{dname}:{nbytes}:{pat}")
        ref = ordered_ref(torch, xs, dt)
        exact = torch.stack([x.double() for x in xs]).sum(0)
        abs_add = torch.stack([x.double().abs() for x in xs]).sum(0)
        beq, ndiff = bit_equal(torch, out, ref)
        gl = [torch.empty_like(out) for _ in range(c.world)]
        c.dist.all_gather(gl, out)
        ident = all(bool(torch.equal(bits(torch, gl[0]), bits(torch, g))) for g in gl[1:])
        nb_eq, nb_diff = bit_equal(torch, out, y_nccl)
        case = {"dtype": dname, "bytes": nbytes, "n": n, "pattern": pat, "inplace": inplace,
                "blocks": blocks or "auto", "bitexact_vs_ordered_ref": beq, "ref_diff_elems": ndiff,
                "identical_all_ranks": ident, "status_ok": ok_status,
                "vs_nccl": {"bit_equal": nb_eq, "diff_elems": nb_diff, **err_stats(torch, out, y_nccl.double())},
                "host_vs_exact": err_stats(torch, out, exact, abs_add),
                "nccl_vs_exact": err_stats(torch, y_nccl, exact, abs_add)}
        res["cases"].append(case)
        if not (beq and ident and ok_status):
            res["fail"].append({k: case[k] for k in ("dtype", "bytes", "pattern", "inplace", "blocks",
                                                     "ref_diff_elems", "identical_all_ranks", "status_ok")})
        all_bitexact &= beq
        all_ident &= ident
        all_status &= ok_status
        if ci % 10 == 0:
            P.progress(f"a: case {ci + 1}/{len(cases)}")
            P["a"] = res
    # aggregate: worst errors per dtype (random patterns only, finite)
    agg = {}
    for case in res["cases"]:
        k = case["dtype"]
        a_ = agg.setdefault(k, {"host_vs_exact_max_rel_randn": 0.0, "nccl_vs_exact_max_rel_randn": 0.0,
                                "host_vs_nccl_max_abs": 0.0, "host_vs_nccl_max_rel": 0.0,
                                "host_max_err_over_abs_addends": 0.0, "nccl_max_err_over_abs_addends": 0.0})
        if case["pattern"] == "randn":
            a_["host_vs_exact_max_rel_randn"] = max(a_["host_vs_exact_max_rel_randn"], case["host_vs_exact"]["max_rel"])
            a_["nccl_vs_exact_max_rel_randn"] = max(a_["nccl_vs_exact_max_rel_randn"], case["nccl_vs_exact"]["max_rel"])
        a_["host_vs_nccl_max_abs"] = max(a_["host_vs_nccl_max_abs"], case["vs_nccl"]["max_abs"])
        a_["host_vs_nccl_max_rel"] = max(a_["host_vs_nccl_max_rel"], case["vs_nccl"]["max_rel"])
        a_["host_max_err_over_abs_addends"] = max(a_["host_max_err_over_abs_addends"],
                                                  case["host_vs_exact"].get("max_err_over_abs_addends", 0.0))
        a_["nccl_max_err_over_abs_addends"] = max(a_["nccl_max_err_over_abs_addends"],
                                                  case["nccl_vs_exact"].get("max_err_over_abs_addends", 0.0))
    res["aggregate"] = agg
    res["n_cases"] = len(cases)
    P["a"] = res
    P.check("a_bitexact_vs_ordered_fp32_ref", all_bitexact)
    P.check("a_identical_all_ranks", all_ident)
    P.check("a_no_kernel_errors", all_status)


# ----------------------------------------------------------------------------------------------- step (b)
def time_eager(c, fn, iters, reps):
    torch = c.torch
    for _ in range(5):
        fn()
    c.barrier()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        c.barrier()
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e) * 1000.0 / iters)
    return out


def capture(c, fn, calls):
    torch = c.torch
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    c.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()
    return g


def time_graph(c, fn, calls, replays, reps):
    torch = c.torch
    g = capture(c, fn, calls)
    g.replay()
    c.barrier()
    out = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        c.barrier()
        s.record()
        for _ in range(replays):
            g.replay()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e) * 1000.0 / (replays * calls))
    del g
    return out


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def collective_med(c, vals):
    """median per rank, then the max over ranks (a collective is as slow as its slowest rank)."""
    allv = c.gather_obj(med(vals))
    allv = [v for v in allv if v is not None]
    return round(max(allv), 2) if allv else None


def nccl_latency(c, sizes, P, key):
    torch = c.torch
    res = {}
    iters, reps, calls, replays = (50, 3, 20, 5) if c.a.quick else (200, 5, 50, 20)
    for nbytes in sizes:
        n = nbytes // 2
        x = torch.randn(n, device=c.dev).to(torch.bfloat16)
        y = torch.empty_like(x)
        r = {}
        r["nccl_eager"] = collective_med(c, time_eager(c, lambda: c.nc.all_reduce(x, y), iters, reps))
        try:
            r["nccl_graph"] = collective_med(c, time_graph(c, lambda: c.nc.all_reduce(x, y), calls, replays, reps))
        except Exception as e:  # noqa: BLE001
            r["nccl_graph_error"] = f"{type(e).__name__}: {e}"
        r["torch_nccl_eager"] = collective_med(c, time_eager(c, lambda: c.dist.all_reduce(x), iters, reps))
        res[str(nbytes)] = r
        P.progress(f"{key}: {nbytes} B {r}")
    return res


def pcie_probe(c):
    torch = c.torch
    nb = 8 * 2**20
    h = torch.empty(nb, dtype=torch.uint8, pin_memory=True)
    d = torch.empty(nb, dtype=torch.uint8, device=c.dev)

    def bw(direction, reps=10):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(2):
            (d.copy_(h, non_blocking=True) if direction == "h2d" else h.copy_(d, non_blocking=True))
        torch.cuda.synchronize()
        s.record()
        for _ in range(reps):
            (d.copy_(h, non_blocking=True) if direction == "h2d" else h.copy_(d, non_blocking=True))
        e.record()
        e.synchronize()
        return round(nb * reps / (s.elapsed_time(e) / 1000.0) / 1e9, 2)

    out = {"bytes": nb}
    for direction in ("h2d", "d2h"):
        solo = None
        for r in range(c.world):
            c.barrier()
            if c.rank == r:
                solo = bw(direction)
            c.barrier()
        out[f"{direction}_solo_GBps"] = c.gather_obj(solo)
        c.barrier()
        conc = bw(direction)
        out[f"{direction}_concurrent_GBps"] = c.gather_obj(conc)
    return out


def step_b(c):
    torch, P = c.torch, c.part
    res = {"pcie_probe": pcie_probe(c)}
    P["b"] = res
    iters, reps, calls, replays = (50, 3, 20, 5) if c.a.quick else (200, 5, 50, 20)
    res["latency"] = nccl_latency(c, SIZES, P, "b")
    for nbytes in SIZES:
        n = nbytes // 2
        x = torch.randn(n, device=c.dev).to(torch.bfloat16)
        y = torch.empty_like(x)
        r = res["latency"][str(nbytes)]
        r["host_eager"] = collective_med(c, time_eager(c, lambda: c.ops.all_reduce(c.h, x, y, 0, 0), iters, reps))
        try:
            r["host_graph"] = collective_med(c, time_graph(c, lambda: c.ops.all_reduce(c.h, x, y, 0, 0), calls,
                                                           replays, reps))
        except Exception as e:  # noqa: BLE001
            r["host_graph_error"] = f"{type(e).__name__}: {e}"
        r["host_status_ok"] = c.status_ok(f"b:{nbytes}")
        P.progress(f"b: host {nbytes} B eager={r.get('host_eager')} graph={r.get('host_graph')}")
        P["b"] = res
    # block sweep at the decode size (graph)
    n = DECODE // 2
    x = torch.randn(n, device=c.dev).to(torch.bfloat16)
    y = torch.empty_like(x)
    sweep = {}
    for b in sorted(set([4, 8, 12, 16, 24, 30, 32, 40, 48, c.max_blocks])):
        if b > c.max_blocks:
            continue
        try:
            sweep[str(b)] = collective_med(c, time_graph(c, lambda: c.ops.all_reduce(c.h, x, y, 0, b), calls,
                                                         replays, reps))
        except Exception as e:  # noqa: BLE001
            sweep[str(b)] = f"error {type(e).__name__}: {e}"
        c.status_ok(f"b:sweep:{b}")
    nums = {k: v for k, v in sweep.items() if isinstance(v, (int, float))}
    best = min(nums, key=nums.get) if nums else None
    res["block_sweep"] = {"us_by_blocks": sweep, "best_blocks": best, "best_us": nums.get(best) if best else None}
    P.progress(f"b: block sweep {res['block_sweep']}")
    # bandwidth model check: bytes moved per GPU per call
    res["model"] = {"bytes_written_per_gpu": DECODE, "bytes_read_per_gpu": DECODE * 3 // 2}
    P["b"] = res
    ok = all(res["latency"][str(s)].get("host_status_ok") for s in SIZES)
    P.check("b_no_kernel_errors", ok)


def step_b_nccl(c):
    c.part["b_nccl"] = nccl_latency(c, SIZES, c.part, "b_nccl")


# ----------------------------------------------------------------------------------------------- step (c)
def step_c(c):
    torch, P = c.torch, c.part
    K = 50
    R = 100 if c.a.quick else 1000
    n = DECODE // 2
    dt = torch.bfloat16
    base = [gen(torch, c.dev, 900, j, K * n, dt, "randn").view(K, n) for j in range(c.world)]
    X = torch.empty(K, n, device=c.dev, dtype=dt)
    Y = torch.empty(K, n, device=c.dev, dtype=dt)
    res = {"calls_per_graph": {}, "replays": R, "n_elems": n}

    def fill(xdst, src, s):
        a_ = 1.0 + ((s * 37) % 16) / 16.0  # 5 significant bits: data change every replay, deterministic everywhere
        xdst.copy_(torch.roll(src, shifts=s % 97, dims=1) * a_)

    all_ok = True
    for kc in (1, K):
        X.zero_()
        Y.zero_()

        def body():
            for k in range(kc):
                c.ops.all_reduce(c.h, X[k], Y[k], 0, 0)
        torch.cuda.synchronize()
        e0 = c.ops.epoch(c.h)
        g = capture(c, body, 1)  # one capture holding kc calls (capture() also runs 3 eager warm-up rounds)
        e_cap = c.ops.epoch(c.h)
        mism_total, first_bad, mism_replays = 0, None, 0
        handle_lost = False
        t_ver = time.time()
        for s in range(R):
            fill(X[:kc], base[c.rank][:kc], s)
            g.replay()
            refx = [torch.roll(base[j][:kc], shifts=s % 97, dims=1) * (1.0 + ((s * 37) % 16) / 16.0)
                    for j in range(c.world)]
            ref = ordered_ref(torch, refx, dt)
            d = int((bits(torch, Y[:kc]) != bits(torch, ref)).sum().item())
            if d:
                mism_total += d
                mism_replays += 1
                if first_bad is None:
                    first_bad = s
            if s % 100 == 99:
                if not c.status_ok(f"c:{kc}:{s}"):  # re-init destroyed the captured handle: g must never replay
                    all_ok = False
                    handle_lost = True
                    break
            if s == 0 and kc == K:
                # eager call between replays: epoch continuity across graph and eager launches
                xe = base[c.rank][0].clone()
                ye = torch.empty_like(xe)
                c.ops.all_reduce(c.h, xe, ye, 0, 0)
                refe = ordered_ref(torch, [base[j][0] for j in range(c.world)], dt)
                res["eager_between_replays_ok"] = bool(torch.equal(bits(torch, ye), bits(torch, refe)))
        torch.cuda.synchronize()
        ver_s = time.time() - t_ver
        if handle_lost:
            res["calls_per_graph"][str(kc)] = {"handle_lost_at_replay": s, "mismatch_elems_this_rank": mism_total,
                                               "first_bad_replay": first_bad, "verify_sec": round(ver_s, 1),
                                               "note": "kernel error -> handle re-created; timing/epoch skipped"}
            P["c"] = res
            del g
            continue
        e_end = c.ops.epoch(c.h)
        # timing without verification
        c.barrier()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        reps_t = 20 if c.a.quick else 100
        ev0.record()
        for _ in range(reps_t):
            g.replay()
        ev1.record()
        ev1.synchronize()
        per_call = ev0.elapsed_time(ev1) * 1000.0 / (reps_t * kc)
        torch.cuda.synchronize()
        e_after = c.ops.epoch(c.h)
        status_ok = c.status_ok(f"c:{kc}:end")
        epochs = c.gather_obj([e0, e_cap, e_end, e_after])
        expected_end = e_cap + kc * R + (1 if kc == K else 0)
        ep_ok = all(ep == epochs[0] for ep in epochs) and e_end == expected_end and e_after == e_end + reps_t * kc
        mism_all = c.gather_obj(mism_total)
        r = {"mismatch_elems_this_rank": mism_total, "mismatch_replays": mism_replays, "first_bad_replay": first_bad,
             "mismatch_elems_all_ranks": mism_all, "verify_sec": round(ver_s, 1),
             "epochs_[start,after_capture,after_replays,after_timing]_all_ranks": epochs,
             "epoch_expected_after_replays": expected_end, "epoch_ok": ep_ok, "status_ok": status_ok,
             "per_call_us_in_graph": collective_med(c, [per_call])}
        res["calls_per_graph"][str(kc)] = r
        all_ok &= (sum(mism_all) == 0) and ep_ok and status_ok
        P.progress(f"c: graph {kc} calls: {r}")
        P["c"] = res
        del g
    P["c"] = res
    P.check("c_graph_replays_bitexact", all_ok)
    P.check("c_eager_between_replays", res.get("eager_between_replays_ok", False))
    # Error path must be LOUD: a deliberately mismatched call (rank 0 grid 4, the others grid 8) fails the call
    # signature carried in every flag -> every block of every rank errors -> out is all-NaN on every rank; the sticky
    # abort keeps the next (correct) call all-NaN too; a collective re-init restores correct results.
    # (Handled here directly, not through status_ok, so the intended error is not counted as a kernel failure.)
    xe = base[c.rank][0].clone()
    y1 = torch.zeros_like(xe)
    c.ops.all_reduce(c.h, xe, y1, 0, 4 if c.rank == 0 else 8)
    torch.cuda.synchronize()
    st1 = [int(v) for v in c.ops.status(c.h)]
    y2 = torch.zeros_like(xe)
    c.ops.all_reduce(c.h, xe, y2, 0, 0)
    torch.cuda.synchronize()
    st2 = [int(v) for v in c.ops.status(c.h)]
    nan1 = bool(torch.isnan(y1.float()).all().item())
    nan2 = bool(torch.isnan(y2.float()).all().item())
    c.new_handle()  # collective: every rank reached this point (the poisoned calls terminate by design)
    y3 = torch.empty_like(xe)
    c.ops.all_reduce(c.h, xe, y3, 0, 0)
    torch.cuda.synchronize()
    ref3 = ordered_ref(torch, [base[j][0] for j in range(c.world)], dt)
    ok3 = bool(torch.equal(bits(torch, y3), bits(torch, ref3)))
    clean3 = c.status_ok("c:error_path:after_reinit")
    me = st1[1 + c.rank]
    ep = {"status_after_bad_call": st1, "decoded": decode_status(me), "status_after_next_call": st2,
          "bad_call_all_nan": nan1, "next_call_all_nan": nan2, "after_reinit_bitexact": ok3,
          "after_reinit_status_ok": clean3,
          "root_cause_kept": st2[1 + c.rank] == me and (me >> 24) in (4, 3) and st1[0] == 1}
    ep["all_ranks"] = c.gather_obj([nan1, nan2, ok3, me >> 24])
    res["error_path"] = ep
    P["c"] = res
    P.progress(f"c: error path {ep}")
    P.check("c_error_path_loud", nan1 and nan2 and ok3 and clean3 and ep["root_cause_kept"]
            and any(code == 4 for *_, code in ep["all_ranks"]))


# ----------------------------------------------------------------------------------------------- step (d)
def int8_bound(torch, xs, exact, n, world, out_dtype):
    """Per-element analytic error bound of the INT8 arm (finite groups): each non-owner contribution is quantized once
    (|err| <= max(amax_group, floor) / 254), the owner's sum is quantized once more (<= max(amax(acc), floor) / 254),
    plus output rounding."""
    G = 32
    per_rank = -(-n // world)                 # ceil(n / world)
    chunk = -(-per_rank // G) * G             # round_up(ceil(n / world), 32) = host_ar_layout.h chunk_len
    npad = -(-n // G) * G
    gidx = torch.arange(npad // G, device=exact.device)
    owner = (gidx * G) // chunk
    peer = torch.zeros(npad // G, device=exact.device, dtype=torch.float64)
    for j, x in enumerate(xs):
        a = torch.zeros(npad, device=exact.device, dtype=torch.float64)
        a[:n] = x.double().abs()
        am = a.view(-1, G).amax(1)
        am = torch.where(am > 0, am.clamp_min(INT8_FLOOR), am)
        peer += torch.where(owner == j, torch.zeros_like(am), am / 254.0)
    ex = torch.zeros(npad, device=exact.device, dtype=torch.float64)
    ex[:n] = exact.abs()
    am_acc = (ex.view(-1, G).amax(1) + peer).clamp_min(INT8_FLOOR)
    gb = (peer + am_acc / 254.0) * 1.001
    per = gb.repeat_interleave(G)[:n]
    ulp = 2.0 ** -8 if out_dtype in (torch.bfloat16,) else (2.0 ** -11 if out_dtype == torch.float16 else 2.0 ** -23)
    add = torch.stack([x.double().abs() for x in xs]).sum(0)
    sub = 1e-40 if out_dtype == torch.bfloat16 else (3e-8 if out_dtype == torch.float16 else 1e-44)  # subnormal step
    return per + (exact.abs() + per) * ulp + add * 1e-6 + sub


def int8_eval(torch, xs, out8, n, world, dt):
    """INT8 arm semantics: a 32-element group with any non-finite input (any rank) must come out all-NaN; every other
    element must be finite and within int8_bound. Shared by step (d) and tests/emulate_numerics.py."""
    G = 32
    npad = -(-n // G) * G
    nfm = torch.zeros(npad, dtype=torch.bool, device=out8.device)
    for x in xs:
        nfm[:n] |= ~torch.isfinite(x)
    bad_group = nfm.view(-1, G).any(1).repeat_interleave(G)[:n]
    exact = torch.stack([x.double() for x in xs]).sum(0)
    o = out8.double()
    nan_ok = bool(torch.isnan(o[bad_group]).all().item()) if bool(bad_group.any().item()) else True
    fin = ~bad_group
    r = {"nonfinite_groups": int(nfm.view(-1, G).any(1).sum().item()), "nonfinite_groups_all_nan": nan_ok}
    if bool(fin.any().item()):
        xs_f = [torch.where(bad_group, torch.zeros_like(x), x) for x in xs]  # bound only over finite groups
        ex_f = torch.where(bad_group, torch.zeros_like(exact), exact)
        bound = int8_bound(torch, xs_f, ex_f, n, world, dt)[fin]
        err = (o[fin] - exact[fin]).abs()
        within = err <= bound  # NaN err -> False
        r["within_bound_frac"] = float(within.double().mean().item())
        r["worst_err_over_bound"] = float((err / bound).nan_to_num(nan=float("inf")).max().item())
        r["int8_vs_exact"] = err_stats(torch, o[fin], exact[fin])
    else:
        r["within_bound_frac"], r["worst_err_over_bound"] = 1.0, 0.0
    r["ok"] = nan_ok and r["within_bound_frac"] == 1.0
    return r


def step_d(c):
    torch, P = c.torch, c.part
    res = {"cases": [], "latency": {}}
    pats = INT8_PATTERNS if not c.a.quick else ["randn", "outlier"]
    all_ident = all_bound = all_status = True
    for dname, dt in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
        for nbytes in SIZES + [983042 if dt == torch.bfloat16 else 983044, 1000]:
            for pat in pats:
                n = nbytes // torch.tensor([], dtype=dt).element_size()
                seed = 5000 + nbytes + 17 * INT8_PATTERNS.index(pat)
                xs = [gen(torch, c.dev, seed, j, n, dt, pat) for j in range(c.world)]
                mine = xs[c.rank]
                out8 = torch.empty_like(mine)
                outn = torch.empty_like(mine)
                c.ops.all_reduce(c.h, mine, out8, 1, 0)
                c.ops.all_reduce(c.h, mine, outn, 0, 0)
                torch.cuda.synchronize()
                st = c.status_ok(f"d:{dname}:{nbytes}:{pat}")
                exact = torch.stack([x.double() for x in xs]).sum(0)
                ev = int8_eval(torch, xs, out8, n, c.world, dt)
                gl = [torch.empty_like(out8) for _ in range(c.world)]
                c.dist.all_gather(gl, out8)
                ident = all(bool(torch.equal(bits(torch, gl[0]), bits(torch, g))) for g in gl[1:])
                case = {"dtype": dname, "bytes": nbytes, "pattern": pat, "status_ok": st, "identical_all_ranks": ident,
                        **ev, "native_vs_exact": err_stats(torch, outn, exact),
                        "int8_vs_native": err_stats(torch, out8, outn.double())}
                res["cases"].append(case)
                all_ident &= ident
                all_bound &= ev["ok"]
                all_status &= st
            P["d"] = res
        P.progress(f"d: {dname} done")
    # latency: INT8 vs native host (graph), bf16
    calls, replays, reps = (20, 5, 3) if c.a.quick else (50, 20, 5)
    for nbytes in SIZES:
        n = nbytes // 2
        x = torch.randn(n, device=c.dev).to(torch.bfloat16)
        y = torch.empty_like(x)
        r = {}
        try:
            r["host_int8_graph"] = collective_med(c, time_graph(c, lambda: c.ops.all_reduce(c.h, x, y, 1, 0), calls,
                                                                replays, reps))
            r["host_native_graph"] = collective_med(c, time_graph(c, lambda: c.ops.all_reduce(c.h, x, y, 0, 0),
                                                                  calls, replays, reps))
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {e}"
        r["status_ok"] = c.status_ok(f"d:lat:{nbytes}")
        all_status &= r["status_ok"]
        res["latency"][str(nbytes)] = r
        P.progress(f"d: latency {nbytes} {r}")
    P["d"] = res
    P.check("d_int8_identical_all_ranks", all_ident)
    P.check("d_int8_within_bound_and_nonfinite_groups_nan", all_bound)
    P.check("d_no_kernel_errors", all_status)


STEPS = {"a": step_a, "b": step_b, "b_nccl": step_b_nccl, "c": step_c, "d": step_d}


def worker(a):
    try:
        c = Ctx(a)
    except Exception as e:  # noqa: BLE001
        p = Part(a.partdir, a.rank)
        p["setup_error"] = f"{type(e).__name__}: {e}"
        p["errors"].append(traceback.format_exc()[-4000:])
        p.save()
        print(traceback.format_exc(), flush=True)
        sys.exit(EXIT_SETUP)
    rc = 0
    for s in a.steps.split(","):
        t0 = time.time()
        c.part.progress(f"step {s} start")
        try:
            STEPS[s](c)
            c.barrier()
        except Exception as e:  # noqa: BLE001
            # the other ranks may now be blocked in a collective: the parent's crash/watchdog timer handles that
            # peers may be blocked in a collective: no teardown (it would block too), exit now; the parent kills the
            # group 60 s after the first non-zero exit or "fatal" part file
            c.part["errors"].append({"step": s, "error": f"{type(e).__name__}: {e}",
                                     "tb": traceback.format_exc()[-4000:]})
            c.part["fatal"] = s
            print(traceback.format_exc(), flush=True)
            sys.stdout.flush()
            os._exit(EXIT_TIMEOUT)
        c.part.progress(f"step {s} done in {time.time() - t0:.1f}s")
    try:
        c.torch.cuda.synchronize()
        c.ops.destroy(c.h)
        c.nc.close()
        c.dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        c.part["errors"].append(f"teardown: {e}")
    c.part.save()
    sys.exit(rc)


def cpu_check(a):
    """CPU-only self check (no GPU): the .so loads in this torch, the ops are registered, the partition model passes,
    init fails cleanly without a GPU. Writes --out and exits 0 when everything is as expected."""
    out = {"mode": "cpu_check"}
    ok = True
    try:
        import torch
        out["torch"] = torch.__version__
        torch.ops.load_library(a.so)
        ops = torch.ops.p5_host_ar
        out["info"] = ops.info()
        out["schemas"] = {}
        for name in ("info", "init", "all_reduce", "status", "epoch", "destroy", "cover_check"):
            out["schemas"][name] = str(getattr(ops, name).default._schema)
        ok &= "Tensor(a!) out" in out["schemas"]["all_reduce"]
        out["all_reduce_has_cuda_kernel"] = torch._C._dispatch_has_kernel_for_dispatch_key(
            "p5_host_ar::all_reduce", "CUDA")
        ok &= bool(out["all_reduce_has_cuda_kernel"])
        out["all_reduce_has_meta_kernel"] = torch._C._dispatch_has_kernel_for_dispatch_key(
            "p5_host_ar::all_reduce", "Meta")
        ok &= bool(out["all_reduce_has_meta_kernel"])
        mi = torch.empty(64, dtype=torch.bfloat16, device="meta")
        ops.all_reduce(0, mi, mi, 0, 0)  # Meta kernel: no-op, no handle lookup
        out["meta_call_ok"] = True
        cc = []
        for n in list(range(0, 300)) + [65536 // 2, 983040 // 2, 983040 // 4, 491521, 1966080 // 2, 3932160 // 4]:
            for world in (2, 4, 8):
                for dt in (0, 1, 2):
                    for mode in (0, 1):
                        r = list(ops.cover_check(n, world, dt, mode, 1 + n % 7))
                        if r[0] != 1:
                            cc.append([n, world, dt, mode, r])
        out["cover_check_fail"] = cc[:20]
        ok &= not cc
        try:
            ops.init(0, 4, "/p5har_cpucheck", 1 << 20, 0, 1000)
            out["init_on_cpu"] = "unexpectedly succeeded"
            ok = False
        except Exception as e:  # noqa: BLE001
            out["init_on_cpu"] = f"raised as expected: {str(e).splitlines()[0][:300]}"
        out["shm_leftover"] = os.path.exists("/dev/shm/p5har_cpucheck")
        ok &= not out["shm_leftover"]
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["tb"] = traceback.format_exc()[-3000:]
        ok = False
    out["ok"] = ok
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("CPU_CHECK " + json.dumps(out)[:3000], flush=True)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so", required=True)
    ap.add_argument("--out", default="/kaggle/working/p5_bench.json")
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--steps", default="a,b,c,d")
    ap.add_argument("--timeout-s", type=float, default=3000.0,
                    help="global wall-clock watchdog (group budgets 900 + 4 x 240 + 900 = 2760 s)")
    ap.add_argument("--kernel-timeout-ms", type=int, default=10000, help="per-wait timeout inside the kernel")
    ap.add_argument("--quick", action="store_true", help="fewer patterns / iterations (smoke)")
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rank", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--partdir", default="", help=argparse.SUPPRESS)
    ap.add_argument("--group", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.cpu_check:
        cpu_check(a)
    elif a.worker:
        worker(a)
    else:
        bad = [s for s in a.steps.split(",") if s not in ("a", "b", "c", "d")]
        if bad:
            ap.error(f"unknown steps {bad}")
        parent(a)


if __name__ == "__main__":
    main()
