"""E25 NB4 arm P: the P5 host-staged world-2 all-reduce wired into Merlin v3.0.0's (vLLM 0.32 fork) TP all-reduce path.

Integration point (Merlin ad42e37, tree 07c16a79 = the tree the Kaggle payload installs, see the #60490 patch header):
  vllm/distributed/device_communicators/cuda_communicator.py (sha256 CC_PRE)
    CudaCommunicator.__init__                       :199  (before `_log_all_reduce_backend_selection`)  -> P5 init
    CudaCommunicator._log_all_reduce_backend_selection :350  -> 'P5_HOST' first in the dispatch list (read-back line)
    CudaCommunicator.all_reduce                     :362  (first statement)                        -> size-gated P5
  Call path of every TP all-reduce: GroupCoordinator.all_reduce (parallel_state.py:724) -> torch.ops.vllm.all_reduce
  (opaque custom op: CudaPlatform.use_custom_op_collectives() is True, platforms/cuda.py:706, so Dynamo never traces the
  dispatcher) -> GroupCoordinator._all_reduce_out_place (:747) -> CudaCommunicator.all_reduce. Under CUDA graph capture
  (PIECEWISE pieces, or FULL) the dispatcher runs once per captured size and the P5 kernel is captured like the NCCL one.

Why it is graph-safe: the P5 kernel reads its epoch from a per-handle device counter (no per-call host argument), flags are
monotonic, it launches on torch's current stream (the capture stream during capture), and it needs no buffer
registration (it stages through its own host slots), so the graph_capture() context needs no P5 hook. NB3 D7 verified
graph capture, 1000 replays bit-exact and eager calls between replays. The routing decision depends only on dtype and
numel (identical on both TP ranks), never on pointers, so both ranks always take the same path.

Fail closed: with NB4_P5 unset the patched file behaves exactly like the original (one attribute = None, one getattr per
call). With NB4_P5=1 any config / sha / load / init / self-test problem raises inside CudaCommunicator.__init__, so the
boot fails and the arm is labelled NOT_INTEGRATED; it never silently runs NCCL. A kernel error (timeout / desync) makes
the handle sticky-NaN; the backend's host monitor sees the abort word and the next eager call raises.

What this file holds (the notebook embeds or imports it):
  PATCH / CC_PRE / CC_POST / MARKER      strict file patch (NB1 #60490 style: sha before, sha after, marker, py_compile)
  BACKEND_SRC                            nb4_p5_backend.py, written next to the tree's vllm/ (on the server PYTHONPATH)
  WORKER_EXT_SRC                         nb4_p5_worker_ext.py: /collective_rpc method nb4_p5_stats (engaged counters)
  GATE_SRC                               nb4_p5_gate.py: 4-GPU synthetic gate (both TP pairs concurrently), run pre-boot
  apply_p5_patch / revert_p5_patch / verify_p5_patch, server_env, parse_log, p5_arm_label, expected_gain
Line numbers above are pre-patch.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MERLIN_COMMIT = "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5"
MERLIN_TREE = "07c16a79d077f1e06c19d0eea84efd64f0b63540"
CC_REL = "vllm/distributed/device_communicators/cuda_communicator.py"
CC_PRE = "2f5368d6ca1856a7b2b49fc423cc4728483d0c9289f913a7bc7cbe0baa040ec0"
CC_POST = "926a2a1fa84f6aba1c13048e7e28b01820d48a56ad43974b6e0fe2d14540d7b0"
MARKER = "# E25 NB4 P5: host-staged world-2 all-reduce backend (env NB4_P5=1)."
BACKEND_REL = "nb4_p5_backend.py"
WORKER_EXT_REL = "nb4_p5_worker_ext.py"
WORKER_EXT_CLS = "nb4_p5_worker_ext.NB4P5WorkerExt"
INTEGRATION = {
    "file": CC_REL,
    "init": f"{CC_REL}:199 CudaCommunicator.__init__, before _log_all_reduce_backend_selection (pre-patch line)",
    "dispatch": f"{CC_REL}:362 CudaCommunicator.all_reduce, first statement (pre-patch line)",
    "readback": f"{CC_REL}:350 _log_all_reduce_backend_selection: 'P5_HOST' first when engaged",
    "call_path": "parallel_state.py:724 GroupCoordinator.all_reduce -> torch.ops.vllm.all_reduce (platforms/cuda.py:706 "
                 "use_custom_op_collectives True) -> parallel_state.py:747 _all_reduce_out_place -> CudaCommunicator.all_reduce",
    "groups": "unique_name 'tp:*' with world 2 only (ep / pp / dcp / etp groups never take P5)",
    "not_patched": "destroy(): the handle lives until process exit (shm name is unlinked after attach; mapping freed at exit)",
}

# The P5 library: NB3 D7 built it in the Merlin venv (torch 2.13.0+cu130, nvcc 13.0.88, sm_89) from the E23 sources
# (E23 prep/P5_host_allreduce/src == Build/nb3_export/p5_src, LF shas below). NB4 rebuilds it the same way
# (build_host_ar.py) and passes the sha of ITS .so via NB4_P5_SO_SHA256; the NB3 sha is a reference only.
P5_SRC_SHA = {"host_ar.cu": "71c8dcf827a991d06fb46bf0ba06f14792c8339249f01f212620e6546ae8f7f9",
              "host_ar_binding.cpp": "1f2a372a257674c59c719a64c740689e94f4e57b6cb6d26cd39095a22232c19c",
              "host_ar_layout.h": "0478edae9458ec751a3bdeb94a71417cf3989f7b25e8aef574e47d72b58d0d5f",
              "host_ar_check.h": "d2bac11df1472e96973485bca84cd9060ec0ddd945f664bed5d0e88d5c96b7a5",
              "build_host_ar.py": "2800594e0860cc69bb813d5493659b37928eee0100d37f25f432e67f1f44b11f"}
P5_SO_NB3_SHA = "3dd9fc6266ad0c01db99b3217561cf29efc8fb49fbac69aae37af582207dc3f5"

TOKEN_BYTES = 5120               # hidden 2560 x bf16 per token
DEFAULT_MAX_BYTES = 327680       # T 64: last size with >= 1.25x (NB3); 1.18x / 1.07x / 1.06x above, 20-40 CTAs spinning
HARD_MAX_BYTES = 655360          # T 128: NB3's largest measured size; nothing above is routed to P5
# NB3 D7 (2026-10-31, nb3_tables.json d7): bytes -> (NCCL LL/SHM graph us, P5 graph us), worst of the two TP pairs, both
# pairs running at the same time (host-arm overlap 0.43-0.97, so "concurrent" is partly serial at small sizes).
NB3_D7 = {5120: (15.97, 11.47), 10240: (18.28, 12.32), 15360: (19.70, 13.60), 20480: (22.28, 15.63),
          30720: (26.06, 16.46), 40960: (31.22, 20.36), 46080: (32.63, 21.18), 51200: (35.53, 25.37),
          61440: (40.86, 25.49), 76800: (47.19, 29.55), 81920: (48.96, 30.65), 92160: (54.41, 33.23),
          102400: (61.12, 36.50), 122880: (61.38, 41.58), 163840: (73.41, 51.61), 184320: (76.70, 58.17),
          245760: (91.78, 73.49), 327680: (118.78, 95.23), 368640: (126.32, 107.06), 491520: (151.98, 142.54),
          655360: (194.55, 182.98)}
# NB1 R2 B1 trace (profile_b1, 40 forwards per rank), measured here: every TP all-reduce is ncclDevKernel_AllReduce_Sum_bf16
# _RING_LL, grid 2 CTAs, all inside graphs, at most one in flight per rank (no AR/AR overlap).
TRACE_B1 = {"stage0": {"ar_per_forward": 51, "ar_ms_per_forward": [3.12, 4.25], "median_us": [51.0, 54.1],
                       "p90_us": [79.1, 173.1], "critical_ms": 50.6},
            "stage1": {"ar_per_forward": 55, "ar_ms_per_forward": [3.18, 3.90], "median_us": [50.3, 52.1],
                       "p90_us": [79.5, 135.3], "critical_ms": 31.6, "note": "46 per target forward + 3 x 3 small"}}

PATCH = r'''# E25 NB4 P5 engine integration onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5,
# tree 07c16a79d077f1e06c19d0eea84efd64f0b63540 = the tree the Kaggle payload installs).
# Adds a size-gated, env-gated (NB4_P5=1) host-staged world-2 all-reduce backend in front of the TP dispatch chain;
# inert when NB4_P5 is unset. The backend module nb4_p5_backend.py lives next to the tree's vllm/ package.
# pre-patch sha256  2f5368d6ca1856a7b2b49fc423cc4728483d0c9289f913a7bc7cbe0baa040ec0
# post-patch sha256 926a2a1fa84f6aba1c13048e7e28b01820d48a56ad43974b6e0fe2d14540d7b0
--- a/vllm/distributed/device_communicators/cuda_communicator.py
+++ b/vllm/distributed/device_communicators/cuda_communicator.py
@@ -196,6 +196,24 @@
             # an MI300 series.
             self.qr_comm = QuickAllReduce(group=self.cpu_group, device=self.device)
 
+        # E25 NB4 P5: host-staged world-2 all-reduce backend (env NB4_P5=1).
+        # Inert unless NB4_P5=1 (TP groups only). Then any import / init /
+        # self-test error raises: the boot fails, it never silently runs NCCL.
+        self.nb4_p5_comm = None
+        if unique_name.split(":")[0] == "tp" and self.world_size > 1:
+            import os as _nb4_os
+
+            if _nb4_os.environ.get("NB4_P5", "0") == "1":
+                from nb4_p5_backend import P5HostAllReduce
+
+                self.nb4_p5_comm = P5HostAllReduce(
+                    group=self.cpu_group,
+                    device=self.device,
+                    unique_name=unique_name,
+                    rank=self.rank,
+                    world_size=self.world_size,
+                )
+
         if self.world_size > 1:
             self._log_all_reduce_backend_selection()
 
@@ -349,6 +367,9 @@
             enabled_ar_backends.append("SYMM_MEM")
         if self.pynccl_comm is not None and not self.pynccl_comm.disabled:
             enabled_ar_backends.append("PYNCCL")
+        if getattr(self, "nb4_p5_comm", None) is not None:  # E25 NB4 P5
+            enabled_ar_backends.insert(0, "P5_HOST")
+            all_potential_ar_backends.insert(0, "P5_HOST")
 
         logger.info_once(
             "Using %s all-reduce backends (in dispatch order) for group "
@@ -360,6 +381,12 @@
         )
 
     def all_reduce(self, input_):
+        # E25 NB4 P5: size-gated host-staged all-reduce first; None = fall through.
+        nb4_p5_comm = getattr(self, "nb4_p5_comm", None)
+        if nb4_p5_comm is not None:
+            out = nb4_p5_comm.maybe_all_reduce(input_)
+            if out is not None:
+                return out
         fi_ar_comm = self.fi_ar_comm
         use_fi_ar = (
             fi_ar_comm is not None
'''

APPLIER_SRC = r'''
def apply_unified(text, patch):
    """Strict unified-diff applier (no fuzz, no offset): every context / removed line must match at its stated line,
    else ValueError. Lines before the first hunk (comments, ---/+++ headers) are ignored. (= NB1 / NB5 applier.)"""
    src = text.splitlines(True)
    hunks, cur = [], None
    for ln in patch.splitlines(True):
        if ln.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
            if not m:
                raise ValueError("bad hunk header: " + ln[:80])
            cur = [int(m.group(1)), int(m.group(2) or 1), []]
            hunks.append(cur)
        elif cur is not None and ln[:1] in (" ", "-", "+"):
            cur[2].append(ln)
        elif cur is not None and ln.startswith("\\"):
            raise ValueError("'No newline at end of file' hunks are not supported")
    if not hunks:
        raise ValueError("patch has no hunks")
    out, pos = [], 0
    for start, n_old, body in hunks:
        i = start - 1
        if i < pos:
            raise ValueError(f"overlapping hunk at line {start}")
        out += src[pos:i]
        old_seen = 0
        for ln in body:
            tag, s = ln[0], ln[1:]
            if tag in " -":
                if i >= len(src) or src[i] != s:
                    raise ValueError(f"context mismatch at line {i + 1}: {s[:60]!r}")
                i += 1
                old_seen += 1
            if tag in " +":
                out.append(s)
        if old_seen != n_old:
            raise ValueError(f"hunk at line {start}: {old_seen} old lines, header says {n_old}")
        pos = i
    return "".join(out + src[pos:])


def reverse_unified(patch):
    """The reverse patch: hunk headers swapped (-a,b +c,d -> -c,d +a,b), '+' and '-' lines swapped."""
    out = []
    started = False
    for ln in patch.splitlines(True):
        if ln.startswith("@@"):
            started = True
            m = re.match(r"@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@(.*)", ln, re.S)
            if not m:
                raise ValueError("bad hunk header: " + ln[:80])
            out.append(f"@@ -{m.group(2)} +{m.group(1)} @@{m.group(3)}")
        elif started and ln[:1] == "+":
            out.append("-" + ln[1:])
        elif started and ln[:1] == "-":
            out.append("+" + ln[1:])
        else:
            out.append(ln)
    return "".join(out)
'''
exec(APPLIER_SRC)  # noqa: S102 - defines apply_unified / reverse_unified here, the notebook embeds the same text

BACKEND_SRC = r'''"""E25 NB4 P5 backend: the P5 host-staged world-2 all-reduce (p5_host_ar.so, built as in NB3 D7) as a size-gated TP
all-reduce backend for Merlin v3.0.0's CudaCommunicator (vLLM 0.32 fork). Imported only by the NB4-patched
cuda_communicator.py when NB4_P5=1.

Fail closed: every config / sha / load / init / self-test problem raises (the boot fails); nothing falls back silently.
Per call: eligible (CUDA tensor, bf16 / fp16 / fp32, NB4_P5_MIN_BYTES <= nbytes <= NB4_P5_MAX_BYTES) -> P5 (out of place,
like every vLLM backend), else None and the caller's chain (PYNCCL here) runs. The decision depends only on dtype and
numel, identical on both TP ranks, so both ranks always take the same path (a split decision would hang the pair).
Non-contiguous or < 16-B aligned inputs are copied first (never a rank-dependent decision).

Env (the NB4 notebook sets it for arm P only):
  NB4_P5=1                read by the patch
  NB4_P5_SO               path of p5_host_ar.so (required)
  NB4_P5_SO_SHA256        its sha256 (required; the file is hashed before it is loaded)
  NB4_P5_MAX_BYTES        largest message routed to P5 (default 327680 = 64 tokens x 5120 B; hard cap 655360)
  NB4_P5_MIN_BYTES        smallest (default 1)
  NB4_P5_TIMEOUT_MS       kernel peer-wait timeout (default 300000, 1000..600000)
  NB4_P5_SELFTEST         1 (default): bit-exact self-test vs an ordered fp32 reference at init; 0 = skip
  NB4_P5_LOG_S            period of the stats log line from a host thread that makes no CUDA call (default 60; 0 = off)

Engaged counters (python, per decision): counts[eager|capture][p5|fallback], fallback_reasons, capture_by_size
{nbytes: [p5, fallback]} (one entry per captured graph call: what every replay of that graph runs) and eager_by_size.
Device truth: stats(read_device=True)["p5_executed"] = the handle's device epoch (every completed P5 kernel, graph replays
included) minus the self-test calls. Log lines: "NB4_P5 {json}" with event init / init_failed / stats / abort.
"""
import hashlib
import json
import os
import threading
import time

import torch

VERSION = "nb4-p5-backend-v1"
DEFAULT_MAX_BYTES = 327680
HARD_MAX_BYTES = 655360
DEFAULT_TIMEOUT_MS = 300000
SELFTEST_BYTES = (5120, 81920, 102400)
_ELIGIBLE = (torch.bfloat16, torch.float16, torch.float32)
_LIB = {"path": None, "sha256": None}
_COMMS = []
_REG_LOCK = threading.Lock()


def log_event(event, **kw):
    rec = {"event": event, "pid": os.getpid(), "t": round(time.time(), 3)}
    rec.update(kw)
    print("NB4_P5 " + json.dumps(rec, sort_keys=True, default=str), flush=True)


def _int_env(env, key, default, lo, hi):
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        v = int(str(raw).strip())
    except ValueError:
        raise ValueError(f"NB4_P5_CONFIG: {key}={raw!r} is not an integer") from None
    if not lo <= v <= hi:
        raise ValueError(f"NB4_P5_CONFIG: {key}={v} outside [{lo}, {hi}]")
    return v


def env_config(environ=None):
    env = os.environ if environ is None else environ
    so = str(env.get("NB4_P5_SO", "") or "")
    sha = str(env.get("NB4_P5_SO_SHA256", "") or "").strip().lower()
    if not so:
        raise ValueError("NB4_P5_CONFIG: NB4_P5_SO is not set")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError(f"NB4_P5_CONFIG: NB4_P5_SO_SHA256 must be 64 hex chars, got {sha!r}")
    max_b = _int_env(env, "NB4_P5_MAX_BYTES", DEFAULT_MAX_BYTES, 16, HARD_MAX_BYTES)
    min_b = _int_env(env, "NB4_P5_MIN_BYTES", 1, 1, max_b)
    tmo = _int_env(env, "NB4_P5_TIMEOUT_MS", DEFAULT_TIMEOUT_MS, 1000, 600000)
    selftest = _int_env(env, "NB4_P5_SELFTEST", 1, 0, 1)
    log_s = _int_env(env, "NB4_P5_LOG_S", 60, 0, 86400)
    return {"so": so, "so_sha256": sha, "max_bytes": max_b, "min_bytes": min_b, "timeout_ms": tmo,
            "selftest": bool(selftest), "log_s": log_s}


def decide(nbytes, dtype, is_cuda, min_bytes, max_bytes):
    """'p5' or the fallback reason. Pure function of values that are identical on every rank of the group."""
    if not is_cuda:
        return "not_cuda"
    if dtype not in _ELIGIBLE:
        return "dtype"
    if nbytes <= 0:
        return "empty"
    if nbytes < min_bytes:
        return "size_lt_min"
    if nbytes > max_bytes:
        return "size_gt_max"
    return "p5"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_library(path, want_sha):
    if _LIB["path"] is not None:
        if _LIB["sha256"] != want_sha:
            raise RuntimeError(f"a different p5_host_ar.so is already loaded ({_LIB['sha256']})")
        return torch.ops.p5_host_ar
    got = sha256_file(path)
    if got != want_sha:
        raise RuntimeError(f"{path} sha256 {got} != expected {want_sha}")
    torch.ops.load_library(path)
    _LIB.update(path=path, sha256=got)
    return torch.ops.p5_host_ar


def selftest_input(seed, rank, n, dtype, device):
    g = torch.Generator(device=device)
    g.manual_seed((seed * 1000003 + rank * 7919 + 17) % (2 ** 63 - 1))
    x = torch.randn(n, generator=g, device=device, dtype=torch.float32)
    x = x * torch.pow(10.0, torch.rand(n, generator=g, device=device, dtype=torch.float32) * 6.0 - 3.0)
    return x.to(dtype)


def ordered_ref(xs, dtype):
    acc = xs[0].float()
    for x in xs[1:]:
        acc = acc + x.float()
    return acc.to(dtype)


def bits_diff(a, b):
    """Elements whose bits differ (NaN == NaN regardless of payload)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), b.numel(), 1)
    v = torch.int16 if a.element_size() == 2 else torch.int32
    same = a.contiguous().view(v) == b.contiguous().view(v)
    both_nan = torch.isnan(a) & torch.isnan(b)
    return int((~(same | both_nan)).sum().item())


def _default_broadcast(obj, group):
    import torch.distributed as dist
    try:
        dist.broadcast_object_list(obj, group=group, group_src=0)
    except TypeError:  # older signature
        dist.broadcast_object_list(obj, src=dist.get_global_rank(group, 0), group=group)


def _snap(d):
    for _ in range(5):
        try:
            return {str(k): (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v))
                    for k, v in list(d.items())}
        except RuntimeError:  # dict changed size while the main thread updated it
            time.sleep(0.001)
    return {"_snapshot": "busy"}


class P5HostAllReduce:
    def __init__(self, group, device, unique_name, rank, world_size, *, ops=None, environ=None, broadcast=None,
                 require_cuda=None, capturing=None):
        self.unique_name = str(unique_name)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        self.handle = None
        self.counts = {"eager": {"p5": 0, "fallback": 0}, "capture": {"p5": 0, "fallback": 0}}
        self.fallback_reasons = {}
        self.capture_by_size = {}
        self.eager_by_size = {}
        self.selftest_calls = 0
        self.aborted = False
        self.abort_status = None
        self._stop = threading.Event()
        self._require_cuda = (ops is None) if require_cuda is None else bool(require_cuda)
        if capturing is not None:
            self._capturing = capturing
        elif torch.cuda.is_available():
            self._capturing = torch.cuda.is_current_stream_capturing
        else:
            self._capturing = lambda: False
        try:
            self.cfg = env_config(environ)
            self.max_bytes, self.min_bytes = self.cfg["max_bytes"], self.cfg["min_bytes"]
            if self.world_size != 2:
                raise RuntimeError(f"world_size {self.world_size} != 2 (NB4 measured P5 at world 2 only)")
            self.ops = ops if ops is not None else load_library(self.cfg["so"], self.cfg["so_sha256"])
            info = str(self.ops.info())
            if "worlds 2" not in info:
                raise RuntimeError(f"unexpected p5_host_ar info {info!r}")
            name = [f"/p5har_nb4_{os.getpid()}_{time.time_ns()}" if self.rank == 0 else None]
            (broadcast or _default_broadcast)(name, group)
            if not isinstance(name[0], str) or not name[0].startswith("/p5har_nb4_"):
                raise RuntimeError(f"bad shm name from rank 0: {name[0]!r}")
            self.shm_name = name[0]
            idx = getattr(device, "index", None)
            if idx is None:
                idx = torch.cuda.current_device() if self._require_cuda else 0
            self.device_index = int(idx)
            self.handle = int(self.ops.init(self.rank, self.world_size, self.shm_name, self.max_bytes,
                                            self.device_index, self.cfg["timeout_ms"]))
            st = [int(v) for v in self.ops.status(self.handle)]
            self.max_blocks = st[-1]
            if st[0] or any(st[1:-1]):
                raise RuntimeError(f"status after init {st}")
            self.selftest = self._selftest() if self.cfg["selftest"] else {"skipped": True}
        except Exception as e:  # noqa: BLE001 - re-raised: fail closed
            log_event("init_failed", group=self.unique_name, rank=self.rank, world=self.world_size,
                      error=f"{type(e).__name__}: {e}"[:2000])
            raise RuntimeError(f"NB4_P5_INIT_FAILED ({self.unique_name} rank {self.rank}): {type(e).__name__}: {e}") from e
        with _REG_LOCK:
            _COMMS.append(self)
        log_event("init", ok=True, version=VERSION, group=self.unique_name, rank=self.rank, world=self.world_size,
                  device=self.device_index, handle=self.handle, shm=self.shm_name, max_bytes=self.max_bytes,
                  min_bytes=self.min_bytes, timeout_ms=self.cfg["timeout_ms"], max_blocks=self.max_blocks,
                  so=self.cfg["so"], so_sha256=self.cfg["so_sha256"], selftest=self.selftest)
        if self.cfg["log_s"] > 0:
            t = threading.Thread(target=self._monitor, name="nb4-p5-monitor", daemon=True)
            t.start()

    # ---- raw P5 call (no counting)
    def _p5(self, inp):
        out = torch.empty_like(inp)
        self.ops.all_reduce(self.handle, inp, out, 0, 0)
        return out

    def _selftest(self):
        cases = []
        for i, nb in enumerate(sorted({b for b in SELFTEST_BYTES + (self.max_bytes,)
                                       if self.min_bytes <= b <= self.max_bytes})):
            for dt, es in ((torch.bfloat16, 2), (torch.float32, 4)):
                n = nb // es
                if n <= 0 or n * es < self.min_bytes:
                    continue
                xs = [selftest_input(7100 + i, r, n, dt, self.device) for r in range(self.world_size)]
                out = self._p5(xs[self.rank])
                self.selftest_calls += 1
                cases.append([n * es, str(dt).replace("torch.", ""), bits_diff(out, ordered_ref(xs, dt))])
        st = [int(v) for v in self.ops.status(self.handle)]
        ok = bool(cases) and all(c[2] == 0 for c in cases) and not st[0] and not any(st[1:-1])
        if not ok:
            raise RuntimeError(f"NB4_P5_SELFTEST_FAILED cases [bytes, dtype, bad_elems] {cases} status {st}")
        return {"ok": True, "cases": cases, "calls": self.selftest_calls}

    # ---- the dispatcher hook
    def maybe_all_reduce(self, input_):
        cap = bool(self._capturing())
        nbytes = input_.numel() * input_.element_size()
        why = decide(nbytes, input_.dtype, input_.is_cuda or not self._require_cuda, self.min_bytes, self.max_bytes)
        c = self.counts["capture" if cap else "eager"]
        by = self.capture_by_size if cap else self.eager_by_size
        if why != "p5":
            c["fallback"] += 1
            self.fallback_reasons[why] = self.fallback_reasons.get(why, 0) + 1
            if nbytes in by or len(by) < 4096:
                by.setdefault(nbytes, [0, 0])[1] += 1
            return None
        if self.aborted and not cap:
            raise RuntimeError(f"NB4_P5_ABORTED ({self.unique_name} rank {self.rank}): status {self.abort_status}; "
                               "the handle returns NaN since the abort")
        inp = input_ if input_.is_contiguous() else input_.contiguous()
        if inp.data_ptr() % 16:
            inp = inp.clone()
        out = self._p5(inp)
        c["p5"] += 1
        if nbytes in by or len(by) < 4096:
            by.setdefault(nbytes, [0, 0])[0] += 1
        return out

    # ---- status / stats
    def poll_status(self):
        st = [int(v) for v in self.ops.status(self.handle)]  # host-mapped words: no CUDA call
        if (st[0] or any(st[1:-1])) and not self.aborted:
            self.aborted = True
            self.abort_status = st
            log_event("abort", group=self.unique_name, rank=self.rank, status=st)
        return st

    def stats(self, read_device=False):
        d = {"group": self.unique_name, "rank": self.rank, "world": self.world_size, "handle": self.handle,
             "max_bytes": self.max_bytes, "min_bytes": self.min_bytes, "timeout_ms": self.cfg["timeout_ms"],
             "max_blocks": self.max_blocks, "counts": {k: dict(v) for k, v in self.counts.items()},
             "fallback_reasons": _snap(self.fallback_reasons), "capture_by_size": _snap(self.capture_by_size),
             "eager_by_size": _snap(self.eager_by_size), "selftest_calls": self.selftest_calls,
             "status": self.poll_status(), "aborted": self.aborted}
        if read_device:  # synchronous device read: call between windows, never inside a capture
            e = int(self.ops.epoch(self.handle))
            d["device_epoch"] = e
            d["p5_executed"] = e - self.selftest_calls
        return d

    def _monitor(self):
        while not self._stop.wait(self.cfg["log_s"]):
            try:
                log_event("stats", **self.stats(read_device=False))
            except Exception as e:  # noqa: BLE001
                log_event("monitor_error", group=self.unique_name, rank=self.rank, error=f"{type(e).__name__}: {e}")


def stats_all(read_device=False):
    with _REG_LOCK:
        comms = list(_COMMS)
    return {"version": VERSION, "pid": os.getpid(), "lib": dict(_LIB), "comms": [c.stats(read_device) for c in comms]}


def stats_json(read_device=False):
    return json.dumps(stats_all(read_device), sort_keys=True, default=str)
'''

WORKER_EXT_SRC = r'''"""E25 NB4 worker extension (read-only): POST /collective_rpc {"method": "nb4_p5_stats", "args": ["1"]} with
VLLM_SERVER_DEV_MODE=1 and --worker-extension-cls nb4_p5_worker_ext.NB4P5WorkerExt. Returns this rank's P5 engaged
counters as JSON text ("1" = also read the device epoch, a synchronous read: call it between windows). Never raises."""
import json


class NB4P5WorkerExt:
    def nb4_p5_stats(self, read_device="1"):
        try:
            import nb4_p5_backend as b
            return b.stats_json(read_device=str(read_device) == "1")
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
'''

GATE_SRC = r'''"""E25 NB4 P5 gate. Run BEFORE the P boot (GPUs free) with the Merlin venv python (the torch the .so was built for):
    python nb4_p5_gate.py --backend-dir DIR --so p5_host_ar.so --so-sha SHA --out gate.json [--max-bytes N]
4 processes = TP pairs (0,1) and (2,3) at the same time, each pair with its own P5 handle created by the REAL
nb4_p5_backend.P5HostAllReduce (the code the engine runs, self-test included). For every decode size 5120 B x T
(T from --tokens, <= max bytes): --eager-calls eager calls, then a captured CUDA graph of --graph-calls back-to-back
all-reduces (one forward's worth; vLLM-style eager warm-up first), replayed --replays times with fresh inputs. Every output
is bit-compared with an ordered fp32 reference built on each rank from the deterministic inputs of both ranks of its pair.
Also: the size / dtype gate (> max bytes and int8 return None, no collective), a non-contiguous input, the engaged
counters (python counts and the device epoch == the P5 calls this rank executed), status words 0. Each start is aligned by
a 4-rank barrier; the pair-overlap of the graph phases is reported. Prints NB4_P5_GATE_JSON {...}; exit 0 only on PASS."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

TOKENS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 32, 36, 48, 64]
TOKEN_BYTES = 5120
PATTERNS = ("randn", "wide", "cancel")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def gen(torch, dev, seed, r, n, pattern):
    g = torch.Generator(device=dev)
    g.manual_seed((seed * 1000003 + r * 7919 + 17) % (2 ** 63 - 1))
    x = torch.randn(n, generator=g, device=dev, dtype=torch.float32)
    if pattern == "wide":
        x = x * torch.pow(10.0, torch.rand(n, generator=g, device=dev) * 10.0 - 6.0)
    elif pattern == "cancel":
        g2 = torch.Generator(device=dev)
        g2.manual_seed(seed * 31 + 5)
        base = torch.randn(n, generator=g2, device=dev) * 1e3
        x = base * (1.0 if r % 2 == 0 else -1.0) + 1e-3 * x
    return x.to(torch.bfloat16)


def fillv(src, s):
    return torch_roll(src, s) * (1.0 + ((s * 37) % 16) / 16.0)


def torch_roll(src, s):
    import torch
    return torch.roll(src, shifts=s % 97, dims=1)


def child(a):
    import torch
    import torch.distributed as dist
    sys.path.insert(0, a.backend_dir)
    import nb4_p5_backend as B
    rank, pair, pr = a.rank, a.rank // 2, a.rank % 2
    rec = {"rank": rank, "pair": pair, "ok": False, "rows": [], "checks": {}}
    try:
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{a.port}", rank=rank, world_size=4,
                                timeout=timedelta(seconds=a.timeout_s))
        groups = [dist.new_group([0, 1], backend="gloo"), dist.new_group([2, 3], backend="gloo")]
        env = {"NB4_P5_SO": a.so, "NB4_P5_SO_SHA256": a.so_sha, "NB4_P5_MAX_BYTES": str(a.max_bytes),
               "NB4_P5_TIMEOUT_MS": str(a.kernel_timeout_ms), "NB4_P5_SELFTEST": "1", "NB4_P5_LOG_S": "0"}
        comm = B.P5HostAllReduce(group=groups[pair], device=dev, unique_name="tp:0", rank=pr, world_size=2, environ=env)
        rec["selftest"] = comm.selftest
        rec["gpu"] = torch.cuda.get_device_name(rank)
        rec["torch"] = torch.__version__
        executed, eager_p5, capture_p5 = 0, 0, 0
        K, bad_total = a.graph_calls, 0
        for T in json.loads(a.tokens):
            nb = T * TOKEN_BYTES
            if nb > comm.max_bytes:
                continue
            n = nb // 2
            row = {"T": T, "bytes": nb}
            dist.barrier()
            t0 = time.monotonic_ns()
            bad = 0
            for s in range(a.eager_calls):
                seed = 1000000 * pair + 1000 * T + s
                xs = [gen(torch, dev, seed, j, n, PATTERNS[s % len(PATTERNS)]) for j in (0, 1)]
                out = comm.maybe_all_reduce(xs[pr])
                if out is None:
                    bad += n
                    continue
                executed += 1
                eager_p5 += 1
                bad += B.bits_diff(out, B.ordered_ref(xs, torch.bfloat16))
            torch.cuda.synchronize()
            row["eager_ns"] = [t0, time.monotonic_ns()]
            row["eager_bad"] = bad
            base = [gen(torch, dev, 5000000 + 1000000 * pair + T, j, K * n, "randn").view(K, n) for j in (0, 1)]
            X = torch.empty(K, n, device=dev, dtype=torch.bfloat16)
            X.copy_(fillv(base[pr], 0))
            outs = [None] * K

            def body():
                for k in range(K):
                    outs[k] = comm.maybe_all_reduce(X[k])
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    body()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            executed += 2 * K
            eager_p5 += 2 * K
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                body()
            torch.cuda.synchronize()
            capture_p5 += K
            if any(o is None for o in outs):
                raise RuntimeError(f"T {T}: P5 declined a call during capture")
            gbad = 0
            dist.barrier()
            t2 = time.monotonic_ns()
            for s in range(a.replays):
                X.copy_(fillv(base[pr], s))
                g.replay()
                executed += K
                ref = B.ordered_ref([fillv(base[j], s) for j in (0, 1)], torch.bfloat16)
                gbad += B.bits_diff(torch.stack(outs), ref)
            torch.cuda.synchronize()
            row["graph_ns"] = [t2, time.monotonic_ns()]
            row["graph_bad"] = gbad
            dist.barrier()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(a.time_replays):
                g.replay()
            e1.record()
            e1.synchronize()
            executed += K * a.time_replays
            row["graph_us_per_call"] = round(e0.elapsed_time(e1) * 1000.0 / (a.time_replays * K), 2)
            del g
            bad_total += bad + gbad
            rec["rows"].append(row)
        # size / dtype gate: no collective is issued (returns None on every rank)
        big = torch.zeros(comm.max_bytes // 2 + 64, device=dev, dtype=torch.bfloat16)
        i8 = torch.zeros(4096, device=dev, dtype=torch.int8)
        rec["checks"]["gate_gt_max_none"] = comm.maybe_all_reduce(big) is None
        rec["checks"]["gate_int8_none"] = comm.maybe_all_reduce(i8) is None
        # non-contiguous input ([16, 2560] transposed view of a [2560, 16] tensor)
        b2 = [gen(torch, dev, 777 + pair, j, 2560 * 16, "randn").view(2560, 16) for j in (0, 1)]
        xnc = b2[pr].t()
        onc = comm.maybe_all_reduce(xnc)
        executed += 1
        eager_p5 += 1
        rec["checks"]["noncontig_bitexact"] = (onc is not None and tuple(onc.shape) == tuple(xnc.shape) and
                                               B.bits_diff(onc.contiguous(), B.ordered_ref([b.t().contiguous() for b in b2], torch.bfloat16)) == 0)
        torch.cuda.synchronize()
        st = comm.stats(read_device=True)
        rec["stats"] = st
        rec["checks"]["bitexact_all_sizes"] = bad_total == 0 and bool(rec["rows"])
        rec["checks"]["device_epoch_counter"] = st["p5_executed"] == executed
        rec["checks"]["python_counts"] = (st["counts"]["eager"]["p5"] == eager_p5 and
                                          st["counts"]["capture"]["p5"] == capture_p5 and
                                          st["counts"]["eager"]["fallback"] == 2 and
                                          st["fallback_reasons"] == {"size_gt_max": 1, "dtype": 1})
        rec["checks"]["status_zero"] = not st["aborted"] and not any(st["status"][:-1])
        rec["executed_expected"] = executed
        rec["ok"] = all(rec["checks"].values())
        dist.barrier()
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"[:3000]
        rec["tb"] = traceback.format_exc()[-4000:]
        print("NB4_P5_GATE_RANK " + json.dumps(rec, default=str), flush=True)
        os._exit(3)
    print("NB4_P5_GATE_RANK " + json.dumps(rec, default=str), flush=True)
    os._exit(0 if rec["ok"] else 4)


def overlap(a, b):
    if not a or not b:
        return None
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return round(max(0, inter) / max(1, min(a[1] - a[0], b[1] - b[0])), 3)


def parent(a):
    t0 = time.time()
    port = free_port()
    outdir = Path(a.out).parent
    outdir.mkdir(parents=True, exist_ok=True)
    procs, logs = [], []
    for r in range(4):
        lf = open(outdir / f"nb4_p5_gate_rank{r}.log", "w")
        logs.append(lf)
        argv = [sys.executable, os.path.abspath(__file__), "--child", "--rank", str(r), "--port", str(port)]
        for k in ("backend_dir", "so", "so_sha", "max_bytes", "tokens", "eager_calls", "graph_calls", "replays",
                  "time_replays", "kernel_timeout_ms", "timeout_s"):
            argv += ["--" + k.replace("_", "-"), str(getattr(a, k))]
        procs.append(subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True))
    deadline = time.time() + a.timeout_s
    first_fail = None
    while time.time() < deadline and any(p.poll() is None for p in procs):
        if first_fail is None and any(p.poll() not in (None, 0) for p in procs):
            first_fail = time.time()
        if first_fail is not None and time.time() - first_fail > 30:
            break
        time.sleep(0.5)
    timed_out = any(p.poll() is None for p in procs)
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                p.kill()
            p.wait()
    for lf in logs:
        lf.close()
    ranks = {}
    for r in range(4):
        for ln in (outdir / f"nb4_p5_gate_rank{r}.log").read_text(errors="replace").splitlines():
            if ln.startswith("NB4_P5_GATE_RANK "):
                ranks[r] = json.loads(ln[len("NB4_P5_GATE_RANK "):])
    res = {"rcs": [p.returncode for p in procs], "timed_out": timed_out, "sec": round(time.time() - t0, 1),
           "ranks": ranks, "max_bytes": a.max_bytes, "graph_calls": a.graph_calls, "replays": a.replays}
    if len(ranks) == 4:
        ov = {}
        r0 = {row["T"]: row for row in ranks[0].get("rows", [])}
        r2 = {row["T"]: row for row in ranks[2].get("rows", [])}
        for T in sorted(set(r0) & set(r2)):
            ov[str(T)] = {"eager": overlap(r0[T].get("eager_ns"), r2[T].get("eager_ns")),
                          "graph": overlap(r0[T].get("graph_ns"), r2[T].get("graph_ns")),
                          "us_per_call_pairs": [r0[T].get("graph_us_per_call"), r2[T].get("graph_us_per_call")]}
        res["pair_overlap"] = ov
        gv = [v["graph"] for v in ov.values() if v["graph"] is not None]
        res["concurrency"] = ("CONCURRENT" if gv and min(gv) >= 0.5 else "CONCURRENCY_UNVERIFIED")
    res["verdict"] = ("PASS" if (len(ranks) == 4 and all(ranks[r].get("ok") for r in ranks) and not timed_out
                                and all(rc == 0 for rc in res["rcs"])) else "FAIL")
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print("NB4_P5_GATE_JSON " + json.dumps({k: res[k] for k in ("verdict", "rcs", "timed_out", "sec")} |
                                           {"concurrency": res.get("concurrency")}), flush=True)
    sys.exit(0 if res["verdict"] == "PASS" else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--backend-dir", required=True)
    ap.add_argument("--so", required=True)
    ap.add_argument("--so-sha", required=True)
    ap.add_argument("--out", default="nb4_p5_gate.json")
    ap.add_argument("--max-bytes", type=int, default=327680)
    ap.add_argument("--tokens", default=json.dumps(TOKENS))
    ap.add_argument("--eager-calls", type=int, default=12)
    ap.add_argument("--graph-calls", type=int, default=51)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--time-replays", type=int, default=20)
    ap.add_argument("--kernel-timeout-ms", type=int, default=30000)
    ap.add_argument("--timeout-s", type=int, default=600)
    a = ap.parse_args()
    child(a) if a.child else parent(a)


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------- helpers (run in the notebook)
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(t):
    return sha256_bytes(t.encode("utf-8"))


BACKEND_SHA = sha256_text(BACKEND_SRC)
WORKER_EXT_SHA = sha256_text(WORKER_EXT_SRC)
GATE_SHA = sha256_text(GATE_SRC)
PATCH_SHA = sha256_text(PATCH)


def payload_files():
    """name -> (text, sha256) of the files the notebook writes / ships in its payload dataset."""
    return {BACKEND_REL: (BACKEND_SRC, BACKEND_SHA), WORKER_EXT_REL: (WORKER_EXT_SRC, WORKER_EXT_SHA),
            "nb4_p5_gate.py": (GATE_SRC, GATE_SHA)}


def write_payload(dst):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    out = {}
    for n, (t, h) in payload_files().items():
        p = dst / n
        p.write_bytes(t.encode("utf-8"))
        got = sha256_bytes(p.read_bytes())
        if got != h:
            raise RuntimeError(f"PAYLOAD_SHA_MISMATCH {n}: {got} != {h}")
        out[n] = {"path": str(p), "sha256": got}
    return out


def _py_compile(py, files):
    cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)"]
                        + [str(f) for f in files], capture_output=True, text=True, timeout=300)
    return {"rc": cp.returncode, "err": cp.stderr[-600:]}


def _drop_pyc(f):
    for pyc in (f.parent / "__pycache__").glob(f.stem + ".*.pyc"):
        pyc.unlink()


def _origin_check(py, tree):
    envp = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    envp["PYTHONPATH"] = str(tree)                # what Merlin's build() exports for the server
    code = ("import importlib.util as u\nfor m in ('vllm', 'nb4_p5_backend', 'nb4_p5_worker_ext'):\n"
            "    print(u.find_spec(m).origin)")
    cp = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=300, env=envp)
    lines = cp.stdout.strip().splitlines()
    ok = cp.returncode == 0 and len(lines) == 3 and all(l.startswith(str(tree)) for l in lines)
    return ok, cp.stdout.strip()[-600:] + cp.stderr[-300:]


def apply_p5_patch(tree, py=None, check_origin=True):
    """Patch the installed tree's cuda_communicator.py (pre-sha must be Merlin ad42e37's CC_PRE, else refuse), verify the
    post-sha, the marker, write nb4_p5_backend.py + nb4_p5_worker_ext.py next to vllm/ (sha-checked), py_compile all three
    with the Merlin venv python, and (check_origin) that `vllm` and both modules resolve inside the tree with
    PYTHONPATH=tree. Any miss raises: no boot is started."""
    tree = Path(tree)
    py = py or sys.executable
    f = tree / CC_REL
    raw = f.read_bytes()
    h0 = sha256_bytes(raw)
    rec = {"file": str(f), "sha_before": h0, "pre_expected": CC_PRE, "post_expected": CC_POST, "patch_sha": PATCH_SHA}
    if h0 == CC_POST:
        rec["applied"] = "already_applied"
    elif h0 != CC_PRE:
        raise RuntimeError(f"PATCH_P5_PRE_SHA_MISMATCH: {h0} (expected Merlin ad42e37 {CC_PRE})")
    else:
        new = apply_unified(raw.decode("utf-8"), PATCH).encode("utf-8")  # noqa: F821 - defined by APPLIER_SRC
        if sha256_bytes(new) != CC_POST:
            raise RuntimeError("PATCH_P5_POST_SHA_MISMATCH: " + sha256_bytes(new))
        tmp = f.with_name(f.name + ".nb4tmp")
        tmp.write_bytes(new)
        os.replace(tmp, f)
        rec["applied"] = "applied"
    _drop_pyc(f)
    data = f.read_bytes()
    rec.update(sha_after=sha256_bytes(data), marker=MARKER.encode() in data)
    rec["modules"] = {}
    for rel, (text, h) in ((BACKEND_REL, (BACKEND_SRC, BACKEND_SHA)), (WORKER_EXT_REL, (WORKER_EXT_SRC, WORKER_EXT_SHA))):
        p = tree / rel
        p.write_bytes(text.encode("utf-8"))
        rec["modules"][rel] = {"file": str(p), "sha256": sha256_bytes(p.read_bytes()), "expected": h}
    rec["py_compile"] = _py_compile(py, [f, tree / BACKEND_REL, tree / WORKER_EXT_REL])
    origin_ok = True
    if check_origin:
        origin_ok, rec["origin"] = _origin_check(py, tree)
    mods_ok = all(m["sha256"] == m["expected"] for m in rec["modules"].values())
    if rec["sha_after"] != CC_POST or not rec["marker"] or rec["py_compile"]["rc"] or not origin_ok or not mods_ok:
        raise RuntimeError("PATCH_P5_VERIFY_FAILED: " + json.dumps(
            {k: rec.get(k) for k in ("sha_after", "marker", "py_compile", "origin", "modules")})[:900])
    rec["ok"] = True
    return rec


def verify_p5_patch(tree):
    """Read-only: 'applied' (post-sha + marker + module shas), 'not_applied' (pre-sha) or 'foreign' (anything else)."""
    tree = Path(tree)
    data = (tree / CC_REL).read_bytes()
    h = sha256_bytes(data)
    mods = {}
    for rel, hh in ((BACKEND_REL, BACKEND_SHA), (WORKER_EXT_REL, WORKER_EXT_SHA)):
        p = tree / rel
        mods[rel] = (sha256_bytes(p.read_bytes()) == hh) if p.is_file() else None
    state = ("applied" if h == CC_POST and MARKER.encode() in data and all(mods.values()) else
             "not_applied" if h == CC_PRE else "foreign")
    return {"state": state, "sha": h, "modules_ok": mods}


def revert_p5_patch(tree, py=None, remove_modules=True):
    """Back to the exact Merlin file (reverse patch, sha-checked == CC_PRE) for the arms after P (T0b). Refuses on a file
    that is neither CC_POST nor CC_PRE."""
    tree = Path(tree)
    py = py or sys.executable
    f = tree / CC_REL
    raw = f.read_bytes()
    h0 = sha256_bytes(raw)
    rec = {"file": str(f), "sha_before": h0}
    if h0 == CC_PRE:
        rec["reverted"] = "not_applied"
    elif h0 != CC_POST:
        raise RuntimeError(f"REVERT_P5_SHA_MISMATCH: {h0} (expected {CC_POST} or {CC_PRE})")
    else:
        old = apply_unified(raw.decode("utf-8"), reverse_unified(PATCH)).encode("utf-8")  # noqa: F821
        if sha256_bytes(old) != CC_PRE:
            raise RuntimeError("REVERT_P5_RESULT_MISMATCH: " + sha256_bytes(old))
        tmp = f.with_name(f.name + ".nb4tmp")
        tmp.write_bytes(old)
        os.replace(tmp, f)
        rec["reverted"] = "reverted"
    _drop_pyc(f)
    if remove_modules:
        for rel in (BACKEND_REL, WORKER_EXT_REL):
            p = tree / rel
            if p.is_file():
                p.unlink()
            _drop_pyc(p)
    rec["sha_after"] = sha256_bytes(f.read_bytes())
    rec["py_compile"] = _py_compile(py, [f])
    if rec["sha_after"] != CC_PRE or rec["py_compile"]["rc"]:
        raise RuntimeError("REVERT_P5_VERIFY_FAILED: " + json.dumps(rec)[:600])
    rec["ok"] = True
    return rec


def server_env(so, so_sha, max_bytes=DEFAULT_MAX_BYTES, timeout_ms=300000, log_s=60, selftest=True):
    """Env added to the server for arm P only (on top of the base env). The tree must be on PYTHONPATH (Merlin's build()
    exports it). Add WORKER_EXT_CLS to --worker-extension-cls (or call nb4_p5_backend.stats_json from the notebook's own
    extension) plus VLLM_SERVER_DEV_MODE=1 for the /collective_rpc stats read."""
    if not 16 <= int(max_bytes) <= HARD_MAX_BYTES:
        raise ValueError(f"max_bytes {max_bytes} outside [16, {HARD_MAX_BYTES}]")
    if len(str(so_sha)) != 64:
        raise ValueError("so_sha must be a sha256 hex digest")
    return {"NB4_P5": "1", "NB4_P5_SO": str(so), "NB4_P5_SO_SHA256": str(so_sha).lower(),
            "NB4_P5_MAX_BYTES": str(int(max_bytes)), "NB4_P5_TIMEOUT_MS": str(int(timeout_ms)),
            "NB4_P5_LOG_S": str(int(log_s)), "NB4_P5_SELFTEST": "1" if selftest else "0"}


def gate_argv(py, gate_path, backend_dir, so, so_sha, out, max_bytes=DEFAULT_MAX_BYTES, timeout_s=600):
    return [str(py), str(gate_path), "--backend-dir", str(backend_dir), "--so", str(so), "--so-sha", str(so_sha),
            "--out", str(out), "--max-bytes", str(int(max_bytes)), "--timeout-s", str(int(timeout_s))]


_EV_RE = re.compile(r"NB4_P5 (\{.*\})\s*$")
_BACKENDS_RE = re.compile(r"Using \[(.*?)\] all-reduce backends \(in dispatch order\) for group '([^']+)'")


def parse_log(text):
    """Server log -> NB4 P5 read-back: per-rank init events (selftest), init failures, aborts, the last stats per
    (pid, group) and the backend-selection line(s) (vLLM prints it once, on the global first rank)."""
    inits, fails, aborts, stats, backends = [], [], [], {}, []
    for ln in text.splitlines():
        m = _EV_RE.search(ln)
        if m:
            try:
                ev = json.loads(m.group(1))
            except ValueError:
                continue
            kind = ev.get("event")
            if kind == "init":
                inits.append(ev)
            elif kind == "init_failed":
                fails.append(ev)
            elif kind == "abort":
                aborts.append(ev)
            elif kind == "stats":
                stats[(ev.get("pid"), ev.get("group"))] = ev
            continue
        b = _BACKENDS_RE.search(ln)
        if b:
            backends.append({"group": b.group(2), "backends": [x.strip().strip("'") for x in b.group(1).split(",") if x.strip()]})
    ok_inits = [e for e in inits if e.get("ok") and (e.get("selftest") or {}).get("ok", (e.get("selftest") or {}).get("skipped"))]
    return {"inits": inits, "inits_ok": len(ok_inits), "init_failed": fails, "aborts": aborts,
            "last_stats": list(stats.values()), "backend_lines": backends,
            "p5_first_in_dispatch": any(b["backends"][:1] == ["P5_HOST"] for b in backends if b["group"].startswith("tp"))}


def p5_arm_label(log_text, stats_by_rank=None, expect_ranks=4):
    """NB4 label pieces for arm P from the read-back: (label, reasons). ENGAGED is required before the arm's speed is
    compared at all; NOT_INTEGRATED / CORRECTNESS_FAIL map straight onto the NB4 labels. stats_by_rank = the parsed
    /collective_rpc nb4_p5_stats results (one dict per rank, each with 'comms')."""
    info = parse_log(log_text)
    reasons = []
    if info["init_failed"]:
        reasons.append("init_failed: " + "; ".join(str(e.get("error"))[:200] for e in info["init_failed"]))
    if info["inits_ok"] < expect_ranks:
        reasons.append(f"inits_ok {info['inits_ok']} < {expect_ranks}")
    if not info["p5_first_in_dispatch"]:
        reasons.append("backend line without P5_HOST first")
    if reasons:
        return "NOT_INTEGRATED", reasons
    comms = [c for s in (stats_by_rank or []) for c in (s.get("comms") or [])]
    if info["aborts"] or any(c.get("aborted") for c in comms):
        return "CORRECTNESS_FAIL", ["P5 abort (sticky NaN handle)"]
    if stats_by_rank is not None:
        if len(comms) < expect_ranks:
            return "NOT_INTEGRATED", [f"stats from {len(comms)} comms < {expect_ranks}"]
        if any(int(c.get("p5_executed") or 0) <= 0 for c in comms):
            return "NOT_INTEGRATED", ["a rank executed 0 P5 all-reduces"]
    return "ENGAGED", []


def engaged_per_step(stats_before, stats_after, steps, ar_per_forward=None):
    """Per-step engaged numbers for one window from two nb4_p5_stats reads (per rank) and the engine's step count over the
    window: p5 = device-epoch delta / steps (graph replays included). nccl_eager = exact python fallback delta (eager
    calls only). If ar_per_forward (from the profiled window, e.g. 51 / 55) is given, nccl_in_graph is estimated as
    ar_per_forward - p5 - nccl_eager per step (graph replays run no python)."""
    out = []
    for b, a in zip(stats_before, stats_after):
        for cb, ca in zip(b.get("comms") or [], a.get("comms") or []):
            p5 = (int(ca.get("p5_executed") or 0) - int(cb.get("p5_executed") or 0)) / max(1, steps)
            ne = (ca["counts"]["eager"]["fallback"] - cb["counts"]["eager"]["fallback"]) / max(1, steps)
            r = {"group": ca.get("group"), "rank": ca.get("rank"), "p5_per_step": round(p5, 2),
                 "nccl_eager_per_step": round(ne, 2)}
            if ar_per_forward:
                r["nccl_in_graph_per_step_est"] = round(max(0.0, ar_per_forward - p5 - ne), 2)
            out.append(r)
    return out


def expected_gain(tokens=(16, 20), ar_per_forward=51, tick_ms=48.2, max_bytes=DEFAULT_MAX_BYTES):
    """Upper-bound P5 saving per micro-batch on the heavy stage from NB3 D7 (concurrent pairs, worst pair): per token
    count T, (NCCL graph - P5 graph) us x ar_per_forward. Only the median AR transfer is removed: the trace's AR tail
    above the median (peer skew) is waiting that P5 also does."""
    rows = []
    for T in tokens:
        nb = T * TOKEN_BYTES
        if nb not in NB3_D7 or nb > max_bytes:
            rows.append({"T": T, "bytes": nb, "note": "not measured in NB3 or above max_bytes: NCCL"})
            continue
        nc, p5 = NB3_D7[nb]
        ms = (nc - p5) * ar_per_forward / 1000.0
        rows.append({"T": T, "bytes": nb, "nccl_us": nc, "p5_us": p5, "save_us_per_ar": round(nc - p5, 2),
                     "save_ms_per_microbatch": round(ms, 3), "pct_of_tick": round(100.0 * ms / tick_ms, 2)})
    return rows


def check_tree(tree):
    """CLI helper: the read-only state of a tree plus the line numbers of the integration points."""
    tree = Path(tree)
    f = tree / CC_REL
    lines = f.read_text(encoding="utf-8").splitlines()
    find = lambda s: next((i + 1 for i, l in enumerate(lines) if l.strip().startswith(s)), None)  # noqa: E731
    return {"verify": verify_p5_patch(tree), "def_all_reduce": find("def all_reduce(self, input_)"),
            "log_call": find("self._log_all_reduce_backend_selection()"),
            "def_log": find("def _log_all_reduce_backend_selection")}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-tree")
    ap.add_argument("--write-payload")
    ap.add_argument("--shas", action="store_true")
    a = ap.parse_args()
    if a.check_tree:
        print(json.dumps(check_tree(a.check_tree), indent=1))
    if a.write_payload:
        print(json.dumps(write_payload(a.write_payload), indent=1))
    if a.shas or not (a.check_tree or a.write_payload):
        print(json.dumps({"CC_PRE": CC_PRE, "CC_POST": CC_POST, "PATCH_SHA": PATCH_SHA, "BACKEND_SHA": BACKEND_SHA,
                          "WORKER_EXT_SHA": WORKER_EXT_SHA, "GATE_SHA": GATE_SHA,
                          "expected_gain": expected_gain()}, indent=1))
