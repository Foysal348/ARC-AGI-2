"""E14-0 load path for the SGLang server on Kaggle 4xL4 (rev 2, 2026-10-23).

Why (research 2026-10-23, Notes/Research - 4xL4 engine bottleneck validation/startup_load_environment.md):
- Since ~2026-10-21 part of /kaggle/input is FUSE.GCSFUSE and NFS slowed too: ~10.5 MB/s single-stream vs ~107 MB/s
  with 16 readers (amn, topic 751344). Our INT4 shard took 657 s on 10-21 (12.8-58.7 s before).
- SGLang 0.5.21's loader reads per file (our target is one 18.6 GB file), so its own prefetch is one stream.
- At 10.5 MB/s a single-stream read of 23.3 GB takes ~37 min; a fixed 30-min first-launch timeout then ends in
  E11Fail -> fallback submission (~0 LB).

Rev 2 follows GPT notes 08/09 as validated in Notes/10: write loop (short writes), `.partial` dir + atomic rename,
per-range blake2b + destination re-read, full sha256 per shard, strict safetensors layout check, cgroup memory, adaptive
worker ramp 4 -> 8 -> 16 (cap 24), an off switch, a phase record (E14State), a fixed JIT manifest.

What this module gives the notebook (stdlib only, no GPU, works on Windows for the CPU tests):
- env_report()         one-shot facts: mount type per path, df, RAM, cgroup memory, nproc, CPU model
- read_probe()         timed 64 MiB read of the largest input file (MB/s per stream)
- parallel_copy()      ranged multi-reader copy of a model dir, verified, published atomically, stall-aware
- warm_page_cache()    the same ranged readers without writing (fallback when no space or the copy failed)
- stage_models()/Staging   the notebook entry point (never raises; fails open to the original paths)
- StallMonitor         launch watchdog: progress = log growth OR the server tree's read() bytes + page faults; never
                       kills a server that is still making progress; gives up only on a stall, a dead process, give_up_at
- E14State             PREPARE -> COPY -> VALIDATE -> LAUNCH -> READY record with deadlines and a terminal reason
- jit_manifest(), save_jit_cache(), restore_jit_cache(), known_answer_ok()   compile caches without the CUDA driver
                       cache (nv/, SGLang #42917), restored only on an exact fingerprint match + a known-answer check
"""
import hashlib
import io
import json
import os
import queue
import shutil
import struct
import tarfile
import threading
import time
from pathlib import Path

MIB = 1 << 20
PROC = "/proc"            # module constants: the CPU tests point them at fake trees
CGROUP = "/sys/fs/cgroup"


# --------------------------------------------------------------------------------------------- environment facts
def _mount_of(path):
    """(mount point, fstype) of the longest /proc/mounts entry containing path, or (None, None) off Linux."""
    try:
        best = ("", None)
        p = os.path.realpath(path) if os.name == "posix" else str(path)
        for line in Path(PROC, "mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                mp = parts[1].replace("\\040", " ")
                if (p == mp or p.startswith(mp.rstrip("/") + "/")) and len(mp) > len(best[0]):
                    best = (mp, parts[2])
        return best if best[1] else (None, None)
    except Exception:
        return (None, None)


def _df(path):
    try:
        u = shutil.disk_usage(path)
        return {"total_gb": round(u.total / 1e9, 1), "free_gb": round(u.free / 1e9, 1)}
    except Exception as e:
        return {"error": repr(e)[:80]}


def _mem_available():
    try:
        for line in Path(PROC, "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _read_int(p):
    try:
        t = Path(p).read_text().strip()
        return None if t in ("", "max") else int(t)
    except Exception:
        return None


def cgroup_memory():
    """(limit, usage) in bytes of this container's memory cgroup, v2 then v1; None where unknown or unlimited.
    /proc/meminfo inside a container can show the HOST, and tmpfs (/dev/shm) pages are charged to the cgroup."""
    lim, use = _read_int(Path(CGROUP, "memory.max")), _read_int(Path(CGROUP, "memory.current"))
    if lim is None and use is None:
        lim = _read_int(Path(CGROUP, "memory", "memory.limit_in_bytes"))
        use = _read_int(Path(CGROUP, "memory", "memory.usage_in_bytes"))
        if lim is not None and lim >= 1 << 60:         # v1 reports "unlimited" as a huge number
            lim = None
    return lim, use


def host_headroom():
    """Bytes that can still be allocated: the smaller of MemAvailable and (cgroup limit - cgroup usage)."""
    vals = []
    ma = _mem_available()
    if ma is not None:
        vals.append(ma)
    lim, use = cgroup_memory()
    if lim is not None:
        vals.append(max(0, lim - (use or 0)))
    return min(vals) if vals else None


def env_report(paths):
    """Facts the 10-22 ERROR post-mortem lacked (startup_load_environment.md): mounts, free space, RAM, cgroup, CPU."""
    rep = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "nproc": os.cpu_count(), "paths": {}, "df": {}}
    for p in paths:
        mp, fs = _mount_of(p)
        rep["paths"][str(p)] = {"mount": mp, "fstype": fs, "exists": Path(p).exists()}
    for d in ("/dev/shm", "/tmp", "/kaggle/working", "/kaggle/tmp"):
        if Path(d).exists():
            rep["df"][d] = _df(d)
    try:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in Path(PROC, "meminfo").read_text().splitlines() if l.split()[1:2]}
        rep["mem_gb"] = {"total": round(mi.get("MemTotal", 0) / 1e6, 1), "available": round(mi.get("MemAvailable", 0) / 1e6, 1)}
    except Exception:
        rep["mem_gb"] = None
    lim, use = cgroup_memory()
    rep["cgroup_gb"] = {"limit": None if lim is None else round(lim / 1e9, 1), "usage": None if use is None else round(use / 1e9, 1)}
    try:
        rep["cpu_model"] = next((l.split(":", 1)[1].strip() for l in Path(PROC, "cpuinfo").read_text().splitlines()
                                 if l.startswith("model name")), None)
    except Exception:
        rep["cpu_model"] = None
    return rep


def _files(src):
    src = Path(src)
    return sorted((p for p in src.rglob("*") if p.is_file()), key=lambda p: -p.stat().st_size)


def read_probe(src, nbytes=64 * MIB, chunk=8 * MIB):
    """MB/s of one sequential stream over the first nbytes of the largest file in src."""
    fs = _files(src)
    if not fs:
        return None
    f, done, t0 = fs[0], 0, time.perf_counter()
    with open(f, "rb", buffering=0) as h:
        while done < nbytes:
            b = h.read(min(chunk, nbytes - done))
            if not b:
                break
            done += len(b)
    dt = max(1e-6, time.perf_counter() - t0)
    return {"file": f.name, "bytes": done, "sec": round(dt, 3), "mb_s": round(done / dt / 1e6, 1)}


# --------------------------------------------------------------------------------------------- phase record
class E14State:
    """PREPARE -> COPY -> VALIDATE -> LAUNCH -> READY (or a terminal FAILED/FALLBACK). Every change is written to a JSON
    file at once (write + rename), so a run killed mid-phase still shows where it stopped and why."""

    def __init__(self, path=None, now=time.perf_counter):
        self.path, self.now = (Path(path) if path else None), now
        self.t0 = now()
        self.rec = {"phases": [], "current": None, "terminal_reason": None}

    def enter(self, phase, deadline=None, **info):
        e = {"phase": phase, "t": round(self.now() - self.t0, 1),
             "deadline_in": None if deadline is None else round(deadline - self.now(), 1), **info}
        self.rec["phases"].append(e)
        self.rec["current"] = phase
        self._flush()

    def note(self, **info):
        if self.rec["phases"]:
            self.rec["phases"][-1].update(info)
            self._flush()

    def end(self, reason):
        self.rec["terminal_reason"] = reason
        self.rec["t_end"] = round(self.now() - self.t0, 1)
        self._flush()

    def _flush(self):
        if not self.path:
            return
        try:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.rec, indent=1, default=str), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            pass


# --------------------------------------------------------------------------------------------- ranged readers
class CopyStalled(RuntimeError):
    pass


class _Progress:
    def __init__(self, total):
        self.total, self.done, self.retries, self.lock = total, 0, 0, threading.Lock()
        self.last_change = time.perf_counter()
        self.abort = threading.Event()
        self.hashes = {}                                   # (dst file, offset) -> blake2b of the source bytes

    def add(self, n):
        with self.lock:
            self.done += n
            self.last_change = time.perf_counter()

    def retry(self):
        with self.lock:
            self.retries += 1


def _backoff(attempt):
    """Seconds to wait before retry `attempt` (0-based) of one range: 1, 2, 4, 8 (cap 30)."""
    return min(30, 2 ** attempt)


def _ranges(size, chunk):
    """Disjoint ranges that cover [0, size) exactly (asserted: a gap would leave preallocated zeros in the copy)."""
    rs = [(o, min(chunk, size - o)) for o in range(0, size, chunk)] or [(0, 0)]
    assert rs[0][0] == 0 and sum(n for _, n in rs) == size
    assert all(a[0] + a[1] == b[0] for a, b in zip(rs, rs[1:]))
    return rs


def _write_all(fo, b):
    """A raw (buffering=0) write may write fewer bytes than asked; loop until all are written."""
    mv, done = memoryview(b), 0
    while done < len(mv):
        n = fo.write(mv[done:])
        if not n:
            raise IOError("write returned 0 bytes")
        done += n


def _copy_range(src, dst, off, n, prog, retries=4):
    """One range with its own handles (portable seek+read/write; os.pread is not on Windows). Short reads raise and are
    retried; short writes loop; the blake2b of the bytes read is kept for the destination re-read."""
    for a in range(retries + 1):
        if prog.abort.is_set():
            return
        try:
            h = hashlib.blake2b(digest_size=16)
            with open(src, "rb", buffering=0) as fi, open(dst, "r+b", buffering=0) as fo:
                fi.seek(off)
                fo.seek(off)
                left = n
                while left > 0:
                    if prog.abort.is_set():
                        return
                    b = fi.read(min(left, 8 * MIB))
                    if not b:
                        raise IOError(f"short read at {off + n - left} in {src}")
                    _write_all(fo, b)
                    h.update(b)
                    left -= len(b)
                    prog.add(len(b))
            prog.hashes[(str(dst), off)] = h.digest()
            return
        except Exception:
            if a == retries:
                raise
            prog.retry()
            time.sleep(_backoff(a))


def _read_range(src, off, n, prog, retries=4):
    for a in range(retries + 1):
        if prog.abort.is_set():
            return
        try:
            with open(src, "rb", buffering=0) as fi:
                fi.seek(off)
                left = n
                while left > 0:
                    if prog.abort.is_set():
                        return
                    b = fi.read(min(left, 8 * MIB))
                    if not b:
                        raise IOError(f"short read in {src}")
                    left -= len(b)
                    prog.add(len(b))
            return
        except Exception:
            if a == retries:
                raise
            prog.retry()
            time.sleep(_backoff(a))


def _levels(workers, levels):
    out = sorted({min(int(x), workers) for x in levels if int(x) >= 1} | {workers})
    return [x for x in out if x <= workers] or [workers]


def _run_ranged(jobs, prog, workers, stall_sec, deadline, report_every, log, levels=(4, 8, 16, 24),
                ramp_bytes=512 * MIB, ramp_gain=1.15):
    """jobs: callables, run by DAEMON threads; aborts on a stall (no byte for stall_sec) or at deadline (a
    time.perf_counter() value, the clock of the notebook's SGL_GIVE_UP_AT). Hung threads are never joined: a read stuck
    inside FUSE would block a join (and, with ThreadPoolExecutor's non-daemon workers, interpreter exit) forever.

    Worker ramp (GPT note 08: no fixed 16-32 default): start at the first level; after ramp_bytes at a level, go up only
    if MB/s rose by ramp_gain over the previous level, else drop back one level and stay. A retry (transient error) at a
    level drops back one level and stays. Returns {"levels": [(workers, MB/s), ...], "final_workers": n}."""
    lv = _levels(workers, levels)
    todo = queue.SimpleQueue()
    for j in jobs:
        todo.put(j)
    errors = []
    cond = threading.Condition()
    allowed = [lv[0]]
    finished = threading.Event()

    def worker(idx):
        while not prog.abort.is_set():
            with cond:
                while idx >= allowed[0] and not prog.abort.is_set() and not finished.is_set():
                    cond.wait(0.5)
            if prog.abort.is_set() or finished.is_set():
                return
            try:
                job = todo.get_nowait()
            except queue.Empty:
                finished.set()
                with cond:
                    cond.notify_all()
                return
            try:
                job()
            except BaseException as e:
                errors.append(e)
                prog.abort.set()
                return

    threads = [threading.Thread(target=worker, args=(i,), name=f"e14load{i}", daemon=True)
               for i in range(max(1, min(lv[-1], len(jobs))))]
    for t in threads:
        t.start()
    li, frozen = 0, len(lv) == 1
    lvl_t, lvl_b, lvl_r, prev_rate = time.perf_counter(), prog.done, prog.retries, None
    hist = []
    last_report = time.perf_counter()

    def set_level(i):
        with cond:
            allowed[0] = lv[i]
            cond.notify_all()

    try:
        while True:
            if errors:
                raise errors[0]
            if not any(t.is_alive() for t in threads):
                return {"levels": hist, "final_workers": lv[li]}
            now = time.perf_counter()
            if now - prog.last_change > stall_sec:
                raise CopyStalled(f"no progress for {stall_sec:.0f} s at {prog.done / 1e9:.2f}/{prog.total / 1e9:.2f} GB")
            if deadline is not None and now > deadline:
                raise CopyStalled(f"deadline reached at {prog.done / 1e9:.2f}/{prog.total / 1e9:.2f} GB")
            if not frozen:
                if prog.retries > lvl_r:
                    hist.append((lv[li], None))
                    li = max(0, li - 1)
                    set_level(li)
                    frozen = True
                elif prog.done - lvl_b >= ramp_bytes:
                    rate = (prog.done - lvl_b) / max(1e-6, now - lvl_t)
                    hist.append((lv[li], round(rate / 1e6, 1)))
                    if prev_rate is not None and rate < prev_rate * ramp_gain:
                        li = max(0, li - 1)
                        set_level(li)
                        frozen = True
                    elif li + 1 < len(lv):
                        li += 1
                        set_level(li)
                        prev_rate, lvl_t, lvl_b, lvl_r = rate, now, prog.done, prog.retries
                    else:
                        frozen = True
            if log and now - last_report >= report_every:
                log(f"E14 load: {prog.done / 1e9:.2f}/{prog.total / 1e9:.2f} GB, {allowed[0]} workers")
                last_report = now
            time.sleep(0.2)
    except BaseException:
        prog.abort.set()
        with cond:
            cond.notify_all()
        raise


# --------------------------------------------------------------------------------------------- verification
_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "U16": 2, "I16": 2, "F16": 2,
                "BF16": 2, "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}


def safetensors_layout(path):
    """(ok, why). The format rule: an 8-byte LE header length, a JSON header, and a byte buffer that is fully indexed
    without holes. Checks: sorted by begin, first begin == 0, each end == the next begin, last end == buffer size, and
    end - begin == dtype bytes x prod(shape) for known dtypes."""
    try:
        size = Path(path).stat().st_size
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                return False, "file shorter than 8 bytes"
            hl = struct.unpack("<Q", raw)[0]
            if hl <= 0 or 8 + hl > size:
                return False, f"bad header length {hl}"
            hdr = json.loads(f.read(hl))
        buf = size - 8 - hl
        ts = sorted(((v["data_offsets"][0], v["data_offsets"][1], k, v) for k, v in hdr.items()
                     if k != "__metadata__" and isinstance(v, dict)), key=lambda x: (x[0], x[1]))
        if not ts:
            return buf == 0, f"no tensors, buffer {buf} bytes"
        if ts[0][0] != 0:
            return False, f"first tensor starts at {ts[0][0]}"
        for (b0, e0, k0, _), (b1, _e1, k1, _v1) in zip(ts, ts[1:]):
            if e0 != b1:
                return False, f"{'overlap' if e0 > b1 else 'hole'} between {k0} and {k1}"
        for b, e, k, v in ts:
            if e < b:
                return False, f"negative extent in {k}"
            db = _DTYPE_BYTES.get(v.get("dtype"))
            if db is not None:
                n = db
                for d in v.get("shape", []):
                    n *= int(d)
                if e - b != n:
                    return False, f"{k}: {e - b} bytes for {v.get('dtype')}{v.get('shape')}"
        if ts[-1][1] != buf:
            return False, f"tensors end at {ts[-1][1]}, buffer is {buf} bytes"
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:200]


def verify_safetensors(path):
    """Kept for the rev-1 callers: the strict layout check."""
    return safetensors_layout(path)


def _hash_range(path, off, n):
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb", buffering=0) as f:
        f.seek(off)
        left = n
        while left > 0:
            b = f.read(min(left, 8 * MIB))
            if not b:
                raise IOError(f"short read while verifying {path} at {off + n - left}")
            h.update(b)
            left -= len(b)
    return h.digest()


def sha256_file(path, chunk=8 * MIB):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def _verify_tree(src, dst, prog=None, ranged=(), workers=8, sha256=False, expected_sha256=None):
    """Everything must hold before the copy is published:
    - every file exists with the source size;
    - ranged files: the re-read blake2b of every destination range == the hash of the source bytes copied into it;
    - small files: byte-identical to the source;
    - .safetensors: strict layout (a failure that the SOURCE shares is accepted as published, and reported);
    - optional full sha256 per shard, compared to expected_sha256 {file name: hex} when given."""
    src, dst = Path(src), Path(dst)
    info = {"layout_as_published": [], "sha256": {}}
    ranged = {str(g): rs for g, rs in ranged}
    for f in _files(src):
        g = dst / f.relative_to(src)
        size = f.stat().st_size
        if not g.exists() or g.stat().st_size != size:
            return False, f"size mismatch: {f.name}", info
        if str(g) not in ranged and Path(f).read_bytes() != Path(g).read_bytes():
            return False, f"content mismatch: {f.name}", info
    if ranged:
        if prog is None:
            return False, "ranged files without copy hashes", info
        bad = []
        jobs = [(g, off, n) for g, rs in ranged.items() for off, n in rs]
        lock = threading.Lock()

        def check(g, off, n):
            if _hash_range(g, off, n) != prog.hashes.get((g, off)):
                with lock:
                    bad.append((Path(g).name, off))

        vprog = _Progress(sum(n for _, _, n in jobs))
        wrapped = [lambda g=g, off=off, n=n: (check(g, off, n), vprog.add(n)) for g, off, n in jobs]
        _run_ranged(wrapped, vprog, workers, 600.0, None, 1e9, None, levels=(workers,))
        if bad:
            return False, f"range hash mismatch: {bad[0][0]} at offset {bad[0][1]} ({len(bad)} ranges)", info
    for f in _files(src):
        g = dst / f.relative_to(src)
        if g.suffix == ".safetensors":
            ok, why = safetensors_layout(g)
            if not ok:
                ok_src, why_src = safetensors_layout(f)
                if ok_src or why_src != why:
                    return False, f"{g.name}: {why}", info
                info["layout_as_published"].append(f"{g.name}: {why}")
        if sha256 and g.suffix == ".safetensors":
            d = sha256_file(g)
            info["sha256"][g.name] = d
            if expected_sha256 and g.name in expected_sha256 and expected_sha256[g.name] != d:
                return False, f"sha256 mismatch: {g.name}", info
    return True, "ok", info


# --------------------------------------------------------------------------------------------- copy / warm
def choose_dest(total_bytes, candidates=("/dev/shm", "/tmp"), margin_bytes=2 * (1 << 30), ram_headroom=24 * (1 << 30)):
    """First candidate with free space for the copy. /dev/shm is RAM (tmpfs, charged to the cgroup): it also needs
    host_headroom() >= copy + ram_headroom, left for the server, the 4 TP workers and the agent."""
    for d in candidates:
        try:
            if not Path(d).is_dir() or shutil.disk_usage(d).free < total_bytes * 1.02 + margin_bytes:
                continue
            if str(d).startswith("/dev/shm"):
                hr = host_headroom()
                if hr is None or hr < total_bytes + ram_headroom:
                    continue
            return d
        except Exception:
            continue
    return None


def staged_name(src, role=None):
    """A destination dir name unique per source: Kaggle model dirs all end in the version ("…/w4a16-g128/1",
    "…/bf16/1"), so the basename alone made target and draft collide (GPT audit 2 P0, reproduced in Notes/20)."""
    src = Path(src)
    tag = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
    return f"{role or 'm'}-{src.parent.name}-{src.name}-{tag}"


def parallel_copy(src, dst_root, *, workers=24, chunk=64 * MIB, stall_sec=600.0, deadline=None, report_every=30.0,
                  log=print, small_bytes=256 * MIB, levels=(4, 8, 16, 24), ramp_bytes=512 * MIB, sha256=True,
                  expected_sha256=None, name=None):
    """Copy every file of src into dst_root/<name>.partial/ with ranged readers, verify, then publish it with one
    atomic rename to dst_root/<name>/ (name defaults to staged_name(src)). Returns (dst_dir, stats). Raises
    (CopyStalled, IOError, ...) after deleting the partial copy: no launch can ever see an unverified directory. The
    caller then launches from src (fail open)."""
    src = Path(src)
    files = _files(src)
    total = sum(f.stat().st_size for f in files)
    name = name or staged_name(src)
    final = Path(dst_root) / name
    work = Path(dst_root) / (name + ".partial")
    for d in (work, final):                           # a stale copy from an earlier attempt is never trusted
        shutil.rmtree(d, ignore_errors=True)
    prog = _Progress(total)
    t0 = time.perf_counter()
    jobs, ranged = [], []
    try:
        work.mkdir(parents=True)
        for f in files:          # largest first (sorted by _files)
            g = work / f.relative_to(src)
            g.parent.mkdir(parents=True, exist_ok=True)
            size = f.stat().st_size
            if size <= small_bytes:
                jobs.append(lambda f=f, g=g, size=size: (shutil.copyfile(f, g), prog.add(size)))
                continue
            with open(g, "wb") as h:                  # preallocate (sparse) so ranged writers can seek anywhere
                h.truncate(size)
            rs = _ranges(size, chunk)
            ranged.append((g, rs))
            for off, n in rs:
                jobs.append(lambda f=f, g=g, off=off, n=n: _copy_range(f, g, off, n, prog))
        ramp = _run_ranged(jobs, prog, workers, stall_sec, deadline, report_every, log, levels=levels,
                           ramp_bytes=ramp_bytes)
        t_copy = time.perf_counter() - t0
        ok, why, vinfo = _verify_tree(src, work, prog, ranged, workers=min(workers, 16), sha256=sha256,
                                      expected_sha256=expected_sha256)
        if not ok:
            raise IOError("copy verification failed: " + why)
        os.replace(work, final)                       # atomic publish (same filesystem)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(final, ignore_errors=True)
        raise
    dt = max(1e-6, time.perf_counter() - t0)
    return final, {"bytes": total, "sec": round(dt, 1), "copy_sec": round(t_copy, 1),
                   "mb_s": round(total / max(1e-6, t_copy) / 1e6, 1), "files": len(files), "workers_max": workers,
                   "ramp": ramp, "retries": prog.retries, "chunk_mib": chunk // MIB, "dst": str(final), **vinfo}


def warm_page_cache(src, *, workers=24, chunk=64 * MIB, stall_sec=600.0, deadline=None, report_every=30.0, log=print,
                    levels=(4, 8, 16, 24), ramp_bytes=512 * MIB):
    files = _files(src)
    total = sum(f.stat().st_size for f in files)
    prog = _Progress(total)
    t0 = time.perf_counter()
    jobs = [lambda f=f, off=off, n=n: _read_range(f, off, n, prog)
            for f in files for off, n in _ranges(f.stat().st_size, chunk)]
    ramp = _run_ranged(jobs, prog, workers, stall_sec, deadline, report_every, log, levels=levels, ramp_bytes=ramp_bytes)
    dt = max(1e-6, time.perf_counter() - t0)
    return {"bytes": total, "sec": round(dt, 1), "mb_s": round(total / dt / 1e6, 1), "files": len(files), "ramp": ramp}


def loadpath_enabled():
    """Off switch: E14_LOADPATH=0 keeps the old single-stream `cat` prefetch path in the notebook."""
    return os.environ.get("E14_LOADPATH", "1").strip() != "0"


def stage_models(model_dirs, *, workers=24, stall_sec=600.0, deadline=None, log=print,
                 dest_candidates=("/dev/shm", "/tmp"), state=None, enabled=None, **copy_kw):
    """The notebook entry point: copy every model dir to the first destination with space; on any failure delete the
    partial copy, warm the page cache in place (ranged, parallel) and launch from the original paths. Never raises
    (except KeyboardInterrupt/SystemExit). A model copied successfully is never read from the source again.
    Returns (path map, report)."""
    model_dirs = [Path(m) for m in model_dirs]
    out = {str(m): str(m) for m in model_dirs}
    if not (loadpath_enabled() if enabled is None else enabled):
        if state:
            state.enter("PREPARE", disabled=True)
        return out, {"disabled": True}
    total = sum(sum(f.stat().st_size for f in _files(m)) for m in model_dirs)
    hr = host_headroom()
    rep = {"total_gb": round(total / 1e9, 2), "dest": None, "copies": {}, "warm": None, "error": None,
           "headroom_gb": None if hr is None else round(hr / 1e9, 1)}
    dest = choose_dest(total, dest_candidates)
    rep["dest"] = dest
    if state:
        state.enter("PREPARE", deadline=deadline, dest=dest, total_gb=rep["total_gb"], headroom_gb=rep["headroom_gb"])
    copied = set()
    names = [staged_name(m, f"m{i}") for i, m in enumerate(model_dirs)]
    if len(set(names)) != len(names) or len({str(m.resolve()) for m in model_dirs}) != len(model_dirs):
        rep["error"] = f"refused: two sources would share a destination: {names}"
        dest = None                                        # never stage over each other; launch from the sources
    if dest:
        try:
            root = Path(dest) / "e14-weights"
            for m, nm in zip(model_dirs, names):
                if state:
                    state.enter("COPY", deadline=deadline, model=m.name, dest_name=nm)
                d, st = parallel_copy(m, root, workers=workers, stall_sec=stall_sec, deadline=deadline, log=log,
                                      name=nm, **copy_kw)
                if str(d) in out.values():
                    raise IOError(f"staged path collision: {d}")
                out[str(m)] = str(d)
                rep["copies"][str(m)] = st
                copied.add(str(m))
                if state:
                    state.note(validated=True, mb_s=st["mb_s"], ramp=st["ramp"])
            if state:
                state.enter("VALIDATE", ok=True)
            return out, rep
        except BaseException as e:
            rep["error"] = f"{type(e).__name__}: {e}"[:300]
            if state:
                state.note(error=rep["error"])
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
    rest = [m for m in model_dirs if str(m) not in copied]   # a verified copy stays in use
    if state:
        state.enter("FALLBACK", deadline=deadline, warm=[m.name for m in rest])
    try:
        rep["warm"] = {str(m): warm_page_cache(m, workers=workers, stall_sec=stall_sec, deadline=deadline, log=log)
                       for m in rest}
    except BaseException as e:
        rep["warm"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
    return out, rep


class Staging(threading.Thread):
    """stage_models on a daemon thread, so the copy overlaps the venv install (the old `cat` prefetch did the same).
    After join(timeout), result is (path map, report), or None when it is still running: then launch from the
    original paths (the page cache is warm for whatever was already read)."""

    def __init__(self, model_dirs, **kw):
        super().__init__(name="e14stage", daemon=True)
        self.model_dirs, self.kw, self.result = list(model_dirs), kw, None

    def run(self):
        self.result = stage_models(self.model_dirs, **self.kw)


# --------------------------------------------------------------------------------------------- launch watchdog
def _proc_tree(pid):
    """pid and all descendants (Linux /proc); [pid] elsewhere."""
    try:
        kids = {}
        for d in Path(PROC).iterdir():
            if d.name.isdigit():
                try:
                    st = (d / "stat").read_text()
                    ppid = int(st.rsplit(")", 1)[1].split()[1])
                    kids.setdefault(ppid, []).append(int(d.name))
                except Exception:
                    continue
        out, todo = [], [pid]
        while todo:
            p = todo.pop()
            out.append(p)
            todo.extend(kids.get(p, []))
        return out
    except Exception:
        return [pid]


def _rchar(pids):
    tot = 0
    for p in pids:
        try:
            for line in Path(PROC, str(p), "io").read_text().splitlines():
                if line.startswith("rchar:"):
                    tot += int(line.split()[1])
        except Exception:
            continue
    return tot


def _faults(pids):
    """minflt + majflt of the tree. SGLang 0.5.21 reads safetensors through mmap (safe_open, weight_utils.py L1160), and
    page faults on a mapping never show in rchar; a process hung in a NCCL spin makes almost none."""
    tot = 0
    for p in pids:
        try:
            f = Path(PROC, str(p), "stat").read_text().rsplit(")", 1)[1].split()
            tot += int(f[7]) + int(f[9])        # fields 10 (minflt) and 12 (majflt); f[0] is field 3 (state)
        except Exception:
            continue
    return tot


def _io_counter(pid):
    """One monotone-ish progress number for the server tree: bytes read() + 4 KiB per page fault."""
    tree = _proc_tree(pid)
    r, flt = _rchar(tree), _faults(tree)
    return r + 4096 * flt, r, flt


class StallMonitor:
    """Waits for the server. Progress = the server tree's I/O counter grew (read() bytes + page faults: the mmap weight
    load, the draft load, compiles) OR the log grew (phase 1: any byte; phase 2, after the first 'Load weight end'
    (model_runner.py L1280): a new line, so a tqdm redraw alone does not count). Stalls: load_stall_sec in phase 1,
    compile_stall_sec in phase 2 (must exceed the ~6 min silent first verify-graph capture: 362 s in E11-06). A server
    that progresses is never given up on before give_up_at (absolute time.perf_counter() value)."""

    LOAD_END = "Load weight end"

    def __init__(self, proc, log_path, *, load_stall_sec=900.0, compile_stall_sec=720.0, poll_sec=5.0, heartbeat_sec=30.0,
                 log=print, now=time.perf_counter):
        self.proc, self.log_path, self.log, self.now = proc, Path(log_path), log, now
        self.load_stall_sec, self.compile_stall_sec = float(load_stall_sec), float(compile_stall_sec)
        self.poll_sec, self.heartbeat_sec = float(poll_sec), float(heartbeat_sec)

    def _log_state(self):
        try:
            t = self.log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            t = ""
        return len(t), t.count("\n"), (self.LOAD_END in t)

    def wait(self, is_ready, give_up_at):
        t0 = self.now()
        size0, lines0, loaded = self._log_state()
        c0 = _io_counter(self.proc.pid)[0]
        last_prog, last_beat = t0, t0
        phase = 2 if loaded else 1
        stats = {"phase_changes": [], "rchar_mb": 0.0, "faults": 0}
        while True:
            if self.proc.poll() is not None:
                return False, f"server exited rc={self.proc.returncode}", stats
            try:
                if is_ready():
                    stats["sec"] = round(self.now() - t0, 1)
                    return True, f"ready in {self.now() - t0:.0f}s", stats
            except Exception:
                pass
            now = self.now()
            size, lines, loaded = self._log_state()
            c, r, flt = _io_counter(self.proc.pid)
            progressed = (c > c0) or ((size > size0) if phase == 1 else (lines > lines0))
            if progressed:
                last_prog = now
            if phase == 1 and loaded:
                phase = 2
                last_prog = now
                stats["phase_changes"].append(round(now - t0, 1))
            size0, lines0, c0 = size, lines, max(c0, c)     # max: a worker that exits must not look like progress later
            # peaks: one unreadable /proc/<pid> sample (a worker exiting) must not shrink the reported totals
            stats["rchar_mb"], stats["faults"] = max(stats["rchar_mb"], round(r / 1e6, 1)), max(stats["faults"], flt)
            limit = self.load_stall_sec if phase == 1 else self.compile_stall_sec
            if now - last_prog > limit:
                stats["sec"] = round(now - t0, 1)
                return False, f"stall in phase {phase}: no progress for {now - last_prog:.0f}s", stats
            if now > give_up_at:
                stats["sec"] = round(now - t0, 1)
                return False, "give-up time reached while still progressing", stats
            if self.log and now - last_beat >= self.heartbeat_sec:
                self.log(f"E14 launch: phase {phase}, {now - t0:.0f}s, log {size} B, rchar {r / 1e9:.2f} GB, faults {flt}")
                last_beat = now
            time.sleep(self.poll_sec)


# --------------------------------------------------------------------------------------------- JIT / compile caches
def jit_cache_env(base="/tmp/e11-cache"):
    """The cache dirs SGLang 0.5.21 uses. SGLANG_JIT_CACHE_DIR is NOT under SGLANG_CACHE_DIR by default
    (environ.py L1220-1222; kernels/jit/utils/compile/cache.py L301-308), so it is set explicitly."""
    b = Path(base)
    return {"SGLANG_CACHE_DIR": str(b / "sglang"), "SGLANG_JIT_CACHE_DIR": str(b / "sglang_jit"),
            "TRITON_CACHE_DIR": str(b / "triton"), "TORCH_EXTENSIONS_DIR": str(b / "torch_ext"),
            "CUDA_CACHE_PATH": str(b / "cuda")}


_NEVER_SHIP = ("sglang/nv", "cuda")     # CUDA driver compute caches: SGLang #42917 (stale nv/ -> PPL 9.98)

# GPT note 08: nv/ exclusion is no compatibility proof. Every field must be present and match exactly.
JIT_MANIFEST_KEYS = ("sglang", "sgl_kernel", "flashinfer", "torch", "triton", "cuda_runtime", "nvcc", "driver",
                     "gpu_name", "sm", "wheels", "jit_sources", "env")
_KERNEL_ENV_PREFIXES = ("TORCH_CUDA_ARCH_LIST", "SGLANG_", "FLASHINFER_", "TRITON_", "NVCC_", "CUDA_HOME")
_CACHE_ENV_KEYS = set(jit_cache_env())


def wheel_inventory(wheel_dir):
    """[[file name, size], ...] of the wheelhouse: names carry version + build tag. (Hashing 5 GB on GCSFUSE would cost
    minutes; the installed JIT sources are hashed instead, by source_fingerprint.)"""
    return sorted([p.name, p.stat().st_size] for p in Path(wheel_dir).glob("*.whl"))


def source_fingerprint(dirs, suffixes=(".py", ".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".c", ".jinja", ".txt")):
    """sha256 over (relative path, content) of the kernel-source files of the installed packages (local venv disk)."""
    h = hashlib.sha256()
    for d in dirs:
        d = Path(d)
        if not d.exists():
            h.update(b"missing:" + str(d.name).encode())
            continue
        for p in sorted(x for x in d.rglob("*") if x.is_file() and x.suffix in suffixes and "__pycache__" not in x.parts):
            h.update(p.relative_to(d.parent).as_posix().encode() + b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()


def kernel_env(env):
    """The env vars that can change compiled kernels (cache-location vars excluded: they differ per run by design)."""
    return {k: v for k, v in sorted(env.items()) if k.startswith(_KERNEL_ENV_PREFIXES) and k not in _CACHE_ENV_KEYS}


def jit_manifest(**fields):
    """A complete manifest or ValueError: a cache is never saved or restored against a partial fingerprint."""
    missing = [k for k in JIT_MANIFEST_KEYS if fields.get(k) in (None, "", [], {})]
    extra = [k for k in fields if k not in JIT_MANIFEST_KEYS]
    if missing or extra:
        raise ValueError(f"jit manifest: missing {missing}, unknown {extra}")
    return json.loads(json.dumps({k: fields[k] for k in JIT_MANIFEST_KEYS}, sort_keys=True))


def save_jit_cache(base, out_tar, manifest):
    """tar the compile caches without the driver caches, with the (complete) manifest inside as MANIFEST.json."""
    manifest = jit_manifest(**manifest)
    base = Path(base)
    out_tar = Path(out_tar)
    with tarfile.open(out_tar, "w") as tf:
        info = tarfile.TarInfo("MANIFEST.json")
        data = json.dumps(manifest, sort_keys=True).encode()
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base).as_posix()
            if any(rel == s or rel.startswith(s + "/") for s in _NEVER_SHIP):
                continue
            if p.is_file():
                tf.add(p, arcname="cache/" + rel)
    return out_tar


def restore_jit_cache(tar_path, base, expect_manifest):
    """Untar only when both manifests are complete and identical; returns (restored, why). Paths are checked against
    traversal. After a restore, known_answer_ok must pass, or the caller wipes the caches and compiles cold."""
    base = Path(base)
    try:
        expect = jit_manifest(**expect_manifest)
        with tarfile.open(tar_path) as tf:
            m = json.loads(tf.extractfile("MANIFEST.json").read())
            if m != expect:
                diff = sorted(k for k in set(m) | set(expect) if m.get(k) != expect.get(k))
                return False, f"manifest mismatch: {diff}"
            for mem in tf.getmembers():
                if not mem.name.startswith("cache/") or not mem.isfile():
                    continue
                rel = mem.name[len("cache/"):]
                if rel.startswith("/") or ".." in Path(rel).parts or any(rel == s or rel.startswith(s + "/") for s in _NEVER_SHIP):
                    continue
                dst = base / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(mem) as src, open(dst, "wb") as out:
                    shutil.copyfileobj(src, out)
        return True, "restored"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:200]


def known_answer_ok(got_tokens, got_logprobs, ref, lp_tol=0.05):
    """After a cache restore: the fixed greedy request must reproduce the cold-cache reference (token ids exactly; the
    first 16 logprobs within lp_tol). On False the caller wipes the caches and relaunches cold."""
    if list(got_tokens) != list(ref["tokens"]):
        return False
    a, b = list(got_logprobs)[:16], list(ref["logprobs"])[:16]
    return len(a) == len(b) and all(abs(x - y) <= lp_tol for x, y in zip(a, b))
