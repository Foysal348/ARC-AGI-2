"""CPU tests for e14_loadpath (run: python -m pytest -q test_e14_loadpath.py). No GPU, no network; Windows or Linux."""
import hashlib
import json
import os
import struct
import tarfile
import threading
import time
from pathlib import Path

import pytest

import e14_loadpath as L

MIB = 1 << 20


def make_safetensors(path, nbytes, seed=0):
    """A valid safetensors file: one U8 tensor of nbytes with pseudo-random content."""
    hdr = json.dumps({"t": {"dtype": "U8", "shape": [nbytes], "data_offsets": [0, nbytes]}}).encode()
    hdr += b" " * (-len(hdr) % 8)
    blob = hashlib.sha256(str(seed).encode()).digest() * (nbytes // 32 + 1)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)) + hdr + blob[:nbytes])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def model(tmp_path):
    m = tmp_path / "src" / "model"
    m.mkdir(parents=True)
    make_safetensors(m / "model.safetensors", 9 * MIB + 123, seed=1)       # 10 ranges of 1 MiB, last one partial
    make_safetensors(m / "draft.safetensors", 3 * MIB, seed=2)
    (m / "config.json").write_text('{"a": 1}')
    (m / "sub").mkdir()
    (m / "sub" / "tok.txt").write_text("x" * 1000)
    return m


# ------------------------------------------------------------------------------------------- copy
def test_copy_exact_and_verified(model, tmp_path):
    dst, st = L.parallel_copy(model, tmp_path / "dst", workers=4, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    for f in model.rglob("*"):
        if f.is_file():
            assert sha(f) == sha(dst / f.relative_to(model)), f.name
    assert st["files"] == 4 and st["bytes"] == sum(f.stat().st_size for f in model.rglob("*") if f.is_file())


def test_verify_safetensors_detects_truncation(model, tmp_path):
    p = tmp_path / "bad.safetensors"
    p.write_bytes((model / "model.safetensors").read_bytes()[:-10])
    ok, _ = L.verify_safetensors(p)
    assert not ok
    ok, _ = L.verify_safetensors(model / "model.safetensors")
    assert ok


def test_verify_tree_detects_unwritten_tail(model, tmp_path):
    """A preallocated (truncate) file has the right size; only the byte sample catches a missing tail range."""
    dst = tmp_path / "d" / "model"
    dst.parent.mkdir()
    import shutil
    shutil.copytree(model, dst)
    g = dst / "model.safetensors"
    data = bytearray(g.read_bytes())
    data[-100:] = b"\0" * 100
    g.write_bytes(bytes(data))
    ok, why, _ = L._verify_tree(model, dst)
    assert not ok and "content mismatch" in why


def test_hung_read_raises_stall_quickly_and_cleans_up(model, tmp_path, monkeypatch):
    """One reader hangs forever (FUSE stall): the copy must give up after stall_sec, not hang, and leave no copy."""
    release = threading.Event()
    real = L._copy_range

    def hang_on_third(src, dst, off, n, prog, retries=4):
        if off == 2 * MIB and Path(src).name == "model.safetensors":
            release.wait(30)            # simulates a read stuck in the kernel
            return
        return real(src, dst, off, n, prog, retries)

    monkeypatch.setattr(L, "_copy_range", hang_on_third)
    t0 = time.monotonic()
    with pytest.raises(L.CopyStalled):
        L.parallel_copy(model, tmp_path / "dst", workers=4, chunk=1 * MIB, small_bytes=64 * 1024, stall_sec=1.0, log=None)
    assert time.monotonic() - t0 < 5
    nm = L.staged_name(model)
    assert not (tmp_path / "dst" / nm).exists() and not (tmp_path / "dst" / (nm + ".partial")).exists()
    release.set()


def test_slow_but_steady_is_not_a_stall(model, tmp_path, monkeypatch):
    """Throttled storage (each range 0.3 s) with stall_sec 5.0 must finish: progress, not speed, is what counts.
    (1.0 and 2.0 s failed now and then on Windows only: gaps of seconds while files are opened, a host effect. Stall
    detection itself is tested by test_hung_reader_* and test_monitor_flat_counters_still_stall.)"""
    real = L._copy_range

    def slow(src, dst, off, n, prog, retries=4):
        time.sleep(0.3)
        return real(src, dst, off, n, prog, retries)

    monkeypatch.setattr(L, "_copy_range", slow)
    dst, _ = L.parallel_copy(model, tmp_path / "dst", workers=2, chunk=1 * MIB, small_bytes=64 * 1024, stall_sec=5.0,
                             log=None)
    assert sha(dst / "model.safetensors") == sha(model / "model.safetensors")


def test_read_error_retries_then_succeeds(model, tmp_path, monkeypatch):
    calls = {"n": 0}
    real_open = open

    def flaky_open(p, mode="r", *a, **k):
        if "rb" in mode and str(p).endswith("model.safetensors") and calls["n"] < 2:
            calls["n"] += 1
            raise OSError("transient EIO")
        return real_open(p, mode, *a, **k)

    monkeypatch.setattr("builtins.open", flaky_open)
    monkeypatch.setattr(L, "_backoff", lambda a: 0)
    dst, _ = L.parallel_copy(model, tmp_path / "dst", workers=1, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    monkeypatch.undo()
    assert sha(dst / "model.safetensors") == sha(model / "model.safetensors")


def test_deadline(model, tmp_path, monkeypatch):
    real = L._copy_range
    monkeypatch.setattr(L, "_copy_range", lambda *a, **k: (time.sleep(0.2), real(*a, **k)))
    with pytest.raises(L.CopyStalled, match="deadline"):
        L.parallel_copy(model, tmp_path / "dst", workers=1, chunk=1 * MIB, small_bytes=64 * 1024, stall_sec=10,
                        deadline=time.perf_counter() + 0.5, log=None)


def test_warm_page_cache(model):
    st = L.warm_page_cache(model, workers=3, chunk=1 * MIB, log=None)
    assert st["bytes"] == sum(f.stat().st_size for f in model.rglob("*") if f.is_file())


def test_stage_models_copies(model, tmp_path):
    d = tmp_path / "shm"
    d.mkdir()
    out, rep = L.stage_models([model], workers=4, log=None, dest_candidates=(str(d),))
    assert out[str(model)] == str(d / "e14-weights" / L.staged_name(model, "m0")) and rep["error"] is None
    assert sha(Path(out[str(model)]) / "model.safetensors") == sha(model / "model.safetensors")


def test_stage_models_fails_open_on_stall(model, tmp_path, monkeypatch):
    d = tmp_path / "shm"
    d.mkdir()

    def stalled(*a, **k):
        raise L.CopyStalled("no progress")

    monkeypatch.setattr(L, "parallel_copy", stalled)
    out, rep = L.stage_models([model], workers=2, log=None, dest_candidates=(str(d),))
    assert out[str(model)] == str(model) and "CopyStalled" in rep["error"] and rep["warm"][str(model)]["bytes"] > 0


def test_stage_models_no_space_warms_in_place(model, tmp_path):
    out, rep = L.stage_models([model], workers=2, log=None, dest_candidates=(str(tmp_path / "missing"),))
    assert rep["dest"] is None and out[str(model)] == str(model) and rep["warm"][str(model)]["files"] == 4


def test_choose_dest_dev_shm_needs_ram(monkeypatch):
    monkeypatch.setattr(L, "cgroup_memory", lambda: (None, None))
    monkeypatch.setattr(L, "_mem_available", lambda: 30 * (1 << 30))
    monkeypatch.setattr(L.Path, "is_dir", lambda self: True)

    class U:
        free = 100 * (1 << 30)

    monkeypatch.setattr(L.shutil, "disk_usage", lambda d: U)
    # 23.3 GB of weights + 24 GiB headroom > 30 GiB available: skip /dev/shm, take /tmp
    assert L.choose_dest(int(23.3e9), ("/dev/shm", "/tmp")) == "/tmp"
    monkeypatch.setattr(L, "_mem_available", lambda: 80 * (1 << 30))
    assert L.choose_dest(int(23.3e9), ("/dev/shm", "/tmp")) == "/dev/shm"


# ------------------------------------------------------------------------------------------- launch watchdog
class FakeProc:
    def __init__(self, pid=999999):
        self.pid, self.returncode = pid, None

    def poll(self):
        return self.returncode


def writer(path, chunks, every, newline=True):
    def run():
        for c in chunks:
            time.sleep(every)
            with open(path, "a", encoding="utf-8") as f:
                f.write(c + ("\n" if newline else ""))
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def _put(path, text):
    """Replace a fake /proc file atomically (real /proc reads never see a half-written file; write_text truncates first,
    so a monitor poll in between read an empty stat/io under CPU load)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    for _ in range(600):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:          # Windows: the monitor (or a file scanner) has the file open for a moment
            time.sleep(0.005)
    path.write_text(text)                # never let the writer thread die silently (it would look like a stall)


def mon(proc, log, **k):
    k.setdefault("poll_sec", 0.05)
    k.setdefault("log", None)
    return L.StallMonitor(proc, log, **k)


def test_ready(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("")
    t0 = time.perf_counter()
    ok, why, _ = mon(FakeProc(), log, load_stall_sec=1, compile_stall_sec=1).wait(
        lambda: time.perf_counter() - t0 > 0.3, give_up_at=time.perf_counter() + 10)
    assert ok and why.startswith("ready")


def test_load_stall(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("start\n")
    t0 = time.perf_counter()
    ok, why, _ = mon(FakeProc(), log, load_stall_sec=0.5, compile_stall_sec=5).wait(lambda: False, time.perf_counter() + 10)
    assert not ok and "phase 1" in why and time.perf_counter() - t0 < 2


def test_long_but_progressing_load_is_kept(tmp_path):
    """Log grows every 0.2 s for >= 2.2 s with load_stall 1.0: a fixed 1 s timeout would kill it, the monitor must not.
    Ready = 0.3 s after the writer finished (event-based: file writes on this Windows host take tens of ms each, so a
    fixed 2.6 s ready time sometimes came before the load-end line)."""
    log = tmp_path / "s.log"
    log.write_text("")
    th = writer(log, ["loading"] * 10 + [L.StallMonitor.LOAD_END], 0.2)
    done = {}

    def ready():
        if th.is_alive():
            return False
        done.setdefault("t", time.perf_counter())
        return time.perf_counter() - done["t"] > 0.3
    ok, why, st = mon(FakeProc(), log, load_stall_sec=1.0, compile_stall_sec=1.5).wait(ready, time.perf_counter() + 20)
    assert ok, why
    assert st["phase_changes"], "phase 2 not entered"


def test_phase2_counts_lines_not_bytes(tmp_path):
    """After Load weight end, bytes without a newline (a progress bar redraw) are not progress."""
    log = tmp_path / "s.log"
    log.write_text(L.StallMonitor.LOAD_END + "\n")
    writer(log, ["#"] * 30, 0.05, newline=False)
    ok, why, _ = mon(FakeProc(), log, load_stall_sec=5, compile_stall_sec=0.6).wait(lambda: False, time.perf_counter() + 10)
    assert not ok and "phase 2" in why


def test_process_exit(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("")
    p = FakeProc()
    threading.Timer(0.3, lambda: setattr(p, "returncode", 1)).start()
    ok, why, _ = mon(p, log, load_stall_sec=5, compile_stall_sec=5).wait(lambda: False, time.perf_counter() + 10)
    assert not ok and "rc=1" in why


def test_give_up_while_progressing(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("")
    writer(log, ["x"] * 100, 0.05)
    ok, why, _ = mon(FakeProc(), log, load_stall_sec=1, compile_stall_sec=1).wait(lambda: False, time.perf_counter() + 0.8)
    assert not ok and "give-up" in why


def test_health_exceptions_are_not_fatal(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("")
    n = {"i": 0}

    def health():
        n["i"] += 1
        if n["i"] < 5:
            raise ConnectionRefusedError
        return True

    ok, _, _ = mon(FakeProc(), log, load_stall_sec=2, compile_stall_sec=2).wait(health, time.perf_counter() + 10)
    assert ok


# ------------------------------------------------------------------------------------------- JIT cache
def full_manifest(**over):
    m = {"sglang": "0.5.21", "sgl_kernel": "0.4.2", "flashinfer": "0.6.18", "torch": "2.11.0", "triton": "3.6.0",
         "cuda_runtime": "13.0", "nvcc": "13.0.88", "driver": "580.95", "gpu_name": "NVIDIA L4", "sm": "8.9",
         "wheels": [["sglang-0.5.21-py3-none-any.whl", 123]], "jit_sources": "ab" * 32, "env": {"TORCH_CUDA_ARCH_LIST": "8.9"}}
    m.update(over)
    return m


def test_jit_cache_roundtrip_excludes_driver_cache(tmp_path):
    base = tmp_path / "cache"
    for rel in ("sglang/jit/a.so", "sglang_jit/k/b.so", "triton/x/c.cubin", "torch_ext/d.so",
                "sglang/nv/ComputeCache/e", "cuda/ComputeCache/f"):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rel)
    man = full_manifest()
    tar = L.save_jit_cache(base, tmp_path / "jit.tar", man)
    with tarfile.open(tar) as tf:
        names = tf.getnames()
    assert not any("/nv/" in n or n.startswith("cache/cuda/") for n in names)
    new = tmp_path / "new"
    ok, why = L.restore_jit_cache(tar, new, man)
    assert ok, why
    assert (new / "triton/x/c.cubin").read_text() == "triton/x/c.cubin"
    assert (new / "sglang_jit/k/b.so").exists()
    assert not (new / "sglang/nv").exists() and not (new / "cuda").exists()


def test_jit_cache_manifest_mismatch(tmp_path):
    base = tmp_path / "cache"
    (base / "triton").mkdir(parents=True)
    (base / "triton" / "a").write_text("a")
    tar = L.save_jit_cache(base, tmp_path / "jit.tar", full_manifest())
    for over in ({"sglang": "0.5.22"}, {"driver": "590.10"}, {"jit_sources": "cd" * 32}, {"env": {"TORCH_CUDA_ARCH_LIST": "9.0"}}):
        ok, why = L.restore_jit_cache(tar, tmp_path / "new", full_manifest(**over))
        assert not ok and why == f"manifest mismatch: {list(over)}" and not (tmp_path / "new").exists(), (over, why)


def test_jit_manifest_must_be_complete(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        L.jit_manifest(sglang="0.5.21")
    with pytest.raises(ValueError, match="unknown"):
        L.jit_manifest(**full_manifest(), extra=1)
    with pytest.raises(ValueError):
        L.save_jit_cache(tmp_path, tmp_path / "x.tar", {"sglang": "0.5.21"})
    base = tmp_path / "c"
    (base / "triton").mkdir(parents=True)
    (base / "triton" / "a").write_text("a")
    tar = L.save_jit_cache(base, tmp_path / "jit.tar", full_manifest())
    ok, why = L.restore_jit_cache(tar, tmp_path / "new", {"sglang": "0.5.21"})     # partial expectation: never restore
    assert not ok and "missing" in why


def test_kernel_env_and_fingerprints(tmp_path):
    env = {"TORCH_CUDA_ARCH_LIST": "8.9", "SGLANG_JIT_CACHE_DIR": "/tmp/x", "SGLANG_CACHE_DIR": "/tmp/y",
           "SGLANG_ENABLE_FOO": "1", "HOME": "/root", "TRITON_CACHE_DIR": "/tmp/z", "FLASHINFER_JIT_VERBOSE": "0"}
    assert L.kernel_env(env) == {"FLASHINFER_JIT_VERBOSE": "0", "SGLANG_ENABLE_FOO": "1", "TORCH_CUDA_ARCH_LIST": "8.9"}
    pkg = tmp_path / "site" / "sgl_kernel"
    (pkg / "csrc").mkdir(parents=True)
    (pkg / "csrc" / "k.cu").write_text("__global__ void k(){}")
    (pkg / "__init__.py").write_text("x = 1")
    a = L.source_fingerprint([pkg])
    (pkg / "csrc" / "k.cu").write_text("__global__ void k(){ }")
    assert L.source_fingerprint([pkg]) != a
    (pkg / "data.bin").write_bytes(b"ignored")
    b = L.source_fingerprint([pkg])
    (pkg / "data.bin").write_bytes(b"changed")
    assert L.source_fingerprint([pkg]) == b                  # only source suffixes count
    wh = tmp_path / "wh"
    wh.mkdir()
    (wh / "b-1.whl").write_bytes(b"12")
    (wh / "a-1.whl").write_bytes(b"1")
    assert L.wheel_inventory(wh) == [["a-1.whl", 1], ["b-1.whl", 2]]


def test_jit_cache_rejects_traversal(tmp_path):
    import io
    tar = tmp_path / "evil.tar"
    man = full_manifest()
    with tarfile.open(tar, "w") as tf:
        for name, data in (("MANIFEST.json", json.dumps(man, sort_keys=True).encode()), ("cache/../../evil.txt", b"x"),
                           ("cache/ok.txt", b"y")):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    ok, _ = L.restore_jit_cache(tar, tmp_path / "new", man)
    assert ok and (tmp_path / "new" / "ok.txt").exists()
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path.parent / "evil.txt").exists()


def test_jit_env_sets_jit_dir_separately():
    e = L.jit_cache_env("/tmp/e11-cache")
    assert e["SGLANG_JIT_CACHE_DIR"] != e["SGLANG_CACHE_DIR"] and "sglang_jit" in e["SGLANG_JIT_CACHE_DIR"]


def test_known_answer():
    ref = {"tokens": [1, 2, 3], "logprobs": [-0.1, -0.2, -0.3]}
    assert L.known_answer_ok([1, 2, 3], [-0.11, -0.2, -0.31], ref)
    assert not L.known_answer_ok([1, 2, 4], [-0.1, -0.2, -0.3], ref)
    assert not L.known_answer_ok([1, 2, 3], [-0.1, -0.5, -0.3], ref)


# ------------------------------------------------------------------------------------------- env facts
def test_env_report_and_probe_do_not_raise(model):
    rep = L.env_report([str(model), "/kaggle/input"])
    assert rep["nproc"] and str(model) in rep["paths"]
    pr = L.read_probe(model, nbytes=2 * MIB)
    assert pr["bytes"] == 2 * MIB and pr["file"] == "model.safetensors"


# ------------------------------------------------------------------------------------------- /proc parsing (Linux)
def stat_line(pid, comm, ppid, minflt=0, majflt=0):
    # /proc/<pid>/stat: pid (comm) state ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt utime ...
    return f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1 4194560 {minflt} 0 {majflt} 0 5 3 0 0 20 0 1 0\n"


@pytest.fixture
def fake_proc(tmp_path, monkeypatch):
    """A /proc tree as on Kaggle: server 100 -> scheduler 101 (comm with spaces and ')') -> TP worker 102; 200 unrelated."""
    pr = tmp_path / "proc"
    tree = {100: (1, "python3"), 101: (100, "sglang::scheduler (TP0)"), 102: (101, "sglang::sched) x"), 200: (1, "bash")}
    rchar = {100: 1_000, 101: 2_000_000, 102: 3_000_000_000, 200: 9}
    for pid, (ppid, comm) in tree.items():
        d = pr / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(stat_line(pid, comm, ppid, minflt=10, majflt=1))
        (d / "io").write_text(f"rchar: {rchar[pid]}\nwchar: 5\nsyscr: 1\n")
    (pr / "self").mkdir()
    (pr / "mounts").write_text(
        "overlay / overlay rw 0 0\n"
        "tmpfs /dev/shm tmpfs rw,size=99g 0 0\n"
        "gcsfuse /kaggle/input/models fuse.gcsfuse ro 0 0\n"
        "10.0.0.2:/data /kaggle/input nfs4 ro 0 0\n"
        "/dev/sdb1 /kaggle/working ext4 rw 0 0\n")
    (pr / "meminfo").write_text("MemTotal:       197000000 kB\nMemFree:        150000000 kB\nMemAvailable:   180000000 kB\n")
    (pr / "cpuinfo").write_text("processor\t: 0\nmodel name\t: AMD EPYC 7B13\n")
    monkeypatch.setattr(L, "PROC", str(pr))
    return pr


def test_proc_tree_and_rchar(fake_proc):
    assert sorted(L._proc_tree(100)) == [100, 101, 102]
    assert L._rchar(L._proc_tree(100)) == 3_002_001_000
    assert L._rchar([12345]) == 0                       # vanished process is skipped
    assert L._faults([100, 101, 102]) == 33               # (10 minflt + 1 majflt) x 3
    assert L._faults([200, 12345]) == 11


def test_mounts_longest_prefix(fake_proc, monkeypatch):
    # keep the path as given on every host. Never fake os.name: on Linux it turns Path() into WindowsPath and the
    # code under test returns None (Kaggle CPU harness 2026-10-23)
    monkeypatch.setattr(L.os.path, "realpath", lambda q: str(q))
    assert L._mount_of("/kaggle/input/models/foysal/q/1") == ("/kaggle/input/models", "fuse.gcsfuse")
    assert L._mount_of("/kaggle/input/datasets/foysal/wh") == ("/kaggle/input", "nfs4")
    assert L._mount_of("/kaggle/input-other") == ("/", "overlay")
    assert L._mount_of("/dev/shm/e14-weights") == ("/dev/shm", "tmpfs")


def test_meminfo_and_env_report(fake_proc, monkeypatch):
    monkeypatch.setattr(L.os.path, "realpath", lambda q: str(q))
    assert L._mem_available() == 180000000 * 1024
    rep = L.env_report(["/kaggle/input/models/x"])
    assert rep["mem_gb"] == {"total": 197.0, "available": 180.0} and rep["cpu_model"] == "AMD EPYC 7B13"
    assert rep["paths"]["/kaggle/input/models/x"]["fstype"] == "fuse.gcsfuse"


def test_monitor_counts_rchar_as_load_progress(tmp_path, fake_proc):
    """Silent log while the TP worker reads weights (rchar grows): phase 1 must not stall."""
    log = tmp_path / "s.log"
    log.write_text("start\n")
    io = fake_proc / "102" / "io"

    def grow():
        for i in range(30):
            time.sleep(0.1)
            _put(io, f"rchar: {3_000_000_000 + (i + 1) * 64 * MIB}\n")
    th = threading.Thread(target=grow, daemon=True)
    th.start()
    # ready = when the writer is done (>= 3 s, event-based: file writes on this Windows host take tens of ms) > stall
    # limit 2 s, so without counting rchar the monitor must stall first; 2 s = 20x the nominal 0.1 s gap
    ok, why, st = mon(FakeProc(pid=100), log, load_stall_sec=2.0, compile_stall_sec=5).wait(
        lambda: not th.is_alive(), time.perf_counter() + 30)
    assert ok, why
    assert st["rchar_mb"] > 3000


def test_parallel_readers_beat_one_stream(model, tmp_path, monkeypatch):
    """GCSFUSE-like storage: each stream is slow (0.3 s per 1 MiB range) but streams add up. 10 workers vs 1 (15 jobs
    = 2 waves + the 0.2 s poll of _run_ranged, so the ideal ratio is ~4-6x; 2.5x proves the ranges run in parallel)."""
    real = L._copy_range

    def per_stream_limited(*a, **k):
        time.sleep(0.3)
        return real(*a, **k)

    monkeypatch.setattr(L, "_copy_range", per_stream_limited)
    _, s1 = L.parallel_copy(model, tmp_path / "d1", workers=1, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    _, s10 = L.parallel_copy(model, tmp_path / "d10", workers=10, chunk=1 * MIB, small_bytes=64 * 1024, log=None,
                             levels=(10,))                 # fixed level: this test is about parallelism, not the ramp
    # copy_sec, not sec: the test is about parallel copying; sec also holds the verify pass (once 1.8 s of 3.6)
    assert s1["copy_sec"] >= 2.5 * s10["copy_sec"], (s1, s10)


def test_hung_reader_does_not_block_interpreter_exit(tmp_path):
    """A read stuck forever must not keep the kernel alive at shutdown (daemon threads, never joined)."""
    import subprocess
    import sys
    m = tmp_path / "m"
    m.mkdir()
    make_safetensors(m / "model.safetensors", 4 * MIB)
    code = f"""
import sys, threading, time
sys.path.insert(0, {str(Path(L.__file__).parent)!r})
import e14_loadpath as L
L._copy_range = lambda *a, **k: threading.Event().wait()
try:
    L.parallel_copy({str(m)!r}, {str(tmp_path / 'dst')!r}, workers=4, chunk=1 << 20, small_bytes=1, stall_sec=0.5, log=None)
except L.CopyStalled:
    print("stalled")
"""
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "stalled" in r.stdout, r.stderr
    assert time.monotonic() - t0 < 10


def test_staging_thread(model, tmp_path):
    d = tmp_path / "shm"
    d.mkdir()
    st = L.Staging([model], workers=4, log=None, dest_candidates=(str(d),))
    st.start()
    st.join(30)
    out, rep = st.result
    assert Path(out[str(model)]).name == L.staged_name(model, "m0") and rep["copies"] and rep["error"] is None


def test_monitor_counts_mmap_faults_as_load_progress(tmp_path, fake_proc):
    """SGLang loads safetensors through mmap: rchar stays flat, the log is silent (one 18.6 GB shard = tqdm 0/1 -> 1/1),
    only page faults grow. That must count as progress (the 37-min slow load the fixed timeout would have killed)."""
    log = tmp_path / "s.log"
    log.write_text("Load weight begin.\n")
    st = fake_proc / "102" / "stat"

    def fault():
        for i in range(30):
            time.sleep(0.1)
            _put(st, stat_line(102, "sglang::sched) x", 101, minflt=10 + (i + 1) * 4096))
    th = threading.Thread(target=fault, daemon=True)
    th.start()
    # ready = writer done (>= 3 s) > stall limit 2 s: without counting faults the monitor must stall first
    ok, why, s = mon(FakeProc(pid=100), log, load_stall_sec=2.0, compile_stall_sec=5).wait(
        lambda: not th.is_alive(), time.perf_counter() + 30)
    assert ok, why
    assert s["faults"] > 40000


def test_monitor_flat_counters_still_stall(tmp_path, fake_proc):
    """Counters present but flat (hung in NCCL init): stall."""
    log = tmp_path / "s.log"
    log.write_text("x\n")
    ok, why, _ = mon(FakeProc(pid=100), log, load_stall_sec=0.5, compile_stall_sec=5).wait(lambda: False,
                                                                                            time.perf_counter() + 10)
    assert not ok and "phase 1" in why


# =========================================================================== rev 2 (GPT notes 08/09, Notes/10)
class _Proxy:
    """Wraps a real file object; subclasses override read/write to inject faults."""

    def __init__(self, f):
        self.f = f

    def __getattr__(self, k):
        return getattr(self.f, k)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.f.close()


def inject_open(monkeypatch, wrapper, when):
    """Make the module's open() return wrapper(real file) when when(path, mode) holds."""
    import builtins
    real = builtins.open

    def o(p, mode="r", *a, **k):
        f = real(p, mode, *a, **k)
        return wrapper(f) if when(str(p), mode) else f

    monkeypatch.setattr(L, "open", o, raising=False)


def no_dirs_left(root, name="model"):
    return not (Path(root) / name).exists() and not (Path(root) / (name + ".partial")).exists()


def test_short_writes_still_give_an_exact_copy(model, tmp_path, monkeypatch):
    """GPT gap 2 item 2: a raw write may write fewer bytes than asked. Rev 1 ignored the count (bug)."""
    class ShortWriter(_Proxy):
        def write(self, b):
            return self.f.write(bytes(memoryview(b)[:max(1, len(b) // 3)]))

    inject_open(monkeypatch, ShortWriter, lambda p, m: m == "r+b")
    dst, _ = L.parallel_copy(model, tmp_path / "dst", workers=4, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    assert sha(dst / "model.safetensors") == sha(model / "model.safetensors")
    assert sha(dst / "draft.safetensors") == sha(model / "draft.safetensors")


def test_write_all_zero_write_raises():
    class Zero:
        def write(self, b):
            return 0
    with pytest.raises(IOError):
        L._write_all(Zero(), b"abc")


def test_zero_byte_read_once_is_retried(model, tmp_path, monkeypatch):
    state = {"left": 1}

    class ZeroOnce(_Proxy):
        def read(self, n=-1):
            if state["left"] > 0:
                state["left"] -= 1
                return b""
            return self.f.read(n)

    monkeypatch.setattr(L, "_backoff", lambda a: 0)
    inject_open(monkeypatch, ZeroOnce, lambda p, m: m == "rb" and p.endswith("model.safetensors") and "src" in p)
    dst, st = L.parallel_copy(model, tmp_path / "dst", workers=2, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    assert sha(dst / "model.safetensors") == sha(model / "model.safetensors") and st["retries"] == 1


def test_zero_byte_reads_forever_leave_no_directory(model, tmp_path, monkeypatch):
    class Zero(_Proxy):
        def read(self, n=-1):
            return b""

    monkeypatch.setattr(L, "_backoff", lambda a: 0)
    inject_open(monkeypatch, Zero, lambda p, m: m == "rb" and p.endswith("model.safetensors") and "src" in p)
    with pytest.raises(IOError, match="short read"):
        L.parallel_copy(model, tmp_path / "dst", workers=2, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    assert no_dirs_left(tmp_path / "dst")


def test_enospc_leaves_no_directory_and_stage_fails_open(model, tmp_path, monkeypatch):
    import errno

    class Full(_Proxy):
        def write(self, b):
            raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(L, "_backoff", lambda a: 0)
    inject_open(monkeypatch, Full, lambda p, m: m == "r+b")
    with pytest.raises(OSError):
        L.parallel_copy(model, tmp_path / "dst", workers=2, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    assert no_dirs_left(tmp_path / "dst")
    d = tmp_path / "shm"
    d.mkdir()
    out, rep = L.stage_models([model], workers=2, log=None, dest_candidates=(str(d),), chunk=1 * MIB,
                              small_bytes=64 * 1024)
    assert out[str(model)] == str(model) and "No space" in rep["error"] and rep["warm"][str(model)]["bytes"] > 0
    assert no_dirs_left(d / "e14-weights")


def test_one_byte_flip_is_caught_by_range_hash(model, tmp_path, monkeypatch):
    """A byte changed in the destination after the write (any write-path bug) must fail verification."""
    real = L._copy_range

    def flip(src, dst, off, n, prog, retries=4):
        real(src, dst, off, n, prog, retries)
        if Path(dst).name == "model.safetensors" and off == 4 * MIB:
            with open(dst, "r+b") as f:
                f.seek(off + 100)
                b = f.read(1)
                f.seek(off + 100)
                f.write(bytes([b[0] ^ 1]))

    monkeypatch.setattr(L, "_copy_range", flip)
    with pytest.raises(IOError, match="range hash mismatch: model.safetensors at offset 4194304"):
        L.parallel_copy(model, tmp_path / "dst", workers=4, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    assert no_dirs_left(tmp_path / "dst")


def test_worker_exception_leaves_no_directory(model, tmp_path, monkeypatch):
    real = L._copy_range

    def bug(src, dst, off, n, prog, retries=4):
        if off == 3 * MIB:
            raise RuntimeError("bug in a worker")
        return real(src, dst, off, n, prog, retries)

    monkeypatch.setattr(L, "_copy_range", bug)
    with pytest.raises(RuntimeError, match="bug in a worker"):
        L.parallel_copy(model, tmp_path / "dst", workers=4, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    assert no_dirs_left(tmp_path / "dst")


def test_stale_partial_and_final_are_replaced(model, tmp_path):
    root = tmp_path / "dst"
    nm = L.staged_name(model)
    for d in (nm, nm + ".partial"):
        (root / d).mkdir(parents=True)
        (root / d / "model.safetensors").write_bytes(b"junk")
        (root / d / "junk.txt").write_text("old")
    dst, _ = L.parallel_copy(model, root, workers=4, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    assert sha(dst / "model.safetensors") == sha(model / "model.safetensors")
    assert not (dst / "junk.txt").exists() and not (root / (nm + ".partial")).exists()


def test_sha256_recorded_and_expected_checked(model, tmp_path):
    dst, st = L.parallel_copy(model, tmp_path / "d1", workers=4, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    assert st["sha256"]["model.safetensors"] == sha(model / "model.safetensors")
    with pytest.raises(IOError, match="sha256 mismatch: model.safetensors"):
        L.parallel_copy(model, tmp_path / "d2", workers=4, chunk=1 * MIB, small_bytes=64 * 1024, log=None,
                        expected_sha256={"model.safetensors": "0" * 64})
    assert no_dirs_left(tmp_path / "d2")
    L.parallel_copy(model, tmp_path / "d3", workers=4, chunk=1 * MIB, small_bytes=64 * 1024, log=None,
                    expected_sha256={"model.safetensors": st["sha256"]["model.safetensors"]})


def test_ranges_cover_exactly():
    for size, chunk in ((0, 4), (1, 4), (8, 4), (9, 4), (64 * MIB + 1, MIB)):
        rs = L._ranges(size, chunk)
        assert sum(n for _, n in rs) == size and rs[0][0] == 0


def make_multi(path, tensors, buf_len, fill=b"\x07"):
    """tensors: [(name, dtype, shape, begin, end)]; header as given, buffer of buf_len bytes."""
    hdr = json.dumps({n: {"dtype": d, "shape": s, "data_offsets": [b, e]} for n, d, s, b, e in tensors}).encode()
    hdr += b" " * (-len(hdr) % 8)
    Path(path).write_bytes(struct.pack("<Q", len(hdr)) + hdr + fill * buf_len)


def test_safetensors_layout_rules(tmp_path):
    p = tmp_path / "x.safetensors"
    # header order differs from offset order (GPT: "last JSON entry" is unsafe) -> still fine
    make_multi(p, [("b", "F16", [4], 8, 16), ("a", "F32", [2], 0, 8)], 16)
    assert L.safetensors_layout(p) == (True, "ok")
    make_multi(p, [("a", "F32", [2], 0, 8), ("b", "F16", [4], 10, 18)], 18)
    assert "hole" in L.safetensors_layout(p)[1]
    make_multi(p, [("a", "F32", [2], 0, 8), ("b", "F16", [4], 6, 14)], 14)
    assert "overlap" in L.safetensors_layout(p)[1]
    make_multi(p, [("a", "F32", [3], 0, 8)], 8)
    assert "bytes for F32" in L.safetensors_layout(p)[1]
    make_multi(p, [("a", "U8", [8], 2, 10)], 10)
    assert "first tensor starts" in L.safetensors_layout(p)[1]
    make_multi(p, [("a", "U8", [8], 0, 8)], 12)
    assert "buffer is 12" in L.safetensors_layout(p)[1]
    p.write_bytes(b"\x01\x02")
    assert not L.safetensors_layout(p)[0]
    p.write_bytes(struct.pack("<Q", 5) + b"{bad}")
    assert not L.safetensors_layout(p)[0]


def test_layout_problem_of_the_source_is_accepted_as_published(tmp_path):
    """If the published file itself breaks the strict rule, the copy is not rejected for it (it is reported)."""
    m = tmp_path / "src" / "model"
    m.mkdir(parents=True)
    make_multi(m / "w.safetensors", [("a", "U8", [3 * MIB], 0, 3 * MIB), ("b", "U8", [8], 3 * MIB + 8, 3 * MIB + 16)],
               3 * MIB + 16)
    dst, st = L.parallel_copy(m, tmp_path / "dst", workers=2, chunk=1 * MIB, small_bytes=64 * 1024, log=None)
    assert st["layout_as_published"] and "hole" in st["layout_as_published"][0]


# ------------------------------------------------------------------------------------------- worker ramp
def _synthetic(prog, n_jobs, mib=1, delay=0.04, sem=None, live=None):
    def job():
        if live is not None:                   # counted before the storage limit: this is the worker concurrency
            with live["lock"]:
                live["now"] += 1
                live["max"] = max(live["max"], live["now"])
        try:
            if sem:
                sem.acquire()
            try:
                time.sleep(delay)
                prog.add(mib * MIB)
            finally:
                if sem:
                    sem.release()
        finally:
            if live is not None:
                with live["lock"]:
                    live["now"] -= 1
    return [job] * n_jobs


def test_ramp_goes_up_while_throughput_scales():
    prog = L._Progress(400 * MIB)
    r = L._run_ranged(_synthetic(prog, 400), prog, 8, 10, None, 1e9, None, levels=(2, 4, 8), ramp_bytes=12 * MIB)
    assert [lv for lv, _ in r["levels"]][:2] == [2, 4] and r["final_workers"] == 8, r


def test_ramp_falls_back_when_storage_saturates():
    """Storage that serves only 2 streams at a time: going 2 -> 4 brings no gain, so it returns to 2 and stays.
    Active workers never exceed the level that was allowed."""
    prog = L._Progress(300 * MIB)
    live = {"lock": threading.Lock(), "now": 0, "max": 0}
    sem = threading.Semaphore(2)
    # 40 MiB windows (~0.8 s at 2 streams): 12 MiB (~0.24 s) against the 0.2 s poll let jitter fake a > 15 % gain
    r = L._run_ranged(_synthetic(prog, 300, sem=sem, live=live), prog, 8, 10, None, 1e9, None, levels=(2, 4, 8),
                      ramp_bytes=40 * MIB)
    assert r["final_workers"] == 2 and [lv for lv, _ in r["levels"]] == [2, 4], r
    assert live["max"] <= 4


def test_ramp_backs_off_on_retry():
    """Level 2 is measured at the first 0.2 s poll (~10 MiB >= 4 MiB) -> level 4; a transient error fires at 14 MiB,
    i.e. inside the level-4 window -> back to 2 and stay."""
    prog = L._Progress(300 * MIB)
    fired = {"x": False}
    lock = threading.Lock()

    def job():
        with lock:
            if not fired["x"] and prog.done >= 14 * MIB:
                fired["x"] = True
                prog.retry()
        time.sleep(0.04)
        prog.add(MIB)

    r = L._run_ranged([job] * 300, prog, 8, 10, None, 1e9, None, levels=(2, 4, 8), ramp_bytes=4 * MIB)
    assert r["final_workers"] == 2 and r["levels"][-1] == (4, None), r


# ------------------------------------------------------------------------------------------- cgroup, state, switch
@pytest.fixture
def fake_cgroup(tmp_path, monkeypatch):
    cg = tmp_path / "cgroup"
    cg.mkdir()
    monkeypatch.setattr(L, "CGROUP", str(cg))
    return cg


def test_cgroup_v2_v1_and_headroom(fake_cgroup, fake_proc):
    (fake_cgroup / "memory.max").write_text("8000000000\n")
    (fake_cgroup / "memory.current").write_text("3000000000\n")
    assert L.cgroup_memory() == (8_000_000_000, 3_000_000_000)
    assert L.host_headroom() == 5_000_000_000            # smaller than MemAvailable (184 GB in the fake /proc)
    (fake_cgroup / "memory.max").write_text("max\n")
    assert L.cgroup_memory() == (None, 3_000_000_000) and L.host_headroom() == 180000000 * 1024
    for f in ("memory.max", "memory.current"):
        (fake_cgroup / f).unlink()
    (fake_cgroup / "memory").mkdir()
    (fake_cgroup / "memory" / "memory.limit_in_bytes").write_text(str(1 << 62))
    (fake_cgroup / "memory" / "memory.usage_in_bytes").write_text("100")
    assert L.cgroup_memory() == (None, 100)
    (fake_cgroup / "memory" / "memory.limit_in_bytes").write_text("40000000000")
    assert L.cgroup_memory() == (40_000_000_000, 100)


def test_dev_shm_skipped_when_cgroup_is_tight(fake_cgroup, fake_proc, monkeypatch):
    (fake_cgroup / "memory.max").write_text(str(40 * (1 << 30)))
    (fake_cgroup / "memory.current").write_text(str(2 * (1 << 30)))
    monkeypatch.setattr(L.Path, "is_dir", lambda self: True)

    class U:
        free = 100 * (1 << 30)

    monkeypatch.setattr(L.shutil, "disk_usage", lambda d: U)
    assert L.choose_dest(int(23.3e9), ("/dev/shm", "/tmp")) == "/tmp"   # 38 GiB headroom < 23.3 GB + 24 GiB


def test_state_record_written_at_every_change(tmp_path, model):
    p = tmp_path / "e14_state.json"
    st = L.E14State(p)
    d = tmp_path / "shm"
    d.mkdir()
    L.stage_models([model], workers=2, log=None, dest_candidates=(str(d),), state=st, chunk=1 * MIB,
                   small_bytes=64 * 1024)
    st.enter("LAUNCH", deadline=time.perf_counter() + 60)
    st.end("ready")
    rec = json.loads(p.read_text())
    assert [x["phase"] for x in rec["phases"]] == ["PREPARE", "COPY", "VALIDATE", "LAUNCH"]
    assert rec["phases"][1]["validated"] is True and rec["terminal_reason"] == "ready"
    assert 0 < rec["phases"][3]["deadline_in"] <= 60


def test_off_switch(model, tmp_path, monkeypatch):
    monkeypatch.setenv("E14_LOADPATH", "0")
    d = tmp_path / "shm"
    d.mkdir()
    out, rep = L.stage_models([model], log=None, dest_candidates=(str(d),))
    assert rep == {"disabled": True} and out[str(model)] == str(model) and not any(d.iterdir())


def test_copied_model_is_not_read_again_after_a_later_failure(model, tmp_path, monkeypatch):
    second = tmp_path / "src" / "draft"
    second.mkdir()
    make_safetensors(second / "d.safetensors", 2 * MIB, seed=3)
    real = L.parallel_copy
    calls = []

    def first_ok_then_fail(m, *a, **k):
        calls.append(Path(m).name)
        if Path(m).name == "draft":
            raise L.CopyStalled("stalled on the draft")
        return real(m, *a, **k)

    monkeypatch.setattr(L, "parallel_copy", first_ok_then_fail)
    d = tmp_path / "shm"
    d.mkdir()
    out, rep = L.stage_models([model, second], workers=2, log=None, dest_candidates=(str(d),), chunk=1 * MIB,
                              small_bytes=64 * 1024)
    assert out[str(model)] == str(d / "e14-weights" / L.staged_name(model, "m0")) and out[str(second)] == str(second)
    assert list(rep["warm"]) == [str(second)] and calls == ["model", "draft"]


def test_stage_models_same_basename_never_collide(tmp_path):
    """Kaggle model dirs both end in "/1" (…/w4a16-g128/1, …/bf16/1). GPT audit 2 P0: with dst/<basename> the draft
    copy deleted the target copy and both paths pointed at the draft. Each source must keep its own verified dir."""
    t = tmp_path / "in" / "qwen3-8-27b-int4-redhat" / "pytorch" / "w4a16-g128" / "1"
    dr = tmp_path / "in" / "qwen3-8-27b-dflash2" / "pytorch" / "bf16" / "1"
    for p, seed, txt in ((t, 1, "target"), (dr, 2, "draft")):
        p.mkdir(parents=True)
        make_safetensors(p / "model.safetensors", 3 * MIB, seed=seed)
        (p / "config.json").write_text(json.dumps({"m": txt}))
    want = {str(t): sha(t / "model.safetensors"), str(dr): sha(dr / "model.safetensors")}
    d = tmp_path / "shm"
    d.mkdir()
    out, rep = L.stage_models([t, dr], workers=2, log=None, dest_candidates=(str(d),), chunk=1 * MIB,
                              small_bytes=64 * 1024)
    assert rep["error"] is None and out[str(t)] != out[str(dr)], out
    for src, staged in out.items():
        assert sha(Path(staged) / "model.safetensors") == want[src]
    assert json.loads((Path(out[str(t)]) / "config.json").read_text())["m"] == "target"
    assert json.loads((Path(out[str(dr)]) / "config.json").read_text())["m"] == "draft"


def test_stage_models_refuses_one_source_twice(model, tmp_path):
    d = tmp_path / "shm"
    d.mkdir()
    out, rep = L.stage_models([model, model], workers=2, log=None, dest_candidates=(str(d),), chunk=1 * MIB,
                              small_bytes=64 * 1024)
    assert rep["error"] and "share a destination" in rep["error"] and out[str(model)] == str(model)
