"""Run model-written `transform` programs in isolated subprocesses.

Each job is (code, inputs). The program runs in a fresh interpreter with a
wall-clock timeout (and, on Linux, an address-space cap), so an infinite loop,
a crash or a memory blow-up costs one job, never the run. Anything the program
prints is swallowed; the result travels back as one marked JSON line.

Per input the result is {"output": grid} or {"error": message}. An output is
only accepted if it is a 2D rectangular grid of ints 0-9, 1..30 on each side.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

MARK = "@@ARC_RESULT@@"

RUNNER = r'''
try:                                    # v1.4: memory cap set by the child itself (no preexec_fn)
    import os as _q27_os, resource as _q27_res
    _q27_lim = int(_q27_os.environ.get("Q27_CHILD_MAX_AS", "0"))
    if _q27_lim > 0:
        _q27_res.setrlimit(_q27_res.RLIMIT_AS, (_q27_lim, _q27_lim))
except Exception:
    pass
# v1.6 guard: a PEP 578 audit hook (cannot be removed by the program). v1.7 (GPT final audit): installation failure or
# an inactive hook now ABORTS the child (it was fail-open), native routes are narrowed (ctypes symbols for file /
# process functions, cffi, _posixsubprocess.fork_exec), sqlite3/ctypes.dlopen paths are checked. This is a best-effort
# guard for model-written code, NOT a security sandbox (Python documents audit hooks as unsuitable for that).
try:
    import sys as _q27_sys, os as _q27_os2, re as _q27_re
    _q27_deny = tuple(p for p in _q27_os2.environ.get("Q27_DENY_PATHS", "/kaggle/input|solution").lower().split("|") if p)
    _q27_busy = []
    _q27_native = _q27_re.compile(
        r"^_*(open|fopen|freopen|fdopen|wopen|wfopen|creat|read|pread|fread|fgets|fgetc|getline|getdelim|fscanf|mmap|"
        r"syscall|system|popen|wpopen|exec|fexecve|fork|vfork|clone|posix_spawn|dlopen|dlmopen|dlsym|sendfile|splice|"
        r"copy_file_range|opendir|fdopendir|scandir|readlink|createfile|createprocess|winexec|shellexecute|"
        r"loadlibrary|ntcreatefile|ntopenfile|ntreadfile)", _q27_re.I)
    def _q27_hook(event, args, _deny=_q27_deny, _busy=_q27_busy, _native=_q27_native):
        if event in ("ctypes.dlsym", "ctypes.dlsym/handle"):
            _n = str(args[1]) if len(args) > 1 else ""
            if _native.match(_n):
                raise PermissionError("[sandbox] not allowed: native " + _n)
            return
        if event == "import" and args and str(args[0]).split(".")[0] in ("cffi", "_cffi_backend"):
            raise ImportError("[sandbox] not allowed: " + str(args[0]))
        if event in ("open", "os.listdir", "os.scandir", "os.chdir", "os.rename", "os.remove", "shutil.copyfile",
                     "shutil.copytree", "glob.glob", "os.walk", "os.symlink", "os.link", "sqlite3.connect",
                     "ctypes.dlopen"):
            if _busy:
                return
            _busy.append(1)
            try:                        # the path as given AND resolved (relative paths after chdir, symlinks)
                _ps = []
                for _a in args[:2]:
                    if isinstance(_a, (str, bytes)) or hasattr(_a, "__fspath__"):
                        _ps.append(str(_a))
                        try:
                            _ps.append(str(_q27_os2.path.realpath(_a)))
                        except Exception:
                            pass
            finally:
                _busy.pop()
            for _p in _ps:
                if any(d in _p.lower().replace("\\", "/") for d in _deny):
                    raise PermissionError("[sandbox] access denied: " + _p)
        elif event in ("subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.fork",
                       "socket.connect"):
            raise PermissionError("[sandbox] not allowed: " + event)
    _q27_sys.addaudithook(_q27_hook)
    try:                                # scipy imports subprocess, so the module stays; its raw fork_exec does not
        import _posixsubprocess as _q27_ps
        def _q27_no_fork(*a, **k):
            raise PermissionError("[sandbox] not allowed: fork_exec")
        _q27_ps.fork_exec = _q27_no_fork
    except ImportError:
        pass
    try:                                # canary: the hook must really deny a forbidden path, else stop here
        open("/" + _q27_deny[0].strip("/") + "/__q27_canary__")
        _q27_ok = False
    except PermissionError:
        _q27_ok = True
    except OSError:
        _q27_ok = False
    if not _q27_ok:
        raise RuntimeError("audit hook inactive")
except BaseException as _q27_e:
    import sys as _q27_sys2
    _q27_sys2.stderr.write("[sandbox] guard failed, refusing to run: %r\n" % (_q27_e,))
    _q27_sys2.stderr.flush()
    _q27_sys2.exit(3)
import sys, json, io, contextlib, copy, traceback
MARK = "@@ARC_RESULT@@"
job = json.loads(sys.stdin.read())

def clean(res):
    try:
        import numpy as np
        if isinstance(res, np.ndarray):
            res = res.tolist()
    except Exception:
        pass
    if isinstance(res, tuple):
        res = list(res)
    if not isinstance(res, list) or not res:
        return None, "output is not a non-empty list of rows"
    rows = []
    for row in res:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)) or not row:
            return None, "output rows must be non-empty lists"
        out = []
        for v in row:
            try:
                iv = int(v)
            except Exception:
                return None, f"non-integer cell {v!r}"
            if iv != v or not 0 <= iv <= 9:
                return None, f"cell value {v!r} outside 0-9"
            out.append(iv)
        rows.append(out)
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        return None, "output is not rectangular"
    if not (1 <= len(rows) <= 30 and 1 <= w <= 30):
        return None, f"output size {len(rows)}x{w} outside 1..30"
    return rows, None

results = []
sink = io.StringIO()
ns = {"__name__": "__arc_program__"}
try:
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(job["code"], "<program>", "exec"), ns)
    fn = ns.get("transform")
    if not callable(fn):
        raise NameError("no function named transform")
except Exception as e:
    msg = "".join(traceback.format_exception_only(type(e), e)).strip()
    results = [{"error": "program failed to load: " + msg[-400:]} for _ in job["inputs"]]
else:
    for grid in job["inputs"]:
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                res = fn(copy.deepcopy(grid))
            rows, err = clean(res)
            results.append({"output": rows} if err is None else {"error": err})
        except Exception as e:
            tb = traceback.format_exc().strip().splitlines()
            results.append({"error": " | ".join(tb[-3:])[-400:]})
sys.stdout.write(MARK + json.dumps(results) + "\n")
sys.stdout.flush()
'''


def _limits(mem_bytes):
    def apply():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass
    return apply if os.name == "posix" else None


def run_one(code, inputs, timeout=10.0, mem_bytes=2 << 30):
    """Run one program on a list of input grids; never raises."""
    try:
        p = subprocess.run(
            [sys.executable, "-c", RUNNER],
            input=json.dumps({"code": code, "inputs": inputs}),
            capture_output=True, text=True, timeout=timeout,
            env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                     MKL_NUM_THREADS="1", PYTHONHASHSEED="0", Q27_CHILD_MAX_AS=str(mem_bytes)))   # v1.4
    except subprocess.TimeoutExpired:
        return [{"error": f"timed out after {timeout:.0f}s"} for _ in inputs]
    for line in reversed(p.stdout.splitlines()):
        if line.startswith(MARK):
            try:
                res = json.loads(line[len(MARK):])
                if len(res) == len(inputs):
                    return res
            except Exception:
                break
    tail = (p.stderr or "").strip().splitlines()[-2:]
    why = " | ".join(tail)[-300:] or f"exit code {p.returncode}"
    return [{"error": "program crashed: " + why} for _ in inputs]


def run_many(jobs, timeout=10.0, workers=None, mem_bytes=2 << 30):
    """jobs: list of (code, inputs). Returns results in the same order."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, c, i, timeout, mem_bytes) for c, i in jobs]
        return [f.result() for f in futs]
